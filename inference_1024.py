import argparse
import cv2
import glob
import numpy as np
import os
import torch
import yaml

from basicsr.archs.vqgan_arch import VQAutoEncoder
from basicsr.archs.codeformer_arch import CodeFormer
from torchvision.transforms.functional import normalize
from basicsr.utils import img2tensor, tensor2img
from basicsr.data.ffhq_blind_dataset import FFHQBlindDataset
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


def calc_mean_std(feat, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.

    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    """Adaptive instance normalization.

    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.

    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def to_tensor(img, is_train = False):
    '''
    convert numpy array to tensor
    for 2-D array [H , W] -> (1, 1, H , W)
    for 3-D array [H, W, 3] -> (1, 3, H, W)
    '''
    if is_train == False:
        if img.ndim == 2:
            img = np.expand_dims(img, 0)
            img = np.expand_dims(img, 0)
        elif img.ndim == 3:
            if img.shape[2] == 3 or img.shape[2] == 2:  # for [H, W, 3] and [H, W, 2]
                img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, 0)

        assert(img.ndim == 4)
        img = torch.from_numpy(img).float()

    else:
        if img.ndim == 2:
            img = np.expand_dims(img, 0)
        elif img.ndim == 3:
            if img.shape[2] == 3:  # for [H, W, 3] only
                img = np.transpose(img, (2, 0, 1))

        assert(img.ndim == 3)
        img = torch.from_numpy(img).float()

    return img

def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        ndarray = tensor.cpu().detach().numpy()
        ndarray = np.squeeze(ndarray)

        if ndarray.shape[0] == 3:   # for [3, H, W] only
            ndarray = np.transpose(ndarray, (1, 2, 0))

        out = ndarray.copy()
    else:
        out = tensor

    return out

def read_single_img(img_pth='', pth = 'tmp/in0.png', pth1 = 'tmp/in1.png', resize=True):
    # read img from
    # img = get_img('s3://ffhq/ffhq_imgs/ffhq_512/00000001.png')
    img = get_img(img_pth)
    # img = get_img('/mnt/lustre/sunjixiang1/code/CodeFormer/results/src_gd_1.0/cropped_faces/125405467-0378-0381_src_00.png')
    # img = get_img('/mnt/lustre/sunjixiang1/dataset/global_test/debug/125405467-0378-0381_src_local_UW_face.png')
    # img = get_img('/mnt/lustre/sunjixiang1/code/CodeFormer/inputs/cropped_faces/125405467-0378-0381_src.png')
    # img = get_img('/mnt/lustre/sunjixiang1/code/CodeFormer/inputs/ref_cropped/125405467-0378-0381_ref.png')
    # img = get_img('/mnt/lustre/sunjixiang1/code/CodeFormer/results/cropped_face_gd/125405467-0378-0381_src.png')
    
    if resize:
        img_in = cv2.resize(img, (512, 512), interpolation = cv2.INTER_LINEAR).astype(np.float32)
    else:
        img_in = img.astype(np.float32)

    cv2.imwrite(pth, img_in*255)

    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)

    

    img_in = to_tensor(img_in, is_train = False)

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    normalize(img_in, mean, std, inplace=True)

    img_in_data = to_numpy(img_in)
    img_in_data = img_in_data * 0.5 + 0.5
    cv2.imwrite(pth1, img_in_data*255)

    return img_in


def get_img(gt_path):
    from petrel_client.client import Client
    from basicsr.utils import imfrombytes
    conf_path = '~/petreloss.conf'
    file_client = Client(conf_path)
    # load gt image
    # match with codebook ckpt data
    img_bytes = file_client.get(gt_path)
    return imfrombytes(img_bytes, float32=True)


def test_vqgan():

    with open('options/VQGAN_512_ds32_nearest_stage1.yml', 'r') as f:
        cfg = yaml.safe_load(f)

    # model_path = '/mnt/lustre/leifei1/project/CodeFormer/experiments/20230503_195457_VQGAN-512-ds32-nearest-stage1/models/net_g_800000.pth'

    model_path = 'experiments/pretrained_models/vqgan/vqgan_code1024.pth'


    # load model
    cfg_vqgan = cfg['network_g']
    model_type = cfg_vqgan.pop('type')
    model = VQAutoEncoder(**cfg_vqgan)

    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    # for k, v in checkpoint['params_ema'].items():
    #     print(k)

    model.load_state_dict(checkpoint['params_ema'])

    model.eval()
    
    img_in = read_single_img(pth = 'tmp/tmp/in.png', pth1 = 'tmp/tmp/in1.png', resize=True)

    # img_in = cv2.resize(img_in, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    # img_in = read_from_dataloader()
    print('img_in: ', img_in.shape, img_in.mean())

    # inference
    out, codebook_loss, quant_stats = model(img_in)
    print('out raw info: ', out.shape, out.mean())

    out = out * 0.5 + 0.5
    out = tensor2img(out, rgb2bgr=True, out_type = np.float32)
    print('out info: ', out.shape, out.mean())

    cv2.imwrite('tmp/tmp/out.png', out * 255)    


def test_3():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # with open('options/CodeFormer_stage2.yml', 'r') as f:
    with open('options/CodeFormer_stage3.yml', 'r') as f:
        cfg = yaml.safe_load(f)
    model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230529_181513_CodeFormer_stage3/models/net_g_latest.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230525_175526_CodeFormer_stage2/models/net_g_latest.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230526_203531_CodeFormer_stage2/models/net_g_latest.pth'
    cfg_codeformer = cfg['network_g']
    model_type = cfg_codeformer.pop('type')
    model = CodeFormer(**cfg_codeformer)
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    # for k, v in checkpoint['params_ema'].items():
    #     print(k)

    model.load_state_dict(checkpoint['params_ema'])
    # model.fuse_generator_block = {}
    # model.connect_list = {}
    model.to(device)
    model.eval()
    img_1 = read_single_img(img_pth='s3://ffhq/ffhq_imgs/ffhq_512/00000001.png', pth='tmp/in1.png', pth1='tmp/in11.png')
    img_2 = read_single_img(img_pth='s3://ffhq/ffhq_imgs/ffhq_512/00000002.png', pth='tmp/in2.png', pth1='tmp/in21.png')

    # img_in = read_from_dataloader()
    print('img_in: ', img_1.shape, img_1.mean())
    img_1 = img_1.to(device)
    img_2 = img_2.to(device)

    # inference
    # out, logits, lq_feat = model(img_in, w=1.0, detach_16=True)
    out1, logits1, lq_feat1, quant_feat1 = proc(model, img_1, 'tmp/out1.png')
    out2, logits2, lq_feat2, quant_feat2 = proc(model, img_2, 'tmp/out2.png')
    print(out2==out1, logits2==logits1, lq_feat2==lq_feat1, quant_feat2==quant_feat1)
    print(np.sum(out2==out1), np.sum(logits2==logits1), np.sum(lq_feat2==lq_feat1), np.sum(quant_feat2==quant_feat1))

def proc(model, img, out_pth='tmp/out.png'):
    out, logits, lq_feat, quant_feat = forward(model, img, w=0.0, detach_16=True)
    print('logits shape:', logits.shape)
    print('lq shape:', lq_feat.shape)
    print('quant feat shape:', quant_feat.shape)
    # out, logits, lq_feat = model(img_in, w=1.0, adain=True)
    # out, codebook_loss, quant_stats = model(img_in)
    print('out raw info: ', out.shape, out.mean())


    out = out.detach().cpu()
    # out = tensor2img(out, rgb2bgr=True, min_max=(-1, 1))
    out = out * 0.5 + 0.5
    out = tensor2img(out, rgb2bgr=True, out_type = np.float32)

    print('out info: ', out.shape, out.mean())
    
    cv2.imwrite(out_pth, out * 255)
    return out, logits.detach().cpu().numpy(), lq_feat.detach().cpu().numpy(), quant_feat.detach().cpu().numpy()


def forward(self, x, w=0, detach_16=True, code_only=False, adain=False):
    # ################### Encoder #####################
    enc_feat_dict = {}
    out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_list]
    for i, block in enumerate(self.encoder.blocks):
        x = block(x) 
        if i in out_list:
            enc_feat_dict[str(x.shape[-1])] = x.clone()

    lq_feat = x
    # ################# Transformer ###################
    # quant_feat, codebook_loss, quant_stats = self.quantize(lq_feat)
    pos_emb = self.position_emb.unsqueeze(1).repeat(1,x.shape[0],1)
    # BCHW -> BC(HW) -> (HW)BC
    feat_emb = self.feat_emb(lq_feat.flatten(2).permute(2,0,1))
    query_emb = feat_emb
    # Transformer encoder
    for layer in self.ft_layers:
        query_emb = layer(query_emb, query_pos=pos_emb)

    # output logits
    logits = self.idx_pred_layer(query_emb) # (hw)bn
    logits = logits.permute(1,0,2) # (hw)bn -> b(hw)n

    if code_only: # for training stage II
        # logits doesn't need softmax before cross_entropy loss
        return logits, lq_feat

    # ################# Quantization ###################
    # if self.training:
    #     quant_feat = torch.einsum('btn,nc->btc', [soft_one_hot, self.quantize.embedding.weight])
    #     # b(hw)c -> bc(hw) -> bchw
    #     quant_feat = quant_feat.permute(0,2,1).view(lq_feat.shape)
    # ------------
    soft_one_hot = F.softmax(logits, dim=2)
    _, top_idx = torch.topk(soft_one_hot, 1, dim=2)
    quant_feat = self.quantize.get_codebook_feat(top_idx, shape=[x.shape[0],16,16,256])
    # preserve gradients
    # quant_feat = lq_feat + (quant_feat - lq_feat).detach()

    if detach_16:
        quant_feat = quant_feat.detach() # for training stage III
    if adain:
        quant_feat = adaptive_instance_normalization(quant_feat, lq_feat)

    # ################## Generator ####################
    x = quant_feat
    fuse_list = [self.fuse_generator_block[f_size] for f_size in self.connect_list]

    for i, block in enumerate(self.generator.blocks):
        x = block(x) 
        if i in fuse_list: # fuse after i-th block
            f_size = str(x.shape[-1])
            if w>0:
                x = self.fuse_convs_dict[f_size](enc_feat_dict[f_size].detach(), x, w)
    out = x
    # logits doesn't need softmax before cross_entropy loss
    return out, logits, lq_feat, quant_feat


if __name__ == '__main__':
    # test_vqgan()
    # test_codeformer()
    test_3()
    # read_from_dataloader()




