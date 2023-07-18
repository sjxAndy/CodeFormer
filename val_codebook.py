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

def read_single_img(pth = 'tmp/in0.png', pth1 = 'tmp/in1.png'):
    # read img from
    # img = get_img('s3://ffhq/ffhq_imgs/ffhq_512/00000001.png')
    # img = get_img('/mnt/lustre/sunjixiang1/code/CodeFormer/results/src_gd_1.0/cropped_faces/125405467-0378-0381_src_00.png')
    # img = get_img('/mnt/lustre/sunjixiang1/dataset/global_test/debug/125405467-0378-0381_src_local_UW_face.png')
    # img = get_img('/mnt/lustre/sunjixiang1/code/CodeFormer/inputs/cropped_faces/125405467-0378-0381_src.png')
    # img = get_img('/mnt/lustre/sunjixiang1/code/CodeFormer/inputs/ref_cropped/125405467-0378-0381_ref.png')
    # img = get_img('/mnt/lustre/sunjixiang1/code/CodeFormer/results/cropped_face_gd/125405467-0378-0381_src.png')
    img = get_img('s3://ffhq/images/00000.png')

    img_in = cv2.resize(img, (512, 512), interpolation = cv2.INTER_LINEAR).astype(np.float32)   

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



def read_single_img2():
    # read img from
    # img = cv2.imread('/mnt/lustre/share/disp_data/ffhq/FFHQ/ffhq/images/00001.png')
    img = cv2.imread('tmp/input1.png')

    img_in = cv2.resize(img, (512, 512), interpolation = cv2.INTER_LINEAR).astype(np.float32) / 255.0
    cv2.imwrite('tmp/in0.png', img_in*255)


    img_in = img2tensor(img_in, bgr2rgb=True, float32=True)

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    normalize(img_in, mean, std, inplace=True)
    img_in = img_in.unsqueeze(0)

    img_in_data = img_in * 0.5 + 0.5
    img_in_data = tensor2img(img_in_data, rgb2bgr=True, out_type = np.float32)
    cv2.imwrite('tmp/in1.png', img_in_data*255)

    return img_in




def read_from_dataloader():
    with open('options/VQGAN_512_ds32_nearest_stage1.yml', 'r') as f:
        cfg = yaml.safe_load(f)

    cfg_data = cfg['datasets']['train']
    dataset = FFHQBlindDataset(cfg_data)
    loader = DataLoader(dataset, batch_size = 1, shuffle=True)
    data = next(iter(loader))

    img_in = data['in']
    gt = data['gt']

    # print(img_in.shape, img_in.mean())
    # print('gt info: ', gt.shape, gt.mean())

    img_in_data = tensor2img(img_in, rgb2bgr=True, out_type = np.float32)
    img_in_data = img_in_data * 0.5 + 0.5

    gt_data = tensor2img(gt, rgb2bgr=True, out_type = np.float32)
    gt_data = gt_data * 0.5 + 0.5

    cv2.imwrite('tmp/in.png', img_in_data)
    cv2.imwrite('tmp/gt.png', gt_data)

    return gt


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
    
    img_in = read_single_img()

    # img_in = read_from_dataloader()
    print('img_in: ', img_in.shape, img_in.mean())

    # inference
    out, codebook_loss, quant_stats = model(img_in)
    print('out raw info: ', out.shape, out.mean())

    out = out * 0.5 + 0.5
    out = tensor2img(out, rgb2bgr=True, out_type = np.float32)
    print('out info: ', out.shape, out.mean())

    cv2.imwrite('tmp/out.png', out * 255)    


def test_codeformer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('options/CodeFormer_stage3.yml', 'r') as f:
        cfg = yaml.safe_load(f)
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230525_175526_CodeFormer_stage2/models/net_g_latest.pth'
    model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230529_181513_CodeFormer_stage3/models/net_g_latest.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/weights/CodeFormer/codeformer.pth'
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
    img_in = read_single_img(pth='tmp/in_gd001.png', pth1='tmp/in_gd0011.png')

    img_in = img_in.to(device)
    # img_in = read_from_dataloader()
    print('img_in: ', img_in.shape, img_in.mean())

    # inference
    if True:
        out, logits, lq_feat = model(img_in, w=1.0, detach_16=True)
        # out, logits, lq_feat = model(img_in, w=-1, adain=True)
        # out, codebook_loss, quant_stats = model(img_in)
        out = out.detach().cpu()
        print('out raw info: ', out.shape, out.mean())

        out = out * 0.5 + 0.5
        out = tensor2img(out, rgb2bgr=True, out_type = np.float32)
        print('out info: ', out.shape, out.mean())
        
        cv2.imwrite('tmp/out_gd001.png', out * 255)
    else:
        for w in np.arange(0, 1.05, 0.05):
            out, logits, lq_feat = model(img_in, w=w, detach_16=True)
            # out, logits, lq_feat = model(img_in, w=-1, adain=True)
            # out, codebook_loss, quant_stats = model(img_in)
            out = out.detach().cpu()
            print('out raw info: ', out.shape, out.mean())

            out = out * 0.5 + 0.5
            out = tensor2img(out, rgb2bgr=True, out_type = np.float32)
            print('out info: ', out.shape, out.mean())
            
            cv2.imwrite('tmp/out_gd' + str(w) + '.png', out * 255)
    # logits, lq_feat = self.net_g(self.input, w=0, code_only=True)


def test_3():
    # with open('options/CodeFormer_stage2.yml', 'r') as f:
    with open('options/CodeFormer_stage3.yml', 'r') as f:
        cfg = yaml.safe_load(f)
    model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230601_194639_CodeFormer_stage3/models/net_g_latest.pth'
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
    model.eval()
    img_in = read_single_img(pth='tmp/tmp/in3.png', pth1='tmp/tmp/in31.png')

    # img_in = read_from_dataloader()
    print('img_in: ', img_in.shape, img_in.mean())

    # inference
    out, logits, lq_feat = model(img_in, w=0.0, detach_16=True)
    # out, logits, lq_feat = model(img_in, w=1.0, adain=True)
    # out, codebook_loss, quant_stats = model(img_in)
    print('out raw info: ', out.shape, out.mean())

    out = out * 0.5 + 0.5
    out = tensor2img(out, rgb2bgr=True, out_type = np.float32)
    print('out info: ', out.shape, out.mean())
    
    cv2.imwrite('tmp/tmp/out6.png', out * 255)
    # logits, lq_feat = self.net_g(self.input, w=0, code_only=True)


def test_cb():
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

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
    model.to(device)
    model.eval()


    with open('options/CodeFormer_stage3.yml', 'r') as f:
        cfg = yaml.safe_load(f)
    latent_gt_path = cfg.get('latent_gt_path', None)
    latent_gt_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/pretrained_models/vqgan/latent_gt_code1024.pth'
    latent_gt_dict = torch.load(latent_gt_path)
    name = '00000'
    img = read_single_img(pth='tmp/00000.png', pth1='tmp/00000_1.png')
    latent_gt = latent_gt_dict['orig'][name]
    print(type(latent_gt))
    latent_gt = torch.tensor(latent_gt)
    idx_gt = latent_gt.view(1, -1)
    idx_gt.to(device)
    # shape = torch.tensor([1,16,16,256]).to(device)
    # shape = shape.to(device)
    model.quantize.to(device)
    quant_feat_gt = model.quantize.get_codebook_feat(idx_gt, shape=[1,16,16,256])

    x = quant_feat_gt
    
    # for i, block in enumerate(model.generator.blocks):
    #     x = block(x)
    x = model. generator(quant_feat_gt)
    out = x
    print('out raw info: ', out.shape, out.mean())

    out = out * 0.5 + 0.5
    out = tensor2img(out, rgb2bgr=True, out_type = np.float32)
    print('out info: ', out.shape, out.mean())
    
    cv2.imwrite('tmp/idx_gt.png', out * 255)

if __name__ == '__main__':
    # test_vqgan()
    # test_codeformer()
    test_cb()
    # read_from_dataloader()




