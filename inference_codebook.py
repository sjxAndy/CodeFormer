import argparse
import cv2
import glob
import numpy as np
import os
import torch
import yaml

from basicsr.archs.vqgan_arch import VQAutoEncoder
from basicsr.archs.vqgan_double_arch import VQDoubleAutoEncoder
from basicsr.archs.vqgan_double_flip_arch import VQDoubleFlipAutoEncoder
from basicsr.archs.vqgan_single_arch import VQSingleAutoEncoder
from basicsr.archs.codeformer_arch import CodeFormer
from basicsr.archs.srformer_arch import SRFormer
from torchvision.transforms.functional import normalize
from basicsr.utils import img2tensor, tensor2img
from basicsr.data.ffhq_blind_dataset import FFHQBlindDataset
from torch.utils.data import Dataset, DataLoader
from basicsr.data import build_dataloader, build_dataset


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


def read_img(pth = 'tmp/in0.png', pth1 = 'tmp/in1.png'):
    # read img from
    size = 512
    img = get_img('s3://Deblur/GoPro/crop/blur_crops/GOPR0868_11_01-000263_s006.png')

    img_in = cv2.resize(img, (size, size), interpolation = cv2.INTER_LINEAR).astype(np.float32)   

    cv2.imwrite(pth, img_in*255)

    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)

    img_in = to_tensor(img_in, is_train = False)

    img_gt = get_img('s3://Deblur/GoPro/crop/sharp_crops/GOPR0868_11_01-000263_s006.png')

    img_gt = cv2.resize(img_gt, (size, size), interpolation = cv2.INTER_LINEAR).astype(np.float32)   

    cv2.imwrite(pth1, img_gt*255)

    # return img_in
    return img_in[:, :, 0:size//2, 0:size//2], img_in[:, :, 0:size//2, size//2:size], img_in[:, :, size//2:size, 0:size//2], img_in[:, :, size//2:size, size//2:size], img_in


def test_vqgan():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('options/double_stage1.yml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    # dataset_opt = cfg['datasets']['train']
    # dataset_opt['phase'] = 'train'

    # train_set = build_dataset(dataset_opt)
    # print(len(train_set))
    # img0_d = train_set[0]
    # img0 = img0_d['gt']

    # img0 = tensor2img(img0, rgb2bgr=True, out_type = np.float32)

    # cv2.imwrite('tmp/img0.png', img0 * 255)   
    # img0 = img0_d['lq']

    # img0 = tensor2img(img0, rgb2bgr=True, out_type = np.float32)

    # cv2.imwrite('tmp/img1.png', img0 * 255)  


    # model_path = '/mnt/lustre/leifei1/project/CodeFormer/experiments/20230503_195457_VQGAN-512-ds32-nearest-stage1/models/net_g_800000.pth'

    # model_path = 'experiments/pretrained_models/vqgan/vqgan_code1024.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230612_204254_VQGAN-512-ds32-nearest-stage1/models/net_g_40000.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230612_214957_VQGAN-512-ds32-nearest-stage1/models/net_g_40000.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230612_204254_VQGAN-512-ds32-nearest-stage1/models/net_g_380000.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230615_151723_VQGAN-512-ds32-nearest-stage1/models/net_g_170000.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230615_151839_VQGAN-512-ds32-nearest-stage1/models/net_g_170000.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230619_165111_VQGAN-512-ds32-nearest-stage1/models/net_g_330000.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230619_165038_VQGAN-512-ds32-nearest-stage1/models/net_g_330000.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230615_151839_VQGAN-512-ds32-nearest-stage1/models/net_g_650000.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230615_151723_VQGAN-512-ds32-nearest-stage1/models/net_g_650000.pth'
    model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230619_164941_VQGAN-512-ds32-nearest-stage1/models/net_g_latest.pth'

    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230619_165038_VQGAN-512-ds32-nearest-stage1/models/net_g_1570000.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230619_165111_VQGAN-512-ds32-nearest-stage1/models/net_g_latest.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230703_144456_VQGAN-512-ds32-nearest-stage1/models/net_g_10000.pth'

    # load model
    cfg_vqgan = cfg['network_g']
    model_type = cfg_vqgan.pop('type')
    model = VQDoubleAutoEncoder(**cfg_vqgan)
    # model = VQSingleAutoEncoder(**cfg_vqgan)

    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    # for k, v in checkpoint['params_ema'].items():
    #     print(k)

    model.load_state_dict(checkpoint['params_ema'])
    model.to(device)
    model.eval()
    
    # img_in = read_img()
    # img_in = img_in.to(device)

    # # img_in = read_from_dataloader()
    # print('img_in: ', img_in.shape, img_in.mean())

    # # inference
    # with torch.no_grad():
    #     out, codebook_loss, quant_stats = model(img_in)
    #     # out = model(img_in)
    # print('out raw info: ', out.shape, out.mean())
    # out = out.detach().cpu()

    # # out = out * 0.5 + 0.5
    # out = tensor2img(out, rgb2bgr=True, out_type = np.float32)
    # print('out info: ', out.shape, out.mean())

    # cv2.imwrite('tmp/out_1600k_double_256.png', out * 255)

    img_in0, img_in1, img_in2, img_in3, img_in = read_img()
    img_in0 = img_in0.to(device)
    img_in1 = img_in1.to(device)
    img_in2 = img_in2.to(device)
    img_in3 = img_in3.to(device)
    img_in = img_in.to(device)
    print(img_in0.shape)

    # inference
    with torch.no_grad():
        out0, codebook_loss, quant_stats = model(img_in0)
        out1, codebook_loss, quant_stats = model(img_in1)
        out2, codebook_loss, quant_stats = model(img_in2)
        out3, codebook_loss, quant_stats = model(img_in3)
        out, codebook_loss, quant_stats = model(img_in)
        out = torch.zeros(img_in.shape).to(device)
        size = img_in.shape[-1]
        out[:, :, 0:size//2, 0:size//2] = out0
        out[:, :, 0:size//2, size//2:size] = out1
        out[:, :, size//2:size, 0:size//2] = out2
        out[:, :, size//2:size, size//2:size] = out3

        # out = model(img_in)
    print('out raw info: ', out.shape, out.mean())
    out = out.detach().cpu()

    # out = out * 0.5 + 0.5
    out = tensor2img(out, rgb2bgr=True, out_type = np.float32)
    print('out info: ', out.shape, out.mean())
    cv2.imwrite('tmp/out_1600k_double_512_crop.png', out * 255)



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


def read_single_img(pth = 'tmp/in0.png', pth1 = 'tmp/in1.png'):
    # read img from
    # img = get_img('s3://ffhq/images_deblur_student6_g_350000/00010_deblur.jpg')
    # img = get_img('/mnt/lustre/sunjixiang1/code/CodeFormer/results/src_gd_1.0/cropped_faces/125405467-0378-0381_src_00.png')
    # img = get_img('/mnt/lustre/sunjixiang1/dataset/global_test/debug/125405467-0378-0381_src_local_UW_face.png')
    # img = get_img('/mnt/lustre/sunjixiang1/code/CodeFormer/inputs/cropped_faces/125405467-0378-0381_src.png')
    # img = get_img('/mnt/lustre/sunjixiang1/code/CodeFormer/inputs/ref_cropped/125405467-0378-0381_ref.png')
    # img = get_img('/mnt/lustre/sunjixiang1/code/CodeFormer/results/cropped_face_gd/125405467-0378-0381_src.png')
    # img = get_img('/mnt/lustre/sunjixiang1/code/CodeFormer/results/cropped_faces/210038924-0139-0142_src_global_result.png')
    # img = get_img('/mnt/lustre/sunjixiang1/code/CodeFormer/results/cropped_faces/102550045-0093-0096_src_global_result.png')
    img = get_img('tmp/codeformer_retrain1/mid0.png')
    # img = get_img('/mnt/lustre/sunjixiang1/code/CodeFormer/results/cropped_faces/125405467-0378-0381_src_global_result.png')
    # img = get_img('/mnt/lustre/sunjixiang1/code/CodeFormer/results/cropped_faces/103306301-0153-0160_src_global_result.png')
    # img = get_img('/mnt/lustre/sunjixiang1/code/CodeFormer/results/cropped_face_gd/125405467-0378-0381_src.png')
    
    # img = get_img('s3://Deblur/GoPro/crop/blur_crops/GOPR0868_11_01-000263_s006.png')

    img_in = cv2.resize(img, (512, 512), interpolation = cv2.INTER_LINEAR).astype(np.float32) 

    cv2.imwrite(pth, img_in*255)

    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)

    img_in = to_tensor(img_in, is_train = False)

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    normalize(img_in, mean, std, inplace=True)

    img_in_data = to_numpy(img_in)
    # img_in_data = img_in_data * 0.5 + 0.5
    cv2.imwrite(pth1, img_in_data*255)

    return img_in


def test_3():
    # with open('options/CodeFormer_stage2.yml', 'r') as f:
    with open('options/CodeFormer_stage3_lowde.yml', 'r') as f:
        cfg = yaml.safe_load(f)
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230601_194639_CodeFormer_stage3/models/net_g_latest.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230602_113339_CodeFormer_stage3/models/net_g_latest.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230525_175526_CodeFormer_stage2/models/net_g_latest.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230526_203531_CodeFormer_stage2/models/net_g_latest.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230602_153451_CodeFormer_stage3/models/net_g_latest.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230602_161809_CodeFormer_stage3/models/net_g_latest.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/weights/CodeFormer/codeformer.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230605_210859_CodeFormer_stage3/models/net_g_85000.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230606_114645_CodeFormer_stage3/models/net_g_15000.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230605_210019_CodeFormer_stage3/models/net_g_latest.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230606_172329_CodeFormer_stage2/models/net_g_500000.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230615_173502_CodeFormer_stage3/models/net_g_latest.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230615_153039_CodeFormer_stage2/models/net_g_latest.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230628_223749_CodeFormer_stage3/models/net_g_latest.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230628_222823_CodeFormer_stage3/models/net_g_815000.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230704_154304_CodeFormer_stage3/models/net_g_720000.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230704_153409_CodeFormer_stage3/models/net_g_720000.pth'
    model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230628_223749_CodeFormer_stage3/models/net_g_latest.pth'

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
    img_in = read_single_img(pth='tmp/glass.png', pth1='tmp/00010_1.png')

    # img_in = read_from_dataloader()
    print('img_in: ', img_in.shape, img_in.mean())

    # inference
    # out, logits, lq_feat = model(img_in, w=1.0, detach_16=True)
    with torch.no_grad():
        out, logits, lq_feat = model(img_in, w=1.0, adain=True)
    # out, codebook_loss, quant_stats = model(img_in)
    print('out raw info: ', out.shape, out.mean())

    out = out * 0.5 + 0.5
    out = tensor2img(out, rgb2bgr=True, out_type = np.float32)
    print('out info: ', out.shape, out.mean())
    
    cv2.imwrite('tmp/14_mid.png', out * 255)
    # logits, lq_feat = self.net_g(self.input, w=0, code_only=True)


def read_an_img(pth):
    img = get_img(pth)

    if img.shape[-2] != 1024:
        print('need to resize:', img.shape[-2])
        img_in = cv2.resize(img, (1024, 1024), interpolation = cv2.INTER_LINEAR).astype(np.float32)
    else:
        img_in = img

    # img_in = cv2.resize(img, (512, 512), interpolation = cv2.INTER_LINEAR).astype(np.float32)

    # cv2.imwrite(pth, img_in*255)

    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)

    img_in = to_tensor(img_in, is_train = False)

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    normalize(img_in, mean, std, inplace=True)

    img_in_data = to_numpy(img_in)
    # img_in_data = img_in_data * 0.5 + 0.5
    # cv2.imwrite(pth1, img_in_data*255)

    return img_in


def test_all():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # with open('options/CodeFormer_stage3_lowde.yml', 'r') as f:
    #     cfg = yaml.safe_load(f)
    # with open('options/CodeFormer_stage3_long.yml', 'r') as f:
    with open('options/SRFormer_stage3_lowde.yml', 'r') as f:
        cfg = yaml.safe_load(f)

    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230619_113550_CodeFormer_stage3/models/net_g_latest.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230628_223749_CodeFormer_stage3/models/net_g_latest.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/weights/CodeFormer/codeformer.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230704_154304_CodeFormer_stage3/models/net_g_latest.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230704_153409_CodeFormer_stage3/models/net_g_latest.pth'


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

    dir = '/mnt/lustre/sunjixiang1/code/CodeFormer/results/cropped_faces/'
    imgs = []
    for pth in os.listdir(dir):
        if pth.endswith('.png') or pth.endswith('.jpg'):
            imgs.append(os.path.join(dir, pth))
    for img_pth in imgs:
        img_in = read_an_img(img_pth)
        img_in = img_in.to(device)

        # inference
        # out, logits, lq_feat = model(img_in, w=1.0, detach_16=True)
        with torch.no_grad():
            out, logits, lq_feat = model(img_in, w=1.0, adain=True)
        out = out.detach().cpu()
        # out, codebook_loss, quant_stats = model(img_in)
        print('out raw info: ', out.shape, out.mean())

        out = out * 0.5 + 0.5
        out = tensor2img(out, rgb2bgr=True, out_type = np.float32)
        print('out info: ', out.shape, out.mean())
        
        cv2.imwrite(os.path.join('tmp/codeformer_lowde_lowweight_whole/', os.path.basename(img_pth)), out * 255)
    # logits, lq_feat = self.net_g(self.input, w=0, code_only=True)


def test_my():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # with open('options/CodeFormer_stage3_lowde.yml', 'r') as f:
    #     cfg = yaml.safe_load(f)
    with open('options/SRFormer_stage3_lowde_o2.yml', 'r') as f:
        cfg = yaml.safe_load(f)

    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230619_113550_CodeFormer_stage3/models/net_g_latest.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230628_223749_CodeFormer_stage3/models/net_g_latest.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/weights/CodeFormer/codeformer.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230710_172551_SRFormer/models/net_g_210000.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230710_173516_SRFormer/models/net_g_210000.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230710_200457_SRFormer/models/net_g_210000.pth'
    # model_path = 'experiments/20230713_145736_SRFormer/models/net_g_100000.pth'
    # model_path = 'experiments/20230713_152942_SRFormer/models/net_g_100000.pth'
    # model_path = 'experiments/20230713_153747_SRFormer/models/net_g_100000.pth'
    # model_path = 'experiments/20230714_113656_SRFormer/models/net_g_350000.pth'
    # model_path = 'experiments/20230714_113943_SRFormer/models/net_g_350000.pth'
    model_path = 'experiments/20230714_113919_SRFormer/models/net_g_360000.pth'

    cfg_codeformer = cfg['network_g']
    model_type = cfg_codeformer.pop('type')
    model = SRFormer(**cfg_codeformer)
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    # for k, v in checkpoint['params_ema'].items():
    #     print(k)

    model.load_state_dict(checkpoint['params_ema'])
    # model.fuse_generator_block = {}
    # model.connect_list = {}
    model.to(device)
    model.eval()

    dir = '/mnt/lustre/sunjixiang1/dataset/test/1'
    imgs = []
    for pth in os.listdir(dir):
        if pth.endswith('.png') or pth.endswith('.jpg'):
            imgs.append(os.path.join(dir, pth))
    for img_pth in imgs:
        # if not '14.' in img_pth:
        #     continue
        img_in = read_an_img(img_pth)
        img_in = img_in.to(device)

        # inference
        # out, logits, lq_feat = model(img_in, w=1.0, detach_16=True)
        with torch.no_grad():
            # out, logits, lq_feat = model(img_in, w=1.0, adain=True)
            # out, mid0, mid1, mid2, mid3 = model(img_in, w=1.0, adain=True)
            # out, mid0, mid1 = model(img_in, w=1.0, adain=True)
            out = model(img_in, w=1.0, adain=True)
        out = out.detach().cpu()
        # mid0 = mid0.detach().cpu()
        # mid1 = mid1.detach().cpu()
        # mid2 = mid2.detach().cpu()
        # mid3 = mid3.detach().cpu()

        # out, codebook_loss, quant_stats = model(img_in)
        print('out raw info: ', out.shape, out.mean())

        out = out * 0.5 + 0.5
        # mid0 = mid0 * 0.5 + 0.5
        # mid1 = mid1 * 0.5 + 0.5
        # mid2 = mid2 * 0.5 + 0.5
        # mid3 = mid3 * 0.5 + 0.5
        out = tensor2img(out, rgb2bgr=True, out_type = np.float32)
        # mid0 = tensor2img(mid0, rgb2bgr=True, out_type = np.float32)
        # mid1 = tensor2img(mid1, rgb2bgr=True, out_type = np.float32)
        # mid2 = tensor2img(mid2, rgb2bgr=True, out_type = np.float32)
        # mid3 = tensor2img(mid3, rgb2bgr=True, out_type = np.float32)
        # out = cv2.resize(out, (1024, 1024), interpolation = cv2.INTER_LINEAR)
        print('out info: ', out.shape, out.mean())
        # print('out info: ', mid0.shape, mid0.mean())
        # print('out info: ', mid1.shape, mid1.mean())
        # print('out info: ', mid2.shape, mid2.mean())
        # print('out info: ', mid3.shape, mid3.mean())
        
        
        cv2.imwrite(os.path.join('tmp/codeformer_option2/1/', os.path.basename(img_pth)), out * 255)
        # cv2.imwrite('tmp/codeformer_retrain2/mid0.png', mid0 * 255)
        # cv2.imwrite('tmp/codeformer_retrain2/mid1.png', mid1 * 255)
        # cv2.imwrite('tmp/codeformer_retrain/mid2.png', mid2 * 255)
        # cv2.imwrite('tmp/codeformer_retrain/mid3.png', mid3 * 255)
    # logits, lq_feat = self.net_g(self.input, w=0, code_only=True)


def upsample():
    dir = '/mnt/lustre/sunjixiang1/code/CodeFormer/tmp/codeformer_2'
    imgs = []
    for pth in os.listdir(dir):
        if pth.endswith('.png') or pth.endswith('.jpg'):
            imgs.append(os.path.join(dir, pth))
    for img_pth in imgs:
        img_in = get_img(img_pth)
        img_in = cv2.resize(img_in, (1024, 1024), interpolation = cv2.INTER_LINEAR)

        
        cv2.imwrite(os.path.join('tmp/codeformer_2_1024/', os.path.basename(img_pth)), img_in * 255)


if __name__ == '__main__':
    # test_vqgan()
    # test_codeformer()
    # test_3()
    # test_all()
    # read_from_dataloader()
    test_my()
    # upsample()



