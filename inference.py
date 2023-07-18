import argparse
import cv2
import glob
import numpy as np
import os
import torch
import yaml

from basicsr.archs.vqgan_arch import VQAutoEncoder, AutoEncoder, DeblurVQAutoEncoder
from basicsr.archs.deblur_arch import DeblurTwoBranch

from torchvision.transforms.functional import normalize
from basicsr.utils import img2tensor, tensor2img
from basicsr.data.ffhq_blind_dataset import FFHQBlindDataset
from torch.utils.data import Dataset, DataLoader
from basicsr.archs import build_network



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

def read_single_img():
    # read img from
    img = cv2.imread('/mnt/lustre/share/disp_data/ffhq/FFHQ/ffhq/images/00001.png')    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_in = cv2.resize(img, (512, 512), interpolation = cv2.INTER_LINEAR).astype(np.float32) / 255.0

    cv2.imwrite('tmp/in0.png', img_in*255)

    img_in = to_tensor(img_in, is_train = False)

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    normalize(img_in, mean, std, inplace=True)

    img_in_data = to_numpy(img_in)
    img_in_data = img_in_data * 0.5 + 0.5
    cv2.imwrite('tmp/in1.png', img_in_data*255)

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

def read_deblur_img():
    img = cv2.imread('/mnt/lustre/leifei1/data/deblur/train/train/blur_crops/GOPR0372_07_00-000058_s001.png').astype(np.float32) / 255.0

    # img = cv2.imread('tmp/in0.png').astype(np.float32) / 255.0

    img = cv2.resize(img, (1280, 720))

    # img = img[:512, :25, :]

    # cv2.imwrite("tmp/in.png", img * 255)

    img = img2tensor(img, bgr2rgb=True, float32=True)
    img = img.unsqueeze(0)


    return img

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

def test_vqgan():

    with open('options/VQGAN_512_ds32_nearest_stage1.yml', 'r') as f:
        cfg = yaml.safe_load(f)

    # model_path = '/mnt/lustre/leifei1/project/CodeFormer/experiments/20230503_195457_VQGAN-512-ds32-nearest-stage1/models/net_g_800000.pth'

    model_path = 'pretrained_models/vqgan_code1024.pth'


    # load model
    cfg_vqgan = cfg['network_g']
    model_type = cfg_vqgan.pop('type')
    model = VQAutoEncoder(**cfg_vqgan)

    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    # for k, v in checkpoint['params_ema'].items():
    #     print(k)

    model.load_state_dict(checkpoint['params_ema'])
    model.eval()
    
    img_in = read_single_img2()

    # img_in = read_from_dataloader()
    print('img_in: ', img_in.shape, img_in.mean())

    # inference
    out, codebook_loss, quant_stats = model(img_in)
    print('out raw info: ', out.shape, out.mean())

    out = out * 0.5 + 0.5
    out = tensor2img(out, rgb2bgr=True, out_type = np.float32)
    print('out info: ', out.shape, out.mean())

    cv2.imwrite('tmp/out.png', out * 255)    

def test_vqgan_deblur():
    with open('options/Deblur_VQGAN.yml', 'r') as f:
        cfg = yaml.safe_load(f)

    model_path = 'experiments/20230609_105240_Deblur-VQGAN-512/models/net_g_520000.pth'
    # model_path = 'experiments/20230607_221807_Deblur-VQGAN-512-ds32-nearest-stage1/models/net_g_760000.pth'

    # load model
    cfg_vqgan = cfg['network_g']
    model_type = cfg_vqgan.pop('type')
    model = DeblurVQAutoEncoder(**cfg_vqgan)
    # model = VQAutoEncoder(**cfg_vqgan)


    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    # for k, v in checkpoint['params_ema'].items():
    #     print(k)

    model.load_state_dict(checkpoint['params_ema'])
    model.eval()


    img_in = read_deblur_img()

    # inference
    print('input raw info: ', img_in.shape, img_in.mean())
    out, codebook_loss, quant_stats = model(img_in)
    print('out raw info: ', out.shape, out.mean())
    img_out = tensor2img(out, rgb2bgr=True, out_type = np.float32)
    print('out info: ', img_out.shape, img_out.mean())

    cv2.imwrite('tmp/out_DeblurVQAutoEncoder_potrait.png', img_out * 255)

def test_ae_deblur():
    with open('options/Deblur_AE.yml', 'r') as f:
        cfg = yaml.safe_load(f)

    model_path = 'experiments/20230608_195210_AE-512-ds32-nearest-stage1/models/net_g_90000.pth'

    # load model
    cfg_vqgan = cfg['network_g']
    model_type = cfg_vqgan.pop('type')
    model = AutoEncoder(**cfg_vqgan)

    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    # for k, v in checkpoint['params_ema'].items():
    #     print(k)

    model.load_state_dict(checkpoint['params_ema'])
    model.eval()
    img_in = read_deblur_img()

    # inference
    print('in info： ', img_in.shape, img_in.mean())
    out = model(img_in)
    print('out raw info: ', out.shape, out.mean())

    img_out = tensor2img(out, rgb2bgr=True, out_type = np.float32)
    print('out info: ', img_out.shape, img_out.mean())

    cv2.imwrite('tmp/out_potrait.png', img_out * 255)


def test_DeblurTwoBranch():
    with open('options/train_DeblurTwoBranch.yml') as f:
        opt = yaml.safe_load(f)


    model_path = 'experiments/20230613_205403_train_deblurTB/models/net_g_165000.pth'

    cfg = opt['network_g']
    model_type = cfg.pop('type')
    model = DeblurTwoBranch(**cfg)

    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['params_ema'])
    model.eval()
    img_in = read_deblur_img()

    print('input shape: ', img_in.shape)
    out = model(img_in, 1.0)


    img_out = out['main_dec']
    print('out shape: ', img_out.shape, img_out.mean())

    img_out = tensor2img(img_out, rgb2bgr=True, out_type = np.float32)

    cv2.imwrite('tmp/out_twoBranch.png', img_out * 255)



if __name__ == '__main__':
    # test_vqgan()
    # read_from_dataloader()
    test_ae_deblur()
    # test_vqgan_deblur()
    # test_DeblurTwoBranch()


