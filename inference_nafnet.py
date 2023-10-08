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
from basicsr.archs.Baseline_arch import BaselineLocal


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


def read_an_img(pth):
    img = get_img(pth)

    img_in = img

    # img_in = cv2.resize(img, (512, 512), interpolation = cv2.INTER_LINEAR).astype(np.float32)

    # cv2.imwrite(pth, img_in*255)

    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)

    img_in = to_tensor(img_in, is_train = False)

    # mean = [0.5, 0.5, 0.5]
    # std = [0.5, 0.5, 0.5]

    # normalize(img_in, mean, std, inplace=True)

    # img_in_data = to_numpy(img_in)
    # img_in_data = img_in_data * 0.5 + 0.5
    # cv2.imwrite(pth1, img_in_data*255)

    return img_in


def test_naf():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('options/Baseline-width64.yml', 'r') as f:
        cfg = yaml.safe_load(f)

    cfg_vqgan = cfg['network_g']
    model_type = cfg_vqgan.pop('type')
    # model = VQDoubleAutoEncoder(**cfg_vqgan)
    # model = VQDoubleFlipAutoEncoder(**cfg_vqgan)
    # model = VQSingleAutoEncoder(**cfg_vqgan)
    model = BaselineLocal(**cfg_vqgan)
    model_path = '/mnt/lustre/sunjixiang1/ckpts/nafnet_baseline_Gopro_400000.pth'
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    # for k, v in checkpoint['params_ema'].items():
    #     print(k)

    model.load_state_dict(checkpoint['params'])
    # model.fuse_generator_block = {}
    # model.connect_list = {}
    model.to(device)
    model.eval()

    # img_pth = 's3://Deblur/GoPro/GOPRO_Large/test/GOPR0384_11_00/blur/000001.png'
    # img_pth = '/mnt/lustre/sunjixiang1/dataset/test/scene3/00/05_blur.png'
    img_pth = '/mnt/lustre/sunjixiang1/dataset/test/scene3/06/01_blur.png'
    img_in = read_an_img(img_pth).to(device)

    with torch.no_grad():
        pred = model(img_in)
        if isinstance(pred, list):
            pred = pred[-1]
    pred = pred.detach().cpu()
    pred = tensor2img(pred, rgb2bgr=True, out_type = np.float32)

    img_in = img_in.detach().cpu()
    img_in = tensor2img(img_in, rgb2bgr=True, out_type = np.float32)
    
    cv2.imwrite('tmp/pred_3_6_1.png', pred * 255)
    # cv2.imwrite('tmp/res_000001.png', ((pred - img_in)*0.5 + 0.5) * 255)
    cv2.imwrite('tmp/resnew_3_6_1.png', (abs(pred - img_in)) * 255)

if __name__ == '__main__':
    #
    test_naf()


