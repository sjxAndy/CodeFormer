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
from basicsr.archs.kernel_arch import KernelArch
from basicsr.archs.kernel_decoder_arch import KernelDecoderArch
from basicsr.archs.kernel_single_arch import KernelSingleArch
from basicsr.data.paired_image_petrel_test_dataset import PairedImagePetrelTestDataset

from basicsr.models.kernel_model import DegradationModel

from tqdm import tqdm

from basicsr.metrics.metric_util import reorder_image, to_y_channel
import skimage.metrics
import matplotlib.pyplot as plt

import argparse


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


def plot_canvas(PSFs, path_to_save):
    if len(PSFs) == 0:
        raise Exception("Please run fit() method first.")
    else:
        plt.close()
        fig, axes = plt.subplots(PSFs.shape[0], PSFs.shape[1], figsize=(10, 10))
        # print(axes.shape, PSFs.shape)
        if PSFs.shape[0] == 1:
            for i in range(PSFs.shape[1]):
                axes[i].imshow(PSFs[0, i], cmap='gray')
        else:
            for i in range(PSFs.shape[0]):
                for j in range(PSFs.shape[1]):
                    axes[i, j].imshow(PSFs[i, j], cmap='gray')
    plt.savefig(path_to_save)


def test_kernel():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    # with open('options/Kernel_weight.yml', 'r') as f:
    #     cfg = yaml.safe_load(f)
    # with open('options/Kernel_weight_shape.yml', 'r') as f:
    #     cfg = yaml.safe_load(f)
    # with open('options/Kernel_freeze_ft.yml', 'r') as f:
    #     cfg = yaml.safe_load(f)
    with open('options/Kernel_single.yml', 'r') as f:
        cfg = yaml.safe_load(f)
    cfg_vqgan = cfg['network_g']
    model_type = cfg_vqgan.pop('type')
    model = KernelSingleArch(**cfg_vqgan)
    # model = KernelDecoderArch(**cfg_vqgan)
    # model = KernelArch(**cfg_vqgan)
    
    # model_path = 'experiments/20230822_170249_Kernel/models/net_g_120000.pth'
    # model_path = 'experiments/20230815_225937_Kernel/models/net_g_785000.pth'
    model_path = 'experiments/20230826_113032_Kernel_single/models/net_g_latest.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230830_164052_Kernel_init/models/net_g_50000.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230831_162837_Kernel_Decoder_weight/models/net_g_70000.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230831_161924_Kernel_freeze/models/net_g_10000.pth'
    # model_path = 'experiments/20230828_180443_Kernel/models/net_g_50000.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230905_121428_Kernel_weight_shape/models/net_g_20000.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230911_183319_Kernel_freeze_ft/models/net_g_80000.pth'
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    # for k, v in checkpoint['params_ema'].items():
    #     print(k)

    model.load_state_dict(checkpoint['params'])
    # model.fuse_generator_block = {}
    # model.connect_list = {}

    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230826_113032_Kernel_single/models/net_g_latest.pth'
    # checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    # cfg_vqgan.pop('initialize')
    # # cfg_vqgan.pop('fix_modules')
    # cfg_vqgan['codebook_size'] = 1024
    # new_model = KernelSingleArch(**cfg_vqgan)
    # new_model.load_state_dict(checkpoint['params'])
    # model.dcnn = new_model.dcnn
    # # model = new_model
    # del new_model
    model.to(device)

    model.eval()


    img_pth = 's3://Deblur/GoPro/GOPRO_Large/test/GOPR0384_11_00/blur/000001.png'
    # img_pth = '/mnt/lustre/sunjixiang1/dataset/test/scene3/00/05_blur.png'
    # img_pth = '/mnt/lustre/sunjixiang1/dataset/test/scene3/06/01_blur.png'
    img_in = read_an_img(img_pth).to(device)

    gt_pth = 's3://Deblur/GoPro/GOPRO_Large/test/GOPR0384_11_00/sharp/000001.png'
    img_gt = read_an_img(gt_pth).to(device)

    degradation_model = DegradationModel(kernel_size=19).to(device)

    with torch.no_grad():
        # kernel, cb_loss = model(img_in)
        kernel = model(img_in)
        pred = degradation_model(img_gt, kernel)
        # if isinstance(pred, list):
        #     pred = pred[-1]
        # print(kernel[0][2863])
    kernel = kernel.view(1, img_in.shape[2], img_in.shape[3], kernel.shape[-2], kernel.shape[-1]).detach().cpu().numpy()[0]
    print(kernel.shape)
    ratio = 160
    kernel_sparse = kernel[slice(0,kernel.shape[0],ratio), slice(0,kernel.shape[1],ratio), :, :]
    print(kernel_sparse.shape)
    print(kernel_sparse[0, 0], kernel_sparse[1, 1], kernel_sparse[2, 2], kernel_sparse[3, 3])
    if True:
        with open('tmp/kernels.txt', 'w') as f:
            for i in range(kernel_sparse.shape[0]):
                for j in range(kernel_sparse.shape[1]):
                    f.writelines([f"{i}, {j}:\n"])
                    for k in range(19):
                        for l in range(19):
                            f.write(str(kernel_sparse[i][j][k][l])+'\t')
                        f.write('\n')

    if True:
        print(np.sum(kernel_sparse[0, 0]), np.sum(kernel_sparse[1, 1]), np.sum(kernel_sparse[2, 2]), np.sum(kernel_sparse) / 40)

    plot_canvas(kernel_sparse, 'tmp/kernel_map.png')


    
    if False:
        def tr(psf):
            ipsf = []
            for i in range(len(psf)):
                ipsf.append(psf[i].reshape(-1))
            return np.array(ipsf).transpose((1,0))


        def itr(ipsf):
            psf = []
            ipsf = np.transpose(ipsf, (1,0))
            for i in range(len(ipsf)):
                psf.append(ipsf[i].reshape((19, 19)))
            return np.array(psf)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=40)
        psfs = kernel.reshape(-1, kernel.shape[-2], kernel.shape[-1])
        newpsfs = itr(pca.fit_transform(tr(psfs)))
        print(len(newpsfs), newpsfs[0].shape, np.sum(newpsfs) / len(newpsfs))
        newpsfs = newpsfs.reshape(5, 8, newpsfs.shape[-2], newpsfs.shape[-1])
        plot_canvas(newpsfs, 'tmp/kernel_pca.png')


    
    pred = pred.detach().cpu()
    pred = tensor2img(pred, rgb2bgr=True, out_type = np.float32)

    img_in = img_in.detach().cpu()
    img_in = tensor2img(img_in, rgb2bgr=True, out_type = np.float32)
    
    img_gt = img_gt.detach().cpu()
    img_gt = tensor2img(img_gt, rgb2bgr=True, out_type = np.float32)

    print(calculate_psnr_new(pred, img_in, 0))
    print(calculate_psnr_new(pred, img_gt, 0))
    print(calculate_psnr_new(img_in, img_in, 0))

    cv2.imwrite('tmp/reblur.png', pred * 255)
    cv2.imwrite('tmp/in.png', img_in * 255)
    cv2.imwrite('tmp/gt.png', img_gt * 255)

    # cv2.imwrite('tmp/res_000001.png', ((pred - img_in)*0.5 + 0.5) * 255)
    # cv2.imwrite('tmp/resnew_3_6_1.png', (abs(pred - img_in)) * 255)


def extract_kernel():
    # device = 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('options/Kernel_single.yml', 'r') as f:
        cfg = yaml.safe_load(f)

    cfg_vqgan = cfg['network_g']
    model_type = cfg_vqgan.pop('type')
    model = KernelSingleArch(**cfg_vqgan)
    # model = KernelArch(**cfg_vqgan)
    
    # model_path = 'experiments/20230822_170249_Kernel/models/net_g_120000.pth'
    # model_path = 'experiments/20230815_225937_Kernel/models/net_g_785000.pth'
    model_path = 'experiments/20230826_113032_Kernel_single/models/net_g_525000.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230829_170400_Kernel/models/net_g_90000.pth'
    # model_path = 'experiments/20230828_180443_Kernel/models/net_g_50000.pth'
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    # for k, v in checkpoint['params_ema'].items():
    #     print(k)

    model.load_state_dict(checkpoint['params'])
    # model.fuse_generator_block = {}
    # model.connect_list = {}
    model.to(device)
    model.eval()

    img_pth = 's3://Deblur/GoPro/GOPRO_Large/test/GOPR0384_11_00/blur/000001.png'
    # img_pth = '/mnt/lustre/sunjixiang1/dataset/test/scene3/00/05_blur.png'
    # img_pth = '/mnt/lustre/sunjixiang1/dataset/test/scene3/06/01_blur.png'
    img_in = read_an_img(img_pth).to(device)

    gt_pth = 's3://Deblur/GoPro/GOPRO_Large/test/GOPR0384_11_00/sharp/000001.png'
    img_gt = read_an_img(gt_pth).to(device)

    degradation_model = DegradationModel(kernel_size=19).to(device)


    import time
    start = time.time()
    with torch.no_grad():
        kernel = model(img_in)
        # pred = degradation_model(img_gt, kernel)
        # if isinstance(pred, list):
        #     pred = pred[-1]
        # print(kernel[0][2863])
    # kernel = kernel.view(1, img_in.shape[2], img_in.shape[3], kernel.shape[-2], kernel.shape[-1]).detach().cpu().numpy()[0]
    kernel = kernel.view(1, kernel.shape[1], -1).detach().cpu().numpy()[0]
    print(kernel.shape)

    # kernel = kernel[:100, :]
    n_clusters = 4
    
    from sklearnex import patch_sklearn, unpatch_sklearn
    patch_sklearn()
    from sklearn.cluster import KMeans
    y_pred = KMeans(n_clusters=n_clusters, random_state=9).fit_predict(kernel)
    print(y_pred, y_pred.shape)
    kernel_cluster = [[]]
    kernels = [[] for i in range(n_clusters)]
    for i in range(n_clusters):
        print(np.sum(y_pred==i))
        kernel_cluster[0].append(kernel[y_pred==i][0].reshape(19, 19))
        for j in range(0, 10000, 2000):
            kernels[i].append(kernel[y_pred==i][j].reshape(19, 19))
        # kernels[i].append(kernel[y_pred==i][0:10000:2000])
    kernel_cluster = np.array(kernel_cluster)
    kernels = np.array(kernels)
    print(kernel_cluster.shape, kernels.shape)
    plot_canvas(kernel_cluster, 'tmp/kernel_cluster2.png')
    plot_canvas(kernels, 'tmp/kernels2.png')
    print('time consumed:', time.time() - start)


def extract_all_kernels():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('options/Kernel_single.yml', 'r') as f:
        cfg = yaml.safe_load(f)

    cfg_vqgan = cfg['network_g']
    model_type = cfg_vqgan.pop('type')
    model = KernelSingleArch(**cfg_vqgan)
    # model = KernelArch(**cfg_vqgan)
    
    # model_path = 'experiments/20230822_170249_Kernel/models/net_g_120000.pth'
    # model_path = 'experiments/20230815_225937_Kernel/models/net_g_785000.pth'
    model_path = 'experiments/20230826_113032_Kernel_single/models/net_g_525000.pth'
    # model_path = 'experiments/20230828_180443_Kernel/models/net_g_50000.pth'
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    # for k, v in checkpoint['params_ema'].items():
    #     print(k)

    model.load_state_dict(checkpoint['params'])
    # model.fuse_generator_block = {}
    # model.connect_list = {}
    model.to(device)
    model.eval()

    # degradation_model = DegradationModel(kernel_size=19).to(device)

    dataloader = get_dataloader()
    pbar = tqdm(total=len(dataloader), unit='image')
    from sklearnex import patch_sklearn, unpatch_sklearn
    patch_sklearn()
    from sklearn.cluster import KMeans
    import random

    for idx, val_data in enumerate(dataloader):
        print(idx)
        if idx < 997:
            continue
        val_data['lq'] = val_data['lq'].to(device)
        # val_data['gt'] = val_data['gt'].to(device)

        with torch.no_grad():
            # kernel, cb_loss = model(img_in)
            kernel = model(val_data['lq'])
            # pred = degradation_model(val_data['gt'], kernel)
        
        

        kernel = kernel.view(1, kernel.shape[1], -1).detach().cpu().numpy()[0]
        print(kernel.shape)

        # kernel = kernel[:100, :]
        n_clusters = 4

        y_pred = KMeans(n_clusters=n_clusters, random_state=9).fit_predict(kernel)
        # print(y_pred, y_pred.shape)
        kernel_cluster = [[]]
        # kernels = [[] for i in range(n_clusters)]
        for i in range(n_clusters):
            num = np.sum(y_pred==i)
            rand_idx = random.randint(0, num-1)
            kernel_cluster[0].append(kernel[y_pred==i][rand_idx].reshape(19, 19))
            # for j in range(0, 10000, 2000):
            #     kernels[i].append(kernel[y_pred==i][j].reshape(19, 19))
            # kernels[i].append(kernel[y_pred==i][0:10000:2000])
        kernel_cluster = np.array(kernel_cluster)
        np.save(f'tmp/kernels/{idx}.npy', kernel_cluster)
        print(idx)
        # kernels = np.array(kernels)
        # print(kernel_cluster.shape, kernels.shape)
        # plot_canvas(kernel_cluster, 'tmp/kernel_cluster2.png')
        # plot_canvas(kernels, 'tmp/kernels2.png')

        torch.cuda.empty_cache()

        pbar.update(1)
        pbar.set_description(f'Test')
    pbar.close()



def get_dataloader():
    with open('options/double_test.yml', 'r') as f:
        cfg = yaml.safe_load(f)

    cfg_data = cfg['datasets']['val']
    dataset = PairedImagePetrelTestDataset(cfg_data)
    loader = DataLoader(dataset, batch_size = 1, shuffle=False)

    return loader




def calculate_psnr_new(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False):

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    if type(img1) == torch.Tensor:
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1,2,0)
    if type(img2) == torch.Tensor:
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1,2,0)
        
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    max_value = 1. if img1.max() <= 1 else 255.
    return 20. * np.log10(max_value / np.sqrt(mse))


def calculate_psnr(img1, img2, crop_border, input_order='HWC', test_y_channel=False):

    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def test_kernel_all():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # with open('options/Kernel_single.yml', 'r') as f:
    #     cfg = yaml.safe_load(f)

    # cfg_vqgan = cfg['network_g']
    # model_type = cfg_vqgan.pop('type')
    # model = KernelSingleArch(**cfg_vqgan)
    # model_path = 'experiments/20230826_113032_Kernel_single/models/net_g_525000.pth'
    # checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    # # for k, v in checkpoint['params_ema'].items():
    # #     print(k)

    # model.load_state_dict(checkpoint['params'])
    # # model.fuse_generator_block = {}
    # # model.connect_list = {}
    # model.to(device)
    # model.eval()


    with open('options/Kernel1.yml', 'r') as f:
        cfg = yaml.safe_load(f)

    cfg_vqgan = cfg['network_g']
    model_type = cfg_vqgan.pop('type')
    # model = KernelSingleArch(**cfg_vqgan)
    # model = KernelDecoderArch(**cfg_vqgan)
    model = KernelArch(**cfg_vqgan)
    
    # model_path = 'experiments/20230822_170249_Kernel/models/net_g_120000.pth'
    # model_path = 'experiments/20230815_225937_Kernel/models/net_g_785000.pth'
    # model_path = 'experiments/20230826_113032_Kernel_single/models/net_g_525000.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230830_164052_Kernel_init/models/net_g_50000.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230831_162837_Kernel_Decoder_weight/models/net_g_70000.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230831_161924_Kernel_freeze/models/net_g_10000.pth'
    # model_path = 'experiments/20230828_180443_Kernel/models/net_g_50000.pth'
    # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230905_121428_Kernel_weight_shape/models/net_g_20000.pth'
    # checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    # for k, v in checkpoint['params_ema'].items():
    #     print(k)

    # model.load_state_dict(checkpoint['params'])
    # model.fuse_generator_block = {}
    # model.connect_list = {}

    model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230826_113032_Kernel_single/models/net_g_latest.pth'
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    cfg_vqgan.pop('initialize')
    # cfg_vqgan.pop('fix_modules')
    cfg_vqgan['codebook_size'] = 1024
    new_model = KernelSingleArch(**cfg_vqgan)
    new_model.load_state_dict(checkpoint['params'])
    model.dcnn = new_model.dcnn
    # model = new_model
    del new_model
    model.to(device)

    model.eval()




    degradation_model = DegradationModel(kernel_size=19).to(device)

    dataloader = get_dataloader()
    pbar = tqdm(total=len(dataloader), unit='image')
    metrics = 0.0
    cnt = 0

    for idx, val_data in enumerate(dataloader):

        val_data['lq'] = val_data['lq'].to(device)
        val_data['gt'] = val_data['gt'].to(device)

        with torch.no_grad():
            kernel, cb_loss = model(val_data['lq'])
            # kernel = model(val_data['lq'])
            pred = degradation_model(val_data['gt'], kernel)
        
        torch.cuda.empty_cache()

        pred = pred.detach().cpu()
        pred = tensor2img(pred, rgb2bgr=True, out_type = np.float32)

        img_in = val_data['lq'].detach().cpu()
        img_in = tensor2img(img_in, rgb2bgr=True, out_type = np.float32)

        metrics += calculate_psnr_new(pred, img_in, 0)
        cnt += 1
        print('curr metric:', metrics / cnt)

        pbar.update(1)
        pbar.set_description(f'Test')
    pbar.close()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test-all", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--extract-all", action="store_true")
    #
    args = parser.parse_args()
    if args.test:
        test_kernel()
    if args.test_all:
        # print('test_all')
        test_kernel_all()
    if args.extract:
        extact_kernel()
    if args.extract_all:
        extract_all_kernels()
    # test_kernel_all()
    # 
    # extract_kernel()
    # extract_all_kernels()
