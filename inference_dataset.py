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
from basicsr.data import build_dataloader, build_dataset


with open('options/CodeFormer_stage3_lowde.yml', 'r') as f:
    cfg = yaml.safe_load(f)


for phase, dataset_opt in cfg['datasets'].items():
    if phase == 'train':
        dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
        train_set = build_dataset(dataset_opt)
        # train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
        # train_loader = build_dataloader(
        #     train_set,
        #     dataset_opt,
        #     num_gpu=cfg['num_gpu'],
        #     dist=cfg['dist'],
        #     sampler=train_sampler,
        #     seed=cfg['manual_seed'])

if train_set:
    return_dict = train_set[0]
    for key in return_dict.keys():
        print(key)
    img_in = return_dict['in']
    # img_in_large_de = return_dict['in_large_de']
    img_gt = return_dict['gt']
    if 'in_large_de' in return_dict.keys():
        img_large = return_dict['in_large_de']
        img_large = img_large * 0.5 + 0.5
        img_large = img_large.detach().numpy().transpose(1,2,0)
        img_large = cv2.cvtColor(img_large, cv2.COLOR_RGB2BGR)
        cv2.imwrite('tmp/img_large_de.png', img_large*255)


    img_in = img_in * 0.5 + 0.5
    # img_in_large_de = img_in_large_de * 0.5 + 0.5
    img_gt = img_gt * 0.5 + 0.5

    img_in = img_in.detach().numpy().transpose(1,2,0)
    img_in = cv2.cvtColor(img_in, cv2.COLOR_RGB2BGR)
    # img_in_large_de = img_in_large_de.detach().numpy().transpose(1,2,0)
    img_gt = img_gt.detach().numpy().transpose(1,2,0)
    img_gt = cv2.cvtColor(img_gt, cv2.COLOR_RGB2BGR)
    # print(type(img_in), type(img_in_large_de), type(img_gt))
    # print(img_in.shape, img_in_large_de.shape, img_gt.shape)
    cv2.imwrite('tmp/img_in.png', img_in*255)
    # cv2.imwrite('tmp/img_in_large_de.png', img_in_large_de*255)
    cv2.imwrite('tmp/img_gt.png', img_gt*255)
