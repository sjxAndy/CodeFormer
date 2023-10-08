import os
import numpy as np
import torch
import random


all_npy = []
ff = np.load('/mnt/lustre/sunjixiang1/code/CodeFormer/tmp/kernels/0.npy')

print(ff.shape)
ff = ff.squeeze()
print(ff.shape)
print(ff.shape[0])
# f = [ff, ff]
# print(np.concatenate(f).shape)


for fname in os.listdir('/mnt/lustre/sunjixiang1/code/CodeFormer/tmp/kernels'):
    print(fname)
    if fname.endswith('.npy'):
        f = np.load(os.path.join('/mnt/lustre/sunjixiang1/code/CodeFormer/tmp/kernels', fname))
        f = f.squeeze()
        all_npy.append(f)

all_npy = np.concatenate(np.array(all_npy))
print(all_npy.shape)
rand_idx = random.sample(list(np.arange(all_npy.shape[0])), 2048)
weight = torch.FloatTensor(all_npy[rand_idx])
print(weight.shape)