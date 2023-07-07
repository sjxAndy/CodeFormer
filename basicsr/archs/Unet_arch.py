#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-03-20 19:48:14
# Adapted from https://github.com/jvanvugt/pytorch-unet

import torch
from torch import nn
import torch.nn.functional as F

def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer

def conv_down(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer

def conv_down1(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=2, padding=1, bias=bias)
    return layer

class UNet(nn.Module):
    def __init__(self, in_chn=3, out_chn=3, wf=32, depth=5, bias=True):
        super(UNet, self).__init__()
        self.depth = depth
        self.down_path = nn.ModuleList()
        prev_channels = in_chn
        for i in range(depth):
            downsample = True if (i+1) < depth else False
            self.down_path.append(UNetConvBlock(prev_channels, (2**i)*wf, downsample, bias=bias))
            prev_channels = (2**i) * wf

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, (2**i)*wf, bias=bias))
            prev_channels = (2**i)*wf

        self.last = conv3x3(prev_channels, out_chn, bias=bias)

    def forward(self, x1):
        blocks = []
        for i, down in enumerate(self.down_path):
            if (i+1) < self.depth:
                x1, x1_up = down(x1)
                blocks.append(x1_up)
            else:
                x1 = down(x1)
        mid = []
        for i, up in enumerate(self.up_path):
            x1, mid_i = up(x1, blocks[-i-1])
            mid.append(mid_i)

        out = self.last(x1)
        return [out] + mid

class UNet_student(nn.Module):
    def __init__(self, in_chn=3, out_chn=3, wf=8, depth=5, bias=True):
        super(UNet_student, self).__init__()
        self.depth = depth
        self.down_path = nn.ModuleList()
        prev_channels = in_chn
        for i in range(depth):
            downsample = True if (i+1) < depth else False
            self.down_path.append(UNetConvBlock2(prev_channels, (2**i)*wf, downsample, bias=bias))
            prev_channels = (2**i) * wf

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock1(prev_channels, (2**i)*wf, bias=bias))
            prev_channels = (2**i)*wf

        self.last = conv3x3(prev_channels, out_chn, bias=bias)

    def forward(self, x1):
        inp = x1
        blocks = []
        for i, down in enumerate(self.down_path):
            print(f'---------------{i}-----------------')
            if (i+1) < self.depth:
                x1, x1_up = down(x1)
                print('x1: ', x1.shape)
                print('x1_up: ', x1_up.shape)
                blocks.append(x1_up)
            else:
                x1 = down(x1)

        print('block len: ', len(blocks))
        print('final x1:', x1.shape)

        for i, up in enumerate(self.up_path):
            ## up and interp x1, add to res
            x1 = up(x1, blocks[-i-1])

        out = self.last(x1)
        return out+inp


class UNet_student_mask(nn.Module):
    def __init__(self, mask_chn=3, out_chn=3, wf=8, depth=5, bias=True, disable_iso = False):
        super(UNet_student_mask, self).__init__()
        self.depth = depth
        self.down_path = nn.ModuleList()
        prev_channels = 6
        for i in range(depth):
            downsample = True if (i+1) < depth else False
            self.down_path.append(UNetConvBlock2(prev_channels, (2**i)*wf, downsample, bias=bias))
            prev_channels = (2**i) * wf

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock1(prev_channels, (2**i)*wf, bias=bias))
            prev_channels = (2**i)*wf

        self.last = conv3x3(prev_channels, out_chn, bias=bias)

        self.mask_conv = conv3x3(mask_chn, wf, bias=True)

        self.disable_iso = disable_iso

    def forward(self, src, ref, seg_mask, occ_mask, iso_mask):
        inp = src
        x1 = torch.cat((src, ref), 1)
        
        if not self.disable_iso:
            other_input = torch.cat((seg_mask, occ_mask, iso_mask), 1)
        else:
            other_input = torch.cat((seg_mask, occ_mask), 1)

        mask_encoder = self.mask_conv(other_input)

        blocks = []
        for i, down in enumerate(self.down_path):
            if (i+1) < self.depth:
                x1, x1_up = down(x1)
                if i == 0:
                    x1 = x1 + mask_encoder

                blocks.append(x1_up)
            else:
                x1 = down(x1)

        for i, up in enumerate(self.up_path):
            x1 = up(x1, blocks[-i-1])

        out = self.last(x1)

        out = out + inp

        out = torch.clamp(out, 0, 1)

        return out



class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, bias=False):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.block = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=bias),
            nn.ReLU(inplace=True))

        if downsample:
            self.downsample = conv_down(out_size, out_size)

    def forward(self, x):
        out = self.block(x)
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out

class UNetConvBlock2(nn.Module):
    def __init__(self, in_size, out_size, downsample, bias=False):
        super(UNetConvBlock2, self).__init__()
        self.downsample = downsample
        self.block = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=bias),
            nn.ReLU(inplace=True))

        if downsample:
            self.downsample = conv_down1(out_size, out_size)

    def forward(self, x):
        out = self.block(x)
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out

class UNetConvBlock1(nn.Module):
    def __init__(self, in_size, out_size, downsample, bias=False):
        super(UNetConvBlock1, self).__init__()
        self.downsample = downsample
        block = [
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=bias),
            nn.ReLU(inplace=True)]
        if in_size != 8 and out_size != 8:
            block.extend([
                nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=bias),
                nn.ReLU(inplace=True)])
        self.block = nn.Sequential(*block)

        if downsample:
            self.downsample = conv_down1(out_size, out_size)

    def forward(self, x):
        out = self.block(x)
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out

class UNetUpBlock1(nn.Module):
    def __init__(self, in_size, out_size, bias=False):
        super(UNetUpBlock1, self).__init__()
        #self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.upconv = nn.Conv2d(in_size, out_size, kernel_size=3, stride=1,padding=1, bias=bias)# k=3,pad=1
        self.conv_block = UNetConvBlock1(in_size//2, out_size, False, bias)

    def forward(self, x, bridge):
        up = self.upconv(x)
        up = F.interpolate(up, scale_factor=2, mode='bilinear')
        #up = self.up(x)
        #up = self.upconv(F.interpolate(x, scale_factor=2, mode='nearest'))
        #out = torch.cat([up, bridge], 1)
        out = up + bridge
        out = self.conv_block(out)

        return out#, up


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, bias=False):
        super(UNetUpBlock, self).__init__()
        #self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.upconv = nn.Conv2d(in_size, out_size, kernel_size=3, stride=1,padding=1, bias=bias)# k=3,pad=1
        self.conv_block = UNetConvBlock(in_size//2, out_size, False, bias)

    def forward(self, x, bridge):
        up = self.upconv(x)
        up = F.interpolate(up, scale_factor=2, mode='nearest')
        #up = self.up(x)
        #up = self.upconv(F.interpolate(x, scale_factor=2, mode='nearest'))
        #out = torch.cat([up, bridge], 1)
        out = up + bridge
        out = self.conv_block(out)

        return out, up


if __name__ == "__main__":
    net = UNet_student_mask(mask_chn = 3, wf=64, depth=5)

    def prepare_input(resolution):
        src = torch.rand(1, 3, 1024, 1024)
        ref = torch.rand(1, 3, 1024, 1024)
        mask1 = torch.rand(1, 1, 512, 512)
        mask2 = torch.rand(1, 1, 512, 512)
        mask3 = torch.rand(1, 1, 512, 512)

        return dict(src=src, ref=ref, seg_mask = mask1, occ_mask = mask2, iso_mask = mask3)


    from ptflops import get_model_complexity_info
    inp_shape = (3, 1024, 1024)

    macs, params = get_model_complexity_info(net, inp_shape, input_constructor=prepare_input, verbose=False, print_per_layer_stat=False, as_strings=True)


    print(macs, params)