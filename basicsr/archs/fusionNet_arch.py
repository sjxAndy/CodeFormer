import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()

        config = [16, 32, 64, 128, 256, 512, 1024]

        self.conv_ds = []
        self.conv_up = []

        self.conv0 = nn.Sequential(
                nn.Conv2d(3, config[0], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
        )

        self.conv0_ds = nn.Sequential(
                nn.Conv2d(config[0], config[0], kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
        )


        self.conv1 = nn.Sequential(
                nn.Conv2d(5, config[0], kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            )

        self.layer1 = nn.Sequential(
                nn.Conv2d(config[0], config[1], kernel_size=3, stride=1, padding=1),
                nn.ReLU()
        )

        self.conv_end = nn.Sequential(
                nn.Conv2d(config[0], 3, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
        )



        ## encoder
        for i in range(len(config) - 1):
            layer_conv = nn.Sequential(
                nn.Conv2d(config[i], config[i], kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(config[i], config[i+1], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
            self.conv_ds.append(layer_conv)


        self.conv_ds = nn.Sequential(*self.conv_ds)


        ## decoder
        for i in range(len(config) - 1, 0, -1):
            layer_conv = nn.Sequential(
                nn.Conv2d(config[i], config[i], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(config[i], config[i-1], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners = True)   
            )

            self.conv_up.append(layer_conv)


        self.conv_up = nn.Sequential(*self.conv_up)

        self._init_weights()


    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Conv2d,
                nn.Conv3d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
                nn.Linear,
            }:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)


    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x



    def forward(self, src):
        B, C, H, W = inp.shape
        inp = self.check_image_size(src)


        other_input = torch.cat((ref, seg_mask, occ_mask), 1)

        src_f1 = self.conv0(src)
        src_feat = self.conv0_ds(src_f1)
        oi_feat = self.conv1(other_input)

        input = src_feat + oi_feat
        ds1 = self.layer1(input)     ## output channel:32, 1/2HW
        ds2 = self.conv_ds[1](ds1)

        ds3 = self.conv_ds[2](ds2)
        ds4 = self.conv_ds[3](ds3)
        ds5 = self.conv_ds[4](ds4)
        ds6 = self.conv_ds[5](ds5)   ## output channel: 1024

        up5 = self.conv_up[0](ds6)
        up4 = self.conv_up[1](up5 + ds5)
        up3 = self.conv_up[2](up4 + ds4)
        up2 = self.conv_up[3](up3 + ds3)
        up1 = self.conv_up[4](up2 + ds2)
        up0 = self.conv_up[5](up1 + ds1)   ## 16, 1/2h, 1/2W

        out = src + self.conv_end(up0 + src_f1)

        out = torch.clamp(out, 0, 1)

        return out


class FusionNet256(nn.Module):
    def __init__(self):
        super(FusionNet256, self).__init__()

        config = [16, 32, 64, 128, 256]

        self.conv_ds = []
        self.conv_up = []

        self.conv0 = nn.Sequential(
                nn.Conv2d(3, config[0], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
        )

        self.conv0_ds = nn.Sequential(
                nn.Conv2d(config[0], config[0], kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
        )


        self.conv1 = nn.Sequential(
                nn.Conv2d(5, config[0], kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            )




        self.layer1 = nn.Sequential(
                nn.Conv2d(config[0], config[1], kernel_size=3, stride=1, padding=1),
                nn.ReLU()
        )

        self.conv_end = nn.Sequential(
                nn.Conv2d(config[0], 3, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
        )



        ## encoder
        for i in range(len(config) - 1):
            layer_conv = nn.Sequential(
                nn.Conv2d(config[i], config[i], kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(config[i], config[i+1], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
            self.conv_ds.append(layer_conv)


        self.conv_ds = nn.Sequential(*self.conv_ds)


        ## decoder
        for i in range(len(config) - 1, 0, -1):
            layer_conv = nn.Sequential(
                nn.Conv2d(config[i], config[i], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(config[i], config[i-1], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners = True)   
            )

            self.conv_up.append(layer_conv)


        self.conv_up = nn.Sequential(*self.conv_up)


    def forward(self, src, ref, seg_mask, occ_mask):
        other_input = torch.cat((ref, seg_mask, occ_mask), 1)

        src_f1 = self.conv0(src)
        src_feat = self.conv0_ds(src_f1)
        oi_feat = self.conv1(other_input)

        input = src_feat + oi_feat
        ds1 = self.layer1(input)     ## output channel:32, 1/2HW
        ds2 = self.conv_ds[1](ds1)
        ds3 = self.conv_ds[2](ds2)
        ds4 = self.conv_ds[3](ds3)

        up3 = self.conv_up[0](ds4)
        up2 = self.conv_up[1](up3 + ds3)
        up1 = self.conv_up[2](up2 + ds2)
        up0 = self.conv_up[3](up1 + ds1)   ## 16, 1/2h, 1/2W

        out = src + self.conv_end(up0 + src_f1)

        out = torch.clamp(out, 0, 1)

        return out


class FusionNet_woRes(nn.Module):
    def __init__(self):
        super(FusionNet_woRes, self).__init__()

        config = [16, 32, 64, 128, 256, 512, 1024]

        self.conv_ds = []
        self.conv_up = []

        self.conv0 = nn.Sequential(
                nn.Conv2d(3, config[0], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
        )

        self.conv0_ds = nn.Sequential(
                nn.Conv2d(config[0], config[0], kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
        )


        self.conv1 = nn.Sequential(
                nn.Conv2d(5, config[0], kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            )




        self.layer1 = nn.Sequential(
                nn.Conv2d(config[0], config[1], kernel_size=3, stride=1, padding=1),
                nn.ReLU()
        )

        self.conv_end = nn.Sequential(
                nn.Conv2d(config[0], 3, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
        )



        ## encoder
        for i in range(len(config) - 1):
            layer_conv = nn.Sequential(
                nn.Conv2d(config[i], config[i], kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(config[i], config[i+1], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
            self.conv_ds.append(layer_conv)


        self.conv_ds = nn.Sequential(*self.conv_ds)


        ## decoder
        for i in range(len(config) - 1, 0, -1):
            layer_conv = nn.Sequential(
                nn.Conv2d(config[i], config[i], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(config[i], config[i-1], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners = True)   
            )

            self.conv_up.append(layer_conv)


        self.conv_up = nn.Sequential(*self.conv_up)




    def forward(self, src, ref, seg_mask, occ_mask):
        other_input = torch.cat((ref, seg_mask, occ_mask), 1)

        src_f1 = self.conv0(src)
        src_feat = self.conv0_ds(src_f1)
        oi_feat = self.conv1(other_input)

        input = src_feat + oi_feat
        ds1 = self.layer1(input)     ## output channel:32, 1/2HW
        ds2 = self.conv_ds[1](ds1)

        ds3 = self.conv_ds[2](ds2)
        ds4 = self.conv_ds[3](ds3)
        ds5 = self.conv_ds[4](ds4)
        ds6 = self.conv_ds[5](ds5)   ## output channel: 1024

        up5 = self.conv_up[0](ds6)
        up4 = self.conv_up[1](up5 + ds5)
        up3 = self.conv_up[2](up4 + ds4)
        up2 = self.conv_up[3](up3 + ds3)
        up1 = self.conv_up[4](up2 + ds2)
        up0 = self.conv_up[5](up1 + ds1)   ## 16, 1/2h, 1/2W

        out = self.conv_end(up0 + src_f1)

        out = torch.clamp(out, 0, 1)

        return out

class FusionNet_woSrc(nn.Module):
    def __init__(self):
        super(FusionNet_woSrc, self).__init__()

        config = [16, 32, 64, 128, 256, 512, 1024]

        self.conv_ds = []
        self.conv_up = []

        self.conv0 = nn.Sequential(
                nn.Conv2d(3, config[0], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
        )

        self.conv0_ds = nn.Sequential(
                nn.Conv2d(config[0], config[0], kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
        )


        self.conv1 = nn.Sequential(
                nn.Conv2d(5, config[0], kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            )




        self.layer1 = nn.Sequential(
                nn.Conv2d(config[0], config[1], kernel_size=3, stride=1, padding=1),
                nn.ReLU()
        )

        self.conv_end = nn.Sequential(
                nn.Conv2d(config[0], 3, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
        )



        ## encoder
        for i in range(len(config) - 1):
            layer_conv = nn.Sequential(
                nn.Conv2d(config[i], config[i], kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(config[i], config[i+1], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
            self.conv_ds.append(layer_conv)


        self.conv_ds = nn.Sequential(*self.conv_ds)


        ## decoder
        for i in range(len(config) - 1, 0, -1):
            layer_conv = nn.Sequential(
                nn.Conv2d(config[i], config[i], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(config[i], config[i-1], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners = True)   
            )

            self.conv_up.append(layer_conv)


        self.conv_up = nn.Sequential(*self.conv_up)




    def forward(self, src, ref, seg_mask, occ_mask):
        other_input = torch.cat((ref, seg_mask, occ_mask), 1)

        oi_feat = self.conv1(other_input)

        input = oi_feat
        ds1 = self.layer1(input)     ## output channel:32, 1/2HW
        ds2 = self.conv_ds[1](ds1)

        ds3 = self.conv_ds[2](ds2)
        ds4 = self.conv_ds[3](ds3)
        ds5 = self.conv_ds[4](ds4)
        ds6 = self.conv_ds[5](ds5)   ## output channel: 1024

        up5 = self.conv_up[0](ds6)
        up4 = self.conv_up[1](up5 + ds5)
        up3 = self.conv_up[2](up4 + ds4)
        up2 = self.conv_up[3](up3 + ds3)
        up1 = self.conv_up[4](up2 + ds2)
        up0 = self.conv_up[5](up1 + ds1)   ## 16, 1/2h, 1/2W

        out = self.conv_end(up0)

        out = torch.clamp(out, 0, 1)

        return out

class FusionNet_swapSrcRef(nn.Module):
    def __init__(self):
        super(FusionNet_swapSrcRef, self).__init__()

        config = [16, 32, 64, 128, 256, 512, 1024]

        self.conv_ds = []
        self.conv_up = []

        self.conv0 = nn.Sequential(
                nn.Conv2d(3, config[0], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
        )

        self.conv0_ds = nn.Sequential(
                nn.Conv2d(config[0], config[0], kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
        )


        self.conv1 = nn.Sequential(
                nn.Conv2d(5, config[0], kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            )




        self.layer1 = nn.Sequential(
                nn.Conv2d(config[0], config[1], kernel_size=3, stride=1, padding=1),
                nn.ReLU()
        )

        self.conv_end = nn.Sequential(
                nn.Conv2d(config[0], 3, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
        )



        ## encoder
        for i in range(len(config) - 1):
            layer_conv = nn.Sequential(
                nn.Conv2d(config[i], config[i], kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(config[i], config[i+1], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
            self.conv_ds.append(layer_conv)


        self.conv_ds = nn.Sequential(*self.conv_ds)


        ## decoder
        for i in range(len(config) - 1, 0, -1):
            layer_conv = nn.Sequential(
                nn.Conv2d(config[i], config[i], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(config[i], config[i-1], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners = True)   
            )

            self.conv_up.append(layer_conv)


        self.conv_up = nn.Sequential(*self.conv_up)



    def forward(self, src, ref, seg_mask, occ_mask):
        ref = F.interpolate(ref, scale_factor = 2, mode = 'bilinear')
        src = F.interpolate(src, scale_factor = 0.5, mode = 'bilinear')

        other_input = torch.cat((src, seg_mask, occ_mask), 1)

        src_f1 = self.conv0(ref)
        src_feat = self.conv0_ds(src_f1)
        oi_feat = self.conv1(other_input)

        input = src_feat + oi_feat
        ds1 = self.layer1(input)     ## output channel:32, 1/2HW
        ds2 = self.conv_ds[1](ds1)

        ds3 = self.conv_ds[2](ds2)
        ds4 = self.conv_ds[3](ds3)
        ds5 = self.conv_ds[4](ds4)
        ds6 = self.conv_ds[5](ds5)   ## output channel: 1024

        up5 = self.conv_up[0](ds6)
        up4 = self.conv_up[1](up5 + ds5)
        up3 = self.conv_up[2](up4 + ds4)
        up2 = self.conv_up[3](up3 + ds3)
        up1 = self.conv_up[4](up2 + ds2)
        up0 = self.conv_up[5](up1 + ds1)   ## 16, 1/2h, 1/2W

        out = ref + self.conv_end(up0 + src_f1)

        out = torch.clamp(out, 0, 1)

        return out

class FusionNet_concat(nn.Module):
    def __init__(self):
        super(FusionNet_concat, self).__init__()

        config = [16, 32, 64, 128, 256, 512, 1024]

        self.conv_ds = []
        self.conv_up = []

        self.conv0 = nn.Sequential(
                nn.Conv2d(6, config[0], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
        )

        self.conv0_ds = nn.Sequential(
                nn.Conv2d(config[0], config[0], kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
        )


        self.conv1 = nn.Sequential(
                nn.Conv2d(2, config[0], kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            )




        self.layer1 = nn.Sequential(
                nn.Conv2d(config[0], config[1], kernel_size=3, stride=1, padding=1),
                nn.ReLU()
        )

        self.conv_end = nn.Sequential(
                nn.Conv2d(config[0], 3, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
        )



        ## encoder
        for i in range(len(config) - 1):
            layer_conv = nn.Sequential(
                nn.Conv2d(config[i], config[i], kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(config[i], config[i+1], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
            self.conv_ds.append(layer_conv)


        self.conv_ds = nn.Sequential(*self.conv_ds)


        ## decoder
        for i in range(len(config) - 1, 0, -1):
            layer_conv = nn.Sequential(
                nn.Conv2d(config[i], config[i], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(config[i], config[i-1], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners = True)   
            )

            self.conv_up.append(layer_conv)


        self.conv_up = nn.Sequential(*self.conv_up)




    def forward(self, src, ref, seg_mask, occ_mask):
        ref = F.interpolate(ref, scale_factor = 2, mode = 'bilinear')

        other_input = torch.cat((seg_mask, occ_mask), 1)
        img_input = torch.cat((src, ref), 1)

        src_f1 = self.conv0(img_input)
        src_feat = self.conv0_ds(src_f1)
        oi_feat = self.conv1(other_input)

        input = src_feat + oi_feat
        ds1 = self.layer1(input)     ## output channel:32, 1/2HW
        ds2 = self.conv_ds[1](ds1)

        ds3 = self.conv_ds[2](ds2)
        ds4 = self.conv_ds[3](ds3)
        ds5 = self.conv_ds[4](ds4)
        ds6 = self.conv_ds[5](ds5)   ## output channel: 1024

        up5 = self.conv_up[0](ds6)
        up4 = self.conv_up[1](up5 + ds5)
        up3 = self.conv_up[2](up4 + ds4)
        up2 = self.conv_up[3](up3 + ds3)
        up1 = self.conv_up[4](up2 + ds2)
        up0 = self.conv_up[5](up1 + ds1)   ## 16, 1/2h, 1/2W

        out = self.conv_end(up0 + src_f1)

        out = torch.clamp(out, 0, 1)

        return out

class FusionNet_SrcOnly(nn.Module):
    def __init__(self):
        super(FusionNet_SrcOnly, self).__init__()

        config = [16, 32, 64, 128, 256, 512, 1024]

        self.conv_ds = []
        self.conv_up = []

        self.conv0 = nn.Sequential(
                nn.Conv2d(3, config[0], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
        )

        self.conv0_ds = nn.Sequential(
                nn.Conv2d(config[0], config[0], kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
        )


        self.conv1 = nn.Sequential(
                nn.Conv2d(2, config[0], kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            )


        self.layer1 = nn.Sequential(
                nn.Conv2d(config[0], config[1], kernel_size=3, stride=1, padding=1),
                nn.ReLU()
        )

        self.conv_end = nn.Sequential(
                nn.Conv2d(config[0], 3, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
        )



        ## encoder
        for i in range(len(config) - 1):
            layer_conv = nn.Sequential(
                nn.Conv2d(config[i], config[i], kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(config[i], config[i+1], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
            self.conv_ds.append(layer_conv)


        self.conv_ds = nn.Sequential(*self.conv_ds)


        ## decoder
        for i in range(len(config) - 1, 0, -1):
            layer_conv = nn.Sequential(
                nn.Conv2d(config[i], config[i], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(config[i], config[i-1], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners = True)   
            )

            self.conv_up.append(layer_conv)


        self.conv_up = nn.Sequential(*self.conv_up)

        self.padder_size = 2 ** len(config)

        print('padder size: ', self.padder_size)



    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

    def forward(self, src):
        B, C, H, W = src.shape
        src = self.check_image_size(src)

        src_f1 = self.conv0(src)
        src_feat = self.conv0_ds(src_f1)
        input = src_feat
        ds1 = self.layer1(input)     ## output channel:32, 1/2HW
        ds2 = self.conv_ds[1](ds1)
        ds3 = self.conv_ds[2](ds2)
        ds4 = self.conv_ds[3](ds3)
        ds5 = self.conv_ds[4](ds4)
        ds6 = self.conv_ds[5](ds5)   ## output channel: 1024

        up5 = self.conv_up[0](ds6)
        up4 = self.conv_up[1](up5 + ds5)
        up3 = self.conv_up[2](up4 + ds4)
        up2 = self.conv_up[3](up3 + ds3)
        up1 = self.conv_up[4](up2 + ds2)
        up0 = self.conv_up[5](up1 + ds1)   ## 16, 1/2h, 1/2W

        out = self.conv_end(up0 + src_f1)
        out = out + src

        return out[:, :, :H, :W]

class FusionNet_SrcOnly_noRes(nn.Module):
    def __init__(self):
        super(FusionNet_SrcOnly_noRes, self).__init__()

        config = [16, 32, 64, 128, 256, 512, 1024]

        self.conv_ds = []
        self.conv_up = []

        self.conv0 = nn.Sequential(
                nn.Conv2d(3, config[0], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
        )

        self.conv0_ds = nn.Sequential(
                nn.Conv2d(config[0], config[0], kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
        )


        self.conv1 = nn.Sequential(
                nn.Conv2d(2, config[0], kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            )


        self.layer1 = nn.Sequential(
                nn.Conv2d(config[0], config[1], kernel_size=3, stride=1, padding=1),
                nn.ReLU()
        )

        self.conv_end = nn.Sequential(
                nn.Conv2d(config[0], 3, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
        )



        ## encoder
        for i in range(len(config) - 1):
            layer_conv = nn.Sequential(
                nn.Conv2d(config[i], config[i], kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(config[i], config[i+1], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
            self.conv_ds.append(layer_conv)


        self.conv_ds = nn.Sequential(*self.conv_ds)


        ## decoder
        for i in range(len(config) - 1, 0, -1):
            layer_conv = nn.Sequential(
                nn.Conv2d(config[i], config[i], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(config[i], config[i-1], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners = True)   
            )

            self.conv_up.append(layer_conv)


        self.conv_up = nn.Sequential(*self.conv_up)



    def forward(self, src, ref, seg_mask, occ_mask):
        src_f1 = self.conv0(src)
        src_feat = self.conv0_ds(src_f1)
        input = src_feat
        ds1 = self.layer1(input)     ## output channel:32, 1/2HW
        ds2 = self.conv_ds[1](ds1)
        ds3 = self.conv_ds[2](ds2)
        ds4 = self.conv_ds[3](ds3)
        ds5 = self.conv_ds[4](ds4)
        ds6 = self.conv_ds[5](ds5)   ## output channel: 1024

        up5 = self.conv_up[0](ds6)
        up4 = self.conv_up[1](up5 + ds5)
        up3 = self.conv_up[2](up4 + ds4)
        up2 = self.conv_up[3](up3 + ds3)
        up1 = self.conv_up[4](up2 + ds2)
        up0 = self.conv_up[5](up1 + ds1)   ## 16, 1/2h, 1/2W

        out = self.conv_end(up0 + src_f1)

        out = torch.clamp(out, 0, 1)

        return out



if __name__ == '__main__':
    Net = FusionNet_woSrc()

    src = torch.rand(1, 3, 640, 640)
    ref = torch.rand(1, 3, 320, 320)
    mask1 = torch.rand(1, 1, 320, 320)
    mask2 = torch.rand(1, 1, 320, 320)


    out = Net.forward(src, ref, mask1, mask2)

    print(out.shape)

