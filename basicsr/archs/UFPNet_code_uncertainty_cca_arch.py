# ------------------------------------------------------------------------
# Modified from NAFNet (https://github.com/megvii-research/NAFNet)
# ------------------------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.arch_util import LayerNorm2d
from basicsr.archs.local_arch import Local_Base
# from basicsr.models.archs.Flow_arch import KernelPrior
from basicsr.archs.kernel_single_arch import KernelSingleArch
# from basicsr.models.archs.my_module import code_extra_mean_var
from basicsr.utils.registry import ARCH_REGISTRY

import numpy as np

class ResBlock(nn.Module):
    def __init__(self, ch):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
                        nn.ReLU(),
                        nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1))

    def forward(self, input):
        res = self.body(input)
        output = res + input
        return output


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)



# self.conv_kernel = nn.Sequential(
#                     nn.Conv2d(kernel_size*kernel_size, out_ch, kernel_size=3, stride=1, padding=1),
#                     nn.GELU(),
#                     nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
#                     nn.GELU()
#                     )
#     self.conv_2 = nn.Sequential(
#                     nn.Conv2d(out_ch*2, out_ch, kernel_size=3, stride=1, padding=1),
#                     nn.Sigmoid()
#                     )

# def forward(self, input, kernel):
#     x = self.conv_1(input)
#     kernel = self.conv_kernel(kernel)
#     att = torch.cat([x, kernel], dim=1)
#     att = self.conv_2(att)
#     x = x * att
#     output = x + input

#     return output

class CCA(nn.Module):
    """
    CCA Block
    """
    def __init__(self, kernel_size, in_ch, out_ch):
        super().__init__()
        # self.conv_x = nn.Sequential(
        #                 nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
        #                 nn.GELU()
        #                 )
        # self.conv_k = nn.Sequential(
        #                 nn.Conv2d(kernel_size*kernel_size, out_ch, kernel_size=3, stride=1, padding=1),
        #                 nn.GELU(),
        #                 nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        #                 nn.GELU()
        #                 )
        self.relu = nn.ReLU(inplace=True)
        self.sca_x = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.sca_k = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=kernel_size*kernel_size, out_channels=out_ch, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.conv_2 = nn.Sequential(
                        nn.Conv2d(out_ch*2, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.Sigmoid()
                        )

    def forward(self, x, kernel):
        x_sca = self.sca_x(x)
        k_sca = self.sca_k(kernel)
        sum_sca = torch.cat([x_sca, k_sca], dim=1)
        sum_sca = self.conv_2(sum_sca)
        # sum_sca = (x_sca + k_sca) / 2.0
        # sum_sca = torch.sigmoid(sum_sca)
        return x * sum_sca
        return self.relu(x * sum_sca)
        # # channel-wise attention
        # avg_pool_x = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        # channel_att_x = self.mlp_x(avg_pool_x)
        # avg_pool_g = F.avg_pool2d( g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        # channel_att_g = self.mlp_g(avg_pool_g)
        # channel_att_sum = (channel_att_x + channel_att_g)/2.0
        # scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        # x_after_channel = x * scale
        # out = self.relu(x_after_channel)
        # return out


class NAFBlock_CCA(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.cca = CCA(kernel_size, dw_channel // 2, dw_channel // 2)
        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp, kernel):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        # x = x * self.sca(x)
        x = self.cca(x, kernel)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class kernel_attention(nn.Module):
    def __init__(self, kernel_size, in_ch, out_ch):
        super(kernel_attention, self).__init__()

        self.conv_1 = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.GELU()
                        )
        self.conv_kernel = nn.Sequential(
                        nn.Conv2d(kernel_size*kernel_size, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.GELU(),
                        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.GELU()
                        )
        self.conv_2 = nn.Sequential(
                        nn.Conv2d(out_ch*2, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.Sigmoid()
                        )

    def forward(self, input, kernel):
        x = self.conv_1(input)
        kernel = self.conv_kernel(kernel)
        att = torch.cat([x, kernel], dim=1)
        att = self.conv_2(att)
        x = x * att
        output = x + input

        return output


class NAFBlock_kernel(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=21):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_atttion = kernel_attention(kernel_size, in_ch=c, out_ch=c)

    def forward(self, inp, kernel):
        x = inp

        # kernel [B, 19*19, H, W]
        x = self.kernel_atttion(x, kernel)

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class UFPNet_code_uncertainty_cca(nn.Module):
    def __init__(self, img_channel=3, width=64, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], kernel_size=19, kernel_kwargs={}):
        super().__init__()

        self.kernel_size = kernel_size
        # self.kernel_extra = code_extra_mean_var(kernel_size)
        self.kernel_extra = KernelSingleArch(**kernel_kwargs)

        # self.flow = KernelPrior(n_blocks=5, input_size=19 ** 2, hidden_size=25, n_hidden=1, kernel_size=19)

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.kernel_down = nn.ModuleList()


        chan = width
        for num in enc_blk_nums:
            if num==1:
                self.encoders.append(nn.Sequential(*[NAFBlock_CCA(chan, kernel_size=kernel_size) for _ in range(num)]))
            else:
                self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            self.kernel_down.append(nn.Conv2d(kernel_size * kernel_size, kernel_size * kernel_size, 2, 2))
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        with torch.no_grad():
            # kernel estimation: size [B, H*W, 19, 19]
            # kernel_code, kernel_var = self.kernel_extra(inp)
            # kernel_code = (kernel_code - torch.mean(kernel_code, dim=[2, 3], keepdim=True)) / torch.std(kernel_code,
            #                                                                                             dim=[2, 3],
            #                                                                                             keepdim=True)
            # # code uncertainty
            # sigma = kernel_var
            # kernel_code_uncertain = kernel_code * torch.sqrt(1 - torch.square(sigma)) + torch.randn_like(kernel_code) * sigma

            # kernel = generate_k(self.flow, kernel_code_uncertain.reshape(kernel_code.shape[0]*kernel_code.shape[1], -1))
            kernel = self.kernel_extra(inp)
            # kernel = kernel.reshape(kernel_code.shape[0], kernel_code.shape[1], self.kernel_size, self.kernel_size)
            kernel_blur = kernel

            kernel = kernel.permute(0, 2, 3, 1).reshape(B, self.kernel_size*self.kernel_size, inp.shape[2], inp.shape[3])

        x = self.intro(inp)

        encs = []

        for encoder, down, kernel_down in zip(self.encoders, self.downs, self.kernel_down):
            if len(encoder) == 1:
                x = encoder[0](x, kernel)
                kernel = kernel_down(kernel)
            else:
                x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        x = torch.clamp(x, 0.0, 1.0)

        return [kernel_blur, x[:, :, :H, :W]]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode='reflect')
        return x

@ARCH_REGISTRY.register()
class UFPNet_code_uncertainty_cca_Local(Local_Base, UFPNet_code_uncertainty_cca):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        UFPNet_code_uncertainty_cca.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)

