import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange


from vqfr.utils.img_util import trunc_normal_
from vqfr.archs.vqganv2_arch import ResnetBlock, VQGANDecoder, VQGANEncoder, build_quantizer
from vqfr.ops.dcn import ModulatedDeformConvPack, modulated_deform_conv
from vqfr.utils import get_root_logger
from basicsr.utils.registry import ARCH_REGISTRY



class ContextFuseModule(nn.Module):
    def __init__(self, channel, cond_channels, cond_downscale_rate):
        super(ContextFuseModule, self).__init__()
        self.cond_downscale_rate = cond_downscale_rate
        self.offset_conv1 = nn.Sequential(
            nn.Conv2d(channel + cond_channels, channel, kernel_size=1),
            nn.GroupNorm(num_groups=32, num_channels=channel, eps=1e-6, affine=True), nn.SiLU(inplace=True),
            nn.Conv2d(channel, channel, groups=channel, kernel_size=7, padding=3),
            nn.GroupNorm(num_groups=32, num_channels=channel, eps=1e-6, affine=True), nn.SiLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=1))

        self.offset_conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.GroupNorm(num_groups=32, num_channels=channel, eps=1e-6, affine=True), nn.SiLU(inplace=True))

    def forward(self, x_main, inpfeat):
        _, _, h, w = inpfeat.shape
        # interp inpfeat to x_main size
        # print('x: ', x_main.shape, 'inp: ', inpfeat.shape)

        inpfeat = F.interpolate(
            inpfeat,
            size=(h // self.cond_downscale_rate, w // self.cond_downscale_rate),
            mode='bilinear',
            align_corners=False)
        x = self.offset_conv1(torch.cat([inpfeat, x_main], dim=1))
        x = self.offset_conv2(x)
        return x


class MainDecoder(nn.Module):
    def __init__(self, base_channels, channel_multipliers, align_opt):
        super(MainDecoder, self).__init__()
        self.num_levels = len(channel_multipliers)  # [1, 2, 2, 4, 4, 8]

        self.decoder_dict = nn.ModuleDict()
        self.pre_upsample_dict = nn.ModuleDict()
        self.merge_func_dict = nn.ModuleDict()

        for i in reversed(range(self.num_levels)):
            if i == self.num_levels - 1:
                channels_prev = base_channels * channel_multipliers[i]
            else:
                channels_prev = base_channels * channel_multipliers[i + 1]
            channels = base_channels * channel_multipliers[i]

            if i != self.num_levels - 1:
                self.pre_upsample_dict['Level_%d' % 2**i] = \
                    nn.Sequential(
                        nn.UpsamplingNearest2d(scale_factor=2),
                        nn.Conv2d(channels_prev, channels, kernel_size=3, padding=1))


            self.merge_func_dict['Level_%d' % (2**i)] = \
                ContextFuseModule(
                    channel=channels,
                    cond_channels=align_opt['cond_channels'],
                    cond_downscale_rate=2**i
                    )

            if i != self.num_levels - 1:
                self.decoder_dict['Level_%d' % 2**i] = ResnetBlock(2 * channels, channels)

    def forward(self, kernel_feat, inpfeat, fidelity_ratio=1.0):
        x = self.merge_func_dict['Level_%d' % 2**(self.num_levels - 1)](
            kernel_feat['Level_%d' % 2**(self.num_levels - 1)], inpfeat)
            
        # print("main decoder 0：", x.shape)

        for scale in reversed(range(self.num_levels - 1)):
            x = self.pre_upsample_dict['Level_%d' % 2**scale](x)
            fuse_feat = self.merge_func_dict['Level_%d' % 2**scale](
                kernel_feat['Level_%d' % 2**scale], inpfeat)
            x = self.decoder_dict['Level_%d' % 2**scale](torch.cat([x, fuse_feat], dim=1))

            # print("main decoder {}：".format(scale), x.shape)

        return x

@ARCH_REGISTRY.register()
class DeblurTwoBranch(nn.Module):
    def __init__(self, base_channels, channel_multipliers, num_enc_blocks, use_enc_attention, num_dec_blocks,
                 use_dec_attention, code_dim, inpfeat_dim, code_selection_mode, align_opt, quantizer_opt):
        super().__init__()

        self.encoder = VQGANEncoder(
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_blocks=num_enc_blocks,
            use_enc_attention=use_enc_attention,
            code_dim=code_dim)

        if code_selection_mode == 'Nearest':
            self.feat2index = None
        elif code_selection_mode == 'Predict':
            self.feat2index = nn.Sequential(
                nn.LayerNorm(quantizer_opt['code_dim']), nn.Linear(quantizer_opt['code_dim'],
                                                                   quantizer_opt['num_code']))

        self.decoder = VQGANDecoder(
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_blocks=num_dec_blocks,
            use_dec_attention=use_dec_attention,
            code_dim=code_dim)

        self.main_branch = MainDecoder(
            base_channels=base_channels, channel_multipliers=channel_multipliers, align_opt=align_opt)
        self.inpfeat_extraction = nn.Conv2d(3, inpfeat_dim, 3, padding=1)

        self.quantizer = build_quantizer(quantizer_opt)

        self.apply(self._init_weights)

    @torch.no_grad()
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_last_layer(self):
        return self.decoder.conv_out[-1].weight

    def check_image_size(self, x, padder_size):
        _, _, h, w = x.size()
        mod_pad_h = (padder_size - h % padder_size) % padder_size
        mod_pad_w = (padder_size - w % padder_size) % padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


    def forward(self, x_lq, fidelity_ratio=1.0):
        _, _, h, w = x_lq.shape

        x_lq = self.check_image_size(x_lq, 32)

        inp_feat = self.inpfeat_extraction(x_lq)
        res = {}
        enc_feat = self.encoder(x_lq)
        res['enc_feat'] = enc_feat

        if self.feat2index is not None:
            # cross-entropy predict token
            enc_feat = rearrange(enc_feat, 'b c h w -> b (h w) c')
            quant_logit = self.feat2index(enc_feat)
            res['quant_logit'] = quant_logit
            quant_index = quant_logit.argmax(-1)
            quant_feat = self.quantizer.get_feature(quant_index)
        else:
            # nearest predict token
            quant_feat, emb_loss, quant_index = self.quantizer(enc_feat)
            res['codebook_loss'] = emb_loss

        # print('quant feat: ', quant_feat.shape)
        with torch.no_grad():
            # texture dec is output RGB of texture branch
            _, texture_feat_dict = self.decoder(quant_feat, return_feat=True)

        # for k,v in texture_feat_dict.items():
            # print('quant feat ', k, v.shape)

        main_feature = self.main_branch(texture_feat_dict, inp_feat, fidelity_ratio=fidelity_ratio)

        main_dec = self.decoder.conv_out(main_feature)
        # print('main dec: ', main_dec.shape)
        # print('texture dec: ', texture_dec.shape)

        res['main_dec'] = main_dec[:,:,:h,:w]
        return res

