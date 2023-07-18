import math
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, List

from basicsr.archs.vqgan_arch import *
from basicsr.archs.codeformer_arch import *
from basicsr.utils import get_root_logger
from basicsr.utils.registry import ARCH_REGISTRY


def normalize(num_groups, in_channels):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
    

@torch.jit.script
def swish(x):
    return x*torch.sigmoid(x)


def calc_mean_std(feat, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.

    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    """Adaptive instance normalization.

    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.

    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = normalize(16, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = normalize(16, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)

        return x + x_in


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerSALayer(nn.Module):
    def __init__(self, embed_dim, nhead=8, dim_mlp=2048, dropout=0.0, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model - MLP
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        
        # self attention
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        # ffn
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt

class Fuse_sft_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.encode_enc = ResBlock(2*in_ch, out_ch)

        self.scale = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

        self.shift = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

    def forward(self, enc_feat, dec_feat, w=1):
        enc_feat = self.encode_enc(torch.cat([enc_feat, dec_feat], dim=1))
        scale = self.scale(enc_feat)
        shift = self.shift(enc_feat)
        residual = w * (dec_feat * scale + shift)
        out = dec_feat + residual
        return out


@ARCH_REGISTRY.register()
class SRFormer(CodeFormer):
    def __init__(self, dim_embd=512, n_head=8, n_layers=9, 
                codebook_size=1024, latent_size=256,
                connect_list=['32', '64', '128', '256'],
                fix_modules=['quantize','generator'], vqgan_path=None, pix_option=0):
        super(SRFormer, self).__init__(dim_embd, n_head, n_layers, codebook_size, latent_size, connect_list, fix_modules, vqgan_path)
        # if codeformer_path is not None:
        #     self.load_state_dict(
        #         torch.load(codeformer_path, map_location='cpu')['params_ema'])
        self.pix_option = pix_option

        all_fix_modules = ['encoder', 'position_emb', 'feat_emb', 'ft_layers', 'idx_pred_layer', 'generator', 'quantize', 'fuse_convs_dict']
        for module in all_fix_modules:
            obj = getattr(self, module)
            if isinstance(obj, nn.Parameter):
                obj.requires_grad = False
            else:
                for param in obj.parameters():
                    param.requires_grad = False
        if self.pix_option == 0:
            self.downsampler = torch.nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=0)
            # self.downsample_normalize = normalize(3)
            self.upsampler = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        if self.pix_option == 1:
            self.downsampler1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=0)
            self.downres1 = ResBlock(16, 16)
            self.downres2 = ResBlock(16, 16)
            self.downsampler2 = torch.nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

            self.upsampler1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
            self.upres1 = ResBlock(16, 16)
            self.upres2 = ResBlock(16, 16)
            self.upsampler2 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        if self.pix_option == 2:
            self.downsampler1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=0)
            self.downres1 = ResBlock(16, 16)
            self.downres2 = ResBlock(16, 16)
            self.downsampler2 = torch.nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

            self.upsampler1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
            self.upres1 = ResBlock(16, 16)
            self.upres2 = ResBlock(16, 16)
            self.upsampler2 = nn.Conv2d(19, 3, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x, w=0, detach_16=True, code_only=False, adain=False):
        x_in = x
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        save_mid = False
        cri_mid= False
        if self.pix_option == 0:
            x = self.downsampler(x)
            if save_mid:
                mid0 = x
            # x = self.downsample_normalize(x)
            x = swish(x)
            if save_mid:
                mid1 = x
            if cri_mid:
                mid = x
        if self.pix_option == 1:
            x = self.downsampler1(x)
            x = self.downres1(x)
            x = self.downres2(x)
            x = self.downsampler2(x)
            if cri_mid:
                mid = x
            if save_mid:
                mid0 = x
        if self.pix_option == 2:
            x = self.downsampler1(x)
            x = self.downres1(x)
            x = self.downres2(x)
            x = self.downsampler2(x)
            if cri_mid:
                mid = x
            if save_mid:
                mid0 = x
        with torch.no_grad():
            # ################### Encoder #####################
            enc_feat_dict = {}
            out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_list]
            for i, block in enumerate(self.encoder.blocks):
                x = block(x) 
                if i in out_list:
                    enc_feat_dict[str(x.shape[-1])] = x.clone()

            lq_feat = x
            # ################# Transformer ###################
            # quant_feat, codebook_loss, quant_stats = self.quantize(lq_feat)
            pos_emb = self.position_emb.unsqueeze(1).repeat(1,x.shape[0],1)
            # BCHW -> BC(HW) -> (HW)BC
            feat_emb = self.feat_emb(lq_feat.flatten(2).permute(2,0,1))
            query_emb = feat_emb
            # Transformer encoder
            for layer in self.ft_layers:
                query_emb = layer(query_emb, query_pos=pos_emb)

            # output logits
            logits = self.idx_pred_layer(query_emb) # (hw)bn
            logits = logits.permute(1,0,2) # (hw)bn -> b(hw)n

            if code_only: # for training stage II
            # logits doesn't need softmax before cross_entropy loss
                return logits, lq_feat

            # ################# Quantization ###################
            # if self.training:
            #     quant_feat = torch.einsum('btn,nc->btc', [soft_one_hot, self.quantize.embedding.weight])
            #     # b(hw)c -> bc(hw) -> bchw
            #     quant_feat = quant_feat.permute(0,2,1).view(lq_feat.shape)
            # ------------
            soft_one_hot = F.softmax(logits, dim=2)
            _, top_idx = torch.topk(soft_one_hot, 1, dim=2)
            quant_feat = self.quantize.get_codebook_feat(top_idx, shape=[x.shape[0],16,16,256])
            # preserve gradients
            # quant_feat = lq_feat + (quant_feat - lq_feat).detach()

            if detach_16:
                quant_feat = quant_feat.detach() # for training stage III
            if adain:
                quant_feat = adaptive_instance_normalization(quant_feat, lq_feat)

            # ################## Generator ####################
            x = quant_feat
            fuse_list = [self.fuse_generator_block[f_size] for f_size in self.connect_list]

            for i, block in enumerate(self.generator.blocks):
                x = block(x) 
                if i in fuse_list: # fuse after i-th block
                    f_size = str(x.shape[-1])
                    if w>0:
                        x = self.fuse_convs_dict[f_size](enc_feat_dict[f_size].detach(), x, w)
        if self.pix_option == 0:
            if save_mid:
                mid2 = x
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
            if save_mid:
                mid3 = x
            x = self.upsampler(x)
        if self.pix_option == 1:
            if save_mid:
                mid1 = x
            x = self.upsampler1(x)
            x = self.upres1(x)
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
            x = self.upres2(x)
            x = self.upsampler2(x)
        if self.pix_option == 2:
            if save_mid:
                mid1 = x
            x = self.upsampler1(x)
            x = self.upres1(x)
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
            x = self.upres2(x)
            x = torch.cat([x_in, x], dim=1)
            x = self.upsampler2(x)
        out = x
        # logits doesn't need softmax before cross_entropy loss
        if save_mid:
            if self.pix_option == 0:
                return out, mid0, mid1, mid2, mid3
            if self.pix_option == 1:
                return out, mid0, mid1
            if self.pix_option == 2:
                return out, mid0, mid1
        if cri_mid:
            return out, mid
        return out

