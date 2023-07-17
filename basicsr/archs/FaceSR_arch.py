import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy


class BlurLayer(nn.Module):
    """Implements the blur layer used in StyleGAN."""

    def __init__(self,
                 channels,
                 kernel=(1, 2, 1),
                 normalize=True,
                 flip=False):
        super().__init__()
        kernel = np.array(kernel, dtype=np.float32).reshape(1, 3)
        kernel = kernel.T.dot(kernel)
        if normalize:
            kernel /= np.sum(kernel)
        if flip:
            kernel = kernel[::-1, ::-1]
        kernel = kernel.reshape(3, 3, 1, 1)
        kernel = np.tile(kernel, [1, 1, channels, 1])
        kernel = np.transpose(kernel, [2, 3, 0, 1])
        self.register_buffer('kernel', torch.from_numpy(kernel))
        self.channels = channels

    def forward(self, x):
        return F.conv2d(x, self.kernel, stride=1, padding=1, groups=self.channels)

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, use_blur_layer=False, activation=nn.ReLU(True)):
        super(DownSample, self).__init__()

        modules = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                   activation]
        if use_blur_layer:
            modules += [BlurLayer(out_channels)]
        self.module = nn.Sequential(*modules)

    def forward(self, input):
        return self.module(input)

class PixelShuffleAlign(nn.Module):
    def __init__(self, upsacel_factor: int = 1, mode: str = 'caffe'):
        """
        :param upsacel_factor: upsample scale
        :param mode: caffe, pytorch
        """
        super(PixelShuffleAlign, self).__init__()
        self.upscale_factor = upsacel_factor
        self.mode = mode

    def forward(self, x):
        N, C, H, W = x.size()
        c = C // (self.upscale_factor ** 2)
        h, w = H * self.upscale_factor, W * self.upscale_factor

        if self.mode == 'caffe':
            # (N, C, H, W) => (N, r, r, c, H, W)
            x = x.reshape(-1, self.upscale_factor,
                          self.upscale_factor, c, H, W)
            x = x.permute(0, 3, 4, 1, 5, 2)
        elif self.mode == 'pytorch':
            # (N, C, H, W) => (N, r, r, c, H, W)
            x = x.reshape(-1, c, self.upscale_factor,
                          self.upscale_factor, H, W)
            x = x.permute(0, 1, 4, 2, 5, 3)
        else:
            raise NotImplementedError(
                "{} mode is not implemented".format(self.mode))

        x = x.reshape(-1, c, h, w)
        return x

def conv_block(in_nc,
               out_nc,
               kernel_size,
               stride=1,
               dilation=1,
               groups=1,
               bias=True,
               padding_type='zero',
               norm_layer=None,
               activation=nn.ReLU(True),
               use_dropout=False,
               conv_unit=None,
               blur_layer=False
               ):
    conv_block = []
    p = 0
    if padding_type == 'reflect':
        conv_block += [nn.ReflectionPad2d(1)]
    elif padding_type == 'replicate':
        conv_block += [nn.ReplicationPad2d(1)]
    elif padding_type == 'zero':
        p = 1
    else:
        raise NotImplementedError('padding [%s] is not implemented' % padding_type)

    conv_block += [conv_unit(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=p, bias=bias)]
    if blur_layer:
        conv_block += [BlurLayer(out_nc)]
    if norm_layer is not None:
        conv_block += [norm_layer(out_nc)]

    conv_block += [deepcopy(activation)]
    if use_dropout:
        conv_block += [nn.Dropout(0.5)]
    return nn.Sequential(*conv_block)

def pixelshuffle_block(in_nc, out_nc,
                       upscale_factor=2,
                       kernel_size=3,
                       stride=1,
                       bias=True,
                       padding_type='zero',
                       norm_layer=nn.InstanceNorm2d,
                       activation=nn.ReLU(True),
                       conv_unit=None,
                       mode_block='pytorch',
                       blur_layer=False,
                       ):
    """
    Pixel shuffle layer
    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
    Neural Network, CVPR17)
    """
    conv_unit = nn.Conv2d if conv_unit is None else conv_unit
    conv = conv_block(in_nc, out_nc * (upscale_factor ** 2), kernel_size, stride,
                      bias=bias,
                      padding_type=padding_type,
                      norm_layer=norm_layer,
                      activation=activation,
                      conv_unit=conv_unit,
                      blur_layer=blur_layer
                      )

    pixel_shuffle = PixelShuffleAlign(upscale_factor, mode=mode_block)

    norm = norm_layer(out_nc) if norm_layer is not None else None
    a = activation
    model_ = [conv, pixel_shuffle, norm, a] if norm_layer is not None else [conv, pixel_shuffle]
    return nn.Sequential(*model_)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, conv_unit=nn.Conv2d, activation=nn.ReLU(True)):
        super(UpSample, self).__init__()
        self.block = pixelshuffle_block(
            in_nc=in_channels,
            out_nc=out_channels,
            norm_layer=None,
            mode_block='caffe',
            blur_layer=False,
            conv_unit=conv_unit,
            activation=activation
        )

    def forward(self, input):
        return self.block(input)

class ToRgb(nn.Module):
    def __init__(self, in_dim, out_dim=3, activation=None, size=None):
        super(ToRgb, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=True)
        self.activation = activation
        self.upsample_layer = self._build_upsample_layer(size=size)

    def _build_upsample_layer(self, size=None):
        if size is None:
            return nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            return nn.UpsamplingBilinear2d(size=size)

    def forward(self, input, skip=None):
        out = self.conv(input)

        # multi-scale output
        if self.activation is not None:
            return self.activation(out)
        # skip connection
        if skip is not None:
            out = out + self.upsample_layer(skip)
        return out


class ResnetBlock(nn.Module):
    def __init__(self, in_dim, padding_type, norm_layer,
                 activation=nn.ReLU(True),
                 use_dropout=False,
                 conv_unit=None,
                 divide_ratio=False,
                 scale=None,
                 size=None,
                 activation_first=False,
                 *args, **kwargs
                 ):
        super(ResnetBlock, self).__init__()
        self.divide_ratio = divide_ratio
        self.scale_factor = scale
        self.size = size
        self.conv_unit = nn.Conv2d if conv_unit is None else conv_unit
        self.conv_block = self.build_conv_block(in_dim, padding_type, norm_layer,
                                                activation, use_dropout, activation_first)
        self.upsample_layer = self._build_upsample_layer()

    def _build_upsample_layer(self):
        if self.scale_factor is not None:
            return nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
        else:
            return None

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout, activation_first):
        conv_block = []
        p = 0
        if not activation_first:
            if padding_type == 'reflect':
                conv_block += [nn.ReflectionPad2d(1)]
            elif padding_type == 'replicate':
                conv_block += [nn.ReplicationPad2d(1)]
            elif padding_type == 'zero':
                p = 1
            else:
                raise NotImplementedError('padding [%s] is not implemented' % padding_type)
            conv_block += [self.conv_unit(dim, dim, kernel_size=3, padding=p)]
            if norm_layer is not None:
                conv_block += [norm_layer(dim)]
            if activation is not None:
                conv_block += [activation]

            if use_dropout:
                conv_block += [nn.Dropout(0.5)]

            p = 0
            if padding_type == 'reflect':
                conv_block += [nn.ReflectionPad2d(1)]
            elif padding_type == 'replicate':
                conv_block += [nn.ReplicationPad2d(1)]
            elif padding_type == 'zero':
                p = 1
            else:
                raise NotImplementedError('padding [%s] is not implemented' % padding_type)
            conv_block += [self.conv_unit(dim, dim, kernel_size=3, padding=p)]
            if norm_layer is not None:
                conv_block += [norm_layer(dim)]
        else:
            # norm+activation
            if norm_layer is not None:
                conv_block += [norm_layer(dim)]
            if activation is not None:
                conv_block += [activation]
            # conv
            conv_block += [self.conv_unit(dim, dim, kernel_size=3, padding=1)]
            if use_dropout:
                conv_block += [nn.Dropout(0.5)]
            # norm+activation
            if norm_layer is not None:
                conv_block += [norm_layer(dim)]
            if activation is not None:
                conv_block += [activation]
            p = 0
            if padding_type == 'reflect':
                conv_block += [nn.ReflectionPad2d(1)]
            elif padding_type == 'replicate':
                conv_block += [nn.ReplicationPad2d(1)]
            elif padding_type == 'zero':
                p = 1
            else:
                raise NotImplementedError('padding [%s] is not implemented' % padding_type)
            # conv
            conv_block += [self.conv_unit(dim, dim, kernel_size=3, padding=p)]

        return nn.Sequential(*conv_block)


    def _unfold(self, x, x_spatial=None):
        for n, m in enumerate(self.conv_block.children()):
            x = m(x)
        return x

    def forward(self, x, x_spatial=None):
        if self.upsample_layer is not None:
            x = self.upsample_layer(x)
        out = x + self.conv_block(x)
        if not self.divide_ratio:
            return out
        else:
            out = out / np.sqrt(2)
            return out


def get_resblock(block_type):
    if block_type == 'Residual':
        return ResnetBlock
    else:
        raise NotImplementedError()

def get_activation(activation_name):
    """ get activation from string to obj
    """
    active_name = activation_name.lower()

    if active_name == "relu":
        return nn.ReLU(True)
    elif active_name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    elif active_name == "prelu":
        return nn.PReLU()
    elif active_name == 'gelu':
        return nn.GELU()
    else:
        raise NotImplementedError("Unidentified activation name {}.".format(active_name))

class LocalGlobalNet(nn.Module):
    def __init__(self,
                 input_nc=1,
                 output_nc=1,
                 ngf=32,
                 n_blocks_local=3,
                 n_blocks_global=9,
                 channel_scale=0,
                 local_scale=True,
                 activation_name='relu',
                 local_block_type='Residual',
                 global_block_type='Residual'
                 ):
        
        super(LocalGlobalNet, self).__init__()
        self._set_args(locals())

        self.ngf = ngf
        self.local_scale = local_scale

        activation = get_activation(activation_name)

        # set up local generator
        ngf_local = self.ngf // 2
        self.in_layer = nn.Sequential(nn.Conv2d(input_nc, ngf_local, kernel_size=3, stride=1, padding=1),
                                      activation)
        self.local_down = DownSample(in_channels=ngf_local, out_channels=ngf_local,activation=activation)
        self.local_scale = nn.Conv2d(self.ngf, ngf_local, kernel_size=1, bias=True)
        # build local enhancer
        resblock_module = get_resblock(local_block_type)
        self.local_enhancer = nn.Sequential(*[resblock_module(in_dim=ngf_local, out_dim=ngf_local, padding_type='zero',
                                                              activation=activation,
                                                              norm_layer=None) for _ in range(n_blocks_local)])
        # local up block
        self.local_up = UpSample(ngf_local, ngf_local, activation=activation)
        self.out_layer = nn.Conv2d(ngf_local, output_nc, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()


        # build global
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.n_blocks_global = n_blocks_global
        self.channel_scale = channel_scale
        self.activation_name = activation_name
        self.global_block_type = global_block_type

        self.global_net = self._build_global_net()

    def _set_args(self, args):
        for key, value in args.items():
            if key != 'self' and key != '__class__':
                setattr(self, key, value)

    def _build_global_net(self):
        return GlobalNet(
            input_nc=self.input_nc,
            output_nc=self.output_nc,
            ngf=self.ngf,
            n_blocks_global=self.n_blocks_global,
            channel_scale=self.channel_scale,
            activation_name=self.activation_name,
            global_block_type=self.global_block_type
        )

    def forward(self, input, input_ref, *args, **kwargs):
        input_down = F.avg_pool2d(input, 3, 2, [1, 1], count_include_pad=True)
        input_ref_down = F.avg_pool2d(input_ref, 3, 2, [1, 1], count_include_pad=True)

        input_h = self.in_layer(input)
        input_h_down = self.local_down(input_h)

        # forward pass of global net
        hidden_global, output_global = self.global_net(input_down, input_ref_down)

        input_h_en = self.local_enhancer(input_h_down + self.local_scale(hidden_global))

        output = self.local_up(input_h_en)
        output = self.out_layer(output + input_h)

        # to rgb
        output = output + F.interpolate(output_global, scale_factor=2, mode='bilinear')
        return self.tanh(output)




class LocalGlobalNet_FaceDeblur(nn.Module):
    def __init__(self,
                 input_nc=1,
                 output_nc=1,
                 ngf=32,
                 n_blocks_local=3,
                 n_blocks_global=9,
                 channel_scale=0,
                 local_scale=True,
                 activation_name='relu',
                 local_block_type='Residual',
                 global_block_type='Residual'
                 ):
        
        super(LocalGlobalNet_FaceDeblur, self).__init__()
        self._set_args(locals())

        self.ngf = ngf
        self.local_scale = local_scale

        activation = get_activation(activation_name)

        # set up local generator
        ngf_local = self.ngf // 2
        self.in_layer = nn.Sequential(nn.Conv2d(input_nc, ngf_local, kernel_size=3, stride=1, padding=1),
                                      activation)
        self.local_down = DownSample(in_channels=ngf_local, out_channels=ngf_local,activation=activation)
        self.local_scale = nn.Conv2d(self.ngf, ngf_local, kernel_size=1, bias=True)
        # build local enhancer
        resblock_module = get_resblock(local_block_type)
        self.local_enhancer = nn.Sequential(*[resblock_module(in_dim=ngf_local, out_dim=ngf_local, padding_type='zero',
                                                              activation=activation,
                                                              norm_layer=None) for _ in range(n_blocks_local)])
        # local up block
        self.local_up = UpSample(ngf_local, ngf_local, activation=activation)
        self.out_layer = nn.Conv2d(ngf_local, output_nc, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()


        # build global
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.n_blocks_global = n_blocks_global
        self.channel_scale = channel_scale
        self.activation_name = activation_name
        self.global_block_type = global_block_type

        self.global_net = self._build_global_net()


        ## mask half size
        self.conv_mask = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            activation,
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            activation,
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            activation,
        )



    def _set_args(self, args):
        for key, value in args.items():
            if key != 'self' and key != '__class__':
                setattr(self, key, value)

    def _build_global_net(self):
        return GlobalNet(
            input_nc=self.input_nc,
            output_nc=self.output_nc,
            ngf=self.ngf,
            n_blocks_global=self.n_blocks_global,
            channel_scale=self.channel_scale,
            activation_name=self.activation_name,
            global_block_type=self.global_block_type
        )

    def forward(self, input, input_ref, seg_mask, occ_mask, iso_mask, *args, **kwargs):
        input_down = F.avg_pool2d(input, 3, 2, [1, 1], count_include_pad=True)
        input_ref_down = F.avg_pool2d(input_ref, 3, 2, [1, 1], count_include_pad=True)

        input_mask = torch.cat((seg_mask, occ_mask, iso_mask), 1)
        input_mask_feat = self.conv_mask(input_mask)

        input_h = self.in_layer(input)
        input_h_down = self.local_down(input_h)

        # forward pass of global net
        hidden_global, output_global = self.global_net(input_down, input_ref_down, input_mask_feat)

        input_h_en = self.local_enhancer(input_h_down + self.local_scale(hidden_global))

        output = self.local_up(input_h_en)
        output = self.out_layer(output + input_h)

        # to rgb
        output = output + F.interpolate(output_global, scale_factor=2, mode='bilinear')
        return self.tanh(output)


class LocalGlobalNet_FaceDeblur_woMask(nn.Module):
    def __init__(self,
                 input_nc=1,
                 output_nc=1,
                 ngf=32,
                 n_blocks_local=3,
                 n_blocks_global=9,
                 channel_scale=0,
                 local_scale=True,
                 activation_name='relu',
                 local_block_type='Residual',
                 global_block_type='Residual'
                 ):
        
        super(LocalGlobalNet_FaceDeblur_woMask, self).__init__()
        self._set_args(locals())

        self.ngf = ngf
        self.local_scale = local_scale

        activation = get_activation(activation_name)

        # set up local generator
        ngf_local = self.ngf // 2
        self.in_layer = nn.Sequential(nn.Conv2d(input_nc, ngf_local, kernel_size=3, stride=1, padding=1),
                                      activation)
        self.local_down = DownSample(in_channels=ngf_local, out_channels=ngf_local,activation=activation)
        self.local_scale = nn.Conv2d(self.ngf, ngf_local, kernel_size=1, bias=True)
        # build local enhancer
        resblock_module = get_resblock(local_block_type)
        self.local_enhancer = nn.Sequential(*[resblock_module(in_dim=ngf_local, out_dim=ngf_local, padding_type='zero',
                                                              activation=activation,
                                                              norm_layer=None) for _ in range(n_blocks_local)])
        # local up block
        self.local_up = UpSample(ngf_local, ngf_local, activation=activation)
        self.out_layer = nn.Conv2d(ngf_local, output_nc, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()


        # build global
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.n_blocks_global = n_blocks_global
        self.channel_scale = channel_scale
        self.activation_name = activation_name
        self.global_block_type = global_block_type

        self.global_net = self._build_global_net()


        ## mask half size
        self.conv_mask = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            activation,
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            activation,
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            activation,
        )



    def _set_args(self, args):
        for key, value in args.items():
            if key != 'self' and key != '__class__':
                setattr(self, key, value)

    def _build_global_net(self):
        return GlobalNet(
            input_nc=self.input_nc,
            output_nc=self.output_nc,
            ngf=self.ngf,
            n_blocks_global=self.n_blocks_global,
            channel_scale=self.channel_scale,
            activation_name=self.activation_name,
            global_block_type=self.global_block_type
        )

    def forward(self, input, input_ref, seg_mask, occ_mask, iso_mask, *args, **kwargs):
        input_down = F.avg_pool2d(input, 3, 2, [1, 1], count_include_pad=True)
        input_ref_down = F.avg_pool2d(input_ref, 3, 2, [1, 1], count_include_pad=True)

        input_mask_feat = None

        input_h = self.in_layer(input)
        input_h_down = self.local_down(input_h)

        # forward pass of global net
        hidden_global, output_global = self.global_net(input_down, input_ref_down, input_mask_feat)

        input_h_en = self.local_enhancer(input_h_down + self.local_scale(hidden_global))

        output = self.local_up(input_h_en)
        output = self.out_layer(output + input_h)

        # to rgb
        output = output + F.interpolate(output_global, scale_factor=2, mode='bilinear')
        return self.tanh(output)







class GlobalNet(nn.Module):
    def __init__(self,
                 input_nc=1,
                 output_nc=1,
                 ngf=32,
                 n_blocks_global=9,
                 channel_scale=0,
                 activation_name='relu',
                 global_block_type='Residual'
                 ):
        super(GlobalNet, self).__init__()
        self._set_args(locals())

        activation = get_activation(activation_name)


        # build global net
        _ngf = ngf // 2
        self.in_layer = nn.Sequential(nn.Conv2d(input_nc, _ngf, kernel_size=3, stride=1, padding=1),
                                      activation)
        self.residual_conv = nn.Sequential(nn.Conv2d(_ngf, ngf, kernel_size=1, stride=1, bias=True))

        self.in_layer_mask = nn.Sequential(nn.Conv2d(3, _ngf, kernel_size=3, stride=1, padding=1),
                                      activation)


        # down sample blocks
        for i in range(3):
            module = nn.Sequential(nn.Conv2d(_ngf, _ngf * 2, kernel_size=3, stride=2, padding=1),
                                   activation)
            setattr(self, 'down_layer_dis_{}'.format(i), module)
            # setattr(self, 'down_layer_ref_{}'.format(i), module)
            # set up from rgb layer
            self.down_sample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=True)
            module_rgb = nn.Conv2d(input_nc, _ngf * 2, kernel_size=1, stride=1, bias=True)
            setattr(self, 'from_rgb_{}'.format(i), module_rgb)
            _ngf = _ngf * 2

        # for channel scaling in trunk blocks
        ngf_scaled = ngf
        if channel_scale > 1:
            mult = 2 ** 3
            ngf_scaled = ngf - channel_scale
            self.scale_conv_in = nn.Conv2d(ngf * mult,  ngf_scaled * mult, kernel_size=1, stride=1)
            self.scale_conv_out = nn.Conv2d(ngf_scaled * mult, ngf * mult, kernel_size=1, stride=1)

        self._build_trunk(activation, ngf_scaled)
        # up sample blocks
        for i in range(3):
            mult = 2 ** (3 - i)
            _ngf_now = ngf * mult
            _ngf_next = ngf * mult // 2
            setattr(self, 'up_layer_{}'.format(i), UpSample(_ngf_now, _ngf_next, activation=activation))
            setattr(self, 'to_rgb_{}'.format(i), ToRgb(_ngf_now, output_nc))

        # output layer
        self.out_layer = nn.Sequential(nn.Conv2d(_ngf_next, output_nc, 3, 1, 1))

    def _set_args(self, args):
        for key, value in args.items():
            if key != 'self' and key != '__class__':
                setattr(self, key, value)

    def _build_trunk(self, activation, ngf):
        resblock_module = get_resblock(self.global_block_type)

        _ngf = ngf * (2 ** 3)
        resblocks = []
        assert self.n_blocks_global > 3, "number of blocks must greater than 3"
        n_depth = self.n_blocks_global // 3
        scale_factors = {n_depth: 0.5, self.n_blocks_global - n_depth: 2}
        for i in range(self.n_blocks_global):
            resblocks += [resblock_module(in_dim=_ngf, out_dim=_ngf, padding_type='zero',
                                          norm_layer=None, activation=activation, scale=scale_factors.get(i))]
        self.global_trunk = nn.Sequential(*resblocks)

    def forward(self, input, input_ref, input_mask_feat, *args, **kwargs):
        input_feat = self.in_layer(input)
        input_ref_feat = self.in_layer(input_ref)


        # compute multi scale inputs for from rgb fusion layer
        input_multi_scale = [self.down_sample(input)]
        for i in range(1, 3):
            input_multi_scale.append(self.down_sample(input_multi_scale[-1]))

        skip = []
        skip.append(self.residual_conv(input_feat))

        input_ref_bf = None
        for i in range(3):
            # _input_ref_plus for fusion
            if i != 0:
                # input_ref_plus = getattr(self, 'down_layer_ref_{}'.format(i))(input_ref_bf)
                input_ref_plus = getattr(self, 'down_layer_dis_{}'.format(i))(input_ref_bf)

            input_feat = getattr(self, 'down_layer_dis_{}'.format(i))(input_feat)

            # input_ref_feat = getattr(self, 'down_layer_ref_{}'.format(i))(input_ref_feat)
            input_ref_feat = getattr(self, 'down_layer_dis_{}'.format(i))(input_ref_feat)

            # fuse with multi-scale inputs
            input_feat = input_feat + getattr(self, 'from_rgb_{}'.format(i))(input_multi_scale[i])

            if i != 0:
                input_ref_bf = input_ref_feat + input_ref_plus
            else:
                input_ref_bf = input_ref_feat

            skip.append(torch.cat([input_feat, input_feat], dim=1))


        # forward pass of trunk blocks
        hidden_feat = torch.cat([input_feat, input_ref_bf], dim=1)
        if input_mask_feat != None:
            hidden_feat = hidden_feat + input_mask_feat
        hidden_feat = self.scale_conv_in(hidden_feat) if self.channel_scale > 0 else hidden_feat  
        hidden_feat = self.global_trunk(hidden_feat)
        hidden_feat = self.scale_conv_out(hidden_feat) if self.channel_scale > 0 else hidden_feat  # channel rescaling


        # forward pass of up sample blocks
        rgb_skip = None
        for i in range(3):
            hidden_feat = hidden_feat + skip.pop()  # shortcut connection
            rgb_skip = getattr(self, 'to_rgb_{}'.format(i))(hidden_feat, rgb_skip)
            hidden_feat = getattr(self, 'up_layer_{}'.format(i))(hidden_feat)

        # final outputs
        hidden_feat = hidden_feat + skip.pop()
        out = self.out_layer(hidden_feat) + F.interpolate(rgb_skip, scale_factor=2, mode='bilinear')
        return hidden_feat, out



if __name__ == '__main__':
    net = GlobalNet(input_nc=3, output_nc=3)
    net_lg = LocalGlobalNet_FaceDeblur(input_nc=3, output_nc=3, ngf=16, channel_scale=7, n_blocks_local=2, n_blocks_global=9)

    img1 = torch.rand(1, 3, 256, 256)
    img2 = torch.rand(1, 3, 256, 256)
    mask = torch.rand(1, 3, 128, 128)


    out = net_lg(img1, img2, mask)



    print('out shape: ', out.shape)

