import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class SE(nn.Module):
    def __init__(self, Cin, Cout, sn=False):
        super(SE, self).__init__()
        num_hidden = max(Cout // 16, 4)
        if not sn:
            self.se = nn.Sequential(nn.Linear(Cin, num_hidden), nn.ReLU(inplace=True),
                                    nn.Linear(num_hidden, Cout), nn.Sigmoid())
        else:
            self.se = nn.Sequential(nn.utils.spectral_norm(nn.Linear(Cin, num_hidden)), nn.ReLU(inplace=True),
                                    nn.utils.spectral_norm(nn.Linear(num_hidden, Cout)), nn.Sigmoid())

    def forward(self, x):
        se = F.adaptive_avg_pool2d(x, 1)
        se = se.view(se.size(0), -1)
        se = self.se(se)
        se = se.view(se.size(0), -1, 1, 1)
        return x * se

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

class DResnetBlock(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 padding_type,
                 norm_layer,
                 activation=nn.LeakyReLU(0.2, False),
                 use_dropout=False,
                 wide=True,
                 down_sample=False,
                 use_sn=False,
                 use_se=False,
                 ):
        super(DResnetBlock, self).__init__()
        self.in_channel, self.out_channel = in_channel, out_channel
        self.hidden_channel = self.out_channel if wide else self.in_channel
        self.downsample = down_sample
        self.sn = use_sn
        self.use_se = use_se

        self.conv_block = self.build_conv_block(padding_type, norm_layer, activation, use_dropout)
        module = []
        if down_sample:
            module += [BlurLayer(in_channel)]
        if use_sn:
            module += [nn.utils.spectral_norm(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2 if down_sample else 1, padding=1))]
        else:
            module += [nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2 if down_sample else 1, padding=1)]
        self.skip = nn.Sequential(*module)

        # add conv1x1 for downsample and channels scaling
        self.learn_able_sc = True if down_sample or in_channel != out_channel else False
        if self.learn_able_sc:
            self.conv_sc = nn.Conv2d(in_channel, out_channel, kernel_size=1)

    def build_conv_block(self, padding_type, norm_layer, activation, use_dropout):
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
        if self.sn:
            conv_block += [
                nn.utils.spectral_norm(nn.Conv2d(self.in_channel, self.hidden_channel, kernel_size=3, padding=p))]
        else:
            conv_block += [nn.Conv2d(self.in_channel, self.hidden_channel, kernel_size=3, padding=p)]

        if norm_layer is not None:
            conv_block += [norm_layer(self.hidden_channel)]
        conv_block += [activation]

        if use_dropout:
            conv_block += [nn.Dropout(use_dropout)]
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        stride = 2 if self.downsample else 1

        if self.downsample:
            conv_block += [BlurLayer(channels=self.hidden_channel)]
        if self.sn:
            conv_block += [
                nn.utils.spectral_norm(nn.Conv2d(self.hidden_channel, self.out_channel, kernel_size=3,
                                                 stride=stride, padding=p))]
        else:
            conv_block += [nn.Conv2d(self.hidden_channel, self.out_channel, kernel_size=3, stride=stride, padding=p)]

        conv_block += [activation]
        if norm_layer is not None:
            conv_block += [norm_layer(self.out_channel)]
        if self.use_se:
            conv_block += [SE(self.out_channel, self.out_channel, sn=self.sn)]
        return nn.Sequential(*conv_block)

    @staticmethod
    def _unfold_forward(module, x):
        for i, _ in enumerate(module):
            x = module[i](x)
        return x

    def forward(self, x, *args, **kwargs):
        h = self._unfold_forward(self.conv_block, x)
        out = h + self._unfold_forward(self.skip, x)
        return out / np.sqrt(2)

class AdaptiveLinear(nn.Linear):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(AdaptiveLinear, self).__init__(in_features, out_features, bias=True)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input):
        n, c = input.size()
        return F.linear(input, self.weight[:, :c], self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class ResidualDiscriminator(nn.Module):
    def __init__(self,
                 input_nc=1,
                 input_size=1024,
                 ndf=64,
                 n_layers=6,
                 atoms=51,
                 use_sn=True,
                 get_feat=False,
                 dense_connect=False
                 ):
        super(ResidualDiscriminator, self).__init__()

        self.get_feat = get_feat
        self.n_layers = n_layers

        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=1),
                     nn.LeakyReLU(negative_slope=0.2, inplace=False)]]

        ndf_now = ndf
        for n in range(1, n_layers + 1):
            ndf_prev = ndf_now
            ndf_now = min(ndf_now * 2, 512)
            sequence += [[DResnetBlock(
                ndf_prev,
                ndf_now,
                padding_type='zero',
                norm_layer=None,
                wide=False,
                down_sample=True,
                use_sn=use_sn,
                use_se=False)
            ]]

        out_dim = ndf_now if dense_connect else 1
        self.final_conv = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(ndf_now + 1, out_dim, 3, 1, 1)),
                                        nn.LeakyReLU(negative_slope=0.2, inplace=False))

        linear_in = (input_size // (2 ** n_layers)) ** 2 * out_dim
        if dense_connect:
            self.final_linear = nn.Sequential(
                nn.utils.spectral_norm(AdaptiveLinear(linear_in, ndf_now, bias=True)),
                nn.LeakyReLU(0.2, False),
                nn.utils.spectral_norm(nn.Linear(ndf_now, atoms))
            )
        else:
            self.final_linear = nn.Sequential(
                nn.utils.spectral_norm(AdaptiveLinear(linear_in, atoms, bias=True)),
            )
        if get_feat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def _apply_stddev(self, out):
        batch, channel, height, width = out.shape
        group = min(batch, 4)
        stddev = out.view(group, -1, 1, channel // 1, height, width
                          )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdim=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)
        return out

    def forward(self, input, dis_input=None):
        assert input.shape[0] < 4 or input.shape[0] % 4 == 0
        if dis_input is not None:
            input = torch.cat([dis_input, input], dim=1)

        if self.get_feat:
            res = [input]
            for n in range(self.n_layers + 1):
                res.append(getattr(self, 'model' + str(n))(res[-1]))
            # apply stddev
            out = self._apply_stddev(res[-1])
            pre_linear = self.final_conv(out)

            final_out = self.final_linear(pre_linear.view(pre_linear.size(0), -1))
            return final_out, res[1:]
        else:
            out = self.model(input)
            out = self._apply_stddev(out)
            pre_linear = self.final_conv(out)
            return self.final_linear(pre_linear.view(pre_linear.size(0), -1))

class MultiscaleDiscriminator(nn.Module):
    def __init__(self,
                 input_nc=1,
                 input_size=1024,
                 ndf=64,
                 n_layers=6,
                 atoms=51,
                 use_sn=True,
                 get_feat=False,
                 dense_connect=False,
                 stddev=True,
                 num_d=1,
                 out_dim=1,
                 ):
        super(MultiscaleDiscriminator, self).__init__()
        self.input_nc = input_nc
        self.input_size = input_size
        self.n_layers = n_layers
        self.num_d = num_d
        self.ndf = ndf
        self.get_feat = get_feat
        self.out_dim = out_dim
        self.stddev = stddev

        self._build_d_net()

    def _build_d_net(self):
        for i in range(self.num_d):
            net_d = ResidualDiscriminatorOld(
                input_nc=self.input_nc,
                input_size=self.input_size,
                ndf=self.ndf,
                n_layers=self.n_layers,
                atoms=51,
                use_sn=True,
                get_feat=self.get_feat,
                out_dim=self.out_dim,
                stddev=self.stddev
            )
            setattr(self, 'layer' + str(i), net_d)

    def forward(self, input, dis_input=None):
        results = []
        if dis_input is not None:
            input = torch.cat([dis_input, input], dim=1)
        for i in range(self.num_d):
            model = getattr(self, 'layer' + str(i))
            results.append(model(input))
        return results[0]

class ResidualDiscriminatorOld(ResidualDiscriminator):
    def __init__(self,
                 input_nc=1,
                 input_size=1024,
                 ndf=64,
                 n_layers=6,
                 atoms=51,
                 use_sn=True,
                 get_feat=False,
                 dense_connect=False,
                 out_dim=1,
                 stddev=True

                 ):
        super(ResidualDiscriminator, self).__init__()

        self.get_feat = get_feat
        self.n_layers = n_layers
        self.stddev = stddev

        sequence = [[DResnetBlock(
            input_nc,
            ndf,
            padding_type='zero',
            norm_layer=None,
            wide=False,
            down_sample=True,
            use_sn=use_sn,
            use_se=False
        )]]

        ndf_now = ndf
        for n in range(1, n_layers + 1):
            ndf_prev = ndf_now
            ndf_now = min(ndf_now * 2, 512)
            wide = (ndf_prev != ndf_now)
            sequence += [[DResnetBlock(
                ndf_prev,
                ndf_now,
                padding_type='zero',
                norm_layer=None,
                wide=wide,
                down_sample=True,
                use_sn=use_sn,
                use_se=False)
            ]]

        # out_dim = ndf_now if dense_connect else 1
        # self.final_conv = nn.Conv2d(ndf_now + 1, out_dim, 3, 1, 1)

        out_dim = ndf_now if dense_connect else out_dim
        if not self.stddev:
            sequence += [[nn.Conv2d(ndf_now, out_dim, kernel_size=3, stride=1, padding=1)]]
        else:
            self.final_conv = nn.Conv2d(ndf_now + 1, out_dim, 3, 1, 1)

        linear_in = (input_size // (2 ** (n_layers + 1))) ** 2 * out_dim
        if dense_connect:
            self.fc = nn.Sequential(
                AdaptiveLinear(linear_in, ndf_now, bias=True),
                nn.LeakyReLU(0.2, False),
                nn.Linear(ndf_now, atoms)
            )
        else:
            self.fc = AdaptiveLinear(linear_in, atoms, bias=True)
        if get_feat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        assert input.shape[0] < 4 or input.shape[0] % 4 == 0
        if self.get_feat:
            res = [input]
            k = 2 if not self.stddev else 1
            for n in range(self.n_layers + k):
                res.append(getattr(self, 'model' + str(n))(res[-1]))
            # apply stddev
            if not self.stddev:
                pre_linear = res[-1]
            else:
                out = self._apply_stddev(res[-1])
                pre_linear = self.final_conv(out)
            final_out = self.fc(pre_linear.view(pre_linear.size(0), -1))
            return final_out, res[1:]
        else:
            out = self.model(input)
            if not self.stddev:
                pre_linear = out
            else:
                out = self._apply_stddev(out)
                pre_linear = self.final_conv(out)
            return self.fc(pre_linear.view(pre_linear.size(0), -1))


class UNetDiscriminatorSN(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)
    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.
    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out