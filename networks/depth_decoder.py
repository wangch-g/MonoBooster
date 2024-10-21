'''
This code was ported from existing repos
[LINK] https://github.com/nianticlabs/monodepth2
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from .transformer import LayerNorm
from .transformer import CrossLevelAttention


class Conv3x3(nn.Module):
    """
    Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, stride=1, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3, stride=stride)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out
    

class ConvBlock(nn.Module):
    """
    Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels, stride)
        # self.nonlin = nn.ELU(inplace=True)
        self.nonlin = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False)

    def forward(self, x):
        return self.conv(x)


class ConvBlock1x1(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock1x1, self).__init__()

        self.conv = Conv1x1(in_channels, out_channels)
        self.nonlin = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


def upsample(x, mode='bilinear'):
    """
    Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode=mode)


class CLABlock(nn.Module):
    def __init__(self, dim, dim_h, dim_l, num_heads=1, dilation=[1, 2, 3], bias=False):
        super(CLABlock, self).__init__()
        
        self.conv_0 = ConvBlock1x1(dim_h + dim_l, dim)
        self.norm_x = LayerNorm(dim, LayerNorm_type='WithBias')
        self.norm_x_h = LayerNorm(dim_h, LayerNorm_type='WithBias')
        self.norm_x_l = LayerNorm(dim_l, LayerNorm_type='WithBias')
        self.xatt = CrossLevelAttention(dim, dim_h, dim_l, num_heads=num_heads, dilation=dilation, bias=bias)
        self.conv_1 = ConvBlock1x1(dim_h + dim_l, dim)

    def forward(self, x_h, x_l):

        x = self.conv_0(torch.cat((x_h, x_l), 1))
        msg_h, msg_l = self.xatt(self.norm_x(x), self.norm_x_h(x_h), self.norm_x_l(x_l))
        x = self.conv_1(torch.cat((x_h+msg_h, x_l+msg_l), 1))

        return x


class MonoBooster(nn.Module):
    def __init__(self,
                 num_ch_enc=np.array([64, 64, 128, 256, 512]),
                 scales=range(4),
                 num_output_channels=1):
        super(MonoBooster, self).__init__()

        self.scales = scales
        self.num_output_channels = num_output_channels
        self.num_heads = 1

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.conv_11_0 = ConvBlock(self.num_ch_enc[1], self.num_ch_dec[1])
        self.xatt_11 = CLABlock(self.num_ch_dec[1], self.num_ch_dec[1], self.num_ch_enc[0], num_heads=self.num_heads, dilation=[1, 2, 3], bias=False)

        self.conv_21_0 = ConvBlock(self.num_ch_enc[2], self.num_ch_dec[2])
        self.xatt_21 = CLABlock(self.num_ch_dec[2], self.num_ch_dec[2], self.num_ch_enc[1], num_heads=self.num_heads, dilation=[1, 2, 3], bias=False)

        self.conv_31_0 = ConvBlock(self.num_ch_enc[3], self.num_ch_dec[3])
        self.xatt_31 = CLABlock(self.num_ch_dec[3], self.num_ch_dec[3], self.num_ch_enc[2], num_heads=self.num_heads, dilation=[1, 2, 3], bias=False)

        self.conv_41_0 = ConvBlock(self.num_ch_enc[4], self.num_ch_dec[4])
        self.xatt_41 = CLABlock(self.num_ch_dec[4], self.num_ch_dec[4], self.num_ch_enc[3], num_heads=self.num_heads, dilation=[1, 2, 3], bias=False)


        self.conv_12_0 = ConvBlock(self.num_ch_dec[2], self.num_ch_dec[1])
        self.xatt_12 = CLABlock(self.num_ch_dec[1], self.num_ch_dec[1], self.num_ch_dec[1], num_heads=self.num_heads, dilation=[1, 2, 3], bias=False)

        self.conv_22_0 = ConvBlock(self.num_ch_dec[3], self.num_ch_dec[2])
        self.xatt_22 = CLABlock(self.num_ch_dec[2], self.num_ch_dec[2], self.num_ch_dec[2], num_heads=self.num_heads, dilation=[1, 2, 3], bias=False)

        self.conv_32_0 = ConvBlock(self.num_ch_dec[4], self.num_ch_dec[3])
        self.xatt_32 = CLABlock(self.num_ch_dec[3], self.num_ch_dec[3], self.num_ch_dec[3], num_heads=self.num_heads, dilation=[1, 2, 3], bias=False)


        self.conv_13_0 = ConvBlock(self.num_ch_dec[2], self.num_ch_dec[1])
        self.xatt_13 = CLABlock(self.num_ch_dec[1], self.num_ch_dec[1], self.num_ch_dec[1], num_heads=self.num_heads, dilation=[1, 2, 3], bias=False)

        self.conv_23_0 = ConvBlock(self.num_ch_dec[3], self.num_ch_dec[2])
        self.xatt_23 = CLABlock(self.num_ch_dec[2], self.num_ch_dec[2], self.num_ch_dec[2], num_heads=self.num_heads, dilation=[1, 2, 3], bias=False)


        self.conv_14_0 = ConvBlock(self.num_ch_dec[2], self.num_ch_dec[1])
        self.xatt_14 = CLABlock(self.num_ch_dec[1], self.num_ch_dec[1], self.num_ch_dec[1], num_heads=self.num_heads, dilation=[1, 2, 3], bias=False)


        self.conv_04_0 = ConvBlock(self.num_ch_dec[1], self.num_ch_dec[0])
        self.conv_04_1 = ConvBlock1x1(self.num_ch_dec[0], self.num_ch_dec[0])

        self.dispconv_0 = Conv3x3(self.num_ch_dec[0], self.num_output_channels)
        self.dispconv_1 = Conv3x3(self.num_ch_dec[1], self.num_output_channels)
        self.dispconv_2 = Conv3x3(self.num_ch_dec[2], self.num_output_channels)
        self.dispconv_3 = Conv3x3(self.num_ch_dec[3], self.num_output_channels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        # [12, 64, 96, 320]
        # [12, 64, 48, 160]
        # [12, 128, 24, 80]
        # [12, 256, 12, 40]
        # [12, 512, 6, 20]

        outputs = {}
        features = {}

        features["x_1"] = input_features[0]
        features["x_2"] = input_features[1]
        features["x_3"] = input_features[2]
        features["x_4"] = input_features[3]
        features["x_5"] = input_features[4]

        # node 11
        x_l = features["x_1"]
        x_h = features["x_2"]
        x_h = upsample(self.conv_11_0(x_h))
        features["x_11"] = self.xatt_11(x_h, x_l)
        
        # node 21
        x_l = features["x_2"]
        x_h = features["x_3"]
        x_h = upsample(self.conv_21_0(x_h))
        features["x_21"] = self.xatt_21(x_h, x_l)
        
        # node 31
        x_l = features["x_3"]
        x_h = features["x_4"]
        x_h = upsample(self.conv_31_0(x_h))
        features["x_31"] = self.xatt_31(x_h, x_l)
        
        # node 41
        x_l = features["x_4"]
        x_h = features["x_5"]
        x_h = upsample(self.conv_41_0(x_h))
        features["x_41"] = self.xatt_41(x_h, x_l)


        # node 12
        x_l = features["x_11"]
        x_h = features["x_21"]
        x_h = upsample(self.conv_12_0(x_h))
        features["x_12"] = self.xatt_12(x_h, x_l)

        # node 22
        x_l = features["x_21"]
        x_h = features["x_31"]
        x_h = upsample(self.conv_22_0(x_h))
        features["x_22"] = self.xatt_22(x_h, x_l)

        # node 32
        x_l = features["x_31"]
        x_h = features["x_41"]
        x_h = upsample(self.conv_32_0(x_h))
        features["x_32"] = self.xatt_32(x_h, x_l)


        # node 13
        x_l = features["x_12"]
        x_h = features["x_22"]
        x_h = upsample(self.conv_13_0(x_h))
        features["x_13"] = self.xatt_13(x_h, x_l)

        # node 23
        x_l = features["x_22"]
        x_h = features["x_32"]
        x_h = upsample(self.conv_23_0(x_h))
        features["x_23"]  = self.xatt_23(x_h, x_l)


        # node 14
        x_l = features["x_13"]
        x_h = features["x_23"]
        x_h = upsample(self.conv_14_0(x_h))
        features["x_14"] = self.xatt_14(x_h, x_l)

        x = features["x_14"]
        x = upsample(self.conv_04_0(x))
        x = self.conv_04_1(x)
        outputs[("disp", 0)] = self.sigmoid(self.dispconv_0(x))
        outputs[("disp", 1)] = self.sigmoid(self.dispconv_1(features["x_14"]))
        outputs[("disp", 2)] = self.sigmoid(self.dispconv_2(features["x_23"]))
        outputs[("disp", 3)] = self.sigmoid(self.dispconv_3(features["x_32"]))
        return outputs