import logging
import torch
import torch.nn as nn
from models.wider_resnet import wresnet,Efficientnet_X3D_v1
from models.basic_modules import ConvBnRelu, ConvTransposeBnRelu, initialize_weights 
from torch.nn import functional as F
from torchvision import models
import torchvision
import math

logger = logging.getLogger(__name__)


class ASTNet(nn.Module):
    def get_name(self):
        return self.model_name

    def __init__(self, config, pretrained=True):
        super(ASTNet, self).__init__()
        frames = config.MODEL.ENCODED_FRAMES
        final_conv_kernel = config.MODEL.EXTRA.FINAL_CONV_KERNEL
        self.model_name = config.MODEL.NAME

        logger.info(self.model_name + ' (AM + TSM) - WiderResNet_layer6')

        self.Efficientnet_X3D=Efficientnet_X3D_v1()

        channels = [192,96,48,24]

        self.conv_x7 = conv_block(ch_in = channels[1] * frames, ch_out = channels[2])
        self.conv_x3 = conv_block(ch_in = channels[3] * frames, ch_out = channels[3])
        self.conv_x2 = conv_block(ch_in = channels[3] * frames, ch_out = channels[3])

        self.tsm_left = TemporalShift(n_segment=4, n_div=16, direction='left', split=False)

        self.up8 = ConvTransposeBnRelu(channels[2], channels[1], kernel_size=4,stride = 4)
        self.up4 = ConvTransposeBnRelu(channels[3] + channels[3], channels[2], kernel_size=2)
        self.up2 = ConvTransposeBnRelu(channels[3] + channels[3], channels[3], kernel_size=2)

        lReLU = nn.LeakyReLU(0.2, True)
        self.attn8 = RCAB(channels[1], channels[3], kernel_size=3, reduction=16, norm='BN', act=lReLU, downscale=True)
        self.attn4 = RCAB(channels[2], channels[3], kernel_size=3, reduction=16, norm='BN', act=lReLU, downscale=True)
        self.attn2 = RCAB(channels[3], channels[3], kernel_size=3, reduction=16, norm='BN', act=lReLU, downscale=True)

        self.final = nn.Sequential(
            ConvBnRelu(channels[3], channels[3], kernel_size=3, padding=1),  # TODO: kernel_size=3
            ConvBnRelu(channels[3], channels[3], kernel_size=5, padding=2),  # TODO: kernel_size=3
            nn.Conv2d(channels[3], 3,
                      kernel_size=final_conv_kernel,
                      padding=(final_conv_kernel-1)//2,
                      bias=False)
        )

        initialize_weights(self.conv_x2, self.conv_x3, self.conv_x7)
        initialize_weights(self.up2, self.up4, self.up8)
        initialize_weights(self.attn2, self.attn4, self.attn8)
        initialize_weights(self.final)

    def forward(self, x):
        x_input=torch.stack(x,dim=2)
        x0,x1,x2=self.Efficientnet_X3D(x_input)


        x0=x0.view(x0.shape[0], -1, x0.shape[-2], x0.shape[-1])
        x1=x1.view(x1.shape[0], -1, x1.shape[-2], x1.shape[-1])
        x2=x2.view(x2.shape[0], -1, x2.shape[-2], x2.shape[-1])
        
        x8 = self.conv_x7(x2)
        x2 = self.conv_x3(x1)
        x0 = self.conv_x2(x0)

        left = self.tsm_left(x8)

        x8 = x8 + left
        x = self.up8(x8)
        x = self.attn8(x)
        print(x.shape)
        print(x2.shape)

        x = self.up4(torch.cat([x2, x], dim=1))     # 1024 + 512    -> 512, 48, 80
        x = self.attn4(x)

        x = self.up2(torch.cat([x0, x], dim=1))     # 512 + 256     -> 256, 96, 160
        x = self.attn2(x)

        return self.final(x)


class TemporalShift(nn.Module):
    def __init__(self, n_segment=4, n_div=8, direction='left', split=False):
        super(TemporalShift, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div
        self.direction = direction
        self.split = split

        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, direction=self.direction, split=self.split)
        return x

    @staticmethod
    def shift(x, n_segment=4, fold_div=8, direction='left', split=False):
        bz, nt, h, w = x.size()
        c = nt // n_segment
        x = x.view(bz, n_segment, c, h, w)

        fold = c // fold_div

        out = torch.zeros_like(x)
        if direction == 'left':
            out[:, :-1, :fold] = x[:, 1:, :fold]
            out[:, :, fold:] = x[:, :, fold:]
        elif direction == 'right':
            out[:, 1:, :fold] = x[:, :-1, :fold]
            out[:, :, fold:] = x[:, :, fold:]
        else:
            out[:, :-1, :fold] = x[:, 1:, :fold]
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]

        if split:
            p1, _ = out.split([fold * 2, c - (fold * 2)], dim=2)
            p1 = p1.reshape(bz, n_segment * fold * 2, h, w)
            return p1
        else:
            return out.view(bz, nt, h, w)


class RCAB(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, reduction,
            norm=False, act=nn.ReLU(True), downscale=False, return_ca=False):
        super(RCAB, self).__init__()

        self.body = nn.Sequential(
            ConvNorm(in_feat, out_feat, kernel_size, stride=1, norm=norm),
            act,
            ConvNorm(out_feat, out_feat, kernel_size, stride=1, norm=norm),
        )
        self.CA = CALayer(out_feat, reduction)
        self.sig = nn.Sigmoid()

        self.downscale = downscale
        if downscale:
            self.downConv = nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1)
        self.return_ca = return_ca

    def forward(self, x):
        res = x
        out = self.body(x)
        ca = self.CA(out)

        if self.downscale:
            res = self.downConv(res)

        return res + (out * self.sig(ca))


class ConvNorm(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, stride=1, norm=False):
        super(ConvNorm, self).__init__()

        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv = nn.Conv2d(in_feat, out_feat, stride=stride, kernel_size=kernel_size, bias=True)

        self.norm = norm
        if norm == 'IN':
            self.norm = nn.InstanceNorm2d(out_feat, track_running_stats=True)
        elif norm == 'BN':
            self.norm = nn.BatchNorm2d(out_feat)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        return out


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return y






from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision
import math
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 3.


class SEModule_small(nn.Module):
    def __init__(self, channel):
        super(SEModule_small, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            Hsigmoid()
        )

    def forward(self, x):
        y = self.fc(x)
        return x * y


class DYCls(nn.Module):
    def __init__(self, inp, oup):
        super(DYCls, self).__init__()
        self.dim = 32
        self.cls = nn.Linear(inp, oup)
        self.cls_q = nn.Linear(inp, self.dim, bias=False)
        self.cls_p = nn.Linear(self.dim, oup, bias=False)

        mid = 32

        self.fc = nn.Sequential(
            nn.Linear(inp, mid, bias=False),
            SEModule_small(mid),
        )
        self.fc_phi = nn.Linear(mid, self.dim ** 2, bias=False)
        self.fc_scale = nn.Linear(mid, oup, bias=False)
        self.hs = Hsigmoid()
        self.bn1 = nn.BatchNorm1d(self.dim)
        self.bn2 = nn.BatchNorm1d(self.dim)

    def forward(self, x):
        # r = self.cls(x)
        b, c = x.size()
        y = self.fc(x)
        dy_phi = self.fc_phi(y).view(b, self.dim, self.dim)
        dy_scale = self.hs(self.fc_scale(y)).view(b, -1)

        r = dy_scale * self.cls(x)

        x = self.cls_q(x)
        x = self.bn1(x)
        x = self.bn2(torch.matmul(dy_phi, x.view(b, self.dim, 1)).view(b, self.dim)) + x
        x = self.cls_p(x)

        return x + r




class conv_dy(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride, padding):
        super(conv_dy, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.dim = int(math.sqrt(inplanes))
        squeeze = max(inplanes, self.dim ** 2) // 16

        self.q = nn.Conv2d(inplanes, self.dim, 1, stride, 0, bias=False)

        self.p = nn.Conv2d(self.dim, planes, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.dim)
        self.bn2 = nn.BatchNorm1d(self.dim)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(inplanes, squeeze, bias=False),
            SEModule_small(squeeze),
        )
        self.fc_phi = nn.Linear(squeeze, self.dim ** 2, bias=False)
        self.fc_scale = nn.Linear(squeeze, planes, bias=False)
        self.hs = Hsigmoid()

    def forward(self, x):
        r = self.conv(x)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        phi = self.fc_phi(y).view(b, self.dim, self.dim)
        scale = self.hs(self.fc_scale(y)).view(b, -1, 1, 1)
        r = scale.expand_as(r) * r

        out = self.bn1(self.q(x))
        _, _, h, w = out.size()

        out = out.view(b, self.dim, -1)
        out = self.bn2(torch.matmul(phi, out)) + out
        out = out.view(b, -1, h, w)
        out = self.p(out) + r
        return out


class Bottleneck_dy(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_dy, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = conv_dy(inplanes, planes, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv_dy(planes, planes, 3, stride, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv_dy(planes, planes * 4, 1, 1, 0)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            conv_dy(ch_out, ch_out, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x



class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

'''size – 根据不同的输入类型制定的输出大小

scale_factor – 指定输出为输入的多少倍数。如果输入为tuple，其也要制定为tuple类型

mode (str, optional) – 可使用的上采样算法，有'nearest', 'linear', 'bilinear', 'bicubic' and 'trilinear'. 默认使用'nearest'

align_corners (bool, optional) – 如果为True，输入的角像素将与输出张量对齐，因此将保存下来这些像素的值。'''

#padding 的操作就是在图像块的周围加上格子, 从而使得图像经过卷积过后大小不会变化,这种操作是使得图像的边缘数据也能被利用到,这样才能更好地扩张整张图像的边缘特征.


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)                  #1x512x64x64->conv(512，256)/B.N.->1x256x64x64
        # 上采样的 l 卷积
        x1 = self.W_x(x)                  #1x512x64x64->conv(512，256)/B.N.->1x256x64x64
        # concat + relu
        psi = self.relu(g1 + x1)          #1x256x64x64di
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)               #得到权重矩阵  1x256x64x64 -> 1x1x64x64 ->sigmoid 结果到（0，1）
        # 返回加权的 x
        return x * psi  