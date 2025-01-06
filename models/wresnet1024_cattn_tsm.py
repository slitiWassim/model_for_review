import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.wider_resnet import wresnet ,Efficientnet_1024,Efficientnet_X3D
from models.basic_modules import ConvBnRelu, ConvTransposeBnRelu, initialize_weights
from grouped_query_attention_pytorch.attention import MultiheadGQA
from models.QGAttention import Attention_QGA,MultiheadGQAConv
from models.Quantizer import Quantizer
import separableconv.nn as ns
import math

logger = logging.getLogger(__name__)
embedding_dim = 128


class ASTNet(nn.Module):
    def get_name(self):
        return self.model_name

    def __init__(self, config, pretrained=True):
        super(ASTNet, self).__init__()
        
        
        frames = config.MODEL.ENCODED_FRAMES
        final_conv_kernel = config.MODEL.EXTRA.FINAL_CONV_KERNEL
        self.model_name = config.MODEL.NAME
        logger.info('=> ' + self.model_name + '_1024: (CATTN + TSM) - Ped2')
        self.batch=config.TRAIN.BATCH_SIZE_PER_GPU
        self._commitment_cost = 0.25

        channels = [192,96,48,24]
        
        
        
        '''   --   Encoder  --   '''
        # Encoder Backbone 
        self.Efficientnet_X3D = Efficientnet_X3D()

        # Grouped Query Self-attention (GQA)

        #self.attn8 = Attention_QGA(embed_dim=72)
        #self.attn2 = Attention_QGA(embed_dim=72)
        #self.attn0 = Attention_QGA(embed_dim=72)
        self.attn = MultiheadGQA(embed_dim=embedding_dim, query_heads=4, kv_heads=2, device="cuda")
    

        self.conv_x8 = conv_block(ch_in=channels[2] * frames, ch_out=channels[3])
        self.conv_x2 = conv_block(ch_in=channels[3] * frames, ch_out=channels[3])
        self.conv_x0 = conv_block(ch_in=channels[3] * frames, ch_out=channels[3])
        
        ''' --- End of Encoder --- '''
        
        
        # Quantization Layer
        self.vq = Quantizer(channels[3],codebook_size =128,Quantizer_name='ResidualVQ')
        

        
        self.conv8 = nn.Conv2d(in_channels=channels[3],out_channels=embedding_dim,kernel_size=1,bias=False)
        self.conv2 = nn.Conv2d(in_channels=channels[3], out_channels=embedding_dim,kernel_size=1,bias=False)
        self.conv0 = nn.Conv2d(in_channels=channels[3],out_channels=embedding_dim,kernel_size=1,bias=False)
        #self.conv0 = ns.SeparableConv2d(in_channels=channels[3], out_channels=embedding_dim,kernel_size=1)
        #self.pool0 = nn.AvgPool2d(kernel_size=4, stride=4) 
        #self.pool0 = nn.MaxPool2d(kernel_size=6, stride=4,padding=1) 
        #self.pool0 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)
        
        
        #self.conv2 = ns.SeparableConv2d(in_channels=channels[3], out_channels=channels[3],kernel_size=1)
        #self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2) 
        #self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.pool2 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False) 
        
        self._pre_vq_conv = nn.Conv2d(in_channels=embedding_dim, out_channels=channels[3],kernel_size=1)
        #self._pre_vq_conv = nn.SeparableConv2d(in_channels=embedding_dim, out_channels=channels[3],kernel_size=1)

        '''   --   Decoder  --   '''
        self.up8 = ConvTransposeBnRelu(channels[3], channels[3] ,kernel_size=2)   # 2048          -> 1024
        self.up4 = ConvTransposeBnRelu(channels[3]+channels[3], channels[3], kernel_size=2)   # 1024  +   256 -> 512
        self.up2 = ConvTransposeBnRelu(channels[3]+channels[3], channels[3], kernel_size=2)   # 512   +   128 -> 256

        ''' SWIGLU  '''
        #self.attn1 = ConvSwiGLU(channels[3], channels[3])
        #self.attn2 = ConvSwiGLU(channels[3], channels[3])
        #self.attn3 = ConvSwiGLU(channels[3], channels[3])

        #self.tsm_left = TemporalShift(n_segment=4, n_div=16, direction='left')

        self.final = nn.Sequential(
            ConvBnRelu(channels[3], channels[2], kernel_size=1, padding=0),
            ConvBnRelu(channels[2], channels[3], kernel_size=3, padding=1),
            nn.Conv2d(channels[3], 3,
                      kernel_size=final_conv_kernel,
                      padding=1 if final_conv_kernel == 3 else 0,
                      bias=False)
        )

        ''' --- End of Decoder --- '''


        # Initialize  Layers  Weights

        initialize_weights(self.conv_x0, self.conv_x2, self.conv_x8)
        initialize_weights(self.up2, self.up4, self.up8)
        initialize_weights(self.vq)
        initialize_weights(self.conv2,self.conv0,self._pre_vq_conv)
        initialize_weights(self.attn)
        #initialize_weights(self.attn1,self.attn2,self.attn3)
        initialize_weights(self.final)

    def forward(self, x):

        '''   --   Encoder  Part  --   '''

        x_input=torch.stack(x,dim=2)
        x0,x1,x2=self.Efficientnet_X3D(x_input)
        # Reshape the features from various backbone stages
        x0=x0.view(x0.shape[0], -1, x0.shape[-2], x0.shape[-1])
        x1=x1.view(x1.shape[0], -1, x1.shape[-2], x1.shape[-1])
        x2=x2.view(x2.shape[0], -1, x2.shape[-2], x2.shape[-1])

        x8 = self.conv_x8(x2)
        x2 = self.conv_x2(x1)
        x0 = self.conv_x0(x0)

        # Conv pre QGA 

        x8_ = self.conv8(x8)
        #x8_shape = x8.shape
        #x2_ = self.pool2(self.conv2(x2))
        x2_ = F.interpolate(self.conv2(x2), scale_factor=0.5, mode='bilinear', align_corners=False) 
        #x2_shape = x2.shape
        x0_ = F.interpolate(self.conv0(x0), scale_factor=0.25, mode='bilinear', align_corners=False) 
        #x0_ = self.pool0(self.conv0(x0))
        x0_shape = x0_.shape
        
        #x8 = self.relu8(x8)
        x8 = x8_.view(x0_shape[0],-1,embedding_dim) 
        x2_ = x2_.view(x0_shape[0],-1,embedding_dim) 
        x0_ = x0_.view(x0_shape[0],-1,embedding_dim) 
        x8 = self.attn(x0_,x2_,x8)[0].view(x0_shape)
        

        
        ''' --- End of Encoder Part --- '''


        ## Apply Quantization to the Encoder output
        x8 =self._pre_vq_conv(x8)
        x8 , _ =self.vq(x8)    # During training, retrieve the _loss_commit to optimize the model.
        #x2 , _loss_commit2 =self.vq2(x2) # During training, retrieve the _loss_commit to optimize the model.
        #x0 , _loss_commit0 =self.vq0(x0) # During training, retrieve the _loss_commit to optimize the model.



        '''   --   Decoder  Part  --   '''

       
        x = self.up8(x8)
        #x = self.up8(torch.cat([x8, x81], dim=1))

       #x = self.attn1(x)
        
        x = self.up4(torch.cat([x2, x], dim=1))
        
        #x = self.attn2(x)

        x = self.up2(torch.cat([x0, x], dim=1))
        
        #x = self.attn3(x)

        ''' --- End of Decoder Part --- '''


        return self.final(x)
    

    def compute_loss(self, x):

        '''   --   Encoder  Part  --   '''

        x_input=torch.stack(x,dim=2)
        x0,x1,x2=self.Efficientnet_X3D(x_input)

        # Reshape the features from various backbone stages
        x0=x0.view(x0.shape[0], -1, x0.shape[-2], x0.shape[-1])
        x1=x1.view(x1.shape[0], -1, x1.shape[-2], x1.shape[-1])
        x2=x2.view(x2.shape[0], -1, x2.shape[-2], x2.shape[-1])

        x8 = self.conv_x8(x2)
        x2 = self.conv_x2(x1)
        x0 = self.conv_x0(x0)

        # Conv pre QGA 

        x8_ = self.conv8(x8)
        #x8_shape = x8.shape
        x2_ = F.interpolate(self.conv2(x2), scale_factor=0.5, mode='bilinear', align_corners=False) 
        #x2_shape = x2.shape
        x0_ = F.interpolate(self.conv0(x0), scale_factor=0.25, mode='bilinear', align_corners=False) 
        x0_shape = x0_.shape
        #x8 = self.relu8(x8)
        x8 = x8_.view(self.batch,-1,embedding_dim) 
        x2_ = x2_.view(self.batch,-1,embedding_dim) 
        x0_ = x0_.view(self.batch,-1,embedding_dim) 
        x8 = self.attn(x0_,x2_,x8)[0].view(x0_shape)
        
        
        ''' --- End of Encoder Part --- '''
        
        ## Apply Quantization to the Encoder output
        x8 = self._pre_vq_conv(x8)
        x8 , _loss_commit =self.vq(x8) # During training, retrieve the _loss_commit to optimize the model.
        #x2 , _loss_commit2 =self.vq2(x2) # During training, retrieve the _loss_commit to optimize the model.
        #x0 , _loss_commit0 =self.vq0(x0) # During training, retrieve the _loss_commit to optimize the model.

        
        '''   --   Decoder  Part  --   '''

        x = self.up8(x8)
        #x = self.up8(torch.cat([x8, x81], dim=1))

        #x = self.attn1(x)
        
        x = self.up4(torch.cat([x2, x], dim=1))
        
        #x = self.attn2(x)

        x = self.up2(torch.cat([x0, x], dim=1))
        
        #x = self.attn3(x)
        
        
        ''' --- End of Decoder Part --- '''

        loss_commit = self._commitment_cost * _loss_commit 

        
        return self.final(x) ,loss_commit
    

class ConvSwiGLU(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(ConvSwiGLU, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(self.conv1(x)) * self.conv2(x)

class ChannelAttention(nn.Module):
    def __init__(self, input_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.layer = nn.Sequential(
            nn.Conv2d(input_channels, input_channels//reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels//reduction, input_channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.layer(y)
        return x * y


class TemporalShift(nn.Module):
    def __init__(self, n_segment=4, n_div=8, direction='left'):
        super(TemporalShift, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div
        self.direction = direction

        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, direction=self.direction)
        return x

    @staticmethod
    def shift(x, n_segment=4, fold_div=8, direction='left'):
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
        return x * psi                    #与low-level feature相乘，将权重矩阵赋值进去

