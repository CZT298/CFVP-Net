# -*- coding: utf-8 -*-
"""
Created on Thu May 29 14:34:40 2025

@author: zyserver
"""
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
import numpy as np
from monai.networks.blocks import  UnetrUpBlock,UnetOutBlock
from typing import Optional, Sequence, Tuple, Type, Union
from timm.models.layers import DropPath
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
def downsample(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm3d(out_dim),
    )  


    
class DiffConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(DiffConv, self).__init__()
        
        # 定义标准的3D卷积层
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
    
    def forward(self, x):
        """
        :param x: 输入张量，形状为 (B, C, D, H, W)
        :return: 差分卷积后的输出张量
        """
        # 获取原始的卷积权重
        conv_weight = self.conv.weight
        
        # 获取卷积核的形状
        kernel_shape = conv_weight.shape
        kernel_size = kernel_shape[2]  # 假设是立方体的卷积核，大小为 kernel_size x kernel_size x kernel_size
        
        # 创建差分卷积核
        diff_kernel = torch.zeros_like(conv_weight)
        
        # 进行差分操作，处理3D卷积核
        # 对于三维卷积，我们希望计算相邻体素之间的差异
        diff_kernel[:, :, kernel_size//2, kernel_size//2, kernel_size//2] = 1  # 中心位置设为1
        diff_kernel[:, :, 0, 0, 0] = -1  # 其他位置设为 -1，模拟差分操作
        
        # 执行卷积运算
        out = F.conv3d(x, diff_kernel, stride=self.conv.stride, padding=self.conv.padding,
                       dilation=self.conv.dilation, groups=self.conv.groups)
        
        return out

class FastConv(nn.Module):
    def __init__(self, in_ch):
        super(FastConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, 3, padding=1),  # in_ch、out_ch是通道数
            nn.BatchNorm3d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm3d(in_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            # print(self.weight.size())
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x
        
class ConvNeXtBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
    
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1, groups=dim)
        self.act = nn.GELU()
        # self.pwconv2 = nn.Linear(4 * dim, dim)
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1, groups=dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1) # (N, C, H, W, D) -> (N, H, W, D, C)
        x = self.norm(x)
        x = x .permute(0, 4, 1, 2, 3)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 3, 4, 1)
        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(0, 4, 1, 2, 3)
        x = input + self.drop_path(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, channel, down=False, r=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )
        self.down = down
        self.conv = nn.Sequential(
            nn.Conv3d(channel, channel//2, (3, 3, 3), padding=1),  # in_ch、out_ch是通道数
            torch.nn.Dropout(0.5),
            nn.BatchNorm3d(channel//2),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        b, c, _, _,_ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1,1)
        # Fusion
        y = torch.mul(x, y)
        if self.down:
            y = self.conv(y)
        return y


    
class LSKmodule3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 深度可分离卷积：kernel_size=5 和 kernel_size=7,dilation=3
        self.conv0 = nn.Conv3d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.convl = nn.Conv3d(dim, dim, kernel_size=7, stride=1, padding=9, groups=dim, dilation=3)

        # 通道降维（再融合）
        self.conv0_s = nn.Conv3d(dim, dim // 2, kernel_size=1)
        self.conv1_s = nn.Conv3d(dim, dim // 2, kernel_size=1)

        # 注意力融合模块
        self.conv_squeeze = nn.Conv3d(2, 2, kernel_size=7, padding=3)

        # 通道恢复
        self.conv_m = nn.Conv3d(dim // 2, dim, kernel_size=1)
        
        self.se = SEBlock(dim // 2)
    def forward(self, x):
        # x: (B, C, D, H, W)
        
        attn1 = self.conv0(x)         # (B,C,D,H,W)
        attn2 = self.convl(attn1)     # (B,C,D,H,W)

        attn1 = self.conv0_s(attn1)   # (B,C/2,D,H,W)
        attn2 = self.conv1_s(attn2)   # (B,C/2,D,H,W)
        # print(attn1.shape)
        
        attn1_se = self.se(attn1)
        attn2_se = self.se(attn2)
        # print(attn1_se.shape)
        attn = torch.cat([attn1, attn2], dim=1)  # (B,C,D,H,W)

        # 融合注意力（通道维度上的 avg 和 max）
        avg_attn = torch.mean(attn, dim=1, keepdim=True)       # (B,1,D,H,W)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)     # (B,1,D,H,W)
        agg = torch.cat([avg_attn, max_attn], dim=1)           # (B,2,D,H,W)

        sig = self.conv_squeeze(agg).sigmoid()  # (B,2,D,H,W)

        # 注意力加权融合
        attn = attn1_se * sig[:, 0:1, :, :, :] + attn2_se * sig[:, 1:2, :, :, :]  # (B,C/2,D,H,W)

        attn = self.conv_m(attn)  # (B,C,D,H,W)

        return x * attn  # 残差增强

class LSKSEBlock(nn.Module):
#     def __init__(self, dim, drop_path=0.0, mlp_ratio=4.0):
#         super().__init__()
#         self.lsk = LSKmodule3D(dim)
#         self.norm = LayerNorm(dim)  
#         self.act = nn.GELU()
#         self.pwconv = nn.Conv3d(int(dim * mlp_ratio), dim, kernel_size=1)
#         self.gamma = nn.Parameter(torch.ones((dim)), requires_grad=True)
#         # self.se = SEBlock(2*dim,down=True)
#         # DropPath 可选（残差连接时的 drop），这里简化为恒等
#         self.drop_path = nn.Identity()

#     def forward(self, x):
#         shortcut = x
        
#         # x_cat = torch.cat([x, x_fast], dim=1)
#         # x = self.se(x_cat)
        
#         x = self.lsk(x)
#         x = self.norm(x)     
#         x = self.act(x)
#         x = self.pwconv(x)
#         x = self.gamma.view(1, -1, 1, 1, 1) * x
#         x = shortcut + self.drop_path(x)
#         return x
    
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.lsk = LSKmodule3D(dim)
        self.norm = LayerNorm(dim, eps=1e-6)
    
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1, groups=dim)
        self.act = nn.GELU()
        # self.pwconv2 = nn.Linear(4 * dim, dim)
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1, groups=dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.lsk(x)
        x = x.permute(0, 2, 3, 4, 1) # (N, C, H, W, D) -> (N, H, W, D, C)
        x = self.norm(x)
        x = x .permute(0, 4, 1, 2, 3)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 3, 4, 1)
        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(0, 4, 1, 2, 3)
        x = input + self.drop_path(x)
        return x

 
class GlobalExtraction(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.avgpool = self.globalavgchannelpool
        self.maxpool = self.globalmaxchannelpool
        self.proj = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=1, stride=1, padding=0),  # 将 Conv2d 改为 Conv3d
            nn.BatchNorm3d(1)  # BatchNorm2d 改为 BatchNorm3d
        )

    def globalavgchannelpool(self, x):
        x = x.mean(1, keepdim=True)
        return x

    def globalmaxchannelpool(self, x):
        x = x.max(dim=1, keepdim=True)[0]
        return x

    def forward(self, x):
        x_ = x.clone()
        x = self.avgpool(x)  # (B, 1, D, H, W)
        x2 = self.maxpool(x_)  # (B, 1, D, H, W)

        cat = torch.cat((x, x2), dim=1)  # (B, 2, D, H, W)

        proj = self.proj(cat)  # (B, 2, D, H, W) --> (B, 1, D, H, W)
        return proj


class ContextExtraction(nn.Module):
    def __init__(self, dim, reduction=None):
        super().__init__()
        self.reduction = 1 if reduction is None else 2

        self.dconv = self.DepthWiseConv3dx2(dim)  # 调整为3D卷积
        self.proj = self.Proj(dim)

    def DepthWiseConv3dx2(self, dim):
        dconv = nn.Sequential(
            nn.Conv3d(in_channels=dim,
                      out_channels=dim,
                      kernel_size=3,
                      padding=1,
                      groups=dim),
            nn.BatchNorm3d(num_features=dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=dim,
                      out_channels=dim,
                      kernel_size=3,
                      padding=2,
                      dilation=2),
            nn.BatchNorm3d(num_features=dim),
            nn.ReLU(inplace=True)
        )
        return dconv

    def Proj(self, dim):
        proj = nn.Sequential(
            nn.Conv3d(in_channels=dim,
                      out_channels=dim // self.reduction,
                      kernel_size=1),
            nn.BatchNorm3d(num_features=dim // self.reduction)
        )
        return proj

    def forward(self, x):
        x = self.dconv(x)  # (B, C, D, H, W) --> (B, C, D, H, W)
        x = self.proj(x)  # (B, C, D, H, W) --> (B, C, D, H, W)
        return x


class MultiscaleFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.local = ContextExtraction(dim)
        self.global_ = GlobalExtraction()
        self.bn = nn.BatchNorm3d(num_features=dim)  # BatchNorm2d 改为 BatchNorm3d

    def forward(self, x, g):
        x = self.local(x)  # 提取局部特征: (B, C, D, H, W) --> (B, C, D, H, W)
        g = self.global_(g)  # 提取全局特征: (B, C, D, H, W) --> (B, 1, D, H, W)
        fuse = self.bn(x + g)  # 融合局部和全局特征: (B, C, D, H, W)
        return fuse


class MultiScaleGatedAttn(nn.Module):
    # Version 1
    def __init__(self, dim):
        super().__init__()
        self.multi = MultiscaleFusion(dim)
        self.selection = nn.Conv3d(dim, 2, 1)  # Conv2d 改为 Conv3d
        self.proj = nn.Conv3d(dim, dim, 1)  # Conv2d 改为 Conv3d
        self.bn = nn.BatchNorm3d(dim)  # BatchNorm2d 改为 BatchNorm3d
        self.bn_2 = nn.BatchNorm3d(dim)  # BatchNorm2d 改为 BatchNorm3d
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels=dim, out_channels=dim,
                      kernel_size=1, stride=1)
        )

    def forward(self, x, g):
        x_ = x.clone()  # (B, C, D, H, W)
        g_ = g.clone()  # (B, C, D, H, W)

        # 1、多尺度特征融合
        multi = self.multi(x, g)  # x:(B, C, D, H, W), g:(B, C, D, H, W) --> multi: (B, C, D, H, W)

        # 2、空间选择
        multi = self.selection(multi)  # 投影到2个通道: (B, C, D, H, W) --> (B, 2, D, H, W)
        attention_weights = F.softmax(multi, dim=1)  # 执行softmax归一化: (B, 2, D, H, W)
        A, B = attention_weights.split(1, dim=1)  # 分割对应于x和g的两个权重: A = B = (B, 1, D, H, W)
        x_att = A.expand_as(x_) * x_  # A扩展到C个通道: (B, 1, D, H, W) -> (B, C, D, H, W)
        g_att = B.expand_as(g_) * g_  # g扩展到C个通道: (B, 1, D, H, W) -> (B, C, D, H, W)
        x_att = x_att + x_  # 添加残差连接
        g_att = g_att + g_  # 添加残差连接

        # 3、空间交互和交叉调制
        x_sig = torch.sigmoid(x_att)  # 为x生成权重
        g_att_2 = x_sig * g_att  # x调整g
        g_sig = torch.sigmoid(g_att)  # 为g生成权重
        x_att_2 = g_sig * x_att  # g调整x
        interaction = x_att_2 * g_att_2  # 执行逐元素乘法使融合后的特征更稳定

        # 4、重新校准
        projected = torch.sigmoid(self.bn(self.proj(interaction)))  # 通过1×1Conv细化融合特征
        weighted = projected * x_  # 注意力权重重新校准初始输入x
        y = self.conv_block(weighted)  # 通过1×1Conv生成输入
        y = self.bn_2(y)
        return y
    

    
class UpDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpDoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, (3, 3, 3), padding=1),  # in_ch、out_ch是通道数
            torch.nn.Dropout(0.5),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            torch.nn.Dropout(0.5),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
class Ours(nn.Module):
    def __init__(self, 
                 in_chans=1,
                 out_chans=3,
                 embed_dim=[64, 128, 256, 512, 1024],
                 depth=[2, 2, 2, 2, 2, 2],
                 spatial_dims: int = 3,
                 norm_name: Union[Tuple, str] = "instance",
                 ):
        super().__init__()
        
        self.blocks_shallow1 = nn.ModuleList()
        for i in range(depth[0]):
            if i == 0:
                self.blocks_shallow1.append( DiffConv(in_channels = in_chans, out_channels=embed_dim[0]) )
            else:
                self.blocks_shallow1.append( DiffConv(in_channels = embed_dim[0], out_channels=embed_dim[0]) )
        self.downsample1 = downsample(in_dim=embed_dim[0], out_dim=embed_dim[1])
        
        self.blocks_shallow2 = nn.ModuleList()
        for i in range(depth[1]):
            self.blocks_shallow2.append( DiffConv(in_channels = embed_dim[1], out_channels=embed_dim[1]) )
        self.downsample2 = downsample(in_dim=embed_dim[1], out_dim=embed_dim[2])
        
        self.blocks_fastdown1 = nn.ModuleList()
        for i in range(depth[2]):
            self.blocks_fastdown1.append(FastConv(in_ch= embed_dim[2]) )
        self.downsample3 = downsample(in_dim=embed_dim[2], out_dim=embed_dim[3])
        
        self.blocks_fastdown2 = nn.ModuleList()
        for i in range(depth[3]):
            self.blocks_fastdown2.append( FastConv(in_ch= embed_dim[3]) )
        
        self.up0 = nn.ConvTranspose3d(embed_dim[3], embed_dim[2], 2, stride=2)
        self.conv_deep1 = UpDoubleConv(embed_dim[3], embed_dim[2])
        self.blocks_deep1 = nn.ModuleList()
        for i in range(depth[4]):
            self.blocks_deep1.append( LSKSEBlock(dim = embed_dim[2]) )
        self.downsample_sub1 = downsample(in_dim=embed_dim[2], out_dim=embed_dim[3])   
        
        self.conv_deep2 = UpDoubleConv(embed_dim[4], embed_dim[3])
        self.blocks_deep2 = nn.ModuleList()
        for i in range(depth[5]):
            self.blocks_deep2.append( LSKSEBlock(dim = embed_dim[3]) )
        self.downsample_sub2 = downsample(in_dim=embed_dim[3], out_dim=embed_dim[4]) 
        
        # self.Bottleneck = FastConv(in_ch = embed_dim[4])
        
        self.up4 = nn.ConvTranspose3d(embed_dim[4], embed_dim[3], 2, stride=2)
        self.MASAG4 = MultiScaleGatedAttn(dim = embed_dim[3])     
        self.conv4 = UpDoubleConv(embed_dim[4], embed_dim[3])
        
        self.up3 = nn.ConvTranspose3d(embed_dim[3], embed_dim[2], 2, stride=2)
        self.MASAG3 = MultiScaleGatedAttn(dim = embed_dim[2])     
        self.conv3 = UpDoubleConv(embed_dim[3], embed_dim[2])
        
        self.up2 = nn.ConvTranspose3d(embed_dim[2], embed_dim[1], 2, stride=2)
        self.MASAG2 = MultiScaleGatedAttn(dim = embed_dim[1])
        self.conv2 = UpDoubleConv(embed_dim[2], embed_dim[1])
        
        self.up1 =  nn.ConvTranspose3d(embed_dim[1], embed_dim[0], 2, stride=2)
        self.MASAG1 = MultiScaleGatedAttn(dim = embed_dim[0])       
        self.conv1 = UpDoubleConv(embed_dim[1], embed_dim[0])
        
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=embed_dim[0], out_channels=out_chans)
        
    def forward(self, x):  
        ##encoder
        for blk in self.blocks_shallow1:
            x = blk(x)
        shallow1 = x #(1,64,96,96,192)
        x = self.downsample1(x) #(1,128,48,48,96)
        
        for blk in self.blocks_shallow2:
            x = blk(x)
        shallow2 = x #(1,128,48,48,96)
        x = self.downsample2(x)#(1,256,24,24,48)
        deep_in = x
        for blk in self.blocks_fastdown1:
            x = blk(x)
        x = self.downsample3(x) # (1, 512, 12, 12, 24)
        
        
        for blk in self.blocks_fastdown2:
            x = blk(x)
        fast2 = x # (1, 512, 12, 12, 24)
        
        x_up = self.up0(fast2) # (1, 256, 24, 24, 48)
        for idx,blk in enumerate(self.blocks_deep1):
            if idx == 0:
                x_cat =  torch.cat([deep_in, x_up], dim=1) # (1, 512, 24, 24, 48)
                x = self.conv_deep1(x_cat) # (1, 256, 24, 24, 48)
            x = blk(x) # (1, 256, 24, 24, 48)
        deep1 = x
        x = self.downsample_sub1(x) # (1, 512, 12, 12, 24)
    
        for idx,blk in enumerate(self.blocks_deep2):
            if idx == 0: 
                x_cat =  torch.cat([x, fast2], dim=1) # (1, 1024, 12, 12, 24)
                x = self.conv_deep2(x_cat) # (1, 512, 12, 12, 24)
            x = blk(x)
        deep2 = x # (1, 512, 12, 12, 24)
        x = self.downsample_sub2(x)# (1, 1024, 6, 6, 12) 
        
        # x = self.Bottleneck(x) # (1, 1024, 6, 6, 12) 
        
        ###decoder
        x = self.up4(x) # (1, 512, 12, 12, 24)
        attn4 = self.MASAG4(x, deep2) # (1, 512, 12, 12, 24)
        x_cat = torch.cat([x, attn4], dim=1) # (1, 1024, 12, 12, 24)
        x = self.conv4(x_cat) # (1, 512, 12, 12, 24)
        
        x = self.up3(x) # (1, 256, 24, 24, 48)
        attn3 = self.MASAG3(x, deep1) # (1, 256, 24, 24, 48))
        x_cat = torch.cat([x, attn3], dim=1) # (1, 512, 24, 24, 48)
        x = self.conv3(x_cat) # # (1, 256, 24, 24, 48)
        
        x = self.up2(x) # #(1,128,48,48,96)
        attn2 = self.MASAG2(x, shallow2) # #(1,128,48,48,96)
        x_cat = torch.cat([x, attn2], dim=1) # #(1,256,48,48,96)
        x = self.conv2(x_cat) # # #(1,128,48,48,96)
        
        x = self.up1(x) # #(1,64,96,96,192)
        attn1 = self.MASAG1(x, shallow1) # #(1,64,96,96,192)
        x_cat = torch.cat([x, attn1], dim=1) # #(1,128,96,96,192)
        x = self.conv1(x_cat) # #(1,64,96,96,192)
        
        out = self.out(x)
        return nn.Softmax(dim=1)(out)
            
            
            
            
            
            
            
if __name__ == '__main__':
    # (B,C,H,W)
    input=torch.randn(1,1,96,96,192).to(device)
    # input=torch.randn(1,256,24,24,48)
    up = nn.ConvTranspose3d(256,128,1,1)
    model = LSKSEBlock(dim=256)
    our= Ours(1,3).to(device)
    fast = FastConv(256) 
    output=our(input)
    print(output.shape)    