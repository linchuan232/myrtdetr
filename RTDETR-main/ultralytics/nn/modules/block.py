# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
import numpy as np
from .conv import Conv, DWConv, GhostConv, LightConv, RepConv
from .transformer import TransformerBlock

from einops import rearrange

__all__ = ('DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3', 'C2f', 'C3x', 'C3TR', 'C3Ghost',
           'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'RepC3', 'ConvNormLayer', 'BasicBlock', 
           'BottleNeck', 'Blocks','C2f_MambaOut_DSA','BasicBlock_Hybrid_Full','SmallObjectEnhancementModule')

def autopad(k, p=None, d=1):
    """自动填充以保持输出尺寸"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


# ============================================================================
# 频域处理组件（来自 SFS_Conv）
# ============================================================================

class FractionalGaborFilter(nn.Module):
    """分数阶Gabor滤波器"""
    def __init__(self, in_channels, out_channels, kernel_size, order, angles, scales):
        super().__init__()
        self.real_weights = nn.ParameterList()
        
        for angle in angles:
            for scale in scales:
                real_weight = self.generate_fractional_gabor(
                    in_channels, out_channels, kernel_size, order, angle, scale
                )
                self.real_weights.append(nn.Parameter(real_weight))

    def generate_fractional_gabor(self, in_channels, out_channels, size, order, angle, scale):
        x, y = np.meshgrid(np.linspace(-1, 1, size[0]), np.linspace(-1, 1, size[1]))
        x_theta = x * np.cos(angle) + y * np.sin(angle)
        y_theta = -x * np.sin(angle) + y * np.cos(angle)
        
        real_part = np.exp(-((x_theta**2 + (y_theta / scale) ** 2) ** order)) * \
                    np.cos(2 * np.pi * x_theta / scale)
        
        real_weight = torch.tensor(real_part, dtype=torch.float32).view(1, 1, size[0], size[1])
        real_weight = real_weight.repeat(out_channels, 1, 1, 1)
        return real_weight

    def forward(self, x):
        real_result = sum(weight * x for weight in self.real_weights)
        return real_result


class FrequencyUnit(nn.Module):
    """频域处理单元"""
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), order=0.25):
        super().__init__()
        angles = [0, np.pi/4, np.pi/2]
        scales = [1, 2]
        
        self.gabor = FractionalGaborFilter(
            in_channels, out_channels, kernel_size, order, angles, scales
        )
        self.t = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]),
            requires_grad=True,
        )
        nn.init.normal_(self.t, std=0.02)
        self.act = nn.GELU()

    def forward(self, x):
        out = self.gabor(self.t)
        out = F.conv2d(x, out, stride=1, padding=(out.shape[-2] - 1) // 2)
        out = self.act(out)
        return out


# ============================================================================
# 门控空间单元（来自 GatedCNNBlock）
# ============================================================================

class GatedSpatialUnit(nn.Module):
    """门控空间处理单元"""
    def __init__(self, dim, kernel_size=7, conv_ratio=0.5, expansion_ratio=2.0):
        super().__init__()
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = nn.GELU()
        
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        
        self.conv = nn.Conv2d(
            conv_channels, conv_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=conv_channels
        )
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        # x: [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1)
        c = c.permute(0, 3, 1, 2)
        c = self.conv(c)
        c = c.permute(0, 2, 3, 1)
        
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=-1))
        return x.permute(0, 3, 1, 2)


# ============================================================================
# HybridBottleneck 三个版本
# ============================================================================

class HybridBottleneck_Full(nn.Module):
    """完整版混合瓶颈块 - 空间+频域"""
    def __init__(self, c1, c2, shortcut=True, kernel_size=7, expansion=0.5):
        super().__init__()
        c_ = int(c2 * expansion)
        
        self.cv1 = Conv(c1, c_, 1, 1)
        self.gated_spatial = GatedSpatialUnit(c_, kernel_size=kernel_size, conv_ratio=0.5)
        self.frequency = FrequencyUnit(c_, c_, kernel_size=(3, 3))
        
        self.fusion = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_ * 2, c_ // 2, 1),
            nn.GELU(),
            nn.Conv2d(c_ // 2, c_ * 2, 1),
            nn.Sigmoid()
        )
        
        self.cv2 = Conv(c_ * 2, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        feat = self.cv1(x)
        spatial_feat = self.gated_spatial(feat)
        freq_feat = self.frequency(feat)
        combined = torch.cat([spatial_feat, freq_feat], dim=1)
        attention = self.fusion(combined)
        fused = combined * attention
        out = self.cv2(fused)
        return x + out if self.add else out


# ============================================================================
# C2f_Hybrid 三个版本 - CSP架构
# ============================================================================

class C2f_Hybrid_Full(nn.Module):
    """
    C2f with HybridBottleneck_Full
    CSP Bottleneck with 2 convolutions - 完整版
    
    特点:
    - 使用 HybridBottleneck_Full 作为基础模块
    - 完整的空间+频域处理
    - CSP架构提供更好的梯度流
    - 适合对精度要求高的任务
    
    参数:
        c1: 输入通道数
        c2: 输出通道数
        n: Bottleneck数量
        shortcut: 是否使用shortcut连接
        g: 分组卷积的组数（保留参数，实际在HybridBottleneck中不使用）
        e: expansion ratio，隐藏层通道扩展比例
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        
        # 使用 HybridBottleneck_Full 替代原始 Bottleneck
        self.m = nn.ModuleList(
            HybridBottleneck_Full(
                self.c, 
                self.c, 
                shortcut=shortcut,
                kernel_size=7,
                expansion=1.0
            ) for _ in range(n)
        )

    def forward(self, x):
        """前向传播 - CSP架构"""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """使用split()而非chunk()的前向传播"""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2f_Hybrid_Lite(nn.Module):
    """
    C2f with HybridBottleneck_Lite
    CSP Bottleneck with 2 convolutions - 轻量版 ⭐ 推荐
    
    特点:
    - 使用 HybridBottleneck_Lite 作为基础模块
    - 选择性的频域处理
    - 平衡精度和效率
    - 适合大多数应用场景
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        
        # 使用 HybridBottleneck_Lite
        self.m = nn.ModuleList(
            HybridBottleneck_Lite(
                self.c, 
                self.c, 
                shortcut=shortcut,
                kernel_size=7,
                expansion=1.0
            ) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2f_Hybrid_Fast(nn.Module):
    """
    C2f with HybridBottleneck_Fast
    CSP Bottleneck with 2 convolutions - 快速版
    
    特点:
    - 使用 HybridBottleneck_Fast 作为基础模块
    - 仅门控机制，无频域处理
    - 最快的推理速度
    - 适合实时应用
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        
        # 使用 HybridBottleneck_Fast
        self.m = nn.ModuleList(
            HybridBottleneck_Fast(
                self.c, 
                self.c, 
                shortcut=shortcut,
                kernel_size=7,
                expansion=1.0
            ) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ============================================================================
# 用于替代BasicBlock的包装类
# ============================================================================

class BasicBlock_Hybrid_Full(nn.Module):
    """
    用C2f_Hybrid_Full替代BasicBlock的包装类
    可直接在ResNet中替换原始BasicBlock
    
    使用方法:
        # 原始: block = BasicBlock(64, 64)
        # 替换: block = BasicBlock_Hybrid_Full(64, 64)
    """
    expansion = 1  # 保持与BasicBlock一致
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, n=1):
        super().__init__()
        
        # 如果stride!=1，需要下采样
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self.downsample = downsample
        
        # 使用C2f_Hybrid_Full
        # 当stride=1时，直接使用C2f
        # 当stride!=1时，先下采样再使用C2f
        if stride == 1:
            self.c2f = C2f_Hybrid_Full(inplanes, planes, n=n, shortcut=True, e=0.5)
        else:
            # 先通过卷积下采样
            self.stride_conv = nn.Sequential(
                Conv(inplanes, planes, 3, stride),
            )
            self.c2f = C2f_Hybrid_Full(planes, planes, n=n, shortcut=True, e=0.5)
        
        self.stride = stride

    def forward(self, x):
        identity = x
        
        if self.stride != 1:
            # 先下采样
            out = self.stride_conv(x)
            out = self.c2f(out)
            if self.downsample is not None:
                identity = self.downsample(x)
        else:
            out = self.c2f(x)
        
        # 残差连接
        if identity.shape == out.shape:
            out += identity
        elif self.downsample is not None:
            out += self.downsample(identity)
            
        return out


class BasicBlock_Hybrid_Lite(nn.Module):
    """
    用C2f_Hybrid_Lite替代BasicBlock的包装类 ⭐ 推荐
    """
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, n=1):
        super().__init__()
        
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self.downsample = downsample
        
        if stride == 1:
            self.c2f = C2f_Hybrid_Lite(inplanes, planes, n=n, shortcut=True, e=0.5)
        else:
            self.stride_conv = nn.Sequential(
                Conv(inplanes, planes, 3, stride),
            )
            self.c2f = C2f_Hybrid_Lite(planes, planes, n=n, shortcut=True, e=0.5)
        
        self.stride = stride

    def forward(self, x):
        identity = x
        
        if self.stride != 1:
            out = self.stride_conv(x)
            out = self.c2f(out)
            if self.downsample is not None:
                identity = self.downsample(x)
        else:
            out = self.c2f(x)
        
        if identity.shape == out.shape:
            out += identity
        elif self.downsample is not None:
            out += self.downsample(identity)
            
        return out


class BasicBlock_Hybrid_Fast(nn.Module):
    """
    用C2f_Hybrid_Fast替代BasicBlock的包装类
    """
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, n=1):
        super().__init__()
        
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self.downsample = downsample
        
        if stride == 1:
            self.c2f = C2f_Hybrid_Fast(inplanes, planes, n=n, shortcut=True, e=0.5)
        else:
            self.stride_conv = nn.Sequential(
                Conv(inplanes, planes, 3, stride),
            )
            self.c2f = C2f_Hybrid_Fast(planes, planes, n=n, shortcut=True, e=0.5)
        
        self.stride = stride

    def forward(self, x):
        identity = x
        
        if self.stride != 1:
            out = self.stride_conv(x)
            out = self.c2f(out)
            if self.downsample is not None:
                identity = self.downsample(x)
        else:
            out = self.c2f(x)
        
        if identity.shape == out.shape:
            out += identity
        elif self.downsample is not None:
            out += self.downsample(identity)
            
        return out



# DropPath definition (required for GatedCNNBlockAdapted)
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

# GatedCNNBlockAdapted (channel-first adapted)
class GatedCNNBlockAdapted(nn.Module):
    def __init__(self, dim, expansion_ratio=8/3, kernel_size=3, conv_ratio=1.0, drop_path=0.):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Conv2d(dim, hidden * 2, 1)
        self.act = nn.GELU()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        self.conv = nn.Conv2d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=conv_channels)
        self.fc2 = nn.Conv2d(hidden, dim, 1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=1)
        c = self.conv(c)
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=1))
        x = self.drop_path(x)
        return x + shortcut

# GatedDSABlock (adapted to match Bottleneck interface)
class GatedDSABlock(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=1.0, drop_path=0.):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels, matching original Bottleneck
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.spatial_gating_unit = GatedCNNBlockAdapted(c2, kernel_size=k[1], drop_path=drop_path)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        out = self.spatial_gating_unit(out)
        return x + out if self.add else out

# Modified C2f with GatedDSABlock replacement
class C2f_MambaOut_DSA(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(GatedDSABlock(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))



class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck with 2 convolutions module with arguments ch_in, ch_out, number, shortcut,
        groups, expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))

################################### RT-DETR PResnet ###################################
def get_activation(act: str, inpace: bool=True):
    '''get activation
    '''
    act = act.lower()
    
    if act == 'silu':
        m = nn.SiLU()

    elif act == 'relu':
        m = nn.ReLU()

    elif act == 'leaky_relu':
        m = nn.LeakyReLU()

    elif act == 'silu':
        m = nn.SiLU()
    
    elif act == 'gelu':
        m = nn.GELU()
        
    elif act is None:
        m = nn.Identity()
    
    elif isinstance(act, nn.Module):
        m = act

    else:
        raise RuntimeError('')  

    if hasattr(m, 'inplace'):
        m.inplace = inpace
    
    return m 

class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
    
    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__()

        self.shortcut = shortcut

        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)

        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 


    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        if self.shortcut:
            short = x
        else:
            short = self.short(x)
        
        out = out + short
        out = self.act(out)

        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__()

        if variant == 'a':
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        width = ch_out 

        self.branch2a = ConvNormLayer(ch_in, width, 1, stride1, act=act)
        self.branch2b = ConvNormLayer(width, width, 3, stride2, act=act)
        self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)

        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.branch2c(out)

        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        out = out + short
        out = self.act(out)

        return out


class Blocks(nn.Module):
    def __init__(self, ch_in, ch_out, block, count, stage_num, act='relu', input_resolution=None, sr_ratio=None, kernel_size=None, kan_name=None, variant='d'):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(count):
            if input_resolution is not None and sr_ratio is not None:
                self.blocks.append(
                    block(
                        ch_in, 
                        ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1, 
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act,
                        input_resolution=input_resolution,
                        sr_ratio=sr_ratio)
                )
            elif kernel_size is not None:
                self.blocks.append(
                    block(
                        ch_in, 
                        ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1, 
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act,
                        kernel_size=kernel_size)
                )
            elif kan_name is not None:
                self.blocks.append(
                    block(
                        ch_in, 
                        ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1, 
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act,
                        kan_name=kan_name)
                )
            else:
                self.blocks.append(
                    block(
                        ch_in, 
                        ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1, 
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act)
                )
            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out









class AHFF(nn.Module):
    """
    Adaptive High-Frequency Fusion模块：自适应权重与高频增强
    修复版 - 支持list输入
    """
    def __init__(self, channels, r=16, alpha=0.1, keep_dim=False):
        super().__init__()
        self.keep_dim = keep_dim
        
        # 通道注意力MLP
        self.ca_mlp = nn.Sequential(
            nn.Linear(channels * 2, channels * 2 // r),
            nn.ReLU(),
            nn.Linear(channels * 2 // r, channels * 2),
            nn.Sigmoid()
        )
        
        # 空间注意力卷积
        self.sa_conv = nn.Conv2d(2, 1, 7, padding=3)
        
        # 高通滤波参数
        self.hpf_d0_alpha = alpha
        
        # 高频偏置融合卷积
        self.bias_conv = nn.Conv2d(channels * 4, channels * 2, 1)
        
        # 可选的降维卷积
        if keep_dim:
            self.reduce_conv = nn.Conv2d(channels * 2, channels, 1)
    
    def forward(self, x):
        """
        前向传播 - 修复版
        
        Args:
            x: 可以是list[feat1, feat2]或单个tensor
        
        Returns:
            融合后的特征
        """
        # ===== 关键修复：处理list输入 =====
        if isinstance(x, list):
            if len(x) != 2:
                raise ValueError(f"AHFF expects 2 inputs, got {len(x)}")
            feat1, feat2 = x[0], x[1]
        else:
            # 如果是单个tensor，尝试按通道分割
            # 这种情况一般不会发生，但作为fallback
            raise ValueError("AHFF requires 2 separate feature inputs")
        
        # 拼接特征
        fused = torch.cat([feat1, feat2], dim=1)  # [B, 2C, H, W]
        
        # === 通道-空间混合自适应权重 ===
        # 通道注意力
        gap = fused.mean(dim=(2, 3))  # GAP [B, 2C]
        wc = self.ca_mlp(gap).unsqueeze(2).unsqueeze(3)  # [B, 2C, 1, 1]
        
        # 空间注意力
        avg_pool = fused.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        max_pool = fused.amax(dim=1, keepdim=True)
        ws = torch.sigmoid(self.sa_conv(torch.cat([avg_pool, max_pool], dim=1)))  # [B, 1, H, W]
        
        # 混合权重
        w = wc * ws.expand_as(fused)  # [B, 2C, H, W]
        
        # 加权特征
        weighted_feat1 = w[:, :feat1.shape[1]] * feat1
        weighted_feat2 = w[:, feat1.shape[1]:] * feat2
        fused_weighted = torch.cat([weighted_feat1, weighted_feat2], dim=1)
        
        # === 高频增强 ===
        # 2D FFT
        fft = torch.fft.fft2(fused_weighted)
        shift = torch.fft.fftshift(fft)
        
        # 生成高通滤波器
        b, c, h, w = fused_weighted.shape
        y = torch.arange(h, device=fused.device).unsqueeze(1) - h / 2
        x_coord = torch.arange(w, device=fused.device) - w / 2
        D = torch.sqrt(y**2 + x_coord**2)
        D = D.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # 高斯高通滤波核
        D0 = self.hpf_d0_alpha * min(h, w)
        H = 1 - torch.exp(-D**2 / (2 * D0**2))
        
        # 应用高通滤波
        shift_hf = shift * H
        ifft = torch.fft.ifft2(torch.fft.ifftshift(shift_hf))
        fused_hf = torch.real(ifft)  # [B, 2C, H, W]
        
        # === 高频偏置融合 ===
        bias = self.bias_conv(torch.cat([fused_weighted, fused_hf], dim=1))
        output = fused_weighted + bias  # [B, 2C, H, W]
        
        # 可选降维
        if self.keep_dim:
            output = self.reduce_conv(output)  # [B, C, H, W]
        
        return output



class ECA(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = self.avg(x).squeeze(-1).transpose(-1, -2)  # B,C → B,1,C
        y = self.conv(y)
        y = self.sig(y).transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)


# ----------------------------
# RepConvExt （YOLO 版）
# ----------------------------
class HybridBottleneck_Lite(nn.Module):
    """轻量版混合瓶颈块 - 选择性频域 ⭐ 推荐"""
    def __init__(self, c1, c2, shortcut=True, kernel_size=7, expansion=0.5):
        super().__init__()
        c_ = int(c2 * expansion)
        c_freq = c_ // 2
        
        self.cv1 = Conv(c1, c_, 1, 1)
        self.gated_spatial = GatedSpatialUnit(c_, kernel_size=kernel_size, conv_ratio=0.5)
        self.frequency = FrequencyUnit(c_freq, c_freq, kernel_size=(3, 3))
        
        self.freq_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_freq, c_freq, 1),
            nn.Sigmoid()
        )
        
        self.cv2 = Conv(c_ + c_freq, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        feat = self.cv1(x)
        spatial_feat = self.gated_spatial(feat)
        freq_input = feat[:, :feat.size(1)//2, :, :]
        freq_feat = self.frequency(freq_input)
        freq_feat = freq_feat * self.freq_gate(freq_feat)
        combined = torch.cat([spatial_feat, freq_feat], dim=1)
        out = self.cv2(combined)
        return x + out if self.add else out
        
class C2f_Hybrid_Lite(nn.Module):
    """
    C2f with HybridBottleneck_Lite - 轻量版 ⭐ 推荐
    
    特点:
    - 使用 HybridBottleneck_Lite 作为基础模块
    - 选择性的频域处理
    - 平衡精度和效率
    - 适合大多数应用场景
    
    性能预期:
    - mAP提升: +1.5~2.0%
    - 参数增加: ~8%
    - 速度: 略慢3-5%
    
    YAML使用:
        - [-1, 3, C2f_Hybrid_Lite, [256, 0.5]]
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        
        self.m = nn.ModuleList(
            HybridBottleneck_Lite(
                self.c, 
                self.c, 
                shortcut=shortcut,
                kernel_size=7,
                expansion=1.0
            ) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SmallObjectEnhancementModule(nn.Module):
    """
    小目标检测增强模块 - 精简版
    
    核心设计理念：
    1. 频域高频增强 - 捕获小目标的边缘细节（最关键）
    2. 局部细节注意力 - 聚焦小目标区域
    3. 轻量级设计 - 最小化参数和计算开销
    
    不包含：
    - 复杂的多尺度结构（增加计算但收益有限）
    - 过深的网络层（容易过拟合）
    - 冗余的注意力机制（通道注意力对小目标帮助不大）
    """
    
    def __init__(self, c1, c2, freq_enhance_ratio=0.5):
        """
        Args:
            c1: 输入通道数
            c2: 输出通道数（通常与c1相同）
            freq_enhance_ratio: 高频增强强度 (0.0-1.0)
        """
        super(SmallObjectEnhancementModule, self).__init__()
        
        self.c1 = c1
        self.c2 = c2
        self.channels = c2  # 保持兼容性
        self.freq_enhance_ratio = freq_enhance_ratio
        
        # ============ 核心1: 频域高频滤波器 ============
        # 这是最关键的部分 - 小目标的边缘信息主要在高频
        self.freq_filter = nn.Parameter(
            torch.ones(1, c2, 1, 1) * 0.5,  # 可学习的频率权重
            requires_grad=True
        )
        
        # 频域特征压缩（减少计算量）
        self.freq_compress = nn.Conv2d(c1 * 2, c2, 1, bias=False)
        self.freq_norm = nn.BatchNorm2d(c2)
        
        # ============ 核心2: 空间细节注意力 ============
        # 专注于捕获小目标的空间位置
        self.spatial_attention = nn.Sequential(
            # 使用最大池化和平均池化捕获显著特征
            # 小目标在这两种池化下表现不同
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        # ============ 核心3: 边缘增强卷积 ============
        # 小目标检测最需要清晰的边缘
        self.edge_conv = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.Conv2d(c2, c2, 1, bias=False),
        )
        
        # 最终融合
        self.fusion = nn.Sequential(
            nn.Conv2d(c2 * 2, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def high_frequency_enhance(self, x):
        """
        高频增强 - 这是小目标检测的关键
        
        原理：
        - 小目标占据像素少，但边缘清晰
        - 边缘信息主要存在于高频分量
        - 通过FFT提取并增强高频部分
        """
        batch, channel, height, width = x.shape
        
        # 2D FFT
        x_fft = torch.fft.rfft2(x, norm='ortho')
        
        # 创建高频掩码（中心是低频，边缘是高频）
        h, w = x_fft.shape[2], x_fft.shape[3]
        
        # 高通滤波器：增强远离中心的频率
        # 使用简单的径向距离作为权重
        center_h, center_w = h // 2, w // 2
        y_coords = torch.arange(h, device=x.device).view(-1, 1).float()
        x_coords = torch.arange(w, device=x.device).view(1, -1).float()
        
        # 计算到中心的归一化距离
        dist = torch.sqrt((y_coords - center_h)**2 + (x_coords / w * h - center_w)**2)
        dist = dist / dist.max()
        
        # 高频掩码：距离越远（高频），权重越大
        high_freq_mask = dist.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # 应用可学习的频率滤波器
        freq_weight = torch.sigmoid(self.freq_filter)  # [1, C, 1, 1]
        high_freq_mask = high_freq_mask * freq_weight * self.freq_enhance_ratio
        
        # 增强高频分量
        x_fft_enhanced = x_fft * (1.0 + high_freq_mask)
        
        # 逆FFT
        x_enhanced = torch.fft.irfft2(x_fft_enhanced, s=(height, width), norm='ortho')
        
        return x_enhanced
    
    def spatial_detail_attention(self, x):
        """
        空间细节注意力
        
        关键：小目标在最大池化和平均池化下的响应不同
        - 平均池化：小目标容易被周围背景稀释
        - 最大池化：能保留小目标的峰值响应
        """
        # 最大池化：保留峰值（对小目标友好）
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        
        # 平均池化：全局上下文
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        
        # 拼接两种池化的互补信息
        pool_concat = torch.cat([max_pool, avg_pool], dim=1)
        
        # 生成空间注意力图
        attention = self.spatial_attention(pool_concat)
        
        return x * attention
    
    def edge_enhance(self, x):
        """
        边缘增强
        
        小目标最显著的特征就是边缘
        使用深度可分离卷积提取边缘，然后加权
        """
        edge_features = self.edge_conv(x)
        
        # 残差连接：保留原始信息
        return x + edge_features * 0.2  # 0.2是经验值，避免过度增强
    
    def forward(self, x):
        """
        前向传播
        
        处理流程：
        1. 频域高频增强（提取边缘）
        2. 空间注意力（定位小目标）
        3. 边缘增强（强化边界）
        4. 特征融合
        """
        identity = x
        
        # 1. 频域高频增强
        freq_enhanced = self.high_frequency_enhance(x)
        
        # 2. 空间细节注意力
        spatial_attended = self.spatial_detail_attention(x)
        
        # 3. 边缘增强
        edge_enhanced = self.edge_enhance(spatial_attended)
        
        # 4. 融合频域和空间域特征
        combined = torch.cat([freq_enhanced, edge_enhanced], dim=1)
        output = self.fusion(combined)
        
        # 残差连接
        output = output + identity
        output = self.relu(output)
        
        return output
