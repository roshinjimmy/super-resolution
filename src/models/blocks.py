"""
Building Blocks for Dual-EDSR Model

Includes:
- ResidualBlock: Standard residual block with Conv-ReLU-Conv
- Upsampler: PixelShuffle-based upsampling module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ResidualBlock(nn.Module):
    """
    Standard residual block for EDSR.
    
    Structure: Conv -> ReLU -> Conv + skip connection
    No batch normalization as per EDSR paper.
    """
    
    def __init__(
        self,
        num_features: int = 64,
        kernel_size: int = 3,
        res_scale: float = 1.0
    ):
        """
        Args:
            num_features: Number of feature channels
            kernel_size: Convolution kernel size
            res_scale: Residual scaling factor
        """
        super().__init__()
        
        self.res_scale = res_scale
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv2d(
            num_features, num_features,
            kernel_size=kernel_size, padding=padding
        )
        self.conv2 = nn.Conv2d(
            num_features, num_features,
            kernel_size=kernel_size, padding=padding
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale
        
        return out + residual


class Upsampler(nn.Module):
    """
    PixelShuffle-based upsampling module.
    
    Upsamples by a factor of 2^n using n PixelShuffle layers.
    """
    
    def __init__(
        self,
        scale: int,
        num_features: int,
        activation: Optional[str] = None
    ):
        """
        Args:
            scale: Upsampling scale factor (must be power of 2 or 3)
            num_features: Number of input feature channels
            activation: Optional activation ('relu', 'prelu', or None)
        """
        super().__init__()
        
        layers = []
        
        if scale & (scale - 1) == 0:  # Power of 2
            for _ in range(int(torch.log2(torch.tensor(scale)).item())):
                layers.append(nn.Conv2d(
                    num_features, num_features * 4,
                    kernel_size=3, padding=1
                ))
                layers.append(nn.PixelShuffle(2))
                if activation == 'relu':
                    layers.append(nn.ReLU(inplace=True))
                elif activation == 'prelu':
                    layers.append(nn.PReLU())
        elif scale == 3:
            layers.append(nn.Conv2d(
                num_features, num_features * 9,
                kernel_size=3, padding=1
            ))
            layers.append(nn.PixelShuffle(3))
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'prelu':
                layers.append(nn.PReLU())
        else:
            raise ValueError(f"Unsupported scale factor: {scale}")
        
        self.upsampler = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsampler(x)


class MeanShift(nn.Module):
    """
    Mean shift layer for RGB normalization.
    
    Shifts input by mean RGB values (useful for pretrained models).
    """
    
    def __init__(
        self,
        rgb_mean: tuple = (0.4488, 0.4371, 0.4040),
        rgb_std: tuple = (1.0, 1.0, 1.0),
        sign: int = -1
    ):
        """
        Args:
            rgb_mean: RGB channel means
            rgb_std: RGB channel standard deviations
            sign: -1 for subtraction, +1 for addition
        """
        super().__init__()
        
        std = torch.Tensor(rgb_std)
        self.register_buffer(
            'weight',
            torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        )
        self.register_buffer(
            'bias',
            sign * torch.Tensor(rgb_mean) / std
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, self.bias)


class ChannelAttention(nn.Module):
    """
    Channel attention module (squeeze-and-excitation).
    
    Optional enhancement for EDSR.
    """
    
    def __init__(self, num_features: int, reduction: int = 16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_features, num_features // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_features // reduction, num_features, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
