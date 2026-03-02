"""
Dual-Branch Modified EDSR for Dual Low-Resolution Super-Resolution

Architecture:
    LR1 → ShallowConv → feat1 ─┐
                                ├→ Concat → FusionConv → ResBlocks → Upsampler → ReconConv → HR
    LR2 → ShallowConv → feat2 ─┘
"""

import torch
import torch.nn as nn
from typing import Tuple

from .blocks import ResidualBlock, Upsampler, MeanShift


class DualEDSR(nn.Module):
    """
    Dual-Branch Enhanced Deep Super-Resolution Network.
    
    Takes two low-resolution inputs (LR1 and LR2) with different degradations
    and fuses their features to produce a high-resolution output.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_residual_blocks: int = 16,
        scale: int = 4,
        res_scale: float = 1.0,
        use_mean_shift: bool = True
    ):
        """
        Args:
            in_channels: Number of input channels (3 for RGB)
            out_channels: Number of output channels (3 for RGB)
            num_features: Number of feature channels in the network
            num_residual_blocks: Number of residual blocks in the body
            scale: Upsampling scale factor
            res_scale: Residual scaling factor
            use_mean_shift: Whether to use mean shift normalization
        """
        super().__init__()
        
        self.scale = scale
        self.use_mean_shift = use_mean_shift
        
        # Mean shift layers
        if use_mean_shift:
            self.sub_mean = MeanShift(sign=-1)
            self.add_mean = MeanShift(sign=1)
        
        # Shallow feature extraction branches (NO ReLU - match SingleEDSR)
        self.shallow_conv_lr1 = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.shallow_conv_lr2 = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        
        # Feature fusion layer (1x1 conv to reduce 128 -> 64, NO ReLU)
        self.fusion_conv = nn.Conv2d(num_features * 2, num_features, kernel_size=1, padding=0)
        
        # Residual blocks (EDSR body)
        body = []
        for _ in range(num_residual_blocks):
            body.append(ResidualBlock(num_features, res_scale=res_scale))
        self.body = nn.Sequential(*body)
        
        # Post-residual convolution
        self.post_conv = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        
        # Upsampling
        self.upsampler = Upsampler(scale, num_features)
        
        # Final reconstruction convolution
        self.reconstruction = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)
        
        # Note: Using PyTorch default initialization (same as SingleEDSR)
    
    # Note: Removed explicit weight initialization to match SingleEDSR behavior
    # PyTorch default init works well for this architecture
    
    def forward(
        self,
        lr1: torch.Tensor,
        lr2: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            lr1: First low-resolution input (bicubic downsampled)
            lr2: Second low-resolution input (blur + bicubic downsampled)
        
        Returns:
            High-resolution output
        """
        # Mean shift
        if self.use_mean_shift:
            lr1 = self.sub_mean(lr1)
            lr2 = self.sub_mean(lr2)
        
        # Shallow feature extraction
        feat1 = self.shallow_conv_lr1(lr1)
        feat2 = self.shallow_conv_lr2(lr2)
        
        # Feature fusion
        fused = torch.cat([feat1, feat2], dim=1)  # (B, 2*F, H, W)
        fused = self.fusion_conv(fused)  # (B, F, H, W)
        
        # Residual blocks with global skip connection
        residual = fused
        body_out = self.body(fused)
        body_out = self.post_conv(body_out)
        body_out = body_out + residual
        
        # Upsampling
        upsampled = self.upsampler(body_out)
        
        # Reconstruction
        hr = self.reconstruction(upsampled)
        
        # Add mean back
        if self.use_mean_shift:
            hr = self.add_mean(hr)
        
        return hr
    
    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SingleEDSR(nn.Module):
    """
    Single-input EDSR for baseline comparison.
    
    Standard EDSR architecture without dual-branch fusion.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_residual_blocks: int = 16,
        scale: int = 4,
        res_scale: float = 1.0,
        use_mean_shift: bool = True
    ):
        super().__init__()
        
        self.scale = scale
        self.use_mean_shift = use_mean_shift
        
        if use_mean_shift:
            self.sub_mean = MeanShift(sign=-1)
            self.add_mean = MeanShift(sign=1)
        
        # Shallow feature extraction
        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        
        # Residual blocks
        body = []
        for _ in range(num_residual_blocks):
            body.append(ResidualBlock(num_features, res_scale=res_scale))
        self.body = nn.Sequential(*body)
        
        self.post_conv = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        
        # Upsampling
        self.upsampler = Upsampler(scale, num_features)
        
        # Reconstruction
        self.tail = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_mean_shift:
            x = self.sub_mean(x)
        
        head = self.head(x)
        body = self.body(head)
        body = self.post_conv(body)
        body = body + head
        
        up = self.upsampler(body)
        out = self.tail(up)
        
        if self.use_mean_shift:
            out = self.add_mean(out)
        
        return out


def create_model(config: dict) -> DualEDSR:
    """
    Create DualEDSR model from configuration.
    
    Args:
        config: Model configuration dictionary
    
    Returns:
        DualEDSR model instance
    """
    return DualEDSR(
        in_channels=config.get('in_channels', 3),
        out_channels=config.get('out_channels', 3),
        num_features=config.get('num_features', 64),
        num_residual_blocks=config.get('num_residual_blocks', 16),
        scale=config.get('scale', 4),
        res_scale=config.get('res_scale', 0.1),
        use_mean_shift=config.get('use_mean_shift', True)
    )


if __name__ == "__main__":
    # Test model
    model = DualEDSR(num_features=64, num_residual_blocks=16, scale=4)
    print(f"DualEDSR created with {model.get_num_params():,} parameters")
    
    # Test forward pass
    lr1 = torch.randn(1, 3, 64, 64)
    lr2 = torch.randn(1, 3, 64, 64)
    hr = model(lr1, lr2)
    print(f"Input shapes: LR1={lr1.shape}, LR2={lr2.shape}")
    print(f"Output shape: HR={hr.shape}")
    
    # Test single EDSR baseline
    baseline = SingleEDSR(num_features=64, num_residual_blocks=16, scale=4)
    hr_baseline = baseline(lr1)
    print(f"\nSingleEDSR baseline output: {hr_baseline.shape}")
