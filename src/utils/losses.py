"""
Loss Functions for Dual-SR

Includes:
- L1Loss: Primary pixel-wise loss
- SSIMLoss: Structural similarity loss
- CombinedLoss: Weighted combination of L1 and SSIM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class L1Loss(nn.Module):
    """L1 (Mean Absolute Error) Loss."""
    
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(pred, target)


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index Loss.
    
    SSIM measures structural similarity between two images.
    Loss = 1 - SSIM (to minimize)
    """
    
    def __init__(
        self,
        window_size: int = 11,
        channel: int = 3,
        size_average: bool = True
    ):
        """
        Args:
            window_size: Size of the Gaussian window
            channel: Number of channels
            size_average: Whether to average over batch
        """
        super().__init__()
        
        self.window_size = window_size
        self.channel = channel
        self.size_average = size_average
        
        # Create Gaussian window
        self.register_buffer('window', self._create_window(window_size, channel))
    
    def _gaussian(self, window_size: int, sigma: float = 1.5) -> torch.Tensor:
        """Create 1D Gaussian kernel."""
        coords = torch.arange(window_size, dtype=torch.float32)
        coords -= window_size // 2
        
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        
        return g
    
    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        """Create 2D Gaussian window."""
        _1D_window = self._gaussian(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t())
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def _ssim(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        window: torch.Tensor,
        channel: int,
        size_average: bool = True
    ) -> torch.Tensor:
        """Calculate SSIM."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(
            img1 * img1, window, padding=self.window_size // 2, groups=channel
        ) - mu1_sq
        sigma2_sq = F.conv2d(
            img2 * img2, window, padding=self.window_size // 2, groups=channel
        ) - mu2_sq
        sigma12 = F.conv2d(
            img1 * img2, window, padding=self.window_size // 2, groups=channel
        ) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate SSIM loss.
        
        Returns:
            1 - SSIM (loss to minimize)
        """
        channel = pred.size(1)
        
        if channel == self.channel and self.window.data.dtype == pred.dtype:
            window = self.window
        else:
            window = self._create_window(self.window_size, channel).to(pred.device).type(pred.dtype)
        
        ssim = self._ssim(pred, target, window, channel, self.size_average)
        return 1 - ssim


class CombinedLoss(nn.Module):
    """
    Combined L1 and SSIM loss.
    
    Loss = l1_weight * L1 + ssim_weight * (1 - SSIM)
    """
    
    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.1,
        use_ssim: bool = True
    ):
        """
        Args:
            l1_weight: Weight for L1 loss
            ssim_weight: Weight for SSIM loss
            use_ssim: Whether to include SSIM loss
        """
        super().__init__()
        
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.use_ssim = use_ssim
        
        self.l1_loss = L1Loss()
        if use_ssim:
            self.ssim_loss = SSIMLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate combined loss."""
        loss = self.l1_weight * self.l1_loss(pred, target)
        
        if self.use_ssim:
            loss = loss + self.ssim_weight * self.ssim_loss(pred, target)
        
        return loss


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (differentiable L1).
    
    L = sqrt((pred - target)^2 + epsilon^2)
    """
    
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff ** 2 + self.epsilon ** 2)
        return loss.mean()
