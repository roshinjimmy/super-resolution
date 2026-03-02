"""
Evaluation Metrics for Dual-SR

Includes:
- PSNR: Peak Signal-to-Noise Ratio
- SSIM: Structural Similarity Index
- NIQE: Natural Image Quality Evaluator (blind metric)
- BRISQUE: Blind/Referenceless Image Spatial Quality Evaluator
"""

import torch
import numpy as np
from typing import Union, Optional
import warnings


def calculate_psnr(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    max_val: float = 1.0,
    data_range: Optional[float] = None
) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio.
    
    Args:
        pred: Predicted image
        target: Ground truth image
        max_val: Maximum pixel value
        data_range: Dynamic range of the images
    
    Returns:
        PSNR value in dB
    """
    if data_range is None:
        data_range = max_val
    
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    
    psnr = 20 * np.log10(data_range) - 10 * np.log10(mse)
    return float(psnr)


def calculate_ssim(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    data_range: float = 1.0,
    channel_axis: Optional[int] = None
) -> float:
    """
    Calculate Structural Similarity Index.
    
    Args:
        pred: Predicted image
        target: Ground truth image
        data_range: Dynamic range of the images
        channel_axis: Axis for color channels
    
    Returns:
        SSIM value (0 to 1)
    """
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        warnings.warn("scikit-image not installed. Using simplified SSIM.")
        return _simplified_ssim(pred, target, data_range)
    
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # Handle batch dimension
    if pred.ndim == 4:
        # (B, C, H, W) -> compute per image and average
        ssim_vals = []
        for i in range(pred.shape[0]):
            img_pred = pred[i].transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
            img_target = target[i].transpose(1, 2, 0)
            ssim_val = ssim(
                img_pred, img_target,
                data_range=data_range,
                channel_axis=2
            )
            ssim_vals.append(ssim_val)
        return float(np.mean(ssim_vals))
    elif pred.ndim == 3:
        # (C, H, W) -> (H, W, C)
        pred = pred.transpose(1, 2, 0)
        target = target.transpose(1, 2, 0)
        return float(ssim(pred, target, data_range=data_range, channel_axis=2))
    else:
        return float(ssim(pred, target, data_range=data_range))


def _simplified_ssim(
    pred: np.ndarray,
    target: np.ndarray,
    data_range: float = 1.0
) -> float:
    """Simplified SSIM for fallback."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    mu_pred = np.mean(pred)
    mu_target = np.mean(target)
    
    sigma_pred = np.std(pred)
    sigma_target = np.std(target)
    sigma_pred_target = np.mean((pred - mu_pred) * (target - mu_target))
    
    ssim = ((2 * mu_pred * mu_target + C1) * (2 * sigma_pred_target + C2)) / \
           ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred ** 2 + sigma_target ** 2 + C2))
    
    return float(ssim)


def calculate_niqe(
    image: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Calculate Natural Image Quality Evaluator score.
    
    Lower NIQE indicates better perceptual quality.
    
    Args:
        image: Input image (no reference needed)
    
    Returns:
        NIQE score
    """
    try:
        import pyiqa
        niqe_metric = pyiqa.create_metric('niqe', device='cpu')
        
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        
        if image.ndim == 3:
            image = image.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            score = niqe_metric(image)
        
        return float(score.item())
    except ImportError:
        warnings.warn("pyiqa not installed. NIQE calculation unavailable.")
        return float('nan')
    except Exception as e:
        warnings.warn(f"NIQE calculation failed: {e}")
        return float('nan')


def calculate_brisque(
    image: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Calculate BRISQUE score.
    
    Lower BRISQUE indicates better perceptual quality.
    
    Args:
        image: Input image (no reference needed)
    
    Returns:
        BRISQUE score
    """
    try:
        import pyiqa
        brisque_metric = pyiqa.create_metric('brisque', device='cpu')
        
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        
        if image.ndim == 3:
            image = image.unsqueeze(0)
        
        with torch.no_grad():
            score = brisque_metric(image)
        
        return float(score.item())
    except ImportError:
        warnings.warn("pyiqa not installed. BRISQUE calculation unavailable.")
        return float('nan')
    except Exception as e:
        warnings.warn(f"BRISQUE calculation failed: {e}")
        return float('nan')


class MetricCalculator:
    """Utility class for calculating multiple metrics."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self._niqe = None
        self._brisque = None
    
    def psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        return calculate_psnr(pred, target)
    
    def ssim(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        return calculate_ssim(pred, target)
    
    def niqe(self, image: torch.Tensor) -> float:
        return calculate_niqe(image)
    
    def brisque(self, image: torch.Tensor) -> float:
        return calculate_brisque(image)
    
    def calculate_all(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        include_blind: bool = False
    ) -> dict:
        """
        Calculate all available metrics.
        
        Args:
            pred: Predicted image
            target: Ground truth image (optional)
            include_blind: Whether to include NIQE/BRISQUE
        
        Returns:
            Dictionary of metric values
        """
        results = {}
        
        if target is not None:
            results['psnr'] = self.psnr(pred, target)
            results['ssim'] = self.ssim(pred, target)
        
        if include_blind:
            results['niqe'] = self.niqe(pred)
            results['brisque'] = self.brisque(pred)
        
        return results
