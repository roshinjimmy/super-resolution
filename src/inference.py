"""
Inference Script for Sentinel-2 Data

Applies trained DualEDSR model to Sentinel-2 satellite imagery.
"""

import os
import sys
import argparse
from pathlib import Path

import yaml
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

sys.path.insert(0, str(Path(__file__).parent))

from models.dual_edsr import create_model
from utils.metrics import calculate_niqe, calculate_brisque


def load_model(checkpoint_path: str, config: dict, device: torch.device):
    """Load trained model from checkpoint."""
    model = create_model(config['model'])
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def create_lr_pair_from_hr(
    hr_image: np.ndarray,
    scale: int = 4,
    blur_sigma: float = 1.5
):
    """Create LR1 and LR2 from HR image for consistency check."""
    h, w = hr_image.shape[:2]
    lr_size = (w // scale, h // scale)
    
    hr_pil = Image.fromarray(hr_image)
    
    # LR1: Bicubic
    lr1_pil = hr_pil.resize(lr_size, Image.BICUBIC)
    lr1 = np.array(lr1_pil)
    
    # LR2: Blur + Bicubic
    blurred = cv2.GaussianBlur(hr_image, (0, 0), blur_sigma)
    blurred_pil = Image.fromarray(blurred)
    lr2_pil = blurred_pil.resize(lr_size, Image.BICUBIC)
    lr2 = np.array(lr2_pil)
    
    return lr1, lr2


def prepare_sentinel_input(
    image_path: str,
    method: str = 'synthetic',
    blur_sigma: float = 1.5
):
    """
    Prepare Sentinel-2 image for super-resolution.
    
    Args:
        image_path: Path to Sentinel-2 image
        method: 'synthetic' (downsample to create pair) or 'native' (use as-is)
        blur_sigma: Sigma for Gaussian blur in LR2
    
    Returns:
        Tuple of (lr1, lr2) numpy arrays
    """
    img = np.array(Image.open(image_path).convert('RGB'))
    
    if method == 'synthetic':
        # Downsample 10m to 20m equivalent, then SR back
        # This gives us ground truth for comparison
        lr1, lr2 = create_lr_pair_from_hr(img, scale=2, blur_sigma=blur_sigma)
        return lr1, lr2, img  # Returns HR for reference
    else:
        # True blind: create LR pair from native resolution
        h, w = img.shape[:2]
        
        # LR1: The image itself (or slight modification)
        lr1 = img
        
        # LR2: Blurred version
        lr2 = cv2.GaussianBlur(img, (0, 0), blur_sigma)
        
        return lr1, lr2, None


def run_sentinel_inference(
    model,
    input_dir: str,
    output_dir: str,
    device: torch.device,
    method: str = 'synthetic',
    scale: int = 4
):
    """
    Run inference on Sentinel-2 images.
    
    Args:
        model: Trained DualEDSR model
        input_dir: Directory containing Sentinel-2 images
        output_dir: Output directory for SR results
        device: Torch device
        method: 'synthetic' or 'native'
        scale: Upscaling factor
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    (output_dir / 'sr').mkdir(parents=True, exist_ok=True)
    
    if method == 'synthetic':
        (output_dir / 'comparisons').mkdir(parents=True, exist_ok=True)
    
    image_files = list(input_dir.glob('*.png')) + \
                  list(input_dir.glob('*.jpg')) + \
                  list(input_dir.glob('*.tif'))
    
    results = []
    
    with torch.no_grad():
        for img_path in tqdm(image_files, desc=f'Processing ({method})'):
            lr1, lr2, hr = prepare_sentinel_input(str(img_path), method=method)
            
            # Convert to tensor
            lr1_t = torch.from_numpy(lr1.transpose(2, 0, 1)).float() / 255.0
            lr2_t = torch.from_numpy(lr2.transpose(2, 0, 1)).float() / 255.0
            
            lr1_t = lr1_t.unsqueeze(0).to(device)
            lr2_t = lr2_t.unsqueeze(0).to(device)
            
            # Super-resolution
            sr = model(lr1_t, lr2_t)
            sr = torch.clamp(sr, 0, 1)
            
            # Convert back to numpy
            sr_np = sr[0].cpu().numpy().transpose(1, 2, 0)
            sr_np = (sr_np * 255).astype(np.uint8)
            
            # Save SR result
            save_name = img_path.stem + '_sr.png'
            Image.fromarray(sr_np).save(output_dir / 'sr' / save_name)
            
            # Calculate blind metrics
            niqe = calculate_niqe(sr[0].cpu())
            brisque = calculate_brisque(sr[0].cpu())
            
            result = {
                'filename': img_path.name,
                'niqe': niqe,
                'brisque': brisque
            }
            
            # If synthetic, compare with bicubic and original
            if method == 'synthetic' and hr is not None:
                # Bicubic upscale for comparison
                bicubic = torch.nn.functional.interpolate(
                    lr1_t, scale_factor=2, mode='bicubic', align_corners=False
                )
                bicubic = torch.clamp(bicubic, 0, 1)
                
                # Calculate PSNR against original 10m
                hr_t = torch.from_numpy(hr.transpose(2, 0, 1)).float() / 255.0
                
                from utils.metrics import calculate_psnr, calculate_ssim
                
                # Need to resize SR to match HR if sizes differ
                if sr.shape[2:] != hr_t.shape[1:]:
                    sr_resized = torch.nn.functional.interpolate(
                        sr, size=hr_t.shape[1:], mode='bicubic', align_corners=False
                    )
                else:
                    sr_resized = sr
                
                psnr = calculate_psnr(sr_resized[0].cpu(), hr_t)
                ssim = calculate_ssim(sr_resized[0].cpu(), hr_t)
                
                result['psnr'] = psnr
                result['ssim'] = ssim
            
            results.append(result)
    
    # Summary
    print('\n' + '=' * 50)
    print(f'Sentinel-2 Inference Results ({method} method)')
    print('=' * 50)
    
    niqe_scores = [r['niqe'] for r in results if not np.isnan(r.get('niqe', np.nan))]
    brisque_scores = [r['brisque'] for r in results if not np.isnan(r.get('brisque', np.nan))]
    
    if niqe_scores:
        print(f"Average NIQE: {np.mean(niqe_scores):.4f} (lower is better)")
    if brisque_scores:
        print(f"Average BRISQUE: {np.mean(brisque_scores):.4f} (lower is better)")
    
    if method == 'synthetic':
        psnr_scores = [r['psnr'] for r in results if 'psnr' in r]
        ssim_scores = [r['ssim'] for r in results if 'ssim' in r]
        if psnr_scores:
            print(f"Average PSNR: {np.mean(psnr_scores):.2f} dB")
            print(f"Average SSIM: {np.mean(ssim_scores):.4f}")
    
    # Save results
    import json
    with open(output_dir / 'sentinel_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f'\nResults saved to {output_dir}')
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run DualEDSR on Sentinel-2 data')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing Sentinel-2 images')
    parser.add_argument('--output_dir', type=str, default='outputs/sentinel',
                        help='Output directory')
    parser.add_argument('--method', type=str, default='native',
                        choices=['synthetic', 'native'],
                        help='Evaluation method')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f'Using device: {device}')
    
    # Load model
    model = load_model(args.checkpoint, config, device)
    
    # Run inference
    run_sentinel_inference(
        model,
        args.input_dir,
        args.output_dir,
        device,
        method=args.method
    )


if __name__ == '__main__':
    main()
