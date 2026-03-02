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
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
    blur_sigma: float = 1.5,
    scale: int = 4,
):
    """
    Prepare Sentinel-2 image for super-resolution.

    Args:
        image_path: Path to Sentinel-2 image
        method: 'synthetic' (downsample to create pair) or 'native' (use as-is)
        blur_sigma: Sigma for Gaussian blur in LR2 (must match training value)
        scale: Upscaling factor (must match trained model)

    Returns:
        Tuple of (lr1, lr2, hr_or_None) numpy arrays
    """
    img = np.array(Image.open(image_path).convert('RGB'))

    if method == 'synthetic':
        # Downsample HR by ×scale (matching training degradation), then SR back
        lr1, lr2 = create_lr_pair_from_hr(img, scale=scale, blur_sigma=blur_sigma)
        return lr1, lr2, img  # Returns HR for reference
    else:
        # True blind: create LR pair from native resolution
        # LR1: the image itself; LR2: blurred version (simulates sensor blur)
        lr1 = img
        lr2 = cv2.GaussianBlur(img, (0, 0), blur_sigma)
        return lr1, lr2, None


def tile_inference(
    model,
    lr1_np: np.ndarray,
    lr2_np: np.ndarray,
    device: torch.device,
    tile_size: int = 64,
    overlap: int = 16,
    scale: int = 4,
) -> np.ndarray:
    """
    Run model on large images via overlapping patch tiling.

    Splits LR inputs into overlapping tiles of `tile_size`, runs the model on
    each, and blends the HR outputs with a linear ramp weight to suppress
    seam artefacts. Matches the paper: "All images are cropped into
    overlapping patches to facilitate efficient inference on large scenes."

    Args:
        model:     Trained DualEDSR (eval mode).
        lr1_np:    LR1 image as uint8 HxWxC numpy array.
        lr2_np:    LR2 image as uint8 HxWxC numpy array.
        device:    Torch device.
        tile_size: LR tile size (default 64 matches training input size).
        overlap:   Overlap in LR pixels between adjacent tiles.
        scale:     SR scale factor.

    Returns:
        SR image as uint8 HxWxC numpy array at scale×LR resolution.
    """
    h, w = lr1_np.shape[:2]
    step = tile_size - overlap
    out_h, out_w = h * scale, w * scale

    sr_sum    = np.zeros((out_h, out_w, 3), dtype=np.float32)
    weight_sum = np.zeros((out_h, out_w, 1), dtype=np.float32)

    # Build a 1-D linear ramp weight that tapers at tile edges
    ramp = np.ones(tile_size, dtype=np.float32)
    ramp[:overlap]  = np.linspace(0, 1, overlap)
    ramp[-overlap:] = np.linspace(1, 0, overlap)
    weight_2d = np.outer(ramp, ramp)[..., None]  # (T, T, 1)
    weight_hr = np.kron(weight_2d,
                        np.ones((scale, scale, 1), dtype=np.float32))  # (T*s, T*s, 1)

    ys = list(range(0, h - tile_size, step)) + [h - tile_size]
    xs = list(range(0, w - tile_size, step)) + [w - tile_size]

    with torch.no_grad():
        for y in ys:
            for x in xs:
                lr1_tile = lr1_np[y:y+tile_size, x:x+tile_size]
                lr2_tile = lr2_np[y:y+tile_size, x:x+tile_size]

                t1 = torch.from_numpy(
                    lr1_tile.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(device)
                t2 = torch.from_numpy(
                    lr2_tile.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(device)

                sr_tile = model(t1, t2).clamp(0, 1)
                sr_tile_np = sr_tile[0].cpu().numpy().transpose(1, 2, 0)  # (H*s, W*s, 3)

                oy, ox = y * scale, x * scale
                th, tw = sr_tile_np.shape[:2]
                w2d = weight_hr[:th, :tw]
                sr_sum[oy:oy+th, ox:ox+tw]    += sr_tile_np * w2d
                weight_sum[oy:oy+th, ox:ox+tw] += w2d

    sr = sr_sum / np.maximum(weight_sum, 1e-8)
    return (np.clip(sr, 0, 1) * 255).astype(np.uint8)


def run_sentinel_inference(
    model,
    input_dir: str,
    output_dir: str,
    device: torch.device,
    method: str = 'synthetic',
    scale: int = 4,
    blur_sigma: float = 1.5,
    tile_size: int = 64,
    overlap: int = 16,
):
    """
    Run inference on Sentinel-2 images using overlapping patch tiling.

    Args:
        model:      Trained DualEDSR model
        input_dir:  Directory containing Sentinel-2 images
        output_dir: Output directory for SR results
        device:     Torch device
        method:     'synthetic' or 'native'
        scale:      Upscaling factor (must match trained model)
        blur_sigma: Gaussian sigma for LR2 (must match training value)
        tile_size:  LR tile size for patch inference (default 64)
        overlap:    LR overlap between tiles (default 16)
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

    for img_path in tqdm(image_files, desc=f'Processing ({method})'):
        lr1, lr2, hr = prepare_sentinel_input(
            str(img_path), method=method, blur_sigma=blur_sigma, scale=scale
        )

        # Use overlapping patch tiling (paper: "cropped into overlapping patches")
        sr_np = tile_inference(
            model, lr1, lr2, device,
            tile_size=tile_size, overlap=overlap, scale=scale
        )

        # Save SR result
        save_name = img_path.stem + '_sr.png'
        Image.fromarray(sr_np).save(output_dir / 'sr' / save_name)

        # Blind metrics on SR
        sr_t = torch.from_numpy(sr_np.transpose(2, 0, 1)).float().div(255.0)
        niqe    = calculate_niqe(sr_t)
        brisque = calculate_brisque(sr_t)

        result = {'filename': img_path.name, 'niqe': niqe, 'brisque': brisque}

        if method == 'synthetic' and hr is not None:
            hr_t  = torch.from_numpy(hr.transpose(2, 0, 1)).float().div(255.0)
            # Bicubic baseline at same scale for comparison
            lr1_t = torch.from_numpy(lr1.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
            bicubic = torch.nn.functional.interpolate(
                lr1_t, scale_factor=scale, mode='bicubic', align_corners=False
            ).squeeze(0).clamp(0, 1)

            from utils.metrics import calculate_psnr, calculate_ssim
            result['psnr']      = calculate_psnr(sr_t, hr_t)
            result['ssim']      = calculate_ssim(sr_t, hr_t)
            result['psnr_bic']  = calculate_psnr(bicubic, hr_t)
            result['ssim_bic']  = calculate_ssim(bicubic, hr_t)

        results.append(result)
    
    # Summary
    print('\n' + '=' * 50)
    print(f'Sentinel-2 Inference Results ({method} method)')
    print('=' * 50)

    niqe_scores    = [r['niqe']    for r in results if not np.isnan(r.get('niqe',    np.nan))]
    brisque_scores = [r['brisque'] for r in results if not np.isnan(r.get('brisque', np.nan))]

    if niqe_scores:
        print(f"Average NIQE:    {np.mean(niqe_scores):.4f} (lower is better)")
    if brisque_scores:
        print(f"Average BRISQUE: {np.mean(brisque_scores):.4f} (lower is better)")

    if method == 'synthetic':
        psnr_scores = [r['psnr']     for r in results if 'psnr' in r]
        ssim_scores = [r['ssim']     for r in results if 'ssim' in r]
        psnr_bic    = [r['psnr_bic'] for r in results if 'psnr_bic' in r]
        ssim_bic    = [r['ssim_bic'] for r in results if 'ssim_bic' in r]
        if psnr_scores:
            print(f"Average PSNR (Dual-EDSR): {np.mean(psnr_scores):.2f} dB")
            print(f"Average SSIM (Dual-EDSR): {np.mean(ssim_scores):.4f}")
        if psnr_bic:
            print(f"Average PSNR (Bicubic):   {np.mean(psnr_bic):.2f} dB")
            print(f"Average SSIM (Bicubic):   {np.mean(ssim_bic):.4f}")
    
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
    parser.add_argument('--tile_size', type=int, default=64,
                        help='LR tile size for patch inference (default 64, matches training input)')
    parser.add_argument('--overlap', type=int, default=16,
                        help='LR overlap between tiles in pixels (default 16)')
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
        method=args.method,
        scale=config['model'].get('scale', 4),
        blur_sigma=config['data'].get('blur_sigma', 1.5),
        tile_size=args.tile_size,
        overlap=args.overlap,
    )


if __name__ == '__main__':
    main()
