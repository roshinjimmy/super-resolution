"""
Inference Script for Sentinel-2 Data

Runs DualEDSR on two real paired Sentinel-2 acquisitions of the same scene.
Produces a full-resolution SR image and a bicubic baseline for comparison.
"""

import argparse
import sys
from pathlib import Path

import yaml
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from models.dual_edsr import create_model


def load_model(checkpoint_path: str, config: dict, device: torch.device):
    model = create_model(config['model'])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def tile_inference(
    model,
    lr1_np: np.ndarray,
    lr2_np: np.ndarray,
    device: torch.device,
    tile_size: int = 64,
    overlap: int = 16,
    scale: int = 4,
) -> np.ndarray:
    h, w = lr1_np.shape[:2]
    step = tile_size - overlap
    out_h, out_w = h * scale, w * scale

    sr_sum     = np.zeros((out_h, out_w, 3), dtype=np.float32)
    weight_sum = np.zeros((out_h, out_w, 1), dtype=np.float32)

    ramp = np.ones(tile_size, dtype=np.float32)
    ramp[:overlap]  = np.linspace(0, 1, overlap)
    ramp[-overlap:] = np.linspace(1, 0, overlap)
    weight_2d = np.outer(ramp, ramp)[..., None]
    weight_hr = np.kron(weight_2d, np.ones((scale, scale, 1), dtype=np.float32))

    ys = list(range(0, h - tile_size, step)) + [h - tile_size]
    xs = list(range(0, w - tile_size, step)) + [w - tile_size]

    with torch.no_grad():
        for y in ys:
            for x in xs:
                t1 = torch.from_numpy(lr1_np[y:y+tile_size, x:x+tile_size]
                     .transpose(2,0,1)).float().div(255.0).unsqueeze(0).to(device)
                t2 = torch.from_numpy(lr2_np[y:y+tile_size, x:x+tile_size]
                     .transpose(2,0,1)).float().div(255.0).unsqueeze(0).to(device)

                sr_tile = model(t1, t2).clamp(0, 1)
                sr_np = sr_tile[0].cpu().numpy().transpose(1, 2, 0)

                oy, ox = y * scale, x * scale
                th, tw = sr_np.shape[:2]
                w2d = weight_hr[:th, :tw]
                sr_sum[oy:oy+th, ox:ox+tw]     += sr_np * w2d
                weight_sum[oy:oy+th, ox:ox+tw]  += w2d

    sr = sr_sum / np.maximum(weight_sum, 1e-8)
    return (np.clip(sr, 0, 1) * 255).astype(np.uint8)


def strip_black_borders(img: np.ndarray, threshold: int = 10) -> tuple:
    """Return the tight bounding box (y0,y1,x0,x1) of non-black content."""
    mask = img.max(axis=2) > threshold          # True where any channel > threshold
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return 0, img.shape[0], 0, img.shape[1]  # nothing to crop
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


def main():
    parser = argparse.ArgumentParser(description='Run DualEDSR on two paired Sentinel-2 images')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--config', default='config/config.yaml', help='Path to config file')
    parser.add_argument('--lr1', required=True, help='First image (date 1)')
    parser.add_argument('--lr2', required=True, help='Second image (date 2, same area)')
    parser.add_argument('--output_dir', default='outputs/sentinel2_visual', help='Output directory')
    parser.add_argument('--tile_size', type=int, default=64, help='LR tile size (default: 64)')
    parser.add_argument('--overlap', type=int, default=16, help='Tile overlap in px (default: 16)')
    parser.add_argument('--device', default='auto', help='cuda / cpu / auto')
    parser.add_argument('--downsample', action='store_true',
                        help='Downsample inputs x4 before SR (required for browser/rendered images)')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if args.device == 'auto' else torch.device(args.device)
    print(f'Using device: {device}')

    model = load_model(args.checkpoint, config, device)
    scale = config['model'].get('scale', 4)

    lr1 = np.array(Image.open(args.lr1).convert('RGB'))
    lr2 = np.array(Image.open(args.lr2).convert('RGB'))

    # Crop to same size
    h = min(lr1.shape[0], lr2.shape[0])
    w = min(lr1.shape[1], lr2.shape[1])
    lr1, lr2 = lr1[:h, :w], lr2[:h, :w]

    # Strip black borders from swath edges (use lr1 to find valid region)
    y0, y1, x0, x1 = strip_black_borders(lr1)
    lr1, lr2 = lr1[y0:y1, x0:x1], lr2[y0:y1, x0:x1]
    h, w = lr1.shape[:2]

    # Downsample to create proper LR inputs matching training distribution
    if args.downsample:
        lr_h, lr_w = h // scale, w // scale
        lr1 = np.array(Image.fromarray(lr1).resize((lr_w, lr_h), Image.BICUBIC))
        lr2 = np.array(Image.fromarray(lr2).resize((lr_w, lr_h), Image.BICUBIC))
        print(f'Downsampled to LR: {lr_w}x{lr_h}  ->  SR output: {lr_w*scale}x{lr_h*scale}')
    else:
        print(f'After border crop: {w}x{h}  ->  SR output: {w*scale}x{h*scale}')

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sr_np = tile_inference(model, lr1, lr2, device,
                           tile_size=args.tile_size, overlap=args.overlap, scale=scale)
    Image.fromarray(sr_np).save(out / 'sr_output.png')
    print(f'SR image saved -> {out}/sr_output.png')

    bicubic = Image.fromarray(lr1).resize((w * scale, h * scale), Image.BICUBIC)
    bicubic.save(out / 'bicubic_output.png')
    print(f'Bicubic saved  -> {out}/bicubic_output.png')


if __name__ == '__main__':
    main()
