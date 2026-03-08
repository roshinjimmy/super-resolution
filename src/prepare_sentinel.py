#!/usr/bin/env python3
"""
Prepare two real Sentinel-2 images for blind dual-SR evaluation.

Crops both images to the same size and tiles them into patches.
No blur or preprocessing is applied — the natural spectral/atmospheric
difference between the two acquisition dates serves as the dual input.

Usage:
    python prepare_sentinel.py \
        --img1 /path/to/date1.png \
        --img2 /path/to/date2.png \
        --output_dir sentinel2

Then evaluate:
    python evaluate.py \
        --config config/config.yaml \
        --checkpoint checkpoints/dual/dual_edsr_best.pth \
        --blind \
        --lr1_dir sentinel2/lr1 \
        --lr2_dir sentinel2/lr2 \
        --output_dir outputs/sentinel2_blind
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str) -> np.ndarray:
    pil = Image.open(path)
    if pil.mode == 'RGBA':
        arr = np.array(pil)
        alpha = arr[:, :, 3]
        if (alpha == 255).mean() < 0.05:
            print(f"ERROR: '{path}' is nearly fully transparent — wrong date or swath edge.")
            sys.exit(1)
        rgb = arr[:, :, :3].copy()
        rgb[alpha < 128] = 0
        return rgb.astype(np.uint8)
    return np.array(pil.convert('RGB'))


def crop_to_same_size(img1: np.ndarray, img2: np.ndarray) -> tuple:
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    return img1[:h, :w], img2[:h, :w]


def tile(img1: np.ndarray, img2: np.ndarray, tile_size: int, max_tiles: int) -> list:
    h, w = img1.shape[:2]
    pairs = []
    for y in range(0, h - tile_size + 1, tile_size):
        for x in range(0, w - tile_size + 1, tile_size):
            t1 = img1[y:y + tile_size, x:x + tile_size]
            t2 = img2[y:y + tile_size, x:x + tile_size]
            if t1.std() >= 8.0:   # skip blank/uniform patches
                pairs.append((t1, t2))
    if len(pairs) > max_tiles:
        idxs = np.linspace(0, len(pairs) - 1, max_tiles, dtype=int)
        pairs = [pairs[i] for i in idxs]
    return pairs


def main():
    parser = argparse.ArgumentParser(description='Tile two Sentinel-2 images into LR pairs')
    parser.add_argument('--img1', required=True, help='Date 1 PNG/JPG/TIF')
    parser.add_argument('--img2', required=True, help='Date 2 PNG/JPG/TIF (same area)')
    parser.add_argument('--output_dir', default='sentinel2', help='Output root directory')
    parser.add_argument('--tile_size', type=int, default=128, help='Tile size in px (default: 128)')
    parser.add_argument('--max_tiles', type=int, default=100, help='Max tile pairs (default: 100)')
    args = parser.parse_args()

    out = Path(args.output_dir)
    lr1_dir = out / 'lr1'
    lr2_dir = out / 'lr2'
    lr1_dir.mkdir(parents=True, exist_ok=True)
    lr2_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.img1} ...")
    img1 = load_image(args.img1)
    print(f"Loading {args.img2} ...")
    img2 = load_image(args.img2)
    print(f"  img1: {img1.shape[1]}x{img1.shape[0]}  img2: {img2.shape[1]}x{img2.shape[0]}")

    img1, img2 = crop_to_same_size(img1, img2)
    print(f"  Cropped to: {img1.shape[1]}x{img1.shape[0]}")

    if img1.shape[0] < args.tile_size or img1.shape[1] < args.tile_size:
        print(f"ERROR: Image is smaller than tile_size ({args.tile_size}). Use a larger image.")
        sys.exit(1)

    pairs = tile(img1, img2, args.tile_size, args.max_tiles)
    print(f"  Extracted {len(pairs)} tile pairs")

    for i, (t1, t2) in enumerate(pairs):
        name = f"tile_{i:04d}.png"
        Image.fromarray(t1).save(lr1_dir / name)
        Image.fromarray(t2).save(lr2_dir / name)

    print(f"\nDone. {len(pairs)} pairs saved to {out}/lr1  and  {out}/lr2")
    print(f"\nNext -- run blind evaluation:")
    print(f"  python evaluate.py \\")
    print(f"    --config config/config.yaml \\")
    print(f"    --checkpoint checkpoints/dual/dual_edsr_best.pth \\")
    print(f"    --blind \\")
    print(f"    --lr1_dir {lr1_dir} \\")
    print(f"    --lr2_dir {lr2_dir} \\")
    print(f"    --output_dir outputs/sentinel2_blind")


if __name__ == '__main__':
    main()
