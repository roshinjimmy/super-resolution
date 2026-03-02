"""
Data Preparation Script for Dual-SR Project

Prepares the NWPU-RESISC45 Remote Sensing dataset and creates dual low-resolution pairs:
  LR1: Bicubic downsampling  (scale=4)
  LR2: Gaussian blur + bicubic downsampling  (scale=4, sigma=1.5)

Dataset: NWPU-RESISC45
  - 31,500 images, 45 scene categories x 700 images each
  - 256x256 pixels, RGB JPEG
  - Reference: Cheng et al., IEEE TGRS 2017

Usage
-----
Option A - Kaggle CLI (recommended):
    pip install kaggle
    kaggle datasets download -d happyyang/nwpu-resisc45 -p downloads/
    unzip downloads/nwpu-resisc45.zip -d downloads/
    python data/prepare_data.py --raw_dir downloads/NWPU-RESISC45

Option B - Manual (Google Drive):
    https://drive.google.com/file/d/1DnPSU5nelbcn2z-zmEWJDhnvMlhHvPKQ/view
    Extract then: python data/prepare_data.py --raw_dir /path/to/NWPU-RESISC45

Option C - Hugging Face datasets:
    pip install datasets
    python data/prepare_data.py --from_hf
"""

import random
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_hf_download(raw_dir: str) -> str:
    """Download NWPU-RESISC45 from Hugging Face datasets and save to raw_dir."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "Hugging Face `datasets` package not found.\n"
            "Install with: pip install datasets"
        )

    print("Downloading NWPU-RESISC45 from Hugging Face (timm/resisc45)...")
    ds = load_dataset("timm/resisc45", split="train")

    out_dir = Path(raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving {len(ds)} images to {out_dir} ...")
    for item in tqdm(ds, desc="Extracting"):
        label = item["label"] if isinstance(item["label"], str) \
            else ds.features["label"].int2str(item["label"])
        class_dir = out_dir / label
        class_dir.mkdir(exist_ok=True)
        img: Image.Image = item["image"]
        idx = len(list(class_dir.glob("*.jpg"))) + 1
        img.save(class_dir / f"{label}_{idx:03d}.jpg")

    print(f"Done. Dataset at: {out_dir}")
    return str(out_dir)


def find_nwpu_root(path: str) -> Path:
    """
    Given the user-supplied --raw_dir, locate the folder that
    directly contains the 45 class sub-directories.
    Handles the common case of a nested zip extraction:
        NWPU-RESISC45/NWPU-RESISC45/airplane/...
    """
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Raw directory not found: {root}")

    entries = [e for e in root.iterdir() if e.is_dir()]
    if len(entries) >= 30:
        return root

    for sub in entries:
        subentries = [e for e in sub.iterdir() if e.is_dir()]
        if len(subentries) >= 30:
            print(f"Found NWPU-RESISC45 class folders at: {sub}")
            return sub

    raise RuntimeError(
        f"Could not locate 45-class folder structure under {root}.\n"
        "Expected: <raw_dir>/<classname>/<image>.jpg"
    )


# ---------------------------------------------------------------------------
# LR pair generation  (unchanged from original)
# ---------------------------------------------------------------------------

def create_lr_pair(
    hr_image: np.ndarray,
    scale: int = 4,
    blur_sigma: float = 1.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create dual low-resolution images from a high-resolution image.

    Args:
        hr_image:   High-resolution image array (H, W, C)
        scale:      Downsampling scale factor
        blur_sigma: Gaussian blur sigma for LR2

    Returns:
        Tuple of (LR1, LR2) numpy arrays
    """
    h, w = hr_image.shape[:2]
    lr_size = (w // scale, h // scale)  # PIL uses (W, H)

    hr_pil = Image.fromarray(hr_image)

    # LR1: pure bicubic downsampling
    lr1 = np.array(hr_pil.resize(lr_size, Image.BICUBIC))

    # LR2: Gaussian blur then bicubic downsampling
    blurred = cv2.GaussianBlur(hr_image, (0, 0), blur_sigma)
    lr2 = np.array(Image.fromarray(blurred).resize(lr_size, Image.BICUBIC))

    return lr1, lr2


# ---------------------------------------------------------------------------
# Main preparation function
# ---------------------------------------------------------------------------

def prepare_nwpu_dataset(
    raw_dir: str,
    dataset_dir: str = "dataset",
    train_split: float = 0.70,
    val_split: float = 0.15,
    test_split: float = 0.15,
    scale: int = 4,
    blur_sigma: float = 1.5,
    seed: int = 42,
) -> None:
    """
    Process a downloaded NWPU-RESISC45 folder into the dual-LR dataset structure
    expected by DualSRDataset.

    Output structure
    ----------------
    dataset/
    ├── train/
    │   ├── hr/   (PNG, 256x256)
    │   ├── lr1/  (PNG, 64x64  — bicubic x4)
    │   └── lr2/  (PNG, 64x64  — Gaussian sigma=1.5 + bicubic x4)
    ├── val/
    │   └── ...
    └── test/
        └── ...
    """
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
        "Splits must sum to 1.0"

    random.seed(seed)

    images_dir = find_nwpu_root(raw_dir)
    print(f"Using source images from: {images_dir}")

    # Create output directory tree
    for split in ("train", "val", "test"):
        for sub in ("hr", "lr1", "lr2"):
            Path(dataset_dir, split, sub).mkdir(parents=True, exist_ok=True)

    # Collect all image paths (JPEG and PNG)
    image_paths = []
    for class_dir in sorted(images_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_file in sorted(class_dir.glob("*")):
            if img_file.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif"):
                image_paths.append(img_file)

    n_classes = len([e for e in images_dir.iterdir() if e.is_dir()])
    print(f"Found {len(image_paths)} images across {n_classes} classes")

    if len(image_paths) == 0:
        raise RuntimeError(
            f"No images found under {images_dir}.\n"
            "Check that the path contains class sub-directories with JPEG files."
        )

    # Shuffle and assign splits
    random.shuffle(image_paths)
    n_total = len(image_paths)
    n_train = int(n_total * train_split)
    n_val   = int(n_total * val_split)

    def get_split(i: int) -> str:
        if i < n_train:
            return "train"
        elif i < n_train + n_val:
            return "val"
        return "test"

    # Process and save
    print("Generating HR / LR1 / LR2 pairs...")
    skipped = 0
    for i, img_path in enumerate(tqdm(image_paths)):
        try:
            hr_image = np.array(Image.open(img_path).convert("RGB"))

            if hr_image.shape[0] < scale or hr_image.shape[1] < scale:
                skipped += 1
                continue

            lr1, lr2 = create_lr_pair(hr_image, scale=scale, blur_sigma=blur_sigma)

            split    = get_split(i)
            out_name = f"{img_path.parent.name}_{img_path.stem}.png"

            Image.fromarray(hr_image).save(Path(dataset_dir, split, "hr",  out_name))
            Image.fromarray(lr1).save(     Path(dataset_dir, split, "lr1", out_name))
            Image.fromarray(lr2).save(     Path(dataset_dir, split, "lr2", out_name))

        except Exception as e:
            print(f"\nWarning: skipping {img_path.name} — {e}")
            skipped += 1

    n_test = n_total - n_train - n_val - skipped
    print("\nDataset preparation complete!")
    print(f"  Total processed : {n_total - skipped}  (skipped: {skipped})")
    print(f"  Train : {n_train}")
    print(f"  Val   : {n_val}")
    print(f"  Test  : {n_test}")
    print(f"  Output: {dataset_dir}/")
    print(f"  HR: 256x256   LR: {256 // scale}x{256 // scale}   scale: x{scale}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare NWPU-RESISC45 dataset for Dual-SR training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--raw_dir", type=str, default=None,
        help="Path to extracted NWPU-RESISC45 folder (contains 45 class sub-dirs)"
    )
    parser.add_argument(
        "--from_hf", action="store_true",
        help="Download from Hugging Face (timm/resisc45) into --raw_dir"
    )
    parser.add_argument("--dataset_dir",  type=str,   default="dataset")
    parser.add_argument("--train_split",  type=float, default=0.70)
    parser.add_argument("--val_split",    type=float, default=0.15)
    parser.add_argument("--test_split",   type=float, default=0.15)
    parser.add_argument("--scale",        type=int,   default=4)
    parser.add_argument("--blur_sigma",   type=float, default=1.5)
    parser.add_argument("--seed",         type=int,   default=42)

    args = parser.parse_args()

    raw_dir = args.raw_dir or "downloads/NWPU-RESISC45"

    if args.from_hf:
        raw_dir = _try_hf_download(raw_dir)
    elif not Path(raw_dir).exists():
        print(
            "\nNWPU-RESISC45 raw directory not found.\n\n"
            "Download options:\n\n"
            "  Option A  Kaggle CLI (fastest):\n"
            "    pip install kaggle\n"
            "    kaggle datasets download -d happyyang/nwpu-resisc45 -p downloads/\n"
            "    unzip downloads/nwpu-resisc45.zip -d downloads/\n"
            f"    python data/prepare_data.py --raw_dir downloads/NWPU-RESISC45\n\n"
            "  Option B  Manual (Google Drive):\n"
            "    https://drive.google.com/file/d/1DnPSU5nelbcn2z-zmEWJDhnvMlhHvPKQ/view\n"
            "    Extract and run:\n"
            f"    python data/prepare_data.py --raw_dir /path/to/NWPU-RESISC45\n\n"
            "  Option C  Hugging Face:\n"
            "    pip install datasets\n"
            f"    python data/prepare_data.py --from_hf --raw_dir downloads/NWPU-RESISC45\n"
        )
        raise SystemExit(1)

    prepare_nwpu_dataset(
        raw_dir=raw_dir,
        dataset_dir=args.dataset_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        scale=args.scale,
        blur_sigma=args.blur_sigma,
        seed=args.seed,
    )
