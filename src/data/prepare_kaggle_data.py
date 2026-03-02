"""
Prepare UC Merced dataset from Kaggle download.

Creates dual LR pairs for training:
- LR1: Bicubic downsample (4x)
- LR2: Gaussian blur + bicubic downsample (4x)
"""

import os
import random
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm


def create_lr_pair(hr_image, scale=4, blur_sigma=1.5):
    """Create dual LR images from HR."""
    h, w = hr_image.shape[:2]
    lr_size = (w // scale, h // scale)
    
    hr_pil = Image.fromarray(hr_image)
    
    # LR1: Bicubic
    lr1 = np.array(hr_pil.resize(lr_size, Image.BICUBIC))
    
    # LR2: Blur + Bicubic  
    blurred = cv2.GaussianBlur(hr_image, (0, 0), blur_sigma)
    lr2 = np.array(Image.fromarray(blurred).resize(lr_size, Image.BICUBIC))
    
    return lr1, lr2


def prepare_from_kaggle(
    kaggle_path: str = r"C:\Users\VICTUS\.cache\kagglehub\datasets\ashikahmmed\uc-merce\versions\1\converted_uc_merced_data",
    output_dir: str = "dataset",
    train_split: float = 0.70,
    val_split: float = 0.15,
    scale: int = 4,
    blur_sigma: float = 1.5,
    seed: int = 42
):
    """Prepare dataset from Kaggle UC Merced download."""
    random.seed(seed)
    kaggle_path = Path(kaggle_path)
    output_dir = Path(output_dir)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        for subdir in ['hr', 'lr1', 'lr2']:
            (output_dir / split / subdir).mkdir(parents=True, exist_ok=True)
    
    # Collect all image paths
    image_paths = []
    for class_dir in kaggle_path.iterdir():
        if class_dir.is_dir():
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                    image_paths.append(img_file)
    
    print(f"Found {len(image_paths)} images in {len(list(kaggle_path.iterdir()))} classes")
    
    # Shuffle and split
    random.shuffle(image_paths)
    n_total = len(image_paths)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    
    splits = ['train'] * n_train + ['val'] * n_val + ['test'] * (n_total - n_train - n_val)
    
    # Process images
    print("Creating dual LR pairs...")
    for img_path, split in tqdm(zip(image_paths, splits), total=n_total):
        # Load HR image
        hr = np.array(Image.open(img_path).convert('RGB'))
        
        # Ensure 256x256
        if hr.shape[0] != 256 or hr.shape[1] != 256:
            hr = np.array(Image.fromarray(hr).resize((256, 256), Image.BICUBIC))
        
        # Create LR pair
        lr1, lr2 = create_lr_pair(hr, scale=scale, blur_sigma=blur_sigma)
        
        # Save
        img_name = f"{img_path.parent.name}_{img_path.stem}.png"
        Image.fromarray(hr).save(output_dir / split / 'hr' / img_name)
        Image.fromarray(lr1).save(output_dir / split / 'lr1' / img_name)
        Image.fromarray(lr2).save(output_dir / split / 'lr2' / img_name)
    
    # Summary
    train_count = len(list((output_dir / 'train' / 'hr').iterdir()))
    val_count = len(list((output_dir / 'val' / 'hr').iterdir()))
    test_count = len(list((output_dir / 'test' / 'hr').iterdir()))
    
    print(f"\nDataset prepared!")
    print(f"  Train: {train_count} images")
    print(f"  Val: {val_count} images")
    print(f"  Test: {test_count} images")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    prepare_from_kaggle()
