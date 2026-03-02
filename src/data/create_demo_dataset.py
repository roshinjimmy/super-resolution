"""
Create a synthetic demo dataset to test the training pipeline.
This is used when the UC Merced dataset cannot be downloaded automatically.
"""

import os
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm


def create_synthetic_dataset(
    dataset_dir: str = "dataset",
    num_train: int = 100,
    num_val: int = 20,
    num_test: int = 20,
    hr_size: int = 256,
    scale: int = 4,
    blur_sigma: float = 1.5
):
    """
    Create a synthetic dataset with random textures and patterns.
    
    This is for testing purposes when the real UC Merced dataset
    cannot be downloaded.
    """
    lr_size = hr_size // scale
    
    # Create directories
    for split in ['train', 'val', 'test']:
        for subdir in ['hr', 'lr1', 'lr2']:
            os.makedirs(os.path.join(dataset_dir, split, subdir), exist_ok=True)
    
    splits = [
        ('train', num_train),
        ('val', num_val),
        ('test', num_test)
    ]
    
    print("Creating synthetic demo dataset...")
    
    for split, num_images in splits:
        print(f"Creating {split} set ({num_images} images)...")
        
        for i in tqdm(range(num_images)):
            # Generate random HR image with patterns
            hr = generate_synthetic_image(hr_size)
            
            # Create LR1: bicubic downsample
            hr_pil = Image.fromarray(hr)
            lr1_pil = hr_pil.resize((lr_size, lr_size), Image.BICUBIC)
            lr1 = np.array(lr1_pil)
            
            # Create LR2: blur + bicubic downsample
            blurred = cv2.GaussianBlur(hr, (0, 0), blur_sigma)
            blurred_pil = Image.fromarray(blurred)
            lr2_pil = blurred_pil.resize((lr_size, lr_size), Image.BICUBIC)
            lr2 = np.array(lr2_pil)
            
            # Save images
            img_name = f"synthetic_{i:04d}.png"
            Image.fromarray(hr).save(os.path.join(dataset_dir, split, 'hr', img_name))
            Image.fromarray(lr1).save(os.path.join(dataset_dir, split, 'lr1', img_name))
            Image.fromarray(lr2).save(os.path.join(dataset_dir, split, 'lr2', img_name))
    
    print(f"\nSynthetic dataset created at: {dataset_dir}")
    print(f"  Train: {num_train} images")
    print(f"  Val: {num_val} images")
    print(f"  Test: {num_test} images")
    print("\nNote: This is a synthetic dataset for testing purposes.")
    print("For actual training, please download the UC Merced dataset manually from:")
    print("  https://www.kaggle.com/datasets/apollo2506/landuse-scene-classification")


def generate_synthetic_image(size: int = 256) -> np.ndarray:
    """Generate a synthetic image with random textures and patterns."""
    # Create base with random colors
    img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    
    # Add some structure - random rectangles and patterns
    num_shapes = np.random.randint(5, 15)
    
    for _ in range(num_shapes):
        shape_type = np.random.choice(['rect', 'circle', 'line'])
        color = tuple(np.random.randint(0, 255, 3).tolist())
        
        if shape_type == 'rect':
            x1, y1 = np.random.randint(0, size, 2)
            x2, y2 = x1 + np.random.randint(20, 100), y1 + np.random.randint(20, 100)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        elif shape_type == 'circle':
            center = tuple(np.random.randint(0, size, 2).tolist())
            radius = np.random.randint(10, 50)
            cv2.circle(img, center, radius, color, -1)
        else:
            x1, y1 = np.random.randint(0, size, 2)
            x2, y2 = np.random.randint(0, size, 2)
            thickness = np.random.randint(1, 10)
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    
    # Apply slight blur to make it more natural
    img = cv2.GaussianBlur(img, (5, 5), 0.5)
    
    return img


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create synthetic demo dataset")
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--num_train", type=int, default=100)
    parser.add_argument("--num_val", type=int, default=20)
    parser.add_argument("--num_test", type=int, default=20)
    
    args = parser.parse_args()
    
    create_synthetic_dataset(
        dataset_dir=args.dataset_dir,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test
    )
