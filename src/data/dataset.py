"""
PyTorch Dataset for Dual-SR

Provides paired (LR1, LR2, HR) images for training and evaluation.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random


class _DualSRBase(Dataset):
    """
    Shared base: on-the-fly LR pair generation, patch crop, augmentation.
    Sub-classes must set: self.scale, self.blur_sigma, self.patch_size, self.augment.
    """

    def _create_lr_pair(
        self, hr: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """LR1 = bicubic, LR2 = Gaussian(sigma) + bicubic."""
        hr_np = np.array(hr)
        h, w = hr_np.shape[:2]
        lr_size = (w // self.scale, h // self.scale)
        lr1 = Image.fromarray(hr_np).resize(lr_size, Image.BICUBIC)
        blurred = cv2.GaussianBlur(hr_np, (0, 0), float(self.blur_sigma))
        lr2 = Image.fromarray(blurred).resize(lr_size, Image.BICUBIC)
        return lr1, lr2

    def _random_crop(
        self, lr1: Image.Image, lr2: Image.Image, hr: Image.Image
    ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        lr_w, lr_h = lr1.size
        ps = self.patch_size
        x = random.randint(0, lr_w - ps)
        y = random.randint(0, lr_h - ps)
        lr1 = lr1.crop((x, y, x + ps, y + ps))
        lr2 = lr2.crop((x, y, x + ps, y + ps))
        hr_x, hr_y = x * self.scale, y * self.scale
        hr_ps = ps * self.scale
        hr = hr.crop((hr_x, hr_y, hr_x + hr_ps, hr_y + hr_ps))
        return lr1, lr2, hr

    def _apply_augmentation(
        self, lr1: Image.Image, lr2: Image.Image, hr: Image.Image
    ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        if random.random() > 0.5:
            lr1, lr2, hr = TF.hflip(lr1), TF.hflip(lr2), TF.hflip(hr)
        if random.random() > 0.5:
            lr1, lr2, hr = TF.vflip(lr1), TF.vflip(lr2), TF.vflip(hr)
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            lr1, lr2, hr = TF.rotate(lr1, angle), TF.rotate(lr2, angle), TF.rotate(hr, angle)
        return lr1, lr2, hr


class DualSRDataset(_DualSRBase):
    """
    Dataset class for dual low-resolution super-resolution.
    
    Returns:
        Tuple of (lr1, lr2, hr) tensors, each of shape (C, H, W)
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        augment: bool = True,
        hr_size: int = 256,
        lr_size: int = 64,
        patch_size: int = 0,
        scale: int = 4
    ):
        """
        Args:
            root_dir: Root directory containing train/val/test splits
            split: One of 'train', 'val', 'test'
            augment: Whether to apply data augmentation (only for training)
            hr_size: Expected HR image size
            lr_size: Expected LR image size
            patch_size: LR patch size for random crop (0 = use full image)
            scale: SR scale factor (used to derive HR patch size)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.augment = augment and (split == "train")
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.patch_size = patch_size if (split == "train" and patch_size > 0) else 0
        self.scale = scale
        self.blur_sigma = 1.5  # used by _DualSRBase._create_lr_pair (disk mode ignores this)
        
        # Setup paths
        self.hr_dir = self.root_dir / split / "hr"
        self.lr1_dir = self.root_dir / split / "lr1"
        self.lr2_dir = self.root_dir / split / "lr2"
        
        # Verify directories exist
        assert self.hr_dir.exists(), f"HR directory not found: {self.hr_dir}"
        assert self.lr1_dir.exists(), f"LR1 directory not found: {self.lr1_dir}"
        assert self.lr2_dir.exists(), f"LR2 directory not found: {self.lr2_dir}"
        
        # Get image filenames
        self.image_names = sorted([
            f for f in os.listdir(self.hr_dir)
            if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))
        ])
        
        # Basic transforms
        self.to_tensor = transforms.ToTensor()
    
    def __len__(self) -> int:
        return len(self.image_names)
    
    def _load_image(self, path: Path) -> Image.Image:
        """Load image as RGB."""
        return Image.open(path).convert('RGB')

    # _random_crop and _apply_augmentation inherited from _DualSRBase
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample.
        
        Returns:
            Tuple of (lr1, lr2, hr) tensors
        """
        img_name = self.image_names[idx]
        
        # Load images
        hr = self._load_image(self.hr_dir / img_name)
        lr1 = self._load_image(self.lr1_dir / img_name)
        lr2 = self._load_image(self.lr2_dir / img_name)
        
        # Random patch crop (training only, when patch_size > 0)
        if self.patch_size > 0:
            lr1, lr2, hr = self._random_crop(lr1, lr2, hr)

        # Apply augmentation
        if self.augment:
            lr1, lr2, hr = self._apply_augmentation(lr1, lr2, hr)
        
        # Convert to tensors (normalized to [0, 1])
        hr_tensor = self.to_tensor(hr)
        lr1_tensor = self.to_tensor(lr1)
        lr2_tensor = self.to_tensor(lr2)
        
        return lr1_tensor, lr2_tensor, hr_tensor


class DualSRInferenceDataset(Dataset):
    """
    Dataset for inference on arbitrary LR image pairs.
    
    Used for Sentinel-2 evaluation where we provide pre-prepared LR pairs.
    """
    
    def __init__(
        self,
        lr1_dir: str,
        lr2_dir: str,
        hr_dir: Optional[str] = None
    ):
        """
        Args:
            lr1_dir: Directory containing LR1 images
            lr2_dir: Directory containing LR2 images
            hr_dir: Optional directory containing HR images (for evaluation)
        """
        self.lr1_dir = Path(lr1_dir)
        self.lr2_dir = Path(lr2_dir)
        self.hr_dir = Path(hr_dir) if hr_dir else None
        
        # Get image filenames (assume same names in both directories)
        self.image_names = sorted([
            f for f in os.listdir(self.lr1_dir)
            if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))
        ])
        
        self.to_tensor = transforms.ToTensor()
    
    def __len__(self) -> int:
        return len(self.image_names)
    
    def __getitem__(self, idx: int):
        img_name = self.image_names[idx]
        
        lr1 = Image.open(self.lr1_dir / img_name).convert('RGB')
        lr2 = Image.open(self.lr2_dir / img_name).convert('RGB')
        
        lr1_tensor = self.to_tensor(lr1)
        lr2_tensor = self.to_tensor(lr2)
        
        if self.hr_dir and (self.hr_dir / img_name).exists():
            hr = Image.open(self.hr_dir / img_name).convert('RGB')
            hr_tensor = self.to_tensor(hr)
            return lr1_tensor, lr2_tensor, hr_tensor, img_name
        
        return lr1_tensor, lr2_tensor, img_name


def get_dataloaders(
    dataset_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    augment: bool = True,
    patch_size: int = 32,
    scale: int = 4
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        dataset_dir: Root directory of the dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        augment: Whether to use data augmentation for training
        patch_size: LR patch size for random crop during training (0 = full image)
        scale: SR scale factor
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = DualSRDataset(dataset_dir, split="train", augment=augment,
                                  patch_size=patch_size, scale=scale)
    val_dataset = DualSRDataset(dataset_dir, split="val", augment=False)
    test_dataset = DualSRDataset(dataset_dir, split="test", augment=False)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# HuggingFace on-the-fly dataset  (no prepare_data.py step needed)
# ---------------------------------------------------------------------------

class DualSRHFDataset(_DualSRBase):
    """
    On-the-fly dataset backed by a HuggingFace Dataset object.
    HR images come from the HF cache; LR1/LR2 are generated in __getitem__.
    Compatible with: timm/resisc45  (NWPU-RESISC45)
    Schema expected: {'image': PIL.Image, 'label': int|str}
    """

    def __init__(
        self,
        hf_dataset,
        split: str = "train",
        augment: bool = True,
        scale: int = 4,
        blur_sigma: float = 1.5,
        patch_size: int = 32,
    ):
        self.data       = hf_dataset
        self.split      = split
        self.augment    = augment and (split == "train")
        self.scale      = scale
        self.blur_sigma = blur_sigma
        self.patch_size = patch_size if (split == "train" and patch_size > 0) else 0
        self.to_tensor  = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hr: Image.Image = self.data[idx]["image"].convert("RGB")
        lr1, lr2 = self._create_lr_pair(hr)
        if self.patch_size > 0:
            lr1, lr2, hr = self._random_crop(lr1, lr2, hr)
        if self.augment:
            lr1, lr2, hr = self._apply_augmentation(lr1, lr2, hr)
        return self.to_tensor(lr1), self.to_tensor(lr2), self.to_tensor(hr)


def get_hf_dataloaders(
    hf_dataset_name: str = "timm/resisc45",
    batch_size: int = 16,
    num_workers: int = 4,
    patch_size: int = 32,
    scale: int = 4,
    blur_sigma: float = 1.5,
    train_split: float = 0.70,
    val_split: float = 0.15,
    seed: int = 42,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Load an HF image dataset and return (train, val, test) DataLoaders with
    on-the-fly LR pair generation. First run downloads & caches (~600 MB);
    subsequent runs load from the HF cache instantly.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "HuggingFace `datasets` not installed.\n"
            "Run: pip install datasets"
        )

    print(f"Loading '{hf_dataset_name}' from HuggingFace (cached after first run)...")
    ds = load_dataset(hf_dataset_name, split="train")
    ds = ds.shuffle(seed=seed)

    n       = len(ds)
    n_train = int(n * train_split)
    n_val   = int(n * val_split)

    train_hf = ds.select(range(n_train))
    val_hf   = ds.select(range(n_train, n_train + n_val))
    test_hf  = ds.select(range(n_train + n_val, n))

    print(f"  Train: {len(train_hf)}  |  Val: {len(val_hf)}  |  Test: {len(test_hf)}")

    kw = dict(scale=scale, blur_sigma=blur_sigma)
    train_dataset = DualSRHFDataset(train_hf, "train", augment=True,  patch_size=patch_size, **kw)
    val_dataset   = DualSRHFDataset(val_hf,   "val",   augment=False, **kw)
    test_dataset  = DualSRHFDataset(test_hf,  "test",  augment=False, **kw)

    pw = dict(num_workers=num_workers, pin_memory=True,
              persistent_workers=(num_workers > 0))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,  drop_last=True, **pw)
    val_loader   = torch.utils.data.DataLoader(
        val_dataset,   batch_size=batch_size, shuffle=False, **pw)
    test_loader  = torch.utils.data.DataLoader(
        test_dataset,  batch_size=1,          shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
