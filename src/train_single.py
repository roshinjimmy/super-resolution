"""
Training Script for Single-EDSR Baseline

Trains the SingleEDSR model (one LR input) using the same config, optimizer,
scheduler, and dataset as train.py — enabling a fair comparison with DualEDSR.

Usage:
    python train_single.py --config config/config.yaml
    python train_single.py --config config/config.yaml --no_hf --dataset_dir dataset
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import random
import torchvision.transforms.functional as TF
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from models.dual_edsr import SingleEDSR
from utils.losses import CombinedLoss
from utils.metrics import calculate_psnr, calculate_ssim


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class SingleHFDataset(Dataset):
    """On-the-fly single-LR dataset from HuggingFace. LR1 = bicubic only."""

    def __init__(self, hf_dataset, split='train', scale=4, patch_size=32, augment=True):
        self.data       = hf_dataset
        self.split      = split
        self.scale      = scale
        self.patch_size = patch_size if (split == 'train' and patch_size > 0) else 0
        self.augment    = augment and (split == 'train')
        self.to_tensor  = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        hr: Image.Image = self.data[idx]['image'].convert('RGB')
        h, w = hr.size[1], hr.size[0]
        lr_size = (w // self.scale, h // self.scale)
        lr = hr.resize(lr_size, Image.BICUBIC)

        if self.patch_size > 0:
            lw, lh = lr.size
            ps = self.patch_size
            x = random.randint(0, lw - ps)
            y = random.randint(0, lh - ps)
            lr = lr.crop((x, y, x + ps, y + ps))
            hr = hr.crop((x * self.scale, y * self.scale,
                          (x + ps) * self.scale, (y + ps) * self.scale))

        if self.augment:
            if random.random() > 0.5:
                lr, hr = TF.hflip(lr), TF.hflip(hr)
            if random.random() > 0.5:
                lr, hr = TF.vflip(lr), TF.vflip(hr)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                lr, hr = TF.rotate(lr, angle), TF.rotate(hr, angle)

        return self.to_tensor(lr), self.to_tensor(hr)


class SingleDiskDataset(Dataset):
    """Disk-based single-LR dataset (uses lr1/ folder)."""

    def __init__(self, dataset_dir, split='train', scale=4, patch_size=32, augment=True):
        self.hr_dir    = Path(dataset_dir) / split / 'hr'
        self.lr_dir    = Path(dataset_dir) / split / 'lr1'
        self.scale     = scale
        self.patch_size = patch_size if (split == 'train' and patch_size > 0) else 0
        self.augment   = augment and (split == 'train')
        self.images    = sorted(self.hr_dir.glob('*.png'))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx].name
        hr = Image.open(self.hr_dir / name).convert('RGB')
        lr = Image.open(self.lr_dir / name).convert('RGB')

        if self.patch_size > 0:
            lw, lh = lr.size
            ps = self.patch_size
            x = random.randint(0, lw - ps)
            y = random.randint(0, lh - ps)
            lr = lr.crop((x, y, x + ps, y + ps))
            hr = hr.crop((x * self.scale, y * self.scale,
                          (x + ps) * self.scale, (y + ps) * self.scale))

        if self.augment:
            if random.random() > 0.5:
                lr, hr = TF.hflip(lr), TF.hflip(hr)
            if random.random() > 0.5:
                lr, hr = TF.vflip(lr), TF.vflip(hr)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                lr, hr = TF.rotate(lr, angle), TF.rotate(hr, angle)

        return self.to_tensor(lr), self.to_tensor(hr)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_single_edsr(config: dict, use_hf: bool, dataset_dir: str, device: torch.device):
    data_cfg  = config['data']
    train_cfg = config['training']
    ckpt_cfg  = config.get('checkpoint', {})
    log_cfg   = config.get('logging', {})

    scale      = config['model'].get('scale', 4)
    patch_size = data_cfg.get('patch_size', 32)
    batch_size = train_cfg['batch_size']
    num_workers = train_cfg.get('num_workers', 4)
    epochs     = train_cfg['epochs']

    # Build dataloaders
    if use_hf:
        try:
            from datasets import load_dataset
        except ImportError:
            raise RuntimeError("Install HF datasets: pip install datasets")
        print(f"Loading '{data_cfg.get('hf_dataset', 'timm/resisc45')}' from HuggingFace...")
        ds = load_dataset(data_cfg.get('hf_dataset', 'timm/resisc45'), split='train')
        ds = ds.shuffle(seed=data_cfg.get('seed', 42))
        n = len(ds)
        n_train = int(n * data_cfg.get('train_split', 0.70))
        n_val   = int(n * data_cfg.get('val_split',   0.15))
        train_ds = SingleHFDataset(ds.select(range(n_train)),          'train', scale, patch_size)
        val_ds   = SingleHFDataset(ds.select(range(n_train, n_train+n_val)), 'val', scale, 0, False)
        print(f"  Train: {n_train}  |  Val: {n_val}")
    else:
        train_ds = SingleDiskDataset(dataset_dir, 'train', scale, patch_size)
        val_ds   = SingleDiskDataset(dataset_dir, 'val',   scale, 0, False)

    pw = dict(num_workers=num_workers, pin_memory=True,
              persistent_workers=(num_workers > 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, **pw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **pw)

    # Model
    model = SingleEDSR(
        num_features=config['model']['num_features'],
        num_residual_blocks=config['model']['num_residual_blocks'],
        scale=scale
    ).to(device)
    print(f"\nSingleEDSR parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss, optimizer, scheduler  (same as DualEDSR for fair comparison)
    criterion = CombinedLoss(
        l1_weight=train_cfg.get('l1_weight', 1.0),
        ssim_weight=train_cfg.get('ssim_weight', 0.1),
        use_ssim=train_cfg.get('use_ssim', True)
    )
    optimizer = optim.Adam(model.parameters(),
                           lr=train_cfg['learning_rate'],
                           weight_decay=train_cfg.get('weight_decay', 0))
    sched_cfg = train_cfg.get('scheduler', {})
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max',
        patience=sched_cfg.get('patience', 10),
        factor=sched_cfg.get('factor', 0.5),
        min_lr=sched_cfg.get('min_lr', 1e-6)
    )

    # Early stopping
    es_cfg = train_cfg.get('early_stopping', {})
    es_patience = es_cfg.get('patience', 20)
    es_delta    = es_cfg.get('min_delta', 0.001)

    # Checkpointing / logging
    ckpt_dir = Path(ckpt_cfg.get('save_dir', 'checkpoints')) / 'single'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_every = ckpt_cfg.get('save_every', 10)

    writer = None
    if log_cfg.get('use_tensorboard', True):
        log_dir = Path(log_cfg.get('log_dir', 'logs')) / f'single_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        writer = SummaryWriter(log_dir)

    best_psnr = 0.0
    best_ssim = 0.0
    best_psnr_epoch = -1
    best_ssim_epoch = -1
    epochs_no_improve = 0

    print(f'\nStarting SingleEDSR training on {device}')
    print(f'Training samples : {len(train_ds)}')
    print(f'Validation samples: {len(val_ds)}')
    print('-' * 50)

    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        total_loss = 0.0
        for lr_img, hr_img in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            optimizer.zero_grad()
            sr = model(lr_img)
            loss = criterion(sr, hr_img)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)

        # ---- Validate ----
        model.eval()
        val_loss = val_psnr = val_ssim = 0.0
        with torch.no_grad():
            for lr_img, hr_img in val_loader:
                lr_img, hr_img = lr_img.to(device), hr_img.to(device)
                sr = torch.clamp(model(lr_img), 0, 1)
                val_loss += criterion(sr, hr_img).item()
                for i in range(sr.size(0)):
                    val_psnr += calculate_psnr(sr[i], hr_img[i])
                    val_ssim += calculate_ssim(sr[i], hr_img[i])
        n_val_samples = len(val_ds)
        val_loss  /= len(val_loader)
        val_psnr  /= n_val_samples
        val_ssim  /= n_val_samples

        scheduler.step(val_psnr)

        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val',   val_loss,   epoch)
            writer.add_scalar('Metrics/val_PSNR', val_psnr, epoch)
            writer.add_scalar('Metrics/val_SSIM', val_ssim, epoch)

        # Best tracking
        is_best_psnr = val_psnr > best_psnr + es_delta
        is_best_ssim = val_ssim > best_ssim
        if is_best_psnr:
            best_psnr = val_psnr
            best_psnr_epoch = epoch + 1
            epochs_no_improve = 0
            ckpt = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'best_psnr': best_psnr, 'best_ssim': best_ssim}
            torch.save(ckpt, ckpt_dir / 'best_model.pth')
        else:
            epochs_no_improve += 1
        if is_best_ssim:
            best_ssim = val_ssim
            best_ssim_epoch = epoch + 1

        if (epoch + 1) % save_every == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'best_psnr': best_psnr}, ckpt_dir / f'checkpoint_epoch_{epoch+1}.pth')

        psnr_mark = ' ← best PSNR' if is_best_psnr else ''
        ssim_mark = ' ← best SSIM' if is_best_ssim else ''
        print(f'Epoch {epoch+1}/{epochs}  LR: {optimizer.param_groups[0]["lr"]:.2e}')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val   Loss: {val_loss:.4f} | PSNR: {val_psnr:.2f} dB{psnr_mark} | SSIM: {val_ssim:.4f}{ssim_mark}')

        if epochs_no_improve >= es_patience:
            print(f'\nEarly stopping after {es_patience} epochs without improvement.')
            break

    if writer:
        writer.close()

    print('\n' + '=' * 50)
    print('SINGLE-EDSR TRAINING COMPLETE')
    print('=' * 50)
    print(f'  Best PSNR : {best_psnr:.4f} dB  (epoch {best_psnr_epoch})')
    print(f'  Best SSIM : {best_ssim:.4f}      (epoch {best_ssim_epoch})')
    print('  → Use these as the Single-Image EDSR row in Table 1.')
    print('=' * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SingleEDSR baseline')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--dataset_dir', type=str, default='dataset')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--use_hf', action='store_true', default=None,
                        help='Load from HuggingFace (overrides config)')
    parser.add_argument('--no_hf', action='store_true',
                        help='Force disk-based loading')
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs:
        config['training']['epochs'] = args.epochs

    use_hf = config['data'].get('use_hf', True)
    if args.use_hf:
        use_hf = True
    if args.no_hf:
        use_hf = False

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f'Using device: {device}')
    train_single_edsr(config, use_hf, args.dataset_dir, device)

    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx].name
        
        hr = np.array(Image.open(self.hr_dir / img_name).convert('RGB'))
        lr = np.array(Image.open(self.lr_dir / img_name).convert('RGB'))
        
        # Convert to tensor (C, H, W) and normalize to [0, 1]
        hr = torch.from_numpy(hr).permute(2, 0, 1).float() / 255.0
        lr = torch.from_numpy(lr).permute(2, 0, 1).float() / 255.0
        
        return lr, hr


def train_single_edsr(
    dataset_dir='dataset',
    epochs=10,
    batch_size=4,
    lr=1e-4,
    checkpoint_dir='checkpoints_single'
):
    """Train SingleEDSR baseline."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = SingleLRDataset(dataset_dir, 'train')
    val_dataset = SingleLRDataset(dataset_dir, 'val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create model
    model = SingleEDSR(
        num_features=64,
        num_residual_blocks=16,
        scale=4
    ).to(device)
    
    print(f"\nSingleEDSR Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Loss and optimizer
    criterion = CombinedLoss(l1_weight=1.0, ssim_weight=0.0)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_psnr = 0.0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for lr_img, hr_img in pbar:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)
            
            optimizer.zero_grad()
            sr = model(lr_img)
            loss = criterion(sr, hr_img)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_psnr = 0.0
        val_ssim = 0.0
        
        with torch.no_grad():
            for lr_img, hr_img in val_loader:
                lr_img = lr_img.to(device)
                hr_img = hr_img.to(device)
                
                sr = model(lr_img)
                sr = torch.clamp(sr, 0, 1)
                
                for i in range(sr.shape[0]):
                    sr_np = sr[i].cpu().numpy().transpose(1, 2, 0)
                    hr_np = hr_img[i].cpu().numpy().transpose(1, 2, 0)
                    val_psnr += calculate_psnr(sr_np, hr_np)
                    # For SSIM with (H, W, C) format, we need channel_axis=2
                    try:
                        from skimage.metrics import structural_similarity as ssim
                        val_ssim += ssim(sr_np, hr_np, data_range=1.0, channel_axis=2)
                    except:
                        val_ssim += calculate_ssim(sr_np, hr_np)
        
        val_psnr /= len(val_dataset)
        val_ssim /= len(val_dataset)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | "
              f"Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f}")
        
        # Save best model
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_psnr': best_psnr
            }, os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"  New best model saved! PSNR: {best_psnr:.2f} dB")
    
    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'best_psnr': best_psnr
    }, os.path.join(checkpoint_dir, 'final_model.pth'))
    
    print(f"\nTraining complete! Best PSNR: {best_psnr:.2f} dB")
    return best_psnr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SingleEDSR baseline")
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    
    args = parser.parse_args()
    
    train_single_edsr(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
