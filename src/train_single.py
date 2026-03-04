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
from pathlib import Path
from datetime import datetime

import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import csv
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import random
import torchvision.transforms.functional as TF

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

def train_single_edsr(config: dict, use_hf: bool, dataset_dir: str, device: torch.device,
                      resume_from: str = None):
    data_cfg  = config['data']
    train_cfg = config['training']
    ckpt_cfg  = config.get('checkpoint', {})
    log_cfg   = config.get('logging', {})

    scale      = config['model'].get('scale', 4)
    patch_size = data_cfg.get('patch_size', 0)
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
        from datasets import concatenate_datasets
        hf_name = data_cfg.get('hf_dataset', 'timm/resisc45')
        ds = concatenate_datasets([
            load_dataset(hf_name, split='train'),
            load_dataset(hf_name, split='validation'),
            load_dataset(hf_name, split='test'),
        ])
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
        in_channels=config['model'].get('in_channels', 3),
        out_channels=config['model'].get('out_channels', 3),
        num_features=config['model']['num_features'],
        num_residual_blocks=config['model']['num_residual_blocks'],
        scale=scale,
        res_scale=config['model'].get('res_scale', 0.1),
        use_mean_shift=config['model'].get('use_mean_shift', True),
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
                           betas=(0.9, 0.999),
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

    # AMP + gradient clipping
    use_amp   = train_cfg.get('amp', False)
    scaler    = (
        torch.amp.GradScaler('cuda')
        if (use_amp and device.type == 'cuda') else None
    )
    grad_clip = train_cfg.get('grad_clip', 0.0)

    # Checkpointing / logging
    ckpt_dir = Path(ckpt_cfg.get('save_dir', 'checkpoints')) / 'single'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_log_path = Path(log_cfg.get('log_dir', 'logs')) / f'single_edsr_log_{timestamp}.csv'
    csv_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['Epoch', 'Train Loss', 'Val Loss',
                                 'Val PSNR (dB)', 'Val SSIM', 'LR', 'Best'])

    best_psnr = 0.0
    best_ssim = 0.0
    best_psnr_epoch = -1
    best_ssim_epoch = -1
    epochs_no_improve = 0
    start_epoch = 0

    # Resume from checkpoint
    if resume_from:
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if scaler and 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch      = ckpt.get('epoch', 0) + 1
        best_psnr        = ckpt.get('best_psnr', 0.0)
        best_ssim        = ckpt.get('best_ssim', 0.0)
        best_psnr_epoch  = ckpt.get('best_psnr_epoch', -1)
        best_ssim_epoch  = ckpt.get('best_ssim_epoch', -1)
        epochs_no_improve = ckpt.get('epochs_no_improve', 0)
        print(f'Resumed from {resume_from}  (epoch {start_epoch}, best PSNR {best_psnr:.4f} dB)')

    print(f'\nStarting SingleEDSR training on {device}')
    print(f'Training samples : {len(train_ds)}')
    print(f'Validation samples: {len(val_ds)}')
    print('-' * 50)

    for epoch in range(start_epoch, epochs):
        # ---- Train ----
        model.train()
        total_loss = 0.0
        for lr_img, hr_img in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=scaler is not None):
                sr = model(lr_img)
                loss = criterion(sr, hr_img)
            if scaler:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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

        # Best tracking
        is_best_psnr = val_psnr > best_psnr + es_delta
        is_best_ssim = val_ssim > best_ssim
        if is_best_psnr:
            best_psnr = val_psnr
            best_psnr_epoch = epoch + 1
            epochs_no_improve = 0
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_psnr': best_psnr,
                'best_ssim': best_ssim,
                'best_psnr_epoch': best_psnr_epoch,
                'best_ssim_epoch': best_ssim_epoch,
                'epochs_no_improve': epochs_no_improve,
            }
            if scaler:
                ckpt['scaler_state_dict'] = scaler.state_dict()
            torch.save(ckpt, ckpt_dir / 'single_edsr_best.pth')
            print(f'  New best model saved! PSNR: {best_psnr:.2f} dB')
        else:
            epochs_no_improve += 1
        if is_best_ssim:
            best_ssim = val_ssim
            best_ssim_epoch = epoch + 1

        # Always save resume checkpoint (overwrites each epoch)
        resume_ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_psnr': best_psnr,
            'best_ssim': best_ssim,
            'best_psnr_epoch': best_psnr_epoch,
            'best_ssim_epoch': best_ssim_epoch,
            'epochs_no_improve': epochs_no_improve,
        }
        if scaler:
            resume_ckpt['scaler_state_dict'] = scaler.state_dict()
        torch.save(resume_ckpt, ckpt_dir / 'single_edsr_resume.pth')

        # CSV row
        lr_now = optimizer.param_groups[0]['lr']
        best_flag = ('PSNR ' if is_best_psnr else '') + ('SSIM' if is_best_ssim else '')
        with open(csv_log_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch + 1,
                f'{train_loss:.6f}', f'{val_loss:.6f}',
                f'{val_psnr:.2f}',   f'{val_ssim:.4f}',
                f'{lr_now:.2e}',     best_flag.strip()
            ])

        psnr_mark = ' ← best PSNR' if is_best_psnr else ''
        ssim_mark = ' ← best SSIM' if is_best_ssim else ''
        print(f'Epoch {epoch+1}/{epochs}  LR: {optimizer.param_groups[0]["lr"]:.2e}')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val   Loss: {val_loss:.4f} | PSNR: {val_psnr:.2f} dB{psnr_mark} | SSIM: {val_ssim:.4f}{ssim_mark}')

        if epochs_no_improve >= es_patience:
            print(f'\nEarly stopping after {es_patience} epochs without improvement.')
            break

    print(f'\n  CSV log saved → {csv_log_path}')
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
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
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

    # Reproducibility + cuDNN tuning
    seed = config['data'].get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True  # Fixed input size → cuDNN autotuning safe

    train_single_edsr(config, use_hf, args.dataset_dir, device, resume_from=args.resume)

