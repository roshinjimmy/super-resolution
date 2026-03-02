"""
Training Script for Dual-SR

Trains the DualEDSR model on NWPU-RESISC45 (via HuggingFace) with dual LR inputs.
Default: loads dataset on-the-fly from HuggingFace — no prepare_data.py needed.
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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import DualSRDataset, get_dataloaders, get_hf_dataloaders
from models.dual_edsr import DualEDSR, create_model
from utils.losses import CombinedLoss
from utils.metrics import calculate_psnr, calculate_ssim


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class Trainer:
    """Training manager for DualEDSR."""
    
    def __init__(
        self,
        config: dict,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device
    ):
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Training config
        train_config = config['training']
        
        # Loss function
        self.criterion = CombinedLoss(
            l1_weight=train_config.get('l1_weight', 1.0),
            ssim_weight=train_config.get('ssim_weight', 0.1),
            use_ssim=train_config.get('use_ssim', True)
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config.get('weight_decay', 0)
        )
        
        # Scheduler
        scheduler_config = train_config.get('scheduler', {})
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # Maximize PSNR
            patience=scheduler_config.get('patience', 10),
            factor=scheduler_config.get('factor', 0.5),
            min_lr=scheduler_config.get('min_lr', 1e-6)
        )
        
        # Checkpointing
        checkpoint_config = config.get('checkpoint', {})
        self.checkpoint_dir = Path(checkpoint_config.get('save_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = checkpoint_config.get('save_every', 10)
        
        # Logging
        log_config = config.get('logging', {})
        self.log_dir = Path(log_config.get('log_dir', 'logs'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if log_config.get('use_tensorboard', True):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.writer = SummaryWriter(self.log_dir / f'run_{timestamp}')
        else:
            self.writer = None
        
        self.log_every = log_config.get('log_every', 100)
        
        # Early stopping
        early_config = train_config.get('early_stopping', {})
        self.early_stopping_patience = early_config.get('patience', 20)
        self.early_stopping_min_delta = early_config.get('min_delta', 0.001)
        self.best_psnr = 0
        self.best_ssim = 0
        self.best_psnr_epoch = -1
        self.best_ssim_epoch = -1
        self.epochs_without_improvement = 0
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
    
    def train_epoch(self) -> tuple:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_psnr = 0
        total_ssim = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        for batch_idx, (lr1, lr2, hr) in enumerate(pbar):
            lr1 = lr1.to(self.device)
            lr2 = lr2.to(self.device)
            hr = hr.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            sr = self.model(lr1, lr2)
            loss = self.criterion(sr, hr)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            self.global_step += 1
            
            # Calculate train metrics (no grad needed)
            with torch.no_grad():
                for i in range(sr.size(0)):
                    total_psnr += calculate_psnr(sr[i], hr[i])
                    total_ssim += calculate_ssim(sr[i], hr[i])
            
            # Log to tensorboard
            if self.writer and self.global_step % self.log_every == 0:
                self.writer.add_scalar('Loss/train', loss.item(), self.global_step)
                self.writer.add_scalar(
                    'LR', self.optimizer.param_groups[0]['lr'], self.global_step
                )
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        n_samples = len(self.train_loader.dataset)
        return (
            total_loss / len(self.train_loader),
            total_psnr / n_samples,
            total_ssim / n_samples
        )
    
    @torch.no_grad()
    def validate(self) -> tuple:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0
        total_psnr = 0
        total_ssim = 0
        
        for lr1, lr2, hr in tqdm(self.val_loader, desc='Validation'):
            lr1 = lr1.to(self.device)
            lr2 = lr2.to(self.device)
            hr = hr.to(self.device)
            
            sr = self.model(lr1, lr2)
            loss = self.criterion(sr, hr)
            
            total_loss += loss.item()
            
            # Calculate metrics per image in batch
            for i in range(sr.size(0)):
                psnr = calculate_psnr(sr[i], hr[i])
                ssim = calculate_ssim(sr[i], hr[i])
                total_psnr += psnr
                total_ssim += ssim
        
        n_samples = len(self.val_loader.dataset)
        avg_loss = total_loss / len(self.val_loader)
        avg_psnr = total_psnr / n_samples
        avg_ssim = total_ssim / n_samples
        
        return avg_loss, avg_psnr, avg_ssim
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_psnr': self.best_psnr,
            'best_ssim': self.best_ssim,
            'best_psnr_epoch': self.best_psnr_epoch,
            'best_ssim_epoch': self.best_ssim_epoch,
            'config': self.config
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f'  New best model saved! PSNR: {self.best_psnr:.2f} dB')
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_psnr = checkpoint.get('best_psnr', 0)
        self.best_ssim = checkpoint.get('best_ssim', 0)
        self.best_psnr_epoch = checkpoint.get('best_psnr_epoch', -1)
        self.best_ssim_epoch = checkpoint.get('best_ssim_epoch', -1)
        
        print(f'Loaded checkpoint from epoch {self.current_epoch}')
    
    def train(self, num_epochs: int, resume_from: str = None):
        """Main training loop."""
        if resume_from:
            self.load_checkpoint(resume_from)
        
        print(f'\nStarting training on {self.device}')
        print(f'Model parameters: {sum(p.numel() for p in self.model.parameters()):,}')
        print(f'Training samples: {len(self.train_loader.dataset)}')
        print(f'Validation samples: {len(self.val_loader.dataset)}')
        print('-' * 50)
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Train
            train_loss, train_psnr, train_ssim = self.train_epoch()
            
            # Validate
            val_loss, val_psnr, val_ssim = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_psnr)
            
            # Log to tensorboard
            if self.writer:
                self.writer.add_scalar('Loss/train_epoch', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Metrics/train_PSNR', train_psnr, epoch)
                self.writer.add_scalar('Metrics/train_SSIM', train_ssim, epoch)
                self.writer.add_scalar('Metrics/val_PSNR', val_psnr, epoch)
                self.writer.add_scalar('Metrics/val_SSIM', val_ssim, epoch)
            
            # Check for improvement
            is_best_psnr = val_psnr > self.best_psnr + self.early_stopping_min_delta
            is_best_ssim = val_ssim > self.best_ssim
            is_best = is_best_psnr

            if is_best_psnr:
                self.best_psnr = val_psnr
                self.best_psnr_epoch = epoch + 1
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            if is_best_ssim:
                self.best_ssim = val_ssim
                self.best_ssim_epoch = epoch + 1
            
            # Save checkpoint
            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', is_best)
            elif is_best:
                self.save_checkpoint('best_model.pth', is_best=True)

            # Print epoch summary
            elapsed = time.time() - start_time
            best_marker_psnr = ' ← best PSNR' if is_best_psnr else ''
            best_marker_ssim = ' ← best SSIM' if is_best_ssim else ''
            print(f'Epoch {epoch+1}/{num_epochs} ({elapsed:.1f}s)')
            print(f'  Train Loss: {train_loss:.4f} | PSNR: {train_psnr:.2f} dB | SSIM: {train_ssim:.4f}')
            print(f'  Val   Loss: {val_loss:.4f} | PSNR: {val_psnr:.2f} dB{best_marker_psnr} | SSIM: {val_ssim:.4f}{best_marker_ssim}')
            print(f'  LR: {self.optimizer.param_groups[0]["lr"]:.2e}')
            
            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f'\nEarly stopping triggered after {self.early_stopping_patience} epochs without improvement')
                break
        
        # Save final model
        self.save_checkpoint('final_model.pth')
        
        if self.writer:
            self.writer.close()
        
        print('\n' + '=' * 50)
        print('TRAINING COMPLETE — Best Validation Results')
        print('=' * 50)
        print(f'  Best PSNR : {self.best_psnr:.4f} dB  (epoch {self.best_psnr_epoch})')
        print(f'  Best SSIM : {self.best_ssim:.4f}      (epoch {self.best_ssim_epoch})')
        print('=' * 50)


def main():
    parser = argparse.ArgumentParser(description='Train DualEDSR model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--dataset_dir', type=str, default='dataset',
                        help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--use_hf', action='store_true', default=None,
                        help='Load dataset from HuggingFace on-the-fly (overrides config)')
    parser.add_argument('--no_hf', action='store_true',
                        help='Force disk-based loading (overrides config use_hf=true)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line args
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f'Using device: {device}')
    
    # Decide whether to use HuggingFace on-the-fly loading or disk
    use_hf = config['data'].get('use_hf', True)
    if args.use_hf:
        use_hf = True
    if args.no_hf:
        use_hf = False

    if use_hf:
        train_loader, val_loader, _ = get_hf_dataloaders(
            hf_dataset_name=config['data'].get('hf_dataset', 'timm/resisc45'),
            batch_size=config['training']['batch_size'],
            num_workers=config['training'].get('num_workers', 4),
            patch_size=config['data'].get('patch_size', 32),
            scale=config['model'].get('scale', 4),
            blur_sigma=config['data'].get('blur_sigma', 1.5),
            train_split=config['data'].get('train_split', 0.70),
            val_split=config['data'].get('val_split', 0.15),
            seed=config['data'].get('seed', 42),
        )
    else:
        train_loader, val_loader, _ = get_dataloaders(
            args.dataset_dir,
            batch_size=config['training']['batch_size'],
            num_workers=config['training'].get('num_workers', 4),
            patch_size=config['data'].get('patch_size', 32),
            scale=config['model'].get('scale', 4)
        )
    
    # Create model
    model = create_model(config['model'])
    
    # Create trainer and train
    trainer = Trainer(config, model, train_loader, val_loader, device)
    trainer.train(
        num_epochs=config['training']['epochs'],
        resume_from=args.resume
    )


if __name__ == '__main__':
    main()
