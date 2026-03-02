"""
Evaluation Script for Dual-SR

Evaluates trained model on test set and compares with baselines.
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
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import DualSRDataset, DualSRHFDataset
from models.dual_edsr import DualEDSR, SingleEDSR, create_model
from utils.metrics import calculate_psnr, calculate_ssim, calculate_niqe, calculate_brisque


def load_model(checkpoint_path: str, config: dict, device: torch.device) -> DualEDSR:
    """Load trained model from checkpoint."""
    model = create_model(config['model'])
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Best PSNR: {checkpoint.get('best_psnr', 'unknown'):.2f} dB")
    
    return model


def bicubic_upscale(lr: torch.Tensor, scale: int = 4) -> torch.Tensor:
    """Upscale using bicubic interpolation (baseline)."""
    import torch.nn.functional as F
    return F.interpolate(lr, scale_factor=scale, mode='bicubic', align_corners=False)


def evaluate_model(
    model: DualEDSR,
    test_dataset: DualSRDataset,
    device: torch.device,
    output_dir: Path = None,
    save_images: bool = True
) -> dict:
    """
    Evaluate model on test dataset.
    
    Returns:
        Dictionary with average PSNR and SSIM
    """
    if output_dir:
        output_dir = Path(output_dir)
        (output_dir / 'sr').mkdir(parents=True, exist_ok=True)
        (output_dir / 'comparisons').mkdir(parents=True, exist_ok=True)
    
    psnr_list = []
    ssim_list = []
    psnr_bicubic_list = []
    ssim_bicubic_list = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc='Evaluating'):
            lr1, lr2, hr = test_dataset[idx]
            lr1 = lr1.unsqueeze(0).to(device)
            lr2 = lr2.unsqueeze(0).to(device)
            hr = hr.unsqueeze(0).to(device)
            
            # Model prediction
            sr = model(lr1, lr2)
            sr = torch.clamp(sr, 0, 1)
            
            # Bicubic baseline
            bicubic = bicubic_upscale(lr1)
            bicubic = torch.clamp(bicubic, 0, 1)
            
            # Calculate metrics
            psnr = calculate_psnr(sr[0], hr[0])
            ssim = calculate_ssim(sr[0], hr[0])
            psnr_bic = calculate_psnr(bicubic[0], hr[0])
            ssim_bic = calculate_ssim(bicubic[0], hr[0])
            
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            psnr_bicubic_list.append(psnr_bic)
            ssim_bicubic_list.append(ssim_bic)
            
            # Save images
            if save_images and output_dir:
                    # DualSRDataset has image_names; DualSRHFDataset uses idx
                    if hasattr(test_dataset, 'image_names'):
                        img_name = test_dataset.image_names[idx]
                    else:
                        img_name = f'{idx:05d}.png'
                # Save SR result
                sr_np = sr[0].cpu().numpy().transpose(1, 2, 0)
                sr_np = (sr_np * 255).astype(np.uint8)
                Image.fromarray(sr_np).save(output_dir / 'sr' / img_name)
                
                # Create comparison figure
                if idx < 20:  # Save first 20 comparisons
                    create_comparison(
                        lr1[0].cpu(), bicubic[0].cpu(), sr[0].cpu(), hr[0].cpu(),
                        psnr_bic, psnr,
                        output_dir / 'comparisons' / f'compare_{idx:03d}.png'
                    )
    
    results = {
        'dual_edsr': {
            'psnr': np.mean(psnr_list),
            'ssim': np.mean(ssim_list),
            'psnr_std': np.std(psnr_list),
            'ssim_std': np.std(ssim_list)
        },
        'bicubic': {
            'psnr': np.mean(psnr_bicubic_list),
            'ssim': np.mean(ssim_bicubic_list),
            'psnr_std': np.std(psnr_bicubic_list),
            'ssim_std': np.std(ssim_bicubic_list)
        }
    }
    
    return results


def create_comparison(
    lr: torch.Tensor,
    bicubic: torch.Tensor,
    sr: torch.Tensor,
    hr: torch.Tensor,
    psnr_bic: float,
    psnr_sr: float,
    save_path: Path
):
    """Create a comparison figure."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    def tensor_to_img(t):
        return t.numpy().transpose(1, 2, 0)
    
    # LR (upscaled for display)
    lr_up = torch.nn.functional.interpolate(
        lr.unsqueeze(0), scale_factor=4, mode='nearest'
    )[0]
    axes[0].imshow(tensor_to_img(lr_up))
    axes[0].set_title('LR Input')
    axes[0].axis('off')
    
    # Bicubic
    axes[1].imshow(np.clip(tensor_to_img(bicubic), 0, 1))
    axes[1].set_title(f'Bicubic\nPSNR: {psnr_bic:.2f} dB')
    axes[1].axis('off')
    
    # SR
    axes[2].imshow(np.clip(tensor_to_img(sr), 0, 1))
    axes[2].set_title(f'Dual-EDSR\nPSNR: {psnr_sr:.2f} dB')
    axes[2].axis('off')
    
    # HR
    axes[3].imshow(tensor_to_img(hr))
    axes[3].set_title('Ground Truth')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_blind(
    model: DualEDSR,
    lr1_dir: str,
    lr2_dir: str,
    device: torch.device,
    output_dir: Path
) -> dict:
    """
    Blind evaluation (no ground truth).
    
    Uses NIQE and BRISQUE metrics.
    """
    from data.dataset import DualSRInferenceDataset
    
    dataset = DualSRInferenceDataset(lr1_dir, lr2_dir)
    
    niqe_list = []
    brisque_list = []
    niqe_bic_list = []
    brisque_bic_list = []
    
    output_dir = Path(output_dir)
    (output_dir / 'sr').mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc='Blind Evaluation'):
            result = dataset[idx]
            if len(result) == 3:
                lr1, lr2, img_name = result
            else:
                lr1, lr2, _, img_name = result
            
            lr1 = lr1.unsqueeze(0).to(device)
            lr2 = lr2.unsqueeze(0).to(device)
            
            # Model prediction
            sr = model(lr1, lr2)
            sr = torch.clamp(sr, 0, 1)
            
            # Bicubic baseline
            bicubic = bicubic_upscale(lr1)
            bicubic = torch.clamp(bicubic, 0, 1)
            
            # Calculate blind metrics
            niqe = calculate_niqe(sr[0].cpu())
            brisque = calculate_brisque(sr[0].cpu())
            niqe_bic = calculate_niqe(bicubic[0].cpu())
            brisque_bic = calculate_brisque(bicubic[0].cpu())
            
            if not np.isnan(niqe):
                niqe_list.append(niqe)
            if not np.isnan(brisque):
                brisque_list.append(brisque)
            if not np.isnan(niqe_bic):
                niqe_bic_list.append(niqe_bic)
            if not np.isnan(brisque_bic):
                brisque_bic_list.append(brisque_bic)
            
            # Save SR result
            sr_np = sr[0].cpu().numpy().transpose(1, 2, 0)
            sr_np = (sr_np * 255).astype(np.uint8)
            Image.fromarray(sr_np).save(output_dir / 'sr' / img_name)
    
    results = {
        'dual_edsr': {
            'niqe': np.mean(niqe_list) if niqe_list else float('nan'),
            'brisque': np.mean(brisque_list) if brisque_list else float('nan')
        },
        'bicubic': {
            'niqe': np.mean(niqe_bic_list) if niqe_bic_list else float('nan'),
            'brisque': np.mean(brisque_bic_list) if brisque_bic_list else float('nan')
        }
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate DualEDSR model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--dataset_dir', type=str, default='dataset',
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--use_hf', action='store_true', default=None,
                        help='Load test set from HuggingFace (overrides config)')
    parser.add_argument('--no_hf', action='store_true',
                        help='Force disk-based test loading')
    parser.add_argument('--blind', action='store_true',
                        help='Run blind evaluation (no ground truth)')
    parser.add_argument('--lr1_dir', type=str, default=None,
                        help='LR1 directory for blind evaluation')
    parser.add_argument('--lr2_dir', type=str, default=None,
                        help='LR2 directory for blind evaluation')
    
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
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.blind:
        # Blind evaluation
        assert args.lr1_dir and args.lr2_dir, "Must provide lr1_dir and lr2_dir for blind evaluation"
        results = evaluate_blind(model, args.lr1_dir, args.lr2_dir, device, output_dir)
        
        print('\n' + '=' * 50)
        print('Blind Evaluation Results (lower is better)')
        print('=' * 50)
        print(f"\nDual-EDSR:")
        print(f"  NIQE: {results['dual_edsr']['niqe']:.4f}")
        print(f"  BRISQUE: {results['dual_edsr']['brisque']:.4f}")
        print(f"\nBicubic Baseline:")
        print(f"  NIQE: {results['bicubic']['niqe']:.4f}")
        print(f"  BRISQUE: {results['bicubic']['brisque']:.4f}")
    else:
        # Standard evaluation with ground truth
        use_hf = config['data'].get('use_hf', True)
        if args.use_hf:  use_hf = True
        if args.no_hf:   use_hf = False

        if use_hf:
            try:
                from datasets import load_dataset
            except ImportError:
                raise RuntimeError("Install HuggingFace datasets: pip install datasets")
            hf_name = config['data'].get('hf_dataset', 'timm/resisc45')
            ds = load_dataset(hf_name, split='train')
            ds = ds.shuffle(seed=config['data'].get('seed', 42))
            n = len(ds)
            n_train = int(n * config['data'].get('train_split', 0.70))
            n_val   = int(n * config['data'].get('val_split', 0.15))
            test_hf = ds.select(range(n_train + n_val, n))
            test_dataset = DualSRHFDataset(
                test_hf, split='test', augment=False,
                scale=config['model'].get('scale', 4),
                blur_sigma=config['data'].get('blur_sigma', 1.5)
            )
            print(f"HF test set: {len(test_dataset)} images")
        else:
            test_dataset = DualSRDataset(args.dataset_dir, split='test', augment=False)
        results = evaluate_model(model, test_dataset, device, output_dir)
        
        print('\n' + '=' * 50)
        print('Evaluation Results')
        print('=' * 50)
        print(f"\nDual-EDSR:")
        print(f"  PSNR: {results['dual_edsr']['psnr']:.2f} ± {results['dual_edsr']['psnr_std']:.2f} dB")
        print(f"  SSIM: {results['dual_edsr']['ssim']:.4f} ± {results['dual_edsr']['ssim_std']:.4f}")
        print(f"\nBicubic Baseline:")
        print(f"  PSNR: {results['bicubic']['psnr']:.2f} ± {results['bicubic']['psnr_std']:.2f} dB")
        print(f"  SSIM: {results['bicubic']['ssim']:.4f} ± {results['bicubic']['ssim_std']:.4f}")
        print(f"\nImprovement over Bicubic:")
        print(f"  ΔPSNR: +{results['dual_edsr']['psnr'] - results['bicubic']['psnr']:.2f} dB")
        print(f"  ΔSSIM: +{results['dual_edsr']['ssim'] - results['bicubic']['ssim']:.4f}")
    
    # Save results
    import json
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'\nResults saved to {output_dir}')


if __name__ == '__main__':
    main()
