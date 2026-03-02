"""
Diagnostic test to identify DualEDSR bug.
Tests: 1) LR1=LR2=bicubic (should match SingleEDSR)
       2) Check data ranges
       3) Check gradient flow
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path

from models.dual_edsr import DualEDSR, SingleEDSR


def test_architecture():
    """Test 1: Verify architecture shapes."""
    print("=" * 60)
    print("TEST 1: Architecture Shape Verification")
    print("=" * 60)
    
    model = DualEDSR(num_features=64, num_residual_blocks=16, scale=4)
    
    lr1 = torch.randn(1, 3, 64, 64)
    lr2 = torch.randn(1, 3, 64, 64)
    
    # Check intermediate shapes
    with torch.no_grad():
        # Shallow features
        feat1 = model.shallow_conv_lr1(lr1)
        feat2 = model.shallow_conv_lr2(lr2)
        print(f"Shallow feat1 shape: {feat1.shape}")  # Should be (1, 64, 64, 64)
        print(f"Shallow feat2 shape: {feat2.shape}")
        
        # Concatenation
        fused = torch.cat([feat1, feat2], dim=1)
        print(f"After concat shape: {fused.shape}")  # Should be (1, 128, 64, 64)
        
        # Fusion conv
        fused = model.fusion_conv(fused)
        print(f"After fusion conv shape: {fused.shape}")  # Should be (1, 64, 64, 64)
        
        # Through body
        body_out = model.body(fused)
        print(f"After body shape: {body_out.shape}")
        
        # Full forward
        hr = model(lr1, lr2)
        print(f"Output shape: {hr.shape}")  # Should be (1, 3, 256, 256)
    
    print("✅ Architecture shapes look correct\n")


def test_identical_inputs():
    """Test 2: Train with LR1 = LR2 (both bicubic)."""
    print("=" * 60)
    print("TEST 2: Identical Inputs (LR1 = LR2 = bicubic)")
    print("=" * 60)
    
    device = torch.device('cpu')
    
    # Load a sample image
    dataset_dir = Path("dataset")
    hr_path = list((dataset_dir / "train" / "hr").glob("*.png"))[0]
    lr1_path = dataset_dir / "train" / "lr1" / hr_path.name
    
    hr = np.array(Image.open(hr_path).convert('RGB')) / 255.0
    lr1 = np.array(Image.open(lr1_path).convert('RGB')) / 255.0
    
    hr_t = torch.from_numpy(hr).permute(2, 0, 1).unsqueeze(0).float()
    lr1_t = torch.from_numpy(lr1).permute(2, 0, 1).unsqueeze(0).float()
    
    print(f"HR range: [{hr_t.min():.3f}, {hr_t.max():.3f}]")
    print(f"LR1 range: [{lr1_t.min():.3f}, {lr1_t.max():.3f}]")
    
    # Test DualEDSR with identical inputs
    dual_model = DualEDSR(num_features=64, num_residual_blocks=16, scale=4)
    with torch.no_grad():
        sr_dual = dual_model(lr1_t, lr1_t)  # LR1 = LR2
    
    print(f"DualEDSR output range: [{sr_dual.min():.3f}, {sr_dual.max():.3f}]")
    
    # Test SingleEDSR
    single_model = SingleEDSR(num_features=64, num_residual_blocks=16, scale=4)
    with torch.no_grad():
        sr_single = single_model(lr1_t)
    
    print(f"SingleEDSR output range: [{sr_single.min():.3f}, {sr_single.max():.3f}]")
    
    # Calculate PSNR
    from utils.metrics import calculate_psnr
    psnr_dual = calculate_psnr(sr_dual.clamp(0, 1), hr_t)
    psnr_single = calculate_psnr(sr_single.clamp(0, 1), hr_t)
    
    print(f"\nPSNR (untrained):")
    print(f"  DualEDSR:   {psnr_dual:.2f} dB")
    print(f"  SingleEDSR: {psnr_single:.2f} dB")


def test_data_ranges():
    """Test 3: Check data ranges and normalization."""
    print("\n" + "=" * 60)
    print("TEST 3: Data Range Verification")
    print("=" * 60)
    
    dataset_dir = Path("dataset")
    
    # Check HR
    hr_files = list((dataset_dir / "train" / "hr").glob("*.png"))[:5]
    lr1_files = list((dataset_dir / "train" / "lr1").glob("*.png"))[:5]
    lr2_files = list((dataset_dir / "train" / "lr2").glob("*.png"))[:5]
    
    print("HR images:")
    for f in hr_files:
        img = np.array(Image.open(f))
        print(f"  {f.name}: shape={img.shape}, range=[{img.min()}, {img.max()}]")
    
    print("\nLR1 images:")
    for f in lr1_files:
        img = np.array(Image.open(f))
        print(f"  {f.name}: shape={img.shape}, range=[{img.min()}, {img.max()}]")
    
    print("\nLR2 images:")
    for f in lr2_files:
        img = np.array(Image.open(f))
        print(f"  {f.name}: shape={img.shape}, range=[{img.min()}, {img.max()}]")


def test_gradient_flow():
    """Test 4: Check if gradients flow through dual branches."""
    print("\n" + "=" * 60)
    print("TEST 4: Gradient Flow Verification")
    print("=" * 60)
    
    model = DualEDSR(num_features=64, num_residual_blocks=16, scale=4)
    
    lr1 = torch.randn(1, 3, 64, 64, requires_grad=True)
    lr2 = torch.randn(1, 3, 64, 64, requires_grad=True)
    target = torch.randn(1, 3, 256, 256)
    
    sr = model(lr1, lr2)
    loss = torch.nn.functional.l1_loss(sr, target)
    loss.backward()
    
    # Check if branches have gradients
    print(f"LR1 has gradient: {lr1.grad is not None}")
    print(f"LR2 has gradient: {lr2.grad is not None}")
    
    # Check branch weights
    print(f"\nShallow conv LR1 weight grad norm: {model.shallow_conv_lr1.weight.grad.norm():.4f}")
    print(f"Shallow conv LR2 weight grad norm: {model.shallow_conv_lr2.weight.grad.norm():.4f}")
    print(f"Fusion conv weight grad norm: {model.fusion_conv.weight.grad.norm():.4f}")
    
    # Check all params have requires_grad
    frozen_params = [n for n, p in model.named_parameters() if not p.requires_grad]
    if frozen_params:
        print(f"\n⚠️ FROZEN PARAMS FOUND: {frozen_params}")
    else:
        print(f"\n✅ All {sum(1 for p in model.parameters())} parameters have requires_grad=True")


def test_mean_shift():
    """Test 5: Check MeanShift layer effect."""
    print("\n" + "=" * 60)
    print("TEST 5: MeanShift Layer Analysis")
    print("=" * 60)
    
    from models.blocks import MeanShift
    
    sub_mean = MeanShift(sign=-1)
    add_mean = MeanShift(sign=1)
    
    x = torch.rand(1, 3, 64, 64)
    print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
    
    y = sub_mean(x)
    print(f"After sub_mean: [{y.min():.3f}, {y.max():.3f}]")
    
    z = add_mean(y)
    print(f"After add_mean: [{z.min():.3f}, {z.max():.3f}]")
    
    diff = (x - z).abs().max()
    print(f"Reconstruction error: {diff:.6f}")
    
    if diff > 1e-5:
        print("⚠️ MeanShift is not invertible!")
    else:
        print("✅ MeanShift is properly invertible")


if __name__ == "__main__":
    test_architecture()
    test_data_ranges()
    test_identical_inputs()
    test_gradient_flow()
    test_mean_shift()
