"""
Deep diagnostic to find the exact layer causing explosion.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path

from models.dual_edsr import DualEDSR, SingleEDSR
from models.blocks import MeanShift


def trace_forward():
    """Trace through forward pass step by step."""
    print("=" * 60)
    print("Tracing DualEDSR Forward Pass")
    print("=" * 60)
    
    model = DualEDSR(num_features=64, num_residual_blocks=16, scale=4)
    
    # Load real data
    dataset_dir = Path("dataset")
    hr_path = list((dataset_dir / "train" / "hr").glob("*.png"))[0]
    lr1_path = dataset_dir / "train" / "lr1" / hr_path.name
    
    lr1 = np.array(Image.open(lr1_path).convert('RGB')) / 255.0
    lr1_t = torch.from_numpy(lr1).permute(2, 0, 1).unsqueeze(0).float()
    lr2_t = lr1_t.clone()  # Same input for both
    
    print(f"\nInput LR1 range: [{lr1_t.min():.3f}, {lr1_t.max():.3f}]")
    
    with torch.no_grad():
        # Step 1: Mean shift
        x1 = model.sub_mean(lr1_t)
        x2 = model.sub_mean(lr2_t)
        print(f"After sub_mean: [{x1.min():.3f}, {x1.max():.3f}]")
        
        # Step 2: Shallow convs
        feat1 = model.shallow_conv_lr1(x1)
        feat2 = model.shallow_conv_lr2(x2)
        print(f"After shallow_conv_lr1: [{feat1.min():.3f}, {feat1.max():.3f}]")
        print(f"After shallow_conv_lr2: [{feat2.min():.3f}, {feat2.max():.3f}]")
        
        # Step 3: Concatenation
        fused = torch.cat([feat1, feat2], dim=1)
        print(f"After concat: [{fused.min():.3f}, {fused.max():.3f}]")
        
        # Step 4: Fusion conv
        fused = model.fusion_conv(fused)
        print(f"After fusion_conv: [{fused.min():.3f}, {fused.max():.3f}]")
        
        # Step 5: Store residual
        residual = fused
        
        # Step 6: Body (16 residual blocks)
        body_out = model.body(fused)
        print(f"After body: [{body_out.min():.3f}, {body_out.max():.3f}]")
        
        # Step 7: Post-conv
        body_out = model.post_conv(body_out)
        print(f"After post_conv: [{body_out.min():.3f}, {body_out.max():.3f}]")
        
        # Step 8: Add residual
        body_out = body_out + residual
        print(f"After residual add: [{body_out.min():.3f}, {body_out.max():.3f}]")
        
        # Step 9: Upsampler
        upsampled = model.upsampler(body_out)
        print(f"After upsampler: [{upsampled.min():.3f}, {upsampled.max():.3f}]")
        
        # Step 10: Reconstruction
        hr = model.reconstruction(upsampled)
        print(f"After reconstruction: [{hr.min():.3f}, {hr.max():.3f}]")
        
        # Step 11: Add mean
        hr = model.add_mean(hr)
        print(f"After add_mean: [{hr.min():.3f}, {hr.max():.3f}]")


def trace_single_edsr():
    """Trace through SingleEDSR for comparison."""
    print("\n" + "=" * 60)
    print("Tracing SingleEDSR Forward Pass")
    print("=" * 60)
    
    model = SingleEDSR(num_features=64, num_residual_blocks=16, scale=4)
    
    # Load real data
    dataset_dir = Path("dataset")
    lr1_path = list((dataset_dir / "train" / "lr1").glob("*.png"))[0]
    lr1 = np.array(Image.open(lr1_path).convert('RGB')) / 255.0
    x = torch.from_numpy(lr1).permute(2, 0, 1).unsqueeze(0).float()
    
    print(f"\nInput range: [{x.min():.3f}, {x.max():.3f}]")
    
    with torch.no_grad():
        # Step 1: Mean shift
        x = model.sub_mean(x)
        print(f"After sub_mean: [{x.min():.3f}, {x.max():.3f}]")
        
        # Step 2: Head conv
        head = model.head(x)
        print(f"After head: [{head.min():.3f}, {head.max():.3f}]")
        
        # Step 3: Body
        body = model.body(head)
        print(f"After body: [{body.min():.3f}, {body.max():.3f}]")
        
        # Step 4: Post-conv
        body = model.post_conv(body)
        print(f"After post_conv: [{body.min():.3f}, {body.max():.3f}]")
        
        # Step 5: Add residual
        body = body + head
        print(f"After residual add: [{body.min():.3f}, {body.max():.3f}]")
        
        # Step 6: Upsampler
        up = model.upsampler(body)
        print(f"After upsampler: [{up.min():.3f}, {up.max():.3f}]")
        
        # Step 7: Tail
        out = model.tail(up)
        print(f"After tail: [{out.min():.3f}, {out.max():.3f}]")
        
        # Step 8: Add mean
        out = model.add_mean(out)
        print(f"After add_mean: [{out.min():.3f}, {out.max():.3f}]")


def check_weight_stats():
    """Compare weight statistics between models."""
    print("\n" + "=" * 60)
    print("Weight Statistics Comparison")
    print("=" * 60)
    
    dual = DualEDSR(num_features=64, num_residual_blocks=16, scale=4)
    single = SingleEDSR(num_features=64, num_residual_blocks=16, scale=4)
    
    print("\nDualEDSR weights:")
    print(f"  shallow_conv_lr1: mean={dual.shallow_conv_lr1.weight.mean():.4f}, std={dual.shallow_conv_lr1.weight.std():.4f}")
    print(f"  shallow_conv_lr2: mean={dual.shallow_conv_lr2.weight.mean():.4f}, std={dual.shallow_conv_lr2.weight.std():.4f}")
    print(f"  fusion_conv: mean={dual.fusion_conv.weight.mean():.4f}, std={dual.fusion_conv.weight.std():.4f}")
    
    print("\nSingleEDSR weights:")
    print(f"  head: mean={single.head.weight.mean():.4f}, std={single.head.weight.std():.4f}")


if __name__ == "__main__":
    trace_forward()
    trace_single_edsr()
    check_weight_stats()
