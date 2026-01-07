#!/usr/bin/env python3
"""Test that 4D aggregation produces same output as original 6D version."""

import sys
from pathlib import Path
import torch
import numpy as np

# Add UVQ source path
UVQ_SOURCE_PATH = Path.home() / "work" / "UVQ" / "uvq"
sys.path.insert(0, str(UVQ_SOURCE_PATH))

from uvq1p5_pytorch.utils import distortionnet


def original_6d_aggregation(patch_features, batch_size):
    """Original 6D aggregation method."""
    # Reshape to (batch, 3, 3, 128, 8, 8)
    features = patch_features.reshape(batch_size, 3, 3, 128, 8, 8)
    
    # Rearrange to (batch, 3*8, 3*8, 128) = (batch, 24, 24, 128)
    features = features.permute(0, 1, 4, 2, 5, 3).contiguous()
    features = features.reshape(batch_size, 24, 24, 128)
    
    return features


def new_4d_aggregation(patch_features, batch_size):
    """New 4D-only aggregation method."""
    # Step 1: Convert to NHWC format: [9, 128, 8, 8] -> [9, 8, 8, 128]
    patches_nhwc = patch_features.permute(0, 2, 3, 1).contiguous()
    
    # Step 2: Reshape to [3, 24, 8, 128] - concatenate 3 patches horizontally per row
    features = patches_nhwc.reshape(3, 24, 8, 128)
    
    # Step 3: Transpose to [3, 8, 24, 128]
    features = features.permute(0, 2, 1, 3).contiguous()
    
    # Step 4: Reshape to [1, 24, 24, 128] - concatenate 3 rows vertically
    features = features.reshape(batch_size, 24, 24, 128)
    
    return features


def test_aggregation():
    """Test that both aggregation methods produce identical output."""
    print("\n" + "="*70)
    print("Testing 4D Aggregation vs Original 6D Aggregation")
    print("="*70)
    
    # Create random patch features (9 patches, NCHW format)
    torch.manual_seed(42)
    patch_features = torch.randn(9, 128, 8, 8)
    batch_size = 1
    
    print(f"\nInput shape: {patch_features.shape} (9 patches in NCHW format)")
    
    # Test original 6D method
    print("\n1. Testing original 6D aggregation...")
    output_6d = original_6d_aggregation(patch_features.clone(), batch_size)
    print(f"   Output shape: {output_6d.shape}")
    print(f"   Output range: [{output_6d.min():.6f}, {output_6d.max():.6f}]")
    
    # Test new 4D method
    print("\n2. Testing new 4D aggregation...")
    output_4d = new_4d_aggregation(patch_features.clone(), batch_size)
    print(f"   Output shape: {output_4d.shape}")
    print(f"   Output range: [{output_4d.min():.6f}, {output_4d.max():.6f}]")
    
    # Compare outputs
    print("\n3. Comparing outputs...")
    abs_diff = torch.abs(output_6d - output_4d)
    max_diff = torch.max(abs_diff).item()
    mean_diff = torch.mean(abs_diff).item()
    
    print(f"   Max absolute difference:  {max_diff:.10f}")
    print(f"   Mean absolute difference: {mean_diff:.10f}")
    
    # Check if they match
    tolerance = 1e-6
    if max_diff < tolerance:
        print(f"\n   ✅ PASSED: Outputs match within tolerance ({tolerance})")
        return True
    else:
        print(f"\n   ❌ FAILED: Outputs differ by more than tolerance ({tolerance})")
        
        # Show where differences occur
        diff_mask = abs_diff > tolerance
        num_diffs = torch.sum(diff_mask).item()
        print(f"   Number of differing elements: {num_diffs} / {output_6d.numel()}")
        
        return False


def test_with_real_model():
    """Test with actual DistortionNet model."""
    print("\n" + "="*70)
    print("Testing with Real DistortionNet Model")
    print("="*70)
    
    # Create model
    print("\n1. Loading DistortionNet...")
    model = distortionnet.DistortionNetCore()
    model.eval()
    
    # Create random input (9 patches)
    torch.manual_seed(42)
    video_patches = torch.randn(9, 3, 360, 640)
    
    print(f"   Input shape: {video_patches.shape}")
    
    # Run through model
    print("\n2. Running inference...")
    with torch.no_grad():
        patch_features = model(video_patches)
    
    print(f"   Patch features shape: {patch_features.shape}")
    
    # Test aggregation
    print("\n3. Testing aggregation on real features...")
    batch_size = 1
    
    output_6d = original_6d_aggregation(patch_features.clone(), batch_size)
    output_4d = new_4d_aggregation(patch_features.clone(), batch_size)
    
    abs_diff = torch.abs(output_6d - output_4d)
    max_diff = torch.max(abs_diff).item()
    mean_diff = torch.mean(abs_diff).item()
    
    print(f"   Max absolute difference:  {max_diff:.10f}")
    print(f"   Mean absolute difference: {mean_diff:.10f}")
    
    tolerance = 1e-6
    if max_diff < tolerance:
        print(f"\n   ✅ PASSED: Real model outputs match within tolerance ({tolerance})")
        return True
    else:
        print(f"\n   ❌ FAILED: Real model outputs differ by more than tolerance ({tolerance})")
        return False


if __name__ == '__main__':
    print("\n" + "="*70)
    print("4D Aggregation Verification Test")
    print("="*70)
    
    # Test 1: Basic aggregation
    test1_passed = test_aggregation()
    
    # Test 2: With real model
    test2_passed = test_with_real_model()
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"Basic aggregation test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Real model test:        {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n✅ All tests PASSED! 4D aggregation is equivalent to 6D aggregation.")
        sys.exit(0)
    else:
        print("\n❌ Some tests FAILED!")
        sys.exit(1)

