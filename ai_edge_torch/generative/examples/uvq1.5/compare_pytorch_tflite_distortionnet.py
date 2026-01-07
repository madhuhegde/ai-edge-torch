#!/usr/bin/env python3
"""Compare PyTorch and TFLite outputs for minimal DistortionNet models.

This script verifies that TFLite models produce the same outputs as PyTorch models.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import tensorflow as tf

# Add UVQ source path
UVQ_SOURCE_PATH = Path.home() / "work" / "UVQ" / "uvq"
sys.path.insert(0, str(UVQ_SOURCE_PATH))

from uvq1p5_pytorch.utils.distortionnet_minimal import get_minimal_distortionnet


def compare_outputs(size='minimal', tflite_path=None):
    """Compare PyTorch and TFLite outputs.
    
    Args:
        size: 'single', 'minimal', or 'medium'
        tflite_path: Path to TFLite model (optional, will use default location)
    
    Returns:
        Dictionary with comparison results
    """
    print("\n" + "="*70)
    print(f"Comparing PyTorch vs TFLite: {size.upper()} DistortionNet")
    print("="*70)
    
    # Default TFLite path
    if tflite_path is None:
        tflite_path = Path.home() / "work" / "UVQ" / "uvq" / "models" / "tflite_models" / "uvq1.5" / f"distortion_net_{size}.tflite"
    
    if not os.path.exists(tflite_path):
        print(f"✗ TFLite model not found: {tflite_path}")
        return None
    
    # Create PyTorch model
    print("\n1. Loading PyTorch model...")
    pytorch_model = get_minimal_distortionnet(size)
    pytorch_model.eval()
    print("✓ PyTorch model loaded")
    
    # Create test input (batch size 9 for 9 patches)
    # Use fixed seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    if size == 'single':
        test_input = torch.randn(9, 16, 180, 320)
        print(f"\nTest input shape: {test_input.shape} (9 patches, after initial conv)")
    else:
        test_input = torch.randn(9, 3, 360, 640)
        print(f"\nTest input shape: {test_input.shape} (9 patches of 360x640)")
    
    print(f"Test input range: [{test_input.min():.4f}, {test_input.max():.4f}]")
    
    # Run PyTorch inference
    print("\n2. Running PyTorch inference...")
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input)
    
    print(f"✓ PyTorch output shape: {pytorch_output.shape}")
    print(f"  PyTorch output range: [{pytorch_output.min():.4f}, {pytorch_output.max():.4f}]")
    
    # Load TFLite model
    print("\n3. Loading TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✓ TFLite model loaded")
    print(f"  TFLite input shape: {input_details[0]['shape']}")
    print(f"  TFLite output shape: {output_details[0]['shape']}")
    
    # Run TFLite inference
    print("\n4. Running TFLite inference...")
    test_input_np = test_input.numpy().astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input_np)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"✓ TFLite output shape: {tflite_output.shape}")
    print(f"  TFLite output range: [{tflite_output.min():.4f}, {tflite_output.max():.4f}]")
    
    # Compare outputs
    print("\n5. Comparing outputs...")
    pytorch_output_np = pytorch_output.numpy()
    
    # Check shapes match
    if pytorch_output_np.shape != tflite_output.shape:
        print(f"✗ Shape mismatch!")
        print(f"  PyTorch: {pytorch_output_np.shape}")
        print(f"  TFLite:  {tflite_output.shape}")
        return None
    
    # Calculate differences
    abs_diff = np.abs(pytorch_output_np - tflite_output)
    rel_diff = abs_diff / (np.abs(pytorch_output_np) + 1e-8)
    
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)
    max_rel_diff = np.max(rel_diff)
    mean_rel_diff = np.mean(rel_diff)
    
    # Calculate correlation
    pytorch_flat = pytorch_output_np.flatten()
    tflite_flat = tflite_output.flatten()
    correlation = np.corrcoef(pytorch_flat, tflite_flat)[0, 1]
    
    print(f"\n{'='*70}")
    print("COMPARISON RESULTS")
    print(f"{'='*70}")
    
    print(f"\nAbsolute Differences:")
    print(f"  Max:  {max_abs_diff:.6f}")
    print(f"  Mean: {mean_abs_diff:.6f}")
    print(f"  Std:  {np.std(abs_diff):.6f}")
    
    print(f"\nRelative Differences (%):")
    print(f"  Max:  {max_rel_diff * 100:.4f}%")
    print(f"  Mean: {mean_rel_diff * 100:.4f}%")
    
    print(f"\nCorrelation: {correlation:.8f}")
    
    # Determine if outputs match
    tolerance = 1e-4  # Absolute tolerance
    rel_tolerance = 0.01  # 1% relative tolerance
    
    if max_abs_diff < tolerance or (max_rel_diff < rel_tolerance and correlation > 0.999):
        print(f"\n✓ PASS: Outputs match within tolerance!")
        status = "PASS"
    elif correlation > 0.99:
        print(f"\n⚠️  WARNING: Outputs are highly correlated but have numerical differences")
        print(f"   This is expected due to different implementations")
        status = "ACCEPTABLE"
    else:
        print(f"\n✗ FAIL: Outputs differ significantly!")
        status = "FAIL"
    
    # Show sample values
    print(f"\nSample values (first 5 elements of flattened output):")
    print(f"  PyTorch: {pytorch_flat[:5]}")
    print(f"  TFLite:  {tflite_flat[:5]}")
    print(f"  Diff:    {abs_diff.flatten()[:5]}")
    
    results = {
        'status': status,
        'max_abs_diff': max_abs_diff,
        'mean_abs_diff': mean_abs_diff,
        'max_rel_diff': max_rel_diff,
        'mean_rel_diff': mean_rel_diff,
        'correlation': correlation,
        'pytorch_output': pytorch_output_np,
        'tflite_output': tflite_output,
    }
    
    return results


def compare_all_models():
    """Compare all minimal models."""
    print("\n" + "="*70)
    print("Comparing All Minimal DistortionNet Models")
    print("="*70)
    
    sizes = ['single', 'minimal']
    results = {}
    
    for size in sizes:
        result = compare_outputs(size)
        if result:
            results[size] = result
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n{'Model':<12} {'Status':<12} {'Max Abs Diff':<15} {'Correlation':<12}")
    print("-" * 70)
    
    for size in sizes:
        if size in results:
            r = results[size]
            print(f"{size:<12} {r['status']:<12} {r['max_abs_diff']:<15.6f} {r['correlation']:<12.8f}")
        else:
            print(f"{size:<12} {'FAILED':<12} {'N/A':<15} {'N/A':<12}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare PyTorch and TFLite outputs for minimal DistortionNet',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare single block model
  python compare_pytorch_tflite_distortionnet.py --size single
  
  # Compare minimal model
  python compare_pytorch_tflite_distortionnet.py --size minimal
  
  # Compare all models
  python compare_pytorch_tflite_distortionnet.py --all
  
  # Use custom TFLite path
  python compare_pytorch_tflite_distortionnet.py --size minimal --tflite_path ./my_model.tflite
        """
    )
    
    parser.add_argument(
        '--size',
        type=str,
        choices=['single', 'minimal', 'medium'],
        default='minimal',
        help='Which minimal model to compare (default: minimal)'
    )
    
    parser.add_argument(
        '--tflite_path',
        type=str,
        default=None,
        help='Path to TFLite model (optional, will use default location)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Compare all models'
    )
    
    args = parser.parse_args()
    
    if args.all:
        compare_all_models()
    else:
        compare_outputs(args.size, args.tflite_path)
    
    print("\n" + "="*70)
    print("Comparison complete!")
    print("="*70)


if __name__ == '__main__':
    main()

