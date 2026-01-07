#!/usr/bin/env python3
"""Convert minimal DistortionNet and immediately verify PyTorch vs TFLite match.

This ensures we're comparing the same model weights.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import ai_edge_torch
import tensorflow as tf

# Add UVQ source path
UVQ_SOURCE_PATH = Path.home() / "work" / "UVQ" / "uvq"
sys.path.insert(0, str(UVQ_SOURCE_PATH))

from uvq1p5_pytorch.utils.distortionnet_minimal import get_minimal_distortionnet


def convert_and_verify(size='minimal', output_dir=None):
    """Convert a minimal DistortionNet and verify PyTorch vs TFLite match.
    
    Args:
        size: 'single', 'minimal', or 'medium'
        output_dir: Directory to save TFLite model (optional)
    """
    print("\n" + "="*70)
    print(f"Convert and Verify: {size.upper()} DistortionNet")
    print("="*70)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create PyTorch model
    print("\n1. Creating PyTorch model...")
    pytorch_model = get_minimal_distortionnet(size)
    pytorch_model.eval()
    print("✓ PyTorch model created")
    
    # Create test input (batch size 9 for 9 patches)
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
    
    # Convert to TFLite
    print("\n3. Converting to TFLite...")
    try:
        edge_model = ai_edge_torch.convert(
            pytorch_model,
            (test_input,)
        )
        
        # Save to temp file or specified directory
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            tflite_path = os.path.join(output_dir, f"distortion_net_{size}.tflite")
        else:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.tflite')
            tflite_path = temp_file.name
            temp_file.close()
        
        edge_model.export(tflite_path)
        file_size = os.path.getsize(tflite_path) / (1024 * 1024)
        print(f"✓ Converted to TFLite")
        print(f"  Saved to: {tflite_path}")
        print(f"  File size: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Load TFLite model
    print("\n4. Loading TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✓ TFLite model loaded")
    
    # Run TFLite inference with THE SAME input
    print("\n5. Running TFLite inference (same input)...")
    test_input_np = test_input.numpy().astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input_np)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"✓ TFLite output shape: {tflite_output.shape}")
    print(f"  TFLite output range: [{tflite_output.min():.4f}, {tflite_output.max():.4f}]")
    
    # Compare outputs
    print("\n6. Comparing outputs...")
    pytorch_output_np = pytorch_output.numpy()
    
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
    
    # Only calculate correlation if there's variance
    if np.std(pytorch_flat) > 1e-6 and np.std(tflite_flat) > 1e-6:
        correlation = np.corrcoef(pytorch_flat, tflite_flat)[0, 1]
    else:
        correlation = 1.0 if max_abs_diff < 1e-6 else 0.0
    
    print(f"\n{'='*70}")
    print("COMPARISON RESULTS")
    print(f"{'='*70}")
    
    print(f"\nAbsolute Differences:")
    print(f"  Max:  {max_abs_diff:.8f}")
    print(f"  Mean: {mean_abs_diff:.8f}")
    print(f"  Std:  {np.std(abs_diff):.8f}")
    
    print(f"\nRelative Differences (%):")
    print(f"  Max:  {max_rel_diff * 100:.6f}%")
    print(f"  Mean: {mean_rel_diff * 100:.6f}%")
    
    print(f"\nCorrelation: {correlation:.10f}")
    
    # Determine if outputs match
    tolerance = 1e-5  # Absolute tolerance
    rel_tolerance = 0.001  # 0.1% relative tolerance
    
    if max_abs_diff < tolerance:
        print(f"\n✓ EXCELLENT: Outputs match within {tolerance} absolute tolerance!")
        status = "EXCELLENT"
    elif max_abs_diff < 1e-4 and correlation > 0.9999:
        print(f"\n✓ GOOD: Outputs match within 1e-4 tolerance (correlation: {correlation:.6f})")
        status = "GOOD"
    elif correlation > 0.999:
        print(f"\n⚠️  ACCEPTABLE: Outputs are highly correlated ({correlation:.6f}) but have small numerical differences")
        print(f"   Max difference: {max_abs_diff:.6f}")
        print(f"   This is expected due to floating point precision differences")
        status = "ACCEPTABLE"
    elif correlation > 0.99:
        print(f"\n⚠️  WARNING: Outputs are correlated ({correlation:.6f}) but have noticeable differences")
        print(f"   Max difference: {max_abs_diff:.6f}")
        status = "WARNING"
    else:
        print(f"\n✗ FAIL: Outputs differ significantly!")
        print(f"   Correlation: {correlation:.6f}")
        print(f"   Max difference: {max_abs_diff:.6f}")
        status = "FAIL"
    
    # Show sample values
    print(f"\nSample values (first 10 elements of flattened output):")
    print(f"  PyTorch: {pytorch_flat[:10]}")
    print(f"  TFLite:  {tflite_flat[:10]}")
    print(f"  Diff:    {abs_diff.flatten()[:10]}")
    
    # Check operators
    print(f"\n{'='*70}")
    print("TFLite Operators")
    print(f"{'='*70}")
    ops = set()
    for op_details in interpreter._get_ops_details():
        ops.add(op_details['op_name'])
    
    for op in sorted(ops):
        print(f"  - {op}")
    
    # Check for problematic ops
    problematic_ops = ['GATHER_ND', 'GATHER', 'SCATTER_ND']
    found_problematic = [op for op in ops if op in problematic_ops]
    
    if found_problematic:
        print(f"\n⚠️  WARNING: Found problematic operators:")
        for op in found_problematic:
            print(f"  - {op}")
    else:
        print(f"\n✓ No problematic operators found")
    
    results = {
        'status': status,
        'max_abs_diff': max_abs_diff,
        'mean_abs_diff': mean_abs_diff,
        'max_rel_diff': max_rel_diff,
        'mean_rel_diff': mean_rel_diff,
        'correlation': correlation,
        'tflite_path': tflite_path,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Convert and verify minimal DistortionNet models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert and verify single block model
  python convert_and_verify_distortionnet.py --size single
  
  # Convert and verify minimal model
  python convert_and_verify_distortionnet.py --size minimal
  
  # Save to specific directory
  python convert_and_verify_distortionnet.py --size minimal --output_dir ~/work/UVQ/uvq/models/tflite_models/uvq1.5
        """
    )
    
    parser.add_argument(
        '--size',
        type=str,
        choices=['single', 'minimal', 'medium'],
        default='minimal',
        help='Which minimal model to convert and verify (default: minimal)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save TFLite model (default: temp file)'
    )
    
    args = parser.parse_args()
    
    result = convert_and_verify(args.size, args.output_dir)
    
    if result:
        print("\n" + "="*70)
        print(f"FINAL STATUS: {result['status']}")
        print("="*70)
    
    print("\nDone!")


if __name__ == '__main__':
    main()

