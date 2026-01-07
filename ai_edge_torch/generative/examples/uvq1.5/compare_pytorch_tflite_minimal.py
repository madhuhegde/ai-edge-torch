#!/usr/bin/env python3
"""Compare PyTorch and TFLite outputs for minimal DistortionNet models.

This script verifies that the TFLite models produce the same outputs as the
PyTorch models after converting to [B, H, W, C] format.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import tensorflow as tf

# Add UVQ source path
UVQ_SOURCE_PATH = Path.home() / "work" / "UVQ" / "uvq"
sys.path.insert(0, str(UVQ_SOURCE_PATH))

from convert_minimal_distortionnet import DistortionNetMinimalWrapper


def compare_models(size='minimal', tflite_dir='./debug_models'):
    """Compare PyTorch and TFLite model outputs.
    
    Args:
        size: Model size ('single', 'minimal', 'medium')
        tflite_dir: Directory containing TFLite models
    
    Returns:
        Dictionary with comparison results
    """
    print("\n" + "="*70)
    print(f"Comparing PyTorch vs TFLite: {size.upper()} DistortionNet")
    print("="*70)
    
    # Create PyTorch model
    print("\n1. Loading PyTorch model...")
    pytorch_model = DistortionNetMinimalWrapper(size)
    pytorch_model.eval()
    
    # Load TFLite model
    print("2. Loading TFLite model...")
    tflite_path = Path(tflite_dir) / f"distortion_net_{size}.tflite"
    if not tflite_path.exists():
        raise FileNotFoundError(f"TFLite model not found: {tflite_path}")
    
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"   TFLite input shape: {input_details[0]['shape']}")
    print(f"   TFLite output shape: {output_details[0]['shape']}")
    
    # Create test input in TensorFlow format [B, H, W, C]
    # All models now accept 3 RGB channels
    print("\n3. Creating test input...")
    test_input_np = np.random.randn(9, 360, 640, 3).astype(np.float32)
    
    print(f"   Input shape: {test_input_np.shape} (TensorFlow format: B, H, W, C)")
    print(f"   Input range: [{test_input_np.min():.4f}, {test_input_np.max():.4f}]")
    
    # Run PyTorch inference
    print("\n4. Running PyTorch inference...")
    test_input_torch = torch.from_numpy(test_input_np)
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input_torch)
    pytorch_output_np = pytorch_output.numpy()
    
    print(f"   PyTorch output shape: {pytorch_output_np.shape}")
    print(f"   PyTorch output range: [{pytorch_output_np.min():.4f}, {pytorch_output_np.max():.4f}]")
    
    # Run TFLite inference
    print("\n5. Running TFLite inference...")
    interpreter.set_tensor(input_details[0]['index'], test_input_np)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"   TFLite output shape: {tflite_output.shape}")
    print(f"   TFLite output range: [{tflite_output.min():.4f}, {tflite_output.max():.4f}]")
    
    # Compare outputs
    print("\n6. Comparing outputs...")
    
    # Check shapes match
    if pytorch_output_np.shape != tflite_output.shape:
        print(f"   ✗ Shape mismatch!")
        print(f"     PyTorch: {pytorch_output_np.shape}")
        print(f"     TFLite:  {tflite_output.shape}")
        return None
    
    print(f"   ✓ Shapes match: {pytorch_output_np.shape}")
    
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
    
    # Print statistics
    print(f"\n   Absolute Difference:")
    print(f"     Max:  {max_abs_diff:.6f}")
    print(f"     Mean: {mean_abs_diff:.6f}")
    print(f"     Std:  {np.std(abs_diff):.6f}")
    
    print(f"\n   Relative Difference:")
    print(f"     Max:  {max_rel_diff:.6f} ({max_rel_diff*100:.2f}%)")
    print(f"     Mean: {mean_rel_diff:.6f} ({mean_rel_diff*100:.2f}%)")
    
    print(f"\n   Correlation: {correlation:.8f}")
    
    # Determine pass/fail
    print("\n7. Evaluation:")
    
    # Thresholds
    CORRELATION_THRESHOLD = 0.99
    MAX_ABS_DIFF_THRESHOLD = 1e-3
    MEAN_ABS_DIFF_THRESHOLD = 1e-4
    
    passed = True
    
    if correlation < CORRELATION_THRESHOLD:
        print(f"   ⚠️  Correlation ({correlation:.6f}) < {CORRELATION_THRESHOLD}")
        passed = False
    else:
        print(f"   ✓ Correlation ({correlation:.6f}) >= {CORRELATION_THRESHOLD}")
    
    if max_abs_diff > MAX_ABS_DIFF_THRESHOLD:
        print(f"   ⚠️  Max abs diff ({max_abs_diff:.6f}) > {MAX_ABS_DIFF_THRESHOLD}")
        passed = False
    else:
        print(f"   ✓ Max abs diff ({max_abs_diff:.6f}) <= {MAX_ABS_DIFF_THRESHOLD}")
    
    if mean_abs_diff > MEAN_ABS_DIFF_THRESHOLD:
        print(f"   ⚠️  Mean abs diff ({mean_abs_diff:.6f}) > {MEAN_ABS_DIFF_THRESHOLD}")
        passed = False
    else:
        print(f"   ✓ Mean abs diff ({mean_abs_diff:.6f}) <= {MEAN_ABS_DIFF_THRESHOLD}")
    
    if passed:
        print(f"\n   ✅ PASSED: PyTorch and TFLite outputs match!")
    else:
        print(f"\n   ⚠️  WARNING: Outputs differ more than expected")
        print(f"      This may be acceptable depending on use case.")
    
    # Return results
    return {
        'size': size,
        'passed': passed,
        'correlation': correlation,
        'max_abs_diff': max_abs_diff,
        'mean_abs_diff': mean_abs_diff,
        'max_rel_diff': max_rel_diff,
        'mean_rel_diff': mean_rel_diff,
        'pytorch_shape': pytorch_output_np.shape,
        'tflite_shape': tflite_output.shape,
    }


def compare_all_models(tflite_dir='./debug_models'):
    """Compare all three model sizes.
    
    Args:
        tflite_dir: Directory containing TFLite models
    """
    print("\n" + "="*70)
    print("Comparing All Minimal DistortionNet Models")
    print("="*70)
    
    results = {}
    
    for size in ['single', 'minimal', 'medium']:
        try:
            result = compare_models(size, tflite_dir)
            if result:
                results[size] = result
        except Exception as e:
            print(f"\n✗ Error comparing {size} model: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    if results:
        print(f"\n{'Model':<12} {'Status':<10} {'Correlation':<12} {'Max Abs Diff':<15} {'Mean Abs Diff':<15}")
        print("-" * 70)
        
        for size, result in results.items():
            status = "✅ PASS" if result['passed'] else "⚠️  WARN"
            print(f"{size:<12} {status:<10} {result['correlation']:.6f}     "
                  f"{result['max_abs_diff']:.6e}      {result['mean_abs_diff']:.6e}")
        
        # Overall status
        all_passed = all(r['passed'] for r in results.values())
        print("\n" + "="*70)
        if all_passed:
            print("✅ All models PASSED: PyTorch and TFLite outputs match!")
        else:
            print("⚠️  Some models have differences larger than thresholds")
            print("   This may be acceptable depending on your use case.")
        print("="*70)
    else:
        print("\n✗ No models were successfully compared")


def main():
    parser = argparse.ArgumentParser(
        description='Compare PyTorch and TFLite outputs for minimal DistortionNet models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare single model
  python compare_pytorch_tflite_minimal.py --size single
  
  # Compare all models
  python compare_pytorch_tflite_minimal.py --all
  
  # Specify TFLite directory
  python compare_pytorch_tflite_minimal.py --all --tflite_dir ./my_models
        """
    )
    
    parser.add_argument(
        '--size',
        type=str,
        choices=['single', 'minimal', 'medium'],
        help='Which model to compare'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Compare all models'
    )
    
    parser.add_argument(
        '--tflite_dir',
        type=str,
        default='./debug_models',
        help='Directory containing TFLite models (default: ./debug_models)'
    )
    
    args = parser.parse_args()
    
    if args.all:
        compare_all_models(args.tflite_dir)
    elif args.size:
        compare_models(args.size, args.tflite_dir)
    else:
        print("Error: Please specify --size or --all")
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()

