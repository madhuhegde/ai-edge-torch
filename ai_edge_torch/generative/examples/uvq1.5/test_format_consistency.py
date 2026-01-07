#!/usr/bin/env python3
"""Test that TFLite models use the correct [B, H, W, C] format.

This script verifies that:
1. TFLite models accept input in [B, H, W, C] format
2. TFLite models produce output in [B, H, W, C] format
3. The format is consistent with expectations
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

def test_model_format(size='minimal', tflite_dir='./debug_models'):
    """Test that a TFLite model uses the correct format.
    
    Args:
        size: Model size ('single', 'minimal', 'medium')
        tflite_dir: Directory containing TFLite models
    """
    print("\n" + "="*70)
    print(f"Testing Format: {size.upper()} DistortionNet")
    print("="*70)
    
    # Load TFLite model
    tflite_path = Path(tflite_dir) / f"distortion_net_{size}.tflite"
    if not tflite_path.exists():
        print(f"✗ Model not found: {tflite_path}")
        return False
    
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']
    output_shape = output_details[0]['shape']
    
    print(f"\nModel: {tflite_path.name}")
    print(f"Input shape:  {input_shape}")
    print(f"Output shape: {output_shape}")
    
    # Expected shapes in [B, H, W, C] format
    expected_shapes = {
        'single': {
            'input': [9, 360, 640, 3],
            'output_batch': 9,
            'output_channels': 24,  # Last dimension should be channels
        },
        'minimal': {
            'input': [9, 360, 640, 3],
            'output_batch': 9,
            'output_channels': 128,  # Last dimension should be channels
        },
        'medium': {
            'input': [9, 360, 640, 3],
            'output_batch': 9,
            'output_channels': 128,  # Last dimension should be channels
        },
    }
    
    expected = expected_shapes[size]
    
    # Check input format
    print(f"\n1. Checking input format...")
    if list(input_shape) == expected['input']:
        print(f"   ✓ Input shape matches expected [B, H, W, C]: {list(input_shape)}")
        input_ok = True
    else:
        print(f"   ✗ Input shape mismatch!")
        print(f"     Expected: {expected['input']}")
        print(f"     Got:      {list(input_shape)}")
        input_ok = False
    
    # Check output format
    print(f"\n2. Checking output format...")
    if output_shape[0] == expected['output_batch']:
        print(f"   ✓ Output batch size correct: {output_shape[0]}")
        batch_ok = True
    else:
        print(f"   ✗ Output batch size mismatch: expected {expected['output_batch']}, got {output_shape[0]}")
        batch_ok = False
    
    # For NHWC format, channels should be last dimension
    if output_shape[-1] == expected['output_channels']:
        print(f"   ✓ Output channels in last dimension (NHWC format): {output_shape[-1]}")
        channels_ok = True
    else:
        print(f"   ⚠️  Output channels not in last dimension")
        print(f"     Expected channels: {expected['output_channels']}")
        print(f"     Last dimension: {output_shape[-1]}")
        channels_ok = False
    
    # Run a test inference
    print(f"\n3. Running test inference...")
    try:
        test_input = np.random.randn(*input_shape).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"   ✓ Inference successful")
        print(f"   Input shape:  {test_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.6f}, {output.max():.6f}]")
        inference_ok = True
    except Exception as e:
        print(f"   ✗ Inference failed: {e}")
        inference_ok = False
    
    # Overall result
    print(f"\n4. Overall result:")
    all_ok = input_ok and batch_ok and channels_ok and inference_ok
    
    if all_ok:
        print(f"   ✅ PASSED: Model uses correct [B, H, W, C] format")
    else:
        print(f"   ✗ FAILED: Model format issues detected")
    
    return all_ok


def test_all_models(tflite_dir='./debug_models'):
    """Test all three model sizes."""
    print("\n" + "="*70)
    print("Testing TFLite Model Format Consistency")
    print("="*70)
    
    results = {}
    for size in ['single', 'minimal', 'medium']:
        try:
            results[size] = test_model_format(size, tflite_dir)
        except Exception as e:
            print(f"\n✗ Error testing {size} model: {e}")
            import traceback
            traceback.print_exc()
            results[size] = False
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    print(f"\n{'Model':<12} {'Status':<10}")
    print("-" * 25)
    
    for size, passed in results.items():
        status = "✅ PASS" if passed else "✗ FAIL"
        print(f"{size:<12} {status:<10}")
    
    all_passed = all(results.values())
    print("\n" + "="*70)
    if all_passed:
        print("✅ All models use correct [B, H, W, C] format!")
    else:
        print("✗ Some models have format issues")
    print("="*70)
    
    return all_passed


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test TFLite model format consistency'
    )
    parser.add_argument(
        '--size',
        type=str,
        choices=['single', 'minimal', 'medium'],
        help='Which model to test'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Test all models'
    )
    parser.add_argument(
        '--tflite_dir',
        type=str,
        default='./debug_models',
        help='Directory containing TFLite models'
    )
    
    args = parser.parse_args()
    
    if args.all:
        test_all_models(args.tflite_dir)
    elif args.size:
        test_model_format(args.size, args.tflite_dir)
    else:
        test_all_models(args.tflite_dir)

