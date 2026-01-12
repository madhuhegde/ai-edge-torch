#!/usr/bin/env python3
# Copyright 2025 The AI Edge Torch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Verify VideoSeal 0.0 TFLite Detector accuracy.

This script compares the TFLite detector output with the PyTorch reference
to verify conversion accuracy.

Usage:
    # Verify FLOAT32 detector
    python verify_detector_tflite.py --tflite_dir ./tflite_models
    
    # Verify INT8 detector
    python verify_detector_tflite.py --quantize int8 --tflite_dir ./tflite_models
    
    # Verify specific model file
    python verify_detector_tflite.py --detector_path ./tflite_models/videoseal00_detector_256.tflite
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import tensorflow as tf

from videoseal00_models import create_detector


def verify_detector(tflite_path, pytorch_model, num_tests=10, image_size=256):
    """Verify TFLite detector against PyTorch reference.
    
    Args:
        tflite_path: Path to TFLite model
        pytorch_model: PyTorch reference model
        num_tests: Number of random images to test
        image_size: Image size for testing
    
    Returns:
        Dictionary with verification metrics
    """
    print("\n" + "="*70)
    print("Verifying TFLite Detector")
    print("="*70)
    print(f"TFLite model: {tflite_path}")
    print(f"Number of tests: {num_tests}")
    print(f"Image size: {image_size}×{image_size}")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"\nTFLite Model Info:")
    print(f"  Input shape: {input_details[0]['shape']}")
    print(f"  Output shape: {output_details[0]['shape']}")
    print(f"  Input dtype: {input_details[0]['dtype']}")
    print(f"  Output dtype: {output_details[0]['dtype']}")
    
    # Collect metrics
    mae_list = []
    bit_accuracy_list = []
    detection_diff_list = []
    
    print(f"\nRunning {num_tests} verification tests...")
    
    for i in range(num_tests):
        # Generate random test image
        img_torch = torch.rand(1, 3, image_size, image_size)
        img_np = img_torch.numpy().astype(np.float32)
        
        # PyTorch inference
        with torch.no_grad():
            output_torch = pytorch_model(img_torch)
        
        # TFLite inference
        interpreter.set_tensor(input_details[0]['index'], img_np)
        interpreter.invoke()
        output_tflite = interpreter.get_tensor(output_details[0]['index'])
        
        # Convert to numpy for comparison
        output_torch_np = output_torch.numpy()
        
        # Calculate metrics
        mae = np.mean(np.abs(output_torch_np - output_tflite))
        mae_list.append(mae)
        
        # Detection bit (channel 0)
        detection_diff = np.abs(output_torch_np[0, 0] - output_tflite[0, 0])
        detection_diff_list.append(detection_diff)
        
        # Message bits (channels 1-96)
        msg_torch = (output_torch_np[0, 1:] > 0).astype(int)
        msg_tflite = (output_tflite[0, 1:] > 0).astype(int)
        bit_accuracy = np.mean(msg_torch == msg_tflite) * 100
        bit_accuracy_list.append(bit_accuracy)
        
        if (i + 1) % max(1, num_tests // 10) == 0:
            print(f"  Test {i+1}/{num_tests}: MAE={mae:.6f}, Bit Acc={bit_accuracy:.2f}%")
    
    # Calculate summary statistics
    results = {
        'mae_mean': np.mean(mae_list),
        'mae_std': np.std(mae_list),
        'mae_max': np.max(mae_list),
        'bit_accuracy_mean': np.mean(bit_accuracy_list),
        'bit_accuracy_std': np.std(bit_accuracy_list),
        'bit_accuracy_min': np.min(bit_accuracy_list),
        'detection_diff_mean': np.mean(detection_diff_list),
        'detection_diff_max': np.max(detection_diff_list),
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Verify VideoSeal 0.0 TFLite Detector accuracy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify FLOAT32 detector
  python verify_detector_tflite.py --tflite_dir ./tflite_models
  
  # Verify INT8 detector
  python verify_detector_tflite.py --quantize int8 --tflite_dir ./tflite_models
  
  # Verify specific model file
  python verify_detector_tflite.py --detector_path ./tflite_models/videoseal00_detector_256.tflite
  
  # Run more tests for better statistics
  python verify_detector_tflite.py --num_tests 100 --tflite_dir ./tflite_models

Expected Results:
  FLOAT32:
    • MAE: <1e-6 (near-identical to PyTorch)
    • Bit Accuracy: 100% (perfect match)
  
  INT8:
    • MAE: <5e-2 (small differences due to quantization)
    • Bit Accuracy: 95-98% (minimal degradation)
        """
    )
    
    parser.add_argument(
        '--tflite_dir',
        type=str,
        default='./videoseal00_tflite',
        help='Directory containing TFLite models (default: ./videoseal00_tflite)'
    )
    
    parser.add_argument(
        '--detector_path',
        type=str,
        default=None,
        help='Path to specific TFLite detector model (overrides --tflite_dir)'
    )
    
    parser.add_argument(
        '--quantize',
        type=str,
        choices=['int8', 'fp16'],
        default=None,
        help='Quantization type to verify (default: None for FLOAT32)'
    )
    
    parser.add_argument(
        '--image_size',
        type=int,
        default=256,
        help='Image size for testing (default: 256)'
    )
    
    parser.add_argument(
        '--num_tests',
        type=int,
        default=10,
        help='Number of random images to test (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Determine TFLite model path
    if args.detector_path:
        tflite_path = Path(args.detector_path)
    else:
        quant_suffix = f"_{args.quantize}" if args.quantize else ""
        tflite_filename = f"videoseal00_detector_{args.image_size}{quant_suffix}.tflite"
        tflite_path = Path(args.tflite_dir) / tflite_filename
    
    if not tflite_path.exists():
        print(f"✗ TFLite model not found: {tflite_path}")
        print(f"\nPlease run convert_detector_to_tflite.py first:")
        if args.quantize:
            print(f"  python convert_detector_to_tflite.py --quantize {args.quantize} --output_dir {args.tflite_dir}")
        else:
            print(f"  python convert_detector_to_tflite.py --output_dir {args.tflite_dir}")
        return 1
    
    # Load PyTorch reference model
    print("\n" + "="*70)
    print("Loading PyTorch Reference Model")
    print("="*70)
    
    pytorch_model = create_detector("videoseal_0.0")
    print("✓ Loaded PyTorch VideoSeal 0.0 detector")
    
    # Verify TFLite model
    results = verify_detector(tflite_path, pytorch_model, args.num_tests, args.image_size)
    
    # Print results
    print("\n" + "="*70)
    print("Verification Results")
    print("="*70)
    
    quant_name = args.quantize.upper() if args.quantize else "FLOAT32"
    print(f"\nQuantization: {quant_name}")
    print(f"Model: {tflite_path.name}")
    print(f"Size: {tflite_path.stat().st_size / (1024 * 1024):.2f} MB")
    
    print(f"\nMean Absolute Error (MAE):")
    print(f"  Mean: {results['mae_mean']:.6f}")
    print(f"  Std:  {results['mae_std']:.6f}")
    print(f"  Max:  {results['mae_max']:.6f}")
    
    print(f"\nBit Accuracy (96-bit message):")
    print(f"  Mean: {results['bit_accuracy_mean']:.2f}%")
    print(f"  Std:  {results['bit_accuracy_std']:.2f}%")
    print(f"  Min:  {results['bit_accuracy_min']:.2f}%")
    
    print(f"\nDetection Confidence Difference:")
    print(f"  Mean: {results['detection_diff_mean']:.6f}")
    print(f"  Max:  {results['detection_diff_max']:.6f}")
    
    # Determine pass/fail
    print("\n" + "="*70)
    print("Verification Status")
    print("="*70)
    
    if args.quantize == 'int8':
        # INT8 thresholds
        mae_threshold = 0.05
        bit_acc_threshold = 95.0
    else:
        # FLOAT32/FP16 thresholds
        mae_threshold = 1e-5
        bit_acc_threshold = 99.5
    
    mae_pass = results['mae_mean'] < mae_threshold
    bit_acc_pass = results['bit_accuracy_mean'] >= bit_acc_threshold
    
    print(f"\nMAE Check: {'✓ PASS' if mae_pass else '✗ FAIL'}")
    print(f"  Threshold: <{mae_threshold}")
    print(f"  Actual: {results['mae_mean']:.6f}")
    
    print(f"\nBit Accuracy Check: {'✓ PASS' if bit_acc_pass else '✗ FAIL'}")
    print(f"  Threshold: ≥{bit_acc_threshold}%")
    print(f"  Actual: {results['bit_accuracy_mean']:.2f}%")
    
    if mae_pass and bit_acc_pass:
        print(f"\n✓ Verification PASSED - TFLite model is accurate!")
        return 0
    else:
        print(f"\n✗ Verification FAILED - TFLite model has accuracy issues")
        return 1


if __name__ == '__main__':
    exit(main())
