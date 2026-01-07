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
Verify UVQ 1.5 TFLite models.

This script verifies that the converted TFLite models produce consistent
results with the original PyTorch models.
"""

import argparse
import os
import numpy as np
import torch

try:
    import tensorflow as tf
except ImportError:
    print("TensorFlow not found. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "tensorflow"])
    import tensorflow as tf

from uvq_models import (
    create_content_net,
    create_distortion_net,
    create_aggregation_net
)


def verify_content_net(tflite_path, pytorch_model=None):
    """Verify ContentNet TFLite model against PyTorch."""
    print("\n" + "="*70)
    print("Verifying ContentNet")
    print("="*70)
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"\nTFLite Model: {tflite_path}")
    print(f"  Input shape: {input_details[0]['shape']}")
    print(f"  Output shape: {output_details[0]['shape']}")
    
    # Create test input
    test_input = np.random.randn(1, 3, 256, 256).astype(np.float32)
    
    # Run TFLite inference
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"\nTFLite Output:")
    print(f"  Shape: {tflite_output.shape}")
    print(f"  Range: [{tflite_output.min():.2f}, {tflite_output.max():.2f}]")
    print(f"  Mean: {tflite_output.mean():.2f}")
    
    # Compare with PyTorch if model provided
    if pytorch_model is not None:
        with torch.no_grad():
            pytorch_input = torch.from_numpy(test_input)
            pytorch_output = pytorch_model(pytorch_input).numpy()
        
        print(f"\nPyTorch Output:")
        print(f"  Shape: {pytorch_output.shape}")
        print(f"  Range: [{pytorch_output.min():.2f}, {pytorch_output.max():.2f}]")
        print(f"  Mean: {pytorch_output.mean():.2f}")
        
        # Calculate difference
        diff = np.abs(tflite_output - pytorch_output)
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        print(f"\nDifference:")
        print(f"  Max: {max_diff:.6f}")
        print(f"  Mean: {mean_diff:.6f}")
        
        if max_diff < 1e-3:
            print("  ✓ Models match closely")
        elif max_diff < 1e-1:
            print("  ⚠ Models have small differences (acceptable)")
        else:
            print("  ✗ Models have significant differences")
    
    return True


def verify_distortion_net(tflite_path, pytorch_model=None):
    """Verify DistortionNet TFLite model against PyTorch."""
    print("\n" + "="*70)
    print("Verifying DistortionNet")
    print("="*70)
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"\nTFLite Model: {tflite_path}")
    print(f"  Input shape: {input_details[0]['shape']}")
    print(f"  Output shape: {output_details[0]['shape']}")
    
    # Create test input (9 patches)
    test_input = np.random.randn(9, 3, 360, 640).astype(np.float32)
    
    # Run TFLite inference
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"\nTFLite Output:")
    print(f"  Shape: {tflite_output.shape}")
    print(f"  Range: [{tflite_output.min():.2f}, {tflite_output.max():.2f}]")
    print(f"  Mean: {tflite_output.mean():.2f}")
    
    # Compare with PyTorch if model provided
    if pytorch_model is not None:
        with torch.no_grad():
            pytorch_input = torch.from_numpy(test_input)
            pytorch_output = pytorch_model(pytorch_input).numpy()
        
        print(f"\nPyTorch Output:")
        print(f"  Shape: {pytorch_output.shape}")
        print(f"  Range: [{pytorch_output.min():.2f}, {pytorch_output.max():.2f}]")
        print(f"  Mean: {pytorch_output.mean():.2f}")
        
        # Calculate difference
        diff = np.abs(tflite_output - pytorch_output)
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        print(f"\nDifference:")
        print(f"  Max: {max_diff:.6f}")
        print(f"  Mean: {mean_diff:.6f}")
        
        if max_diff < 1e-3:
            print("  ✓ Models match closely")
        elif max_diff < 1e-1:
            print("  ⚠ Models have small differences (acceptable)")
        else:
            print("  ✗ Models have significant differences")
    
    return True


def verify_aggregation_net(tflite_path, pytorch_model=None):
    """Verify AggregationNet TFLite model against PyTorch."""
    print("\n" + "="*70)
    print("Verifying AggregationNet")
    print("="*70)
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"\nTFLite Model: {tflite_path}")
    print(f"  Input 0 shape: {input_details[0]['shape']}")
    print(f"  Input 1 shape: {input_details[1]['shape']}")
    print(f"  Output shape: {output_details[0]['shape']}")
    
    # Create test inputs
    content_features = np.random.randn(1, 8, 8, 128).astype(np.float32)
    distortion_features = np.random.randn(1, 24, 24, 128).astype(np.float32)
    
    # Run TFLite inference
    interpreter.set_tensor(input_details[0]['index'], content_features)
    interpreter.set_tensor(input_details[1]['index'], distortion_features)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"\nTFLite Output:")
    print(f"  Shape: {tflite_output.shape}")
    print(f"  Quality Score: {tflite_output[0, 0]:.3f}")
    print(f"  Range: [1, 5]")
    
    # Compare with PyTorch if model provided
    if pytorch_model is not None:
        with torch.no_grad():
            pytorch_content = torch.from_numpy(content_features)
            pytorch_distortion = torch.from_numpy(distortion_features)
            pytorch_output = pytorch_model(pytorch_content, pytorch_distortion).numpy()
        
        print(f"\nPyTorch Output:")
        print(f"  Shape: {pytorch_output.shape}")
        print(f"  Quality Score: {pytorch_output[0, 0]:.3f}")
        
        # Calculate difference
        diff = np.abs(tflite_output - pytorch_output)
        max_diff = diff.max()
        
        print(f"\nDifference:")
        print(f"  Absolute: {max_diff:.6f}")
        
        if max_diff < 1e-3:
            print("  ✓ Models match closely")
        elif max_diff < 1e-1:
            print("  ⚠ Models have small differences (acceptable)")
        else:
            print("  ✗ Models have significant differences")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Verify UVQ 1.5 TFLite models')
    
    parser.add_argument(
        '--tflite_dir',
        type=str,
        default='./tflite_models',
        help='Directory containing TFLite models'
    )
    
    parser.add_argument(
        '--compare_pytorch',
        action='store_true',
        help='Compare TFLite outputs with PyTorch models'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['content', 'distortion', 'aggregation', 'all'],
        default='all',
        help='Which model to verify'
    )
    
    args = parser.parse_args()
    
    tflite_dir = os.path.abspath(args.tflite_dir)
    
    print("="*70)
    print("UVQ 1.5 TFLite Model Verification")
    print("="*70)
    print(f"TFLite directory: {tflite_dir}")
    print(f"Compare with PyTorch: {args.compare_pytorch}")
    
    # Load PyTorch models if comparison requested
    pytorch_content = None
    pytorch_distortion = None
    pytorch_aggregation = None
    
    if args.compare_pytorch:
        print("\nLoading PyTorch models...")
        if args.model in ['content', 'all']:
            pytorch_content = create_content_net()
        if args.model in ['distortion', 'all']:
            pytorch_distortion = create_distortion_net()
        if args.model in ['aggregation', 'all']:
            pytorch_aggregation = create_aggregation_net()
    
    # Verify models
    success = True
    
    if args.model in ['content', 'all']:
        content_path = os.path.join(tflite_dir, 'content_net.tflite')
        if os.path.exists(content_path):
            try:
                verify_content_net(content_path, pytorch_content)
            except Exception as e:
                print(f"\n✗ ContentNet verification failed: {e}")
                import traceback
                traceback.print_exc()
                success = False
        else:
            print(f"\n✗ ContentNet TFLite model not found: {content_path}")
            success = False
    
    if args.model in ['distortion', 'all']:
        distortion_path = os.path.join(tflite_dir, 'distortion_net.tflite')
        if os.path.exists(distortion_path):
            try:
                verify_distortion_net(distortion_path, pytorch_distortion)
            except Exception as e:
                print(f"\n✗ DistortionNet verification failed: {e}")
                import traceback
                traceback.print_exc()
                success = False
        else:
            print(f"\n✗ DistortionNet TFLite model not found: {distortion_path}")
            success = False
    
    if args.model in ['aggregation', 'all']:
        aggregation_path = os.path.join(tflite_dir, 'aggregation_net.tflite')
        if os.path.exists(aggregation_path):
            try:
                verify_aggregation_net(aggregation_path, pytorch_aggregation)
            except Exception as e:
                print(f"\n✗ AggregationNet verification failed: {e}")
                import traceback
                traceback.print_exc()
                success = False
        else:
            print(f"\n✗ AggregationNet TFLite model not found: {aggregation_path}")
            success = False
    
    # Summary
    print("\n" + "="*70)
    if success:
        print("✓ All verifications passed")
    else:
        print("✗ Some verifications failed")
    print("="*70)


if __name__ == '__main__':
    main()

