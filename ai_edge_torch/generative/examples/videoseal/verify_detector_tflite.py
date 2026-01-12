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
Verify VideoSeal Detector TFLite model against PyTorch reference.

This script tests the TFLite Detector model to ensure it produces similar results
to the original PyTorch model.
"""

import argparse
import os
import numpy as np
import torch
import tensorflow as tf

from videoseal_models import create_detector


def load_tflite_model(model_path):
    """Load a TFLite model."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def run_tflite_detector(interpreter, img):
    """Run TFLite detector inference."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set input
    interpreter.set_tensor(input_details[0]['index'], img)
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    output = interpreter.get_tensor(output_details[0]['index'])
    return output


def calculate_metrics(pytorch_output, tflite_output):
    """Calculate comparison metrics between PyTorch and TFLite outputs."""
    # Convert to numpy if needed
    if isinstance(pytorch_output, torch.Tensor):
        pytorch_output = pytorch_output.cpu().numpy()
    
    # Calculate metrics
    mse = np.mean((pytorch_output - tflite_output) ** 2)
    mae = np.mean(np.abs(pytorch_output - tflite_output))
    max_diff = np.max(np.abs(pytorch_output - tflite_output))
    
    # Calculate relative error
    rel_error = np.mean(np.abs(pytorch_output - tflite_output) / (np.abs(pytorch_output) + 1e-8))
    
    return {
        'mse': mse,
        'mae': mae,
        'max_diff': max_diff,
        'rel_error': rel_error,
    }


def verify_detector(tflite_path, model_name="videoseal", image_size=256):
    """Verify detector TFLite model against PyTorch reference."""
    print("\n" + "="*70)
    print(f"Verifying Detector: {os.path.basename(tflite_path)}")
    print("="*70)
    
    # Load TFLite model
    print("Loading TFLite model...")
    tflite_model = load_tflite_model(tflite_path)
    
    # Load PyTorch model
    print("Loading PyTorch model...")
    pytorch_model = create_detector(model_name=model_name, simple=True)
    pytorch_model.eval()
    
    # Create test input
    print("\nCreating test input...")
    img = torch.rand(1, 3, image_size, image_size)
    
    print(f"Input image shape: {img.shape}")
    
    # Run PyTorch inference
    print("\nRunning PyTorch inference...")
    with torch.no_grad():
        pytorch_output = pytorch_model(img)
    
    print(f"PyTorch output shape: {pytorch_output.shape}")
    print(f"PyTorch output range: [{pytorch_output.min():.4f}, {pytorch_output.max():.4f}]")
    
    # Extract message
    pytorch_msg = (pytorch_output[0, 1:] > 0).float()
    print(f"PyTorch detected message (first 32 bits): {pytorch_msg[:32].numpy()}")
    
    # Run TFLite inference
    print("\nRunning TFLite inference...")
    img_np = img.numpy()
    tflite_output = run_tflite_detector(tflite_model, img_np)
    
    print(f"TFLite output shape: {tflite_output.shape}")
    print(f"TFLite output range: [{tflite_output.min():.4f}, {tflite_output.max():.4f}]")
    
    # Extract message
    tflite_msg = (tflite_output[0, 1:] > 0).astype(np.float32)
    print(f"TFLite detected message (first 32 bits): {tflite_msg[:32]}")
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(pytorch_output, tflite_output)
    
    print("\n" + "-"*70)
    print("Comparison Metrics:")
    print("-"*70)
    print(f"  MSE (Mean Squared Error):     {metrics['mse']:.6e}")
    print(f"  MAE (Mean Absolute Error):    {metrics['mae']:.6e}")
    print(f"  Max Absolute Difference:      {metrics['max_diff']:.6e}")
    print(f"  Relative Error:               {metrics['rel_error']:.6e}")
    
    # Calculate bit accuracy
    bit_accuracy = np.mean(pytorch_msg.numpy() == tflite_msg) * 100
    print(f"  Bit Accuracy:                 {bit_accuracy:.2f}%")
    
    # Determine if verification passed
    passed = metrics['mae'] < 1e-2 and bit_accuracy > 95.0
    
    print("\n" + "="*70)
    if passed:
        print("✓ VERIFICATION PASSED")
        print("  TFLite model produces similar results to PyTorch reference")
    else:
        print("✗ VERIFICATION FAILED")
        print("  TFLite model outputs differ significantly from PyTorch")
    print("="*70)
    
    return passed, metrics


def main():
    parser = argparse.ArgumentParser(
        description='Verify VideoSeal Detector TFLite model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--detector_path',
        type=str,
        help='Path to detector TFLite model'
    )
    
    parser.add_argument(
        '--model_name',
        type=str,
        default='videoseal',
        choices=['videoseal', 'pixelseal', 'chunkyseal'],
        help='VideoSeal model variant (default: videoseal)'
    )
    
    parser.add_argument(
        '--image_size',
        type=int,
        default=256,
        help='Image size used during conversion (default: 256)'
    )
    
    parser.add_argument(
        '--tflite_dir',
        type=str,
        default='./videoseal_tflite',
        help='Directory containing TFLite models (default: ./videoseal_tflite)'
    )
    
    args = parser.parse_args()
    
    # Auto-detect model path if not provided
    if not args.detector_path:
        args.detector_path = os.path.join(
            args.tflite_dir,
            f"videoseal_detector_{args.model_name}_{args.image_size}.tflite"
        )
    
    # Verify detector
    if os.path.exists(args.detector_path):
        passed, metrics = verify_detector(
            args.detector_path,
            args.model_name,
            args.image_size
        )
        
        if passed:
            print("\n✓ Detector model verified successfully!")
            return 0
        else:
            print("\n✗ Detector model failed verification")
            return 1
    else:
        print(f"\n✗ Detector model not found: {args.detector_path}")
        print("\nPlease run convert_detector_to_tflite.py first:")
        print(f"  python convert_detector_to_tflite.py --output_dir {args.tflite_dir}")
        return 1


if __name__ == '__main__':
    exit(main())

