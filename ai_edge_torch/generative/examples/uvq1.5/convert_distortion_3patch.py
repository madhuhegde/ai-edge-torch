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
Convert DistortionNet 3-patch model to TFLite format.

This script converts the DistortionNet model with batch_size=3 (processing 3 patches
at a time) to TFLite format.

Usage:
    python convert_distortion_3patch.py
"""

import os
from pathlib import Path

import torch
import ai_edge_torch

from uvq_models import create_distortion_net_3patch


def convert_distortion_net_3patch(output_dir):
    """Convert DistortionNet 3-patch model to TFLite.
    
    Args:
        output_dir: Directory to save the TFLite model
    
    Returns:
        Path to the saved TFLite model
    """
    print("\n" + "="*70)
    print("Converting DistortionNet 3-Patch Model to TFLite")
    print("="*70)
    
    # Create model
    model = create_distortion_net_3patch()
    
    # Create sample input: (3 patches, height=360, width=640, channels=3)
    # TensorFlow format: [B, H, W, C]
    # Values in [-1, 1] range
    sample_input = torch.randn(3, 360, 640, 3)
    
    print(f"Input shape: {sample_input.shape} (3 patches, TensorFlow format: B, H, W, C)")
    print(f"  Patch size: 360x640")
    print(f"  Input range: [{sample_input.min():.2f}, {sample_input.max():.2f}]")
    
    # Test forward pass
    with torch.no_grad():
        output = model(sample_input)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.2f}, {output.max():.2f}]")
    print(f"Expected output shape: [1, 8, 24, 128] (1 row of 3 patches)")
    
    print("\nConverting to TFLite...")
    
    # Convert to TFLite
    edge_model = ai_edge_torch.convert(
        model,
        (sample_input,)
    )
    
    # Save model
    output_file = os.path.join(output_dir, "distortion_net_3patch.tflite")
    edge_model.export(output_file)
    
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n✓ DistortionNet 3-patch saved to: {output_file}")
    print(f"  File size: {file_size:.2f} MB")
    
    return output_file


def main():
    # Output directory
    output_dir = os.path.join(os.path.dirname(__file__), "uvq1p5_tflite")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("DistortionNet 3-Patch Model to TFLite Conversion")
    print("="*70)
    print(f"Output directory: {output_dir}")
    
    try:
        output_file = convert_distortion_net_3patch(output_dir)
        
        print("\n" + "="*70)
        print("Conversion Summary")
        print("="*70)
        print(f"\n✓ Successfully converted model:")
        print(f"  - distortion_net_3patch: {output_file}")
        
    except Exception as e:
        print(f"\n✗ Failed to convert model: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

