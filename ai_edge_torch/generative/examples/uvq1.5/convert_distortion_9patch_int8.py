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
Convert DistortionNet 9-patch model to TFLite format with INT8 quantization.

This script converts the DistortionNet model with batch_size=9 (processing 9 patches
at a time) to TFLite format with INT8 quantization. The model outputs individual patch 
features [9, 8, 8, 128] without aggregation. The aggregation is handled in the 
application code using 4D operations.

Usage:
    python convert_distortion_9patch_int8.py
"""

import os
from pathlib import Path

import torch
import ai_edge_torch
from ai_edge_torch.generative.quantize import quant_recipes
from ai_edge_torch.generative.quantize import quant_attrs

from uvq_models import create_distortion_net_9patch


def convert_distortion_net_9patch_int8(output_dir):
    """Convert DistortionNet 9-patch model to TFLite with INT8 quantization.
    
    Args:
        output_dir: Directory to save the TFLite model
    
    Returns:
        Path to the saved TFLite model
    """
    print("\n" + "="*70)
    print("Converting DistortionNet 9-Patch Model to TFLite with INT8 Quantization")
    print("="*70)
    
    # Create model
    model = create_distortion_net_9patch()
    
    # Create sample input: (9 patches, height=360, width=640, channels=3)
    # TensorFlow format: [B, H, W, C]
    # Values in [-1, 1] range
    sample_input = torch.randn(9, 360, 640, 3)
    
    print(f"Input shape: {sample_input.shape} (9 patches, TensorFlow format: B, H, W, C)")
    print(f"  Patch size: 360x640")
    print(f"  Input range: [{sample_input.min():.2f}, {sample_input.max():.2f}]")
    
    # Test forward pass
    with torch.no_grad():
        output = model(sample_input)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.2f}, {output.max():.2f}]")
    print(f"Expected output shape: [9, 8, 8, 128] (9 individual patch features)")
    print(f"Note: Aggregation to [1, 24, 24, 128] is done in application code using 4D ops")
    
    print("\nConverting to TFLite with INT8 quantization...")
    print("Using dynamic_qi8_recipe (dynamic INT8 quantization)")
    
    # Create quantization config using dynamic INT8 recipe
    quant_config = quant_recipes.full_dynamic_recipe(
        mcfg=None,
        weight_dtype=quant_attrs.Dtype.INT8,
        granularity=quant_attrs.Granularity.CHANNELWISE
    )
    
    # Convert to TFLite with quantization
    edge_model = ai_edge_torch.convert(
        model,
        (sample_input,),
        quant_config=quant_config
    )
    
    # Save model
    output_file = os.path.join(output_dir, "distortion_net_9patch_int8.tflite")
    edge_model.export(output_file)
    
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n✓ DistortionNet 9-patch (INT8) saved to: {output_file}")
    print(f"  File size: {file_size:.2f} MB")
    
    # Compare with FLOAT32 version if it exists
    float32_file = os.path.join(output_dir, "distortion_net_9patch.tflite")
    if os.path.exists(float32_file):
        float32_size = os.path.getsize(float32_file) / (1024 * 1024)
        reduction = (1 - file_size / float32_size) * 100
        print(f"\nSize comparison:")
        print(f"  FLOAT32: {float32_size:.2f} MB")
        print(f"  INT8:    {file_size:.2f} MB")
        print(f"  Reduction: {reduction:.1f}%")
    
    return output_file


def main():
    # Output directory
    output_dir = os.path.join(os.path.dirname(__file__), "uvq1p5_tflite")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("DistortionNet 9-Patch Model to TFLite Conversion (INT8)")
    print("="*70)
    print(f"Output directory: {output_dir}")
    
    try:
        output_file = convert_distortion_net_9patch_int8(output_dir)
        
        print("\n" + "="*70)
        print("Conversion Summary")
        print("="*70)
        print(f"\n✓ Successfully converted model:")
        print(f"  - distortion_net_9patch_int8: {output_file}")
        
    except Exception as e:
        print(f"\n✗ Failed to convert model: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

