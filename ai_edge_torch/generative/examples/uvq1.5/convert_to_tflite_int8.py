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
Convert UVQ 1.5 models to TFLite format with INT8 quantization.

This script converts the three UVQ 1.5 PyTorch models (ContentNet, DistortionNet,
and AggregationNet) to TFLite format with dynamic INT8 quantization using the
dynamic_qi8_recipe from ai_edge_torch.

Usage:
    # Convert all models with INT8 quantization
    python convert_to_tflite_int8.py --output_dir ~/work/UVQ/uvq/models/tflite_models/uvq1.5
    
    # Convert specific model
    python convert_to_tflite_int8.py --model content --output_dir ~/work/UVQ/uvq/models/tflite_models/uvq1.5
"""

import argparse
import os
from pathlib import Path

import torch
import ai_edge_torch
from ai_edge_torch.generative.quantize import quant_recipes
from ai_edge_torch.generative.quantize import quant_attrs

from uvq_models import (
    create_content_net,
    create_distortion_net,
    create_aggregation_net,
)


def convert_content_net_int8(output_dir, model_path=None):
    """Convert ContentNet to TFLite with INT8 quantization.
    
    Args:
        output_dir: Directory to save the TFLite model
        model_path: Path to PyTorch model checkpoint
    
    Returns:
        Path to the saved TFLite model
    """
    print("\n" + "="*70)
    print("Converting ContentNet to TFLite with INT8 Quantization")
    print("="*70)
    
    # Create model
    model = create_content_net(model_path)
    
    # Create sample input: (batch=1, height=256, width=256, channels=3)
    # TensorFlow format: [B, H, W, C]
    sample_input = torch.randn(1, 256, 256, 3)
    
    print(f"Input shape: {sample_input.shape} (TensorFlow format: B, H, W, C)")
    print(f"Input range: [{sample_input.min():.2f}, {sample_input.max():.2f}]")
    
    # Test forward pass
    with torch.no_grad():
        output = model(sample_input)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.2f}, {output.max():.2f}]")
    
    print("\nConverting to TFLite with INT8 quantization...")
    print("Using dynamic_qi8_recipe (dynamic INT8 quantization)")
    
    # Create quantization config using dynamic INT8 recipe
    # This applies dynamic range quantization with INT8 weights
    quant_config = quant_recipes.full_dynamic_recipe(
        mcfg=None,  # No model config needed for non-transformer models
        weight_dtype=quant_attrs.Dtype.INT8,
        granularity=quant_attrs.Granularity.CHANNELWISE
    )
    
    # Convert to TFLite with quantization
    edge_model = ai_edge_torch.convert(
        model,
        (sample_input,),
        quant_config=quant_config
    )
    
    # Save model with _int8 suffix
    output_file = os.path.join(output_dir, "content_net_int8.tflite")
    edge_model.export(output_file)
    
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n✓ ContentNet (INT8) saved to: {output_file}")
    print(f"  File size: {file_size:.2f} MB")
    
    return output_file


def convert_distortion_net_int8(output_dir, model_path=None):
    """Convert DistortionNet to TFLite with INT8 quantization.
    
    Args:
        output_dir: Directory to save the TFLite model
        model_path: Path to PyTorch model checkpoint
    
    Returns:
        Path to the saved TFLite model
    """
    print("\n" + "="*70)
    print("Converting DistortionNet to TFLite with INT8 Quantization")
    print("="*70)
    
    # Create model
    model = create_distortion_net(model_path)
    
    # Create sample input: (9 patches, height=360, width=640, channels=3)
    # TensorFlow format: [B, H, W, C]
    sample_input = torch.randn(9, 360, 640, 3)
    
    print(f"Input shape: {sample_input.shape} (9 patches per frame, TensorFlow format: B, H, W, C)")
    print(f"  Patch size: 360x640")
    print(f"  Input range: [{sample_input.min():.2f}, {sample_input.max():.2f}]")
    
    # Test forward pass
    with torch.no_grad():
        output = model(sample_input)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.2f}, {output.max():.2f}]")
    
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
    
    # Save model with _int8 suffix
    output_file = os.path.join(output_dir, "distortion_net_int8.tflite")
    edge_model.export(output_file)
    
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n✓ DistortionNet (INT8) saved to: {output_file}")
    print(f"  File size: {file_size:.2f} MB")
    
    return output_file


def convert_aggregation_net_int8(output_dir, model_path=None):
    """Convert AggregationNet to TFLite with INT8 quantization.
    
    Args:
        output_dir: Directory to save the TFLite model
        model_path: Path to PyTorch model checkpoint
    
    Returns:
        Path to the saved TFLite model
    """
    print("\n" + "="*70)
    print("Converting AggregationNet to TFLite with INT8 Quantization")
    print("="*70)
    
    # Create model
    model = create_aggregation_net(model_path)
    
    # Create sample inputs
    content_features = torch.randn(1, 8, 8, 128)
    distortion_features = torch.randn(1, 24, 24, 128)
    
    print(f"Content features shape: {content_features.shape}")
    print(f"Distortion features shape: {distortion_features.shape}")
    
    # Test forward pass
    with torch.no_grad():
        output = model(content_features, distortion_features)
    print(f"Output shape: {output.shape}")
    print(f"Quality score: {output.item():.3f} (range: [1, 5])")
    
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
        (content_features, distortion_features),
        quant_config=quant_config
    )
    
    # Save model with _int8 suffix
    output_file = os.path.join(output_dir, "aggregation_net_int8.tflite")
    edge_model.export(output_file)
    
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n✓ AggregationNet (INT8) saved to: {output_file}")
    print(f"  File size: {file_size:.2f} MB")
    
    return output_file


def convert_all_models_int8(output_dir):
    """Convert all UVQ 1.5 models to TFLite with INT8 quantization.
    
    Args:
        output_dir: Directory to save the TFLite models
    
    Returns:
        Dictionary of model names to file paths
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("UVQ 1.5 PyTorch to TFLite Conversion (INT8 Quantization)")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print(f"Quantization: Dynamic INT8 (dynamic_qi8_recipe)")
    print(f"  - Weight dtype: INT8")
    print(f"  - Granularity: CHANNELWISE")
    
    models = {}
    
    # Convert each model
    try:
        models['content_net'] = convert_content_net_int8(output_dir)
    except Exception as e:
        print(f"\n✗ Failed to convert ContentNet: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        models['distortion_net'] = convert_distortion_net_int8(output_dir)
    except Exception as e:
        print(f"\n✗ Failed to convert DistortionNet: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        models['aggregation_net'] = convert_aggregation_net_int8(output_dir)
    except Exception as e:
        print(f"\n✗ Failed to convert AggregationNet: {e}")
        import traceback
        traceback.print_exc()
    
    # Print summary
    print("\n" + "="*70)
    print("Conversion Summary")
    print("="*70)
    
    if models:
        print(f"\n✓ Successfully converted {len(models)}/3 models with INT8 quantization:")
        for name, path in models.items():
            size = os.path.getsize(path) / (1024 * 1024)
            print(f"  - {name}: {path} ({size:.2f} MB)")
        
        # Compare with float32 versions if they exist
        print("\nSize comparison with FLOAT32 versions:")
        for name, path in models.items():
            int8_size = os.path.getsize(path) / (1024 * 1024)
            float32_path = path.replace('_int8.tflite', '.tflite')
            if os.path.exists(float32_path):
                float32_size = os.path.getsize(float32_path) / (1024 * 1024)
                reduction = (1 - int8_size / float32_size) * 100
                print(f"  - {name}: {float32_size:.2f} MB → {int8_size:.2f} MB ({reduction:.1f}% reduction)")
            else:
                print(f"  - {name}: {int8_size:.2f} MB (FLOAT32 version not found)")
    else:
        print("\n✗ No models were successfully converted")
    
    return models


def main():
    parser = argparse.ArgumentParser(
        description='Convert UVQ 1.5 models to TFLite format with INT8 quantization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all models with INT8 quantization
  python convert_to_tflite_int8.py --output_dir ~/work/UVQ/uvq/models/tflite_models/uvq1.5
  
  # Convert specific model
  python convert_to_tflite_int8.py --model content --output_dir ~/work/UVQ/uvq/models/tflite_models/uvq1.5

Note:
  This script uses the dynamic_qi8_recipe from ai_edge_torch, which applies:
  - Dynamic range quantization
  - INT8 weights
  - Channelwise granularity
  
  The output files will have the _int8.tflite suffix.
        """
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='~/work/UVQ/uvq/models/tflite_models/uvq1.5',
        help='Directory to save TFLite models (default: ~/work/UVQ/uvq/models/tflite_models/uvq1.5)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['content', 'distortion', 'aggregation', 'all'],
        default='all',
        help='Which model to convert (default: all)'
    )
    
    parser.add_argument(
        '--content_path',
        type=str,
        default=None,
        help='Path to ContentNet checkpoint (default: ~/work/models/UVQ/uvq1.5/content_net.pth)'
    )
    
    parser.add_argument(
        '--distortion_path',
        type=str,
        default=None,
        help='Path to DistortionNet checkpoint (default: ~/work/models/UVQ/uvq1.5/distortion_net.pth)'
    )
    
    parser.add_argument(
        '--aggregation_path',
        type=str,
        default=None,
        help='Path to AggregationNet checkpoint (default: ~/work/models/UVQ/uvq1.5/aggregation_net.pth)'
    )
    
    args = parser.parse_args()
    
    # Expand user path
    output_dir = os.path.expanduser(args.output_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert models
    if args.model == 'all':
        convert_all_models_int8(output_dir)
    elif args.model == 'content':
        convert_content_net_int8(output_dir, model_path=args.content_path)
    elif args.model == 'distortion':
        convert_distortion_net_int8(output_dir, model_path=args.distortion_path)
    elif args.model == 'aggregation':
        convert_aggregation_net_int8(output_dir, model_path=args.aggregation_path)


if __name__ == '__main__':
    main()

