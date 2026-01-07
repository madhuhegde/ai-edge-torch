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
Convert UVQ 1.5 models to TFLite format.

This script converts the three UVQ 1.5 PyTorch models (ContentNet, DistortionNet,
and AggregationNet) to TFLite format for deployment on mobile and edge devices.

Usage:
    # Convert all models
    python convert_to_tflite.py --output_dir ./tflite_models
    
    # Convert specific model
    python convert_to_tflite.py --model content --output_dir ./tflite_models
    
    # Convert with quantization
    python convert_to_tflite.py --quantize --output_dir ./tflite_models
"""

import argparse
import os
from pathlib import Path

import torch
import ai_edge_torch

from uvq_models import (
    create_content_net,
    create_distortion_net,
    create_aggregation_net,
    create_uvq1p5_full
)


def convert_content_net(output_dir, model_path=None, quantize=False):
    """Convert ContentNet to TFLite.
    
    Args:
        output_dir: Directory to save the TFLite model
        model_path: Path to PyTorch model checkpoint
        quantize: Whether to apply quantization
    
    Returns:
        Path to the saved TFLite model
    """
    print("\n" + "="*70)
    print("Converting ContentNet to TFLite")
    print("="*70)
    
    # Create model
    model = create_content_net(model_path)
    
    # Create sample input: (batch=1, height=256, width=256, channels=3)
    # TensorFlow format: [B, H, W, C]
    # Values in [-1, 1] range
    sample_input = torch.randn(1, 256, 256, 3)
    
    print(f"Input shape: {sample_input.shape} (TensorFlow format: B, H, W, C)")
    print(f"Input range: [{sample_input.min():.2f}, {sample_input.max():.2f}]")
    
    # Test forward pass
    with torch.no_grad():
        output = model(sample_input)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.2f}, {output.max():.2f}]")
    
    print("\nConverting to TFLite...")
    
    # Convert to TFLite
    edge_model = ai_edge_torch.convert(
        model,
        (sample_input,)
    )
    
    # Save model
    output_file = os.path.join(output_dir, "content_net.tflite")
    edge_model.export(output_file)
    
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n✓ ContentNet saved to: {output_file}")
    print(f"  File size: {file_size:.2f} MB")
    
    return output_file


def convert_distortion_net(output_dir, model_path=None, quantize=False):
    """Convert DistortionNet to TFLite.
    
    Args:
        output_dir: Directory to save the TFLite model
        model_path: Path to PyTorch model checkpoint
        quantize: Whether to apply quantization
    
    Returns:
        Path to the saved TFLite model
    """
    print("\n" + "="*70)
    print("Converting DistortionNet to TFLite")
    print("="*70)
    
    # Create model
    model = create_distortion_net(model_path)
    
    # Create sample input: (batch * 9 patches, height=360, width=640, channels=3)
    # TensorFlow format: [B, H, W, C]
    # For 1 frame: 9 patches of 360x640
    # Values in [-1, 1] range
    sample_input = torch.randn(9, 360, 640, 3)
    
    print(f"Input shape: {sample_input.shape} (9 patches per frame, TensorFlow format: B, H, W, C)")
    print(f"  Patch size: 360x640")
    print(f"  Input range: [{sample_input.min():.2f}, {sample_input.max():.2f}]")
    
    # Test forward pass
    with torch.no_grad():
        output = model(sample_input)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.2f}, {output.max():.2f}]")
    
    print("\nConverting to TFLite...")
    
    # Convert to TFLite
    edge_model = ai_edge_torch.convert(
        model,
        (sample_input,)
    )
    
    # Save model
    output_file = os.path.join(output_dir, "distortion_net.tflite")
    edge_model.export(output_file)
    
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n✓ DistortionNet saved to: {output_file}")
    print(f"  File size: {file_size:.2f} MB")
    
    return output_file


def convert_aggregation_net(output_dir, model_path=None, quantize=False):
    """Convert AggregationNet to TFLite.
    
    Args:
        output_dir: Directory to save the TFLite model
        model_path: Path to PyTorch model checkpoint
        quantize: Whether to apply quantization
    
    Returns:
        Path to the saved TFLite model
    """
    print("\n" + "="*70)
    print("Converting AggregationNet to TFLite")
    print("="*70)
    
    # Create model
    model = create_aggregation_net(model_path)
    
    # Create sample inputs
    # Content features: (batch=1, height=8, width=8, channels=128)
    # Distortion features: (batch=1, height=24, width=24, channels=128)
    content_features = torch.randn(1, 8, 8, 128)
    distortion_features = torch.randn(1, 24, 24, 128)
    
    print(f"Content features shape: {content_features.shape}")
    print(f"Distortion features shape: {distortion_features.shape}")
    
    # Test forward pass
    with torch.no_grad():
        output = model(content_features, distortion_features)
    print(f"Output shape: {output.shape}")
    print(f"Quality score: {output.item():.3f} (range: [1, 5])")
    
    print("\nConverting to TFLite...")
    
    # Convert to TFLite
    edge_model = ai_edge_torch.convert(
        model,
        (content_features, distortion_features)
    )
    
    # Save model
    output_file = os.path.join(output_dir, "aggregation_net.tflite")
    edge_model.export(output_file)
    
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n✓ AggregationNet saved to: {output_file}")
    print(f"  File size: {file_size:.2f} MB")
    
    return output_file


def convert_all_models(output_dir, quantize=False):
    """Convert all UVQ 1.5 models to TFLite.
    
    Args:
        output_dir: Directory to save the TFLite models
        quantize: Whether to apply quantization
    
    Returns:
        Dictionary of model names to file paths
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("UVQ 1.5 PyTorch to TFLite Conversion")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print(f"Quantization: {'Enabled' if quantize else 'Disabled'}")
    
    models = {}
    
    # Convert each model
    try:
        models['content_net'] = convert_content_net(output_dir, quantize=quantize)
    except Exception as e:
        print(f"\n✗ Failed to convert ContentNet: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        models['distortion_net'] = convert_distortion_net(output_dir, quantize=quantize)
    except Exception as e:
        print(f"\n✗ Failed to convert DistortionNet: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        models['aggregation_net'] = convert_aggregation_net(output_dir, quantize=quantize)
    except Exception as e:
        print(f"\n✗ Failed to convert AggregationNet: {e}")
        import traceback
        traceback.print_exc()
    
    # Print summary
    print("\n" + "="*70)
    print("Conversion Summary")
    print("="*70)
    
    if models:
        print(f"\n✓ Successfully converted {len(models)}/3 models:")
        for name, path in models.items():
            size = os.path.getsize(path) / (1024 * 1024)
            print(f"  - {name}: {path} ({size:.2f} MB)")
    else:
        print("\n✗ No models were successfully converted")
    
    return models


def main():
    parser = argparse.ArgumentParser(
        description='Convert UVQ 1.5 models to TFLite format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all models
  python convert_to_tflite.py --output_dir ./tflite_models
  
  # Convert specific model
  python convert_to_tflite.py --model content --output_dir ./tflite_models
  
  # Convert with quantization (not yet implemented)
  python convert_to_tflite.py --quantize --output_dir ./tflite_models
        """
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./uvq1p5_tflite',
        help='Directory to save TFLite models (default: ./uvq1p5_tflite)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['content', 'distortion', 'aggregation', 'all'],
        default='all',
        help='Which model to convert (default: all)'
    )
    
    parser.add_argument(
        '--quantize',
        action='store_true',
        help='Apply quantization during conversion (not yet implemented)'
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
    
    # Create output directory
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert models
    if args.model == 'all':
        convert_all_models(output_dir, quantize=args.quantize)
    elif args.model == 'content':
        convert_content_net(output_dir, model_path=args.content_path, quantize=args.quantize)
    elif args.model == 'distortion':
        convert_distortion_net(output_dir, model_path=args.distortion_path, quantize=args.quantize)
    elif args.model == 'aggregation':
        convert_aggregation_net(output_dir, model_path=args.aggregation_path, quantize=args.quantize)


if __name__ == '__main__':
    main()

