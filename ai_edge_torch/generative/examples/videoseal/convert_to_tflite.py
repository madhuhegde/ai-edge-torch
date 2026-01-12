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
Convert VideoSeal models to TFLite format.

This script converts VideoSeal watermarking models (Embedder and Detector)
to TFLite format for deployment on mobile and edge devices.

VideoSeal is a state-of-the-art invisible watermarking model that can:
- Embed 256-bit watermarks into images (Embedder)
- Detect and extract watermarks from images (Detector)

Usage:
    # Convert both models
    python convert_to_tflite.py --output_dir ./tflite_models
    
    # Convert only embedder
    python convert_to_tflite.py --model embedder --output_dir ./tflite_models
    
    # Convert with different VideoSeal variant
    python convert_to_tflite.py --model_name pixelseal --output_dir ./tflite_models
    
    # Convert with custom image size
    python convert_to_tflite.py --image_size 512 --output_dir ./tflite_models
"""

import argparse
import os
from pathlib import Path

import torch
import ai_edge_torch

from videoseal_models import create_embedder, create_detector


def convert_embedder(output_dir, model_name="videoseal", image_size=256, simple=True):
    """Convert VideoSeal Embedder to TFLite.
    
    Args:
        output_dir: Directory to save the TFLite model
        model_name: VideoSeal model variant ('videoseal', 'pixelseal', 'chunkyseal')
        image_size: Input image size (default: 256)
        simple: Use simplified version for fixed-size images
    
    Returns:
        Path to the saved TFLite model
    """
    print("\n" + "="*70)
    print(f"Converting VideoSeal Embedder ({model_name}) to TFLite")
    print("="*70)
    
    # Create model
    model = create_embedder(model_name=model_name, simple=simple)
    
    # Create sample inputs
    # Image: (batch=1, height, width, channels=3) in [0, 1] range (NHWC format)
    # Message: (batch=1, 256 bits) binary vector
    sample_img = torch.rand(1, image_size, image_size, 3)
    sample_msg = torch.randint(0, 2, (1, 256)).float()
    
    print(f"Input image shape: {sample_img.shape}")
    print(f"Input image range: [{sample_img.min():.2f}, {sample_img.max():.2f}]")
    print(f"Input message shape: {sample_msg.shape}")
    print(f"Input message (first 32 bits): {sample_msg[0, :32].numpy()}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        output = model(sample_img, sample_msg)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.2f}, {output.max():.2f}]")
    
    # Calculate PSNR (should be high for invisible watermark)
    mse = torch.mean((output - sample_img) ** 2)
    psnr = 10 * torch.log10(1.0 / mse)
    print(f"PSNR (higher is better): {psnr:.2f} dB")
    
    print("\nConverting to TFLite...")
    
    try:
        # Convert to TFLite
        edge_model = ai_edge_torch.convert(
            model,
            (sample_img, sample_msg)
        )
        
        # Save model
        output_file = os.path.join(output_dir, f"videoseal_embedder_{model_name}_{image_size}.tflite")
        edge_model.export(output_file)
        
        file_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"\n✓ Embedder saved to: {output_file}")
        print(f"  File size: {file_size:.2f} MB")
        
        return output_file
    
    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def convert_detector(output_dir, model_name="videoseal", image_size=256, simple=True):
    """Convert VideoSeal Detector to TFLite.
    
    Args:
        output_dir: Directory to save the TFLite model
        model_name: VideoSeal model variant ('videoseal', 'pixelseal', 'chunkyseal')
        image_size: Input image size (default: 256)
        simple: Use simplified version for fixed-size images
    
    Returns:
        Path to the saved TFLite model
    """
    print("\n" + "="*70)
    print(f"Converting VideoSeal Detector ({model_name}) to TFLite")
    print("="*70)
    
    # Create model
    model = create_detector(model_name=model_name, simple=simple)
    
    # Create sample input
    # Image: (batch=1, height, width, channels=3) in [0, 1] range (NHWC format)
    sample_img = torch.rand(1, image_size, image_size, 3)
    
    print(f"Input image shape: {sample_img.shape}")
    print(f"Input image range: [{sample_img.min():.2f}, {sample_img.max():.2f}]")
    
    # Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        output = model(sample_img)
    
    print(f"Output shape: {output.shape}")
    print(f"  Channel 0: Detection mask")
    print(f"  Channels 1-256: Message bits")
    print(f"Output range: [{output.min():.2f}, {output.max():.2f}]")
    
    # Extract message (apply threshold)
    detected_msg = (output[0, 1:] > 0).float()
    print(f"Detected message (first 32 bits): {detected_msg[:32].numpy()}")
    
    print("\nConverting to TFLite...")
    
    try:
        # Convert to TFLite
        edge_model = ai_edge_torch.convert(
            model,
            (sample_img,)
        )
        
        # Save model
        output_file = os.path.join(output_dir, f"videoseal_detector_{model_name}_{image_size}.tflite")
        edge_model.export(output_file)
        
        file_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"\n✓ Detector saved to: {output_file}")
        print(f"  File size: {file_size:.2f} MB")
        
        return output_file
    
    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def convert_all_models(output_dir, model_name="videoseal", image_size=256, simple=True):
    """Convert both VideoSeal models to TFLite.
    
    Args:
        output_dir: Directory to save the TFLite models
        model_name: VideoSeal model variant
        image_size: Input image size
        simple: Use simplified version
    
    Returns:
        Dictionary of model names to file paths
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("VideoSeal PyTorch to TFLite Conversion")
    print("="*70)
    print(f"Model variant: {model_name}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Output directory: {output_dir}")
    print(f"Mode: {'Simple (fixed-size)' if simple else 'Dynamic (variable-size)'}")
    
    models = {}
    
    # Convert embedder
    try:
        embedder_path = convert_embedder(output_dir, model_name, image_size, simple)
        if embedder_path:
            models['embedder'] = embedder_path
    except Exception as e:
        print(f"\n✗ Failed to convert Embedder: {e}")
        import traceback
        traceback.print_exc()
    
    # Convert detector
    try:
        detector_path = convert_detector(output_dir, model_name, image_size, simple)
        if detector_path:
            models['detector'] = detector_path
    except Exception as e:
        print(f"\n✗ Failed to convert Detector: {e}")
        import traceback
        traceback.print_exc()
    
    # Print summary
    print("\n" + "="*70)
    print("Conversion Summary")
    print("="*70)
    
    if models:
        print(f"\n✓ Successfully converted {len(models)}/2 models:")
        for name, path in models.items():
            size = os.path.getsize(path) / (1024 * 1024)
            print(f"  - {name}: {path} ({size:.2f} MB)")
    else:
        print("\n✗ No models were successfully converted")
    
    return models


def main():
    parser = argparse.ArgumentParser(
        description='Convert VideoSeal models to TFLite format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert both models with default settings
  python convert_to_tflite.py --output_dir ./tflite_models
  
  # Convert only embedder
  python convert_to_tflite.py --model embedder --output_dir ./tflite_models
  
  # Use PixelSeal variant (SOTA imperceptibility & robustness)
  python convert_to_tflite.py --model_name pixelseal --output_dir ./tflite_models
  
  # Use ChunkySeal variant (1024-bit capacity)
  python convert_to_tflite.py --model_name chunkyseal --output_dir ./tflite_models
  
  # Convert with 512x512 image size
  python convert_to_tflite.py --image_size 512 --output_dir ./tflite_models
  
  # Use dynamic version (supports variable image sizes, but larger model)
  python convert_to_tflite.py --no_simple --output_dir ./tflite_models

Available VideoSeal Models:
  - videoseal: VideoSeal v1.0 (256-bit, stable, recommended)
  - pixelseal: PixelSeal (SOTA imperceptibility & robustness)
  - chunkyseal: ChunkySeal (1024-bit high capacity)
        """
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./videoseal_tflite',
        help='Directory to save TFLite models (default: ./videoseal_tflite)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['embedder', 'detector', 'all'],
        default='all',
        help='Which model to convert (default: all)'
    )
    
    parser.add_argument(
        '--model_name',
        type=str,
        choices=['videoseal', 'pixelseal', 'chunkyseal'],
        default='videoseal',
        help='VideoSeal model variant to use (default: videoseal)'
    )
    
    parser.add_argument(
        '--image_size',
        type=int,
        default=256,
        help='Input image size (default: 256)'
    )
    
    parser.add_argument(
        '--no_simple',
        action='store_true',
        help='Use dynamic version instead of simplified fixed-size version'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    simple = not args.no_simple
    
    # Convert models
    if args.model == 'all':
        convert_all_models(output_dir, args.model_name, args.image_size, simple)
    elif args.model == 'embedder':
        convert_embedder(output_dir, args.model_name, args.image_size, simple)
    elif args.model == 'detector':
        convert_detector(output_dir, args.model_name, args.image_size, simple)


if __name__ == '__main__':
    main()

