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
Convert VideoSeal 0.0 Embedder to TFLite format.

VideoSeal 0.0 is the baseline watermarking model with 96-bit capacity.
This script converts the VideoSeal 0.0 Embedder to TFLite format for 
deployment on mobile and edge devices.

The Embedder takes an image and a 96-bit message and outputs:
- Watermarked image with embedded message

Supports FLOAT32 quantization (INT8 not recommended for embedder):
- FLOAT32: Full precision, best quality

Usage:
    # Convert with default settings (256x256, FLOAT32)
    python convert_embedder_to_tflite.py --output_dir ./tflite_models
    
    # Convert with custom image size
    python convert_embedder_to_tflite.py --image_size 512 --output_dir ./tflite_models

Note: INT8 quantization is not recommended for embedders as it can
significantly degrade watermark quality (PSNR).
"""

import argparse
import os
from pathlib import Path

import torch
import ai_edge_torch
from ai_edge_torch.generative.quantize import quant_recipes

from videoseal00_models import create_embedder


def convert_embedder(output_dir, model_name="videoseal_0.0", image_size=256, simple=True, quantize=None):
    """Convert VideoSeal 0.0 Embedder to TFLite.
    
    Args:
        output_dir: Directory to save the TFLite model
        model_name: VideoSeal model variant (default: 'videoseal_0.0')
        image_size: Input image size (default: 256)
        simple: Use simplified version for fixed-size images
        quantize: Quantization type ('fp16' or None for FLOAT32)
    
    Returns:
        Path to the saved TFLite model, or None if conversion failed
    """
    # Determine quantization suffix and name
    if quantize == 'fp16':
        quant_suffix = "_fp16"
    else:
        quant_suffix = ""
    
    quant_name = quantize.upper() if quantize else "FLOAT32"
    
    print("\n" + "="*70)
    print(f"Converting VideoSeal 0.0 Embedder to TFLite ({quant_name})")
    print("="*70)
    print("\nVideoSeal 0.0 Embedder Features:")
    print("  • 96-bit message input")
    print("  • UNet-Small2 architecture")
    print("  • Message processor: binary+concat type")
    print("  • 8 UNet blocks with RMS normalization")
    
    # Create model
    print("\nLoading VideoSeal 0.0 embedder...")
    model = create_embedder(model_name=model_name, simple=simple)
    
    # Create sample inputs
    # Image: (batch=1, height, width, channels=3) in [0, 1] range (NHWC format)
    # Message: (batch=1, 96) with binary values (0 or 1) - INT32 for HW delegate compatibility
    sample_img = torch.rand(1, image_size, image_size, 3)
    sample_msg = torch.randint(0, 2, (1, 96), dtype=torch.int32)
    
    print(f"\nInput shapes:")
    print(f"  Image: {sample_img.shape} in range [{sample_img.min():.2f}, {sample_img.max():.2f}] (NHWC format)")
    print(f"  Message: {sample_msg.shape} with {sample_msg.sum().int().item()}/96 bits set to 1")
    
    # Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        output = model(sample_img, sample_msg)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Calculate PSNR
    mse = torch.mean((output - sample_img) ** 2)
    psnr = 10 * torch.log10(1.0 / mse)
    print(f"PSNR: {psnr:.2f} dB")
    
    print(f"\nConverting to TFLite ({quant_name})...")
    
    # Prepare quantization config if requested
    quant_config = None
    if quantize == 'fp16':
        print("Using FP16 quantization")
        print("  - All weights and activations: FP16")
        print("  - ~50% size reduction")
        print("  ⚠️  May slightly reduce PSNR")
        quant_config = quant_recipes.full_fp16_recipe(mcfg=None)
    
    try:
        # Convert to TFLite
        edge_model = ai_edge_torch.convert(
            model,
            (sample_img, sample_msg),
            quant_config=quant_config
        )
        
        # Save model
        output_file = os.path.join(output_dir, f"videoseal00_embedder_{image_size}{quant_suffix}.tflite")
        edge_model.export(output_file)
        
        file_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"\n✓ Embedder ({quant_name}) saved to: {output_file}")
        print(f"  File size: {file_size:.2f} MB")
        
        # Compare with FLOAT32 version if quantized
        if quantize:
            float32_file = os.path.join(output_dir, f"videoseal00_embedder_{image_size}.tflite")
            if os.path.exists(float32_file):
                float32_size = os.path.getsize(float32_file) / (1024 * 1024)
                reduction = (1 - file_size / float32_size) * 100
                print(f"\n  Comparison with FLOAT32:")
                print(f"    FLOAT32: {float32_size:.2f} MB")
                print(f"    {quant_name}: {file_size:.2f} MB")
                print(f"    Reduction: {reduction:.1f}%")
                print(f"    Savings: {float32_size - file_size:.2f} MB")
        
        return output_file
    
    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Convert VideoSeal 0.0 Embedder to TFLite format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert with default settings (FLOAT32)
  python convert_embedder_to_tflite.py --output_dir ./tflite_models
  
  # Convert with FP16 quantization (~50% smaller)
  python convert_embedder_to_tflite.py --quantize fp16 --output_dir ./tflite_models
  
  # Convert with 512x512 image size
  python convert_embedder_to_tflite.py --image_size 512 --output_dir ./tflite_models

VideoSeal 0.0 Embedder:
  • Architecture: UNet-Small2 with message processor
  • Message capacity: 96 bits
  • Input: RGB image + 96-bit binary message
  • Output: Watermarked RGB image
  • Quality: High PSNR (typically >40 dB)

Quantization Options:
  - None (default): FLOAT32 - Full precision, best quality
  - fp16: FP16 - ~50% smaller, minimal quality loss
  
  ⚠️  INT8 quantization is NOT recommended for embedders:
      - Significant PSNR degradation
      - May affect watermark detectability
      - Use FLOAT32 or FP16 instead

Note:
  Attenuation is disabled for TFLite compatibility.
  For best results, use the PyTorch version with attenuation enabled.
        """
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./videoseal00_tflite',
        help='Directory to save TFLite model (default: ./videoseal00_tflite)'
    )
    
    parser.add_argument(
        '--model_name',
        type=str,
        default='videoseal_0.0',
        help='VideoSeal model variant to use (default: videoseal_0.0)'
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
    
    parser.add_argument(
        '--quantize',
        type=str,
        choices=['fp16'],
        default=None,
        help='Quantization type (default: None for FLOAT32). Only FP16 supported for embedder.'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    simple = not args.no_simple
    quant_name = args.quantize.upper() if args.quantize else "FLOAT32"
    
    print("\n" + "="*70)
    print("VideoSeal 0.0 Embedder PyTorch to TFLite Conversion")
    print("="*70)
    print(f"Model variant: {args.model_name}")
    print(f"Image size: {args.image_size}×{args.image_size}")
    print(f"Output directory: {output_dir}")
    print(f"Mode: {'Simple (fixed-size)' if simple else 'Dynamic (variable-size)'}")
    print(f"Quantization: {quant_name}")
    
    # Convert embedder
    embedder_path = convert_embedder(output_dir, args.model_name, args.image_size, simple, args.quantize)
    
    # Print summary
    print("\n" + "="*70)
    print("Conversion Summary")
    print("="*70)
    
    if embedder_path:
        size = os.path.getsize(embedder_path) / (1024 * 1024)
        print(f"\n✓ Successfully converted Embedder ({quant_name}):")
        print(f"  Path: {embedder_path}")
        print(f"  Size: {size:.2f} MB")
        print(f"  Message capacity: 96 bits")
        print(f"\nYou can now use this model to embed 96-bit watermarks into images!")
        print(f"See example_usage.py for usage examples.")
        
        return 0
    else:
        print("\n✗ Conversion failed")
        print("See error messages above for details.")
        return 1


if __name__ == '__main__':
    exit(main())
