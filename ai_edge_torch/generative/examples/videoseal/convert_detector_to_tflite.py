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
Convert VideoSeal Detector to TFLite format.

This script converts the VideoSeal Detector (watermark detection and extraction)
to TFLite format for deployment on mobile and edge devices.

The Detector takes an image and outputs:
- Detection confidence (channel 0)
- 256-bit watermark message (channels 1-256)

Supports both FLOAT32 (default) and INT8 quantization:
- FLOAT32: Full precision, best accuracy
- INT8: ~75% smaller, faster inference, minimal accuracy loss

Usage:
    # Convert with default settings (256x256, FLOAT32)
    python convert_detector_to_tflite.py --output_dir ./tflite_models
    
    # Convert with INT8 quantization
    python convert_detector_to_tflite.py --quantize int8 --output_dir ./tflite_models
    
    # Convert with different VideoSeal variant
    python convert_detector_to_tflite.py --model_name pixelseal --output_dir ./tflite_models
    
    # Convert with custom image size
    python convert_detector_to_tflite.py --image_size 512 --output_dir ./tflite_models

Note: The Embedder conversion is not yet supported due to ai-edge-torch limitations
with VideoSeal's dynamic tensor operations. Use PyTorch for embedding watermarks.
"""

import argparse
import os
from pathlib import Path

import torch
import ai_edge_torch
from ai_edge_torch.generative.quantize import quant_recipes
from ai_edge_torch.generative.quantize import quant_attrs

from videoseal_models import create_detector


def convert_detector(output_dir, model_name="videoseal", image_size=256, simple=True, quantize=None):
    """Convert VideoSeal Detector to TFLite.
    
    Args:
        output_dir: Directory to save the TFLite model
        model_name: VideoSeal model variant ('videoseal', 'pixelseal', 'chunkyseal')
        image_size: Input image size (default: 256)
        simple: Use simplified version for fixed-size images
        quantize: Quantization type ('int8', 'fp16', or None for FLOAT32)
    
    Returns:
        Path to the saved TFLite model, or None if conversion failed
    """
    # For INT8, use _int8 suffix; for FP16, use _fp16 suffix; for FLOAT32, no suffix
    if quantize == 'int8':
        quant_suffix = "_int8"
    elif quantize == 'fp16':
        quant_suffix = "_fp16"
    else:
        quant_suffix = ""
    
    quant_name = quantize.upper() if quantize else "FLOAT32"
    
    print("\n" + "="*70)
    print(f"Converting VideoSeal Detector ({model_name}) to TFLite ({quant_name})")
    print("="*70)
    
    # Create model
    print("Loading VideoSeal model...")
    model = create_detector(model_name=model_name, simple=simple)
    
    # Create sample input
    # Image: (batch=1, channels=3, height, width) in [0, 1] range
    sample_img = torch.rand(1, 3, image_size, image_size)
    
    print(f"\nInput image shape: {sample_img.shape}")
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
    print(f"Detected message (first 32 bits): {detected_msg[:32].int().tolist()}")
    
    print(f"\nConverting to TFLite ({quant_name})...")
    
    # Prepare quantization config if requested
    quant_config = None
    if quantize == 'int8':
        print("Using dynamic_qi8_recipe (dynamic INT8 quantization)")
        print("  - Weight dtype: INT8")
        print("  - Granularity: CHANNELWISE")
        print("  - Activations: Dynamic INT8 at runtime")
        print("  - Inputs/Outputs: FLOAT32 for compatibility")
        quant_config = quant_recipes.full_dynamic_recipe(
            mcfg=None,  # No model config needed for non-transformer models
            weight_dtype=quant_attrs.Dtype.INT8,
            granularity=quant_attrs.Granularity.CHANNELWISE
        )
    elif quantize == 'fp16':
        print("Using FP16 quantization")
        print("  - All weights and activations: FP16")
        print("  - ~50% size reduction, minimal accuracy loss")
        quant_config = quant_recipes.full_fp16_recipe(mcfg=None)
    
    try:
        # Convert to TFLite
        edge_model = ai_edge_torch.convert(
            model,
            (sample_img,),
            quant_config=quant_config
        )
        
        # Save model
        output_file = os.path.join(output_dir, f"videoseal_detector_{model_name}_{image_size}{quant_suffix}.tflite")
        edge_model.export(output_file)
        
        file_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"\n✓ Detector ({quant_name}) saved to: {output_file}")
        print(f"  File size: {file_size:.2f} MB")
        
        # Compare with FLOAT32 version if quantized
        if quantize:
            float32_file = os.path.join(output_dir, f"videoseal_detector_{model_name}_{image_size}.tflite")
            if os.path.exists(float32_file):
                float32_size = os.path.getsize(float32_file) / (1024 * 1024)
                reduction = (1 - file_size / float32_size) * 100
                print(f"\n  Comparison with FLOAT32:")
                print(f"    FLOAT32: {float32_size:.2f} MB")
                print(f"    {quant_name}: {file_size:.2f} MB")
                print(f"    Reduction: {reduction:.1f}%")
        
        return output_file
    
    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Convert VideoSeal Detector to TFLite format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert with default settings (FLOAT32)
  python convert_detector_to_tflite.py --output_dir ./tflite_models
  
  # Convert with INT8 quantization (~75% smaller, faster)
  python convert_detector_to_tflite.py --quantize int8 --output_dir ./tflite_models
  
  # Convert with FP16 quantization (~50% smaller)
  python convert_detector_to_tflite.py --quantize fp16 --output_dir ./tflite_models
  
  # Use PixelSeal variant (SOTA imperceptibility & robustness)
  python convert_detector_to_tflite.py --model_name pixelseal --quantize int8 --output_dir ./tflite_models
  
  # Use ChunkySeal variant (1024-bit capacity)
  python convert_detector_to_tflite.py --model_name chunkyseal --output_dir ./tflite_models
  
  # Convert with 512x512 image size
  python convert_detector_to_tflite.py --image_size 512 --output_dir ./tflite_models

Available VideoSeal Models:
  - videoseal: VideoSeal v1.0 (256-bit, stable, recommended)
  - pixelseal: PixelSeal (SOTA imperceptibility & robustness)
  - chunkyseal: ChunkySeal (1024-bit high capacity)

Quantization Options:
  - None (default): FLOAT32 - Full precision, best accuracy
  - int8: Dynamic INT8 - ~75% smaller, faster inference, minimal accuracy loss
  - fp16: FP16 - ~50% smaller, minimal accuracy loss
  
  INT8 benefits:
  - Significantly smaller model size
  - Faster inference on ARM/x86 CPUs with INT8 support
  - Lower memory bandwidth requirements
  - Typical accuracy loss: <2% for detection tasks

Note:
  The Embedder (watermark insertion) is not yet supported for TFLite conversion
  due to ai-edge-torch limitations. Use PyTorch for embedding watermarks on the
  server side, and deploy the TFLite Detector for mobile/edge verification.
        """
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./videoseal_tflite',
        help='Directory to save TFLite model (default: ./videoseal_tflite)'
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
    
    parser.add_argument(
        '--quantize',
        type=str,
        choices=['int8', 'fp16'],
        default=None,
        help='Quantization type (default: None for FLOAT32). Options: int8 (~75%% smaller), fp16 (~50%% smaller)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    simple = not args.no_simple
    quant_name = args.quantize.upper() if args.quantize else "FLOAT32"
    
    print("\n" + "="*70)
    print("VideoSeal Detector PyTorch to TFLite Conversion")
    print("="*70)
    print(f"Model variant: {args.model_name}")
    print(f"Image size: {args.image_size}×{args.image_size}")
    print(f"Output directory: {output_dir}")
    print(f"Mode: {'Simple (fixed-size)' if simple else 'Dynamic (variable-size)'}")
    print(f"Quantization: {quant_name}")
    
    # Convert detector
    detector_path = convert_detector(output_dir, args.model_name, args.image_size, simple, args.quantize)
    
    # Print summary
    print("\n" + "="*70)
    print("Conversion Summary")
    print("="*70)
    
    if detector_path:
        size = os.path.getsize(detector_path) / (1024 * 1024)
        print(f"\n✓ Successfully converted Detector ({quant_name}):")
        print(f"  Path: {detector_path}")
        print(f"  Size: {size:.2f} MB")
        print(f"\nYou can now use this model to detect watermarks in images!")
        print(f"See verify_detector_tflite.py for usage and accuracy verification.")
        return 0
    else:
        print("\n✗ Conversion failed")
        print("See error messages above for details.")
        return 1


if __name__ == '__main__':
    exit(main())

