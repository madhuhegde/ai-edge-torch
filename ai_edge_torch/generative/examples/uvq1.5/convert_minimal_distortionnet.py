#!/usr/bin/env python3
"""Convert minimal DistortionNet models to TFLite for debugging.

This script converts the minimal DistortionNet models (single, minimal, medium)
to TFLite format to help isolate and debug conversion issues.

Usage:
    # Convert single block model
    python convert_minimal_distortionnet.py --size single --output_dir ./debug_models
    
    # Convert minimal model (5 layers)
    python convert_minimal_distortionnet.py --size minimal --output_dir ./debug_models
    
    # Convert medium model (10 layers)
    python convert_minimal_distortionnet.py --size medium --output_dir ./debug_models
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import ai_edge_torch

# Add UVQ source path
UVQ_SOURCE_PATH = Path.home() / "work" / "UVQ" / "uvq"
sys.path.insert(0, str(UVQ_SOURCE_PATH))

from uvq1p5_pytorch.utils.distortionnet_minimal import get_minimal_distortionnet


class DistortionNetMinimalWrapper(nn.Module):
    """Wrapper for minimal DistortionNet models that handles TensorFlow format.
    
    This wrapper converts input from TensorFlow format [B, H, W, C] to PyTorch
    format [B, C, H, W] before passing through the model, then converts the
    output back to TensorFlow format.
    """
    
    def __init__(self, size='minimal'):
        super().__init__()
        self.model = get_minimal_distortionnet(size)
        self.model.eval()
        self.size = size
    
    def forward(self, x):
        """
        Args:
            x: Tensor in TensorFlow format [B, H, W, C]
               For 'single': [9, 180, 320, 16]
               For 'minimal'/'medium': [9, 360, 640, 3]
        
        Returns:
            features: Tensor in TensorFlow format [B, H, W, C]
        """
        # Convert from TensorFlow format (B, H, W, C) to PyTorch format (B, C, H, W)
        # Use contiguous() to ensure memory layout is optimal for TFLite conversion
        x = x.permute(0, 3, 1, 2).contiguous()
        
        # Process through the minimal distortion net
        features = self.model(x)
        
        # The model already outputs in NHWC format due to PermuteLayerNHWC
        # So we don't need to permute again
        return features


def convert_minimal_distortionnet(size='minimal', output_dir='./debug_models', batch_size=9):
    """Convert a minimal DistortionNet model to TFLite.
    
    Args:
        size: 'single', 'minimal', or 'medium'
        output_dir: Directory to save the TFLite model
        batch_size: Batch size for input (default: 9 for 9 patches)
    
    Returns:
        Path to the saved TFLite model
    """
    print("\n" + "="*70)
    print(f"Converting Minimal DistortionNet ({size.upper()}) to TFLite")
    print(f"Batch size: {batch_size}")
    print("="*70)
    
    # Create model with wrapper to handle TensorFlow format
    model = DistortionNetMinimalWrapper(size)
    model.eval()
    
    # Create sample input based on model size
    # TensorFlow format: [B, H, W, C]
    # All models now accept 3 RGB channels
    sample_input = torch.randn(batch_size, 360, 640, 3)
    if batch_size == 1:
        print(f"Input shape: {sample_input.shape} (1 patch of 360x640, TensorFlow format: B, H, W, C)")
    else:
        print(f"Input shape: {sample_input.shape} ({batch_size} patches of 360x640, TensorFlow format: B, H, W, C)")
    
    print(f"Input range: [{sample_input.min():.2f}, {sample_input.max():.2f}]")
    
    # Test forward pass
    print("\nTesting PyTorch model...")
    with torch.no_grad():
        output = model(sample_input)
    print(f"✓ PyTorch inference successful")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.2f}, {output.max():.2f}]")
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"\nModel statistics:")
    print(f"  Parameters: {params:,}")
    print(f"  Size (MB): {params * 4 / (1024**2):.2f}")
    
    print("\nConverting to TFLite...")
    
    # Convert to TFLite
    try:
        edge_model = ai_edge_torch.convert(
            model,
            (sample_input,)
        )
        
        # Save model
        os.makedirs(output_dir, exist_ok=True)
        if batch_size == 1:
            output_file = os.path.join(output_dir, f"distortion_net_{size}_single_batch.tflite")
        else:
            output_file = os.path.join(output_dir, f"distortion_net_{size}.tflite")
        edge_model.export(output_file)
        
        file_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"\n✓ Conversion successful!")
        print(f"✓ Model saved to: {output_file}")
        print(f"  File size: {file_size:.2f} MB")
        
        return output_file
        
    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def verify_tflite_model(tflite_path, size='minimal', batch_size=9):
    """Verify the converted TFLite model.
    
    Args:
        tflite_path: Path to the TFLite model
        size: Model size ('single', 'minimal', 'medium')
        batch_size: Batch size used in the model (default: 9)
    """
    print("\n" + "="*70)
    print(f"Verifying TFLite Model: {tflite_path}")
    print("="*70)
    
    try:
        import tensorflow as tf
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"\n✓ Model loaded successfully")
        print(f"\nInput details:")
        print(f"  Shape: {input_details[0]['shape']}")
        print(f"  Type: {input_details[0]['dtype']}")
        
        print(f"\nOutput details:")
        print(f"  Shape: {output_details[0]['shape']}")
        print(f"  Type: {output_details[0]['dtype']}")
        
        # Create test input
        # TensorFlow format: [B, H, W, C]
        # All models now accept 3 RGB channels
        import numpy as np
        test_input = np.random.randn(batch_size, 360, 640, 3).astype(np.float32)
        
        # Run inference
        print(f"\nRunning TFLite inference...")
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"✓ Inference successful!")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.2f}, {output.max():.2f}]")
        
        # Check operators
        print(f"\nOperators used:")
        ops = set()
        for op_details in interpreter._get_ops_details():
            ops.add(op_details['op_name'])
        
        for op in sorted(ops):
            print(f"  - {op}")
        
        # Check for problematic ops
        problematic_ops = ['GATHER_ND', 'GATHER', 'SCATTER_ND']
        found_problematic = [op for op in ops if op in problematic_ops]
        
        if found_problematic:
            print(f"\n⚠️  WARNING: Found problematic operators:")
            for op in found_problematic:
                print(f"  - {op}")
        else:
            print(f"\n✓ No problematic operators found")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Convert minimal DistortionNet models to TFLite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single block model (isolate depthwise conv)
  python convert_minimal_distortionnet.py --size single --output_dir ./debug_models
  
  # Convert minimal model (5 layers)
  python convert_minimal_distortionnet.py --size minimal --output_dir ./debug_models
  
  # Convert medium model (10 layers)
  python convert_minimal_distortionnet.py --size medium --output_dir ./debug_models
  
  # Convert and verify
  python convert_minimal_distortionnet.py --size minimal --output_dir ./debug_models --verify
        """
    )
    
    parser.add_argument(
        '--size',
        type=str,
        choices=['single', 'minimal', 'medium'],
        default='minimal',
        help='Which minimal model to convert (default: minimal)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./debug_models',
        help='Directory to save TFLite models (default: ./debug_models)'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify the converted TFLite model'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=9,
        help='Batch size for input (default: 9)'
    )
    
    args = parser.parse_args()
    
    # Convert model
    tflite_path = convert_minimal_distortionnet(args.size, args.output_dir, args.batch_size)
    
    # Verify if requested
    if args.verify and tflite_path:
        verify_tflite_model(tflite_path, args.size, args.batch_size)
    
    print("\n" + "="*70)
    print("Conversion complete!")
    print("="*70)


if __name__ == '__main__':
    main()

