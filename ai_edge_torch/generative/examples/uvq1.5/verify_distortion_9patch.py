#!/usr/bin/env python3
"""Verify DistortionNet 9-patch model against PyTorch implementation.

This script verifies that the 9-patch TFLite model with application-side 5D aggregation
produces the same output as the PyTorch batch-9 model.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

# Add UVQ source path
UVQ_SOURCE_PATH = Path.home() / "work" / "UVQ" / "uvq"
sys.path.insert(0, str(UVQ_SOURCE_PATH))

from uvq1p5_pytorch.utils import distortionnet
from uvq1p5_pytorch.utils.tflite_aggregation import aggregate_9patch_features

# Import TensorFlow Lite
try:
    import tensorflow as tf
except ImportError:
    print("TensorFlow not found. Please install: pip install tensorflow")
    raise


def load_tflite_model(model_path):
    """Load TFLite model and return interpreter."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def run_pytorch_model(patches):
    """Run PyTorch DistortionNet model.
    
    Args:
        patches: numpy array of shape (9, 360, 640, 3)
    
    Returns:
        features: numpy array of shape (1, 24, 24, 128)
    """
    model_path = Path.home() / "work" / "models" / "UVQ" / "uvq1.5" / "distortion_net.pth"
    
    model = distortionnet.DistortionNet(
        model_path=str(model_path),
        eval_mode=True,
        pretrained=True,
    )
    model.eval()
    
    # Convert to PyTorch format [B, C, H, W]
    patches_torch = torch.from_numpy(patches).permute(0, 3, 1, 2).float()
    
    with torch.no_grad():
        # Get individual patch features [9, 8, 8, 128] in NHWC
        patch_features = model.model(patches_torch)
        
        # Aggregate using PyTorch's 6D logic
        # This is what the batch-9 model does internally
        features = patch_features.reshape(1, 3, 3, 8, 8, 128)
        features = features.permute(0, 1, 3, 2, 4, 5).contiguous()
        features = features.reshape(1, 24, 24, 128)
    
    return features.numpy()


def run_tflite_9patch_model(interpreter, patches):
    """Run TFLite 9-patch model with application-side aggregation.
    
    Args:
        interpreter: TFLite interpreter
        patches: numpy array of shape (9, 360, 640, 3)
    
    Returns:
        features: numpy array of shape (1, 24, 24, 128)
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], patches.astype(np.float32))
    interpreter.invoke()
    
    # Get output [9, 8, 8, 128]
    patch_features = interpreter.get_tensor(output_details[0]['index'])
    
    # Aggregate using 5D operations
    features = aggregate_9patch_features(patch_features)
    
    return features


def compare_outputs(pytorch_output, tflite_output, tolerance=1e-4):
    """Compare PyTorch and TFLite outputs.
    
    Args:
        pytorch_output: numpy array from PyTorch
        tflite_output: numpy array from TFLite
        tolerance: maximum allowed difference
    
    Returns:
        bool: True if outputs match within tolerance
    """
    print("\nComparison Results:")
    print("=" * 70)
    
    # Shape check
    print(f"PyTorch output shape: {pytorch_output.shape}")
    print(f"TFLite output shape:  {tflite_output.shape}")
    
    if pytorch_output.shape != tflite_output.shape:
        print("❌ Shape mismatch!")
        return False
    
    # Value comparison
    diff = np.abs(pytorch_output - tflite_output)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"\nValue comparison:")
    print(f"  Max difference:  {max_diff:.6e}")
    print(f"  Mean difference: {mean_diff:.6e}")
    print(f"  Tolerance:       {tolerance:.6e}")
    
    # Correlation
    pytorch_flat = pytorch_output.flatten()
    tflite_flat = tflite_output.flatten()
    correlation = np.corrcoef(pytorch_flat, tflite_flat)[0, 1]
    print(f"  Correlation:     {correlation:.10f}")
    
    # Sample values
    print(f"\nSample values (first 5 elements):")
    print(f"  PyTorch: {pytorch_flat[:5]}")
    print(f"  TFLite:  {tflite_flat[:5]}")
    
    # Check tolerance
    if max_diff <= tolerance:
        print(f"\n✅ Verification PASSED (max diff: {max_diff:.6e} <= {tolerance:.6e})")
        return True
    else:
        print(f"\n❌ Verification FAILED (max diff: {max_diff:.6e} > {tolerance:.6e})")
        return False


def main():
    print("=" * 70)
    print("DistortionNet 9-Patch Model Verification")
    print("=" * 70)
    
    # Model paths
    tflite_model_path = Path.home() / "work" / "UVQ" / "uvq" / "models" / "tflite_models" / "uvq1.5" / "distortion_net_9patch.tflite"
    
    if not tflite_model_path.exists():
        print(f"❌ TFLite model not found: {tflite_model_path}")
        return
    
    print(f"\nTFLite model: {tflite_model_path}")
    
    # Load TFLite model
    print("\nLoading TFLite model...")
    interpreter = load_tflite_model(str(tflite_model_path))
    
    # Create test input
    print("\nCreating test input...")
    np.random.seed(42)
    test_patches = np.random.randn(9, 360, 640, 3).astype(np.float32)
    
    print(f"Input shape: {test_patches.shape}")
    print(f"Input range: [{test_patches.min():.2f}, {test_patches.max():.2f}]")
    
    # Run PyTorch model
    print("\nRunning PyTorch model...")
    pytorch_output = run_pytorch_model(test_patches)
    print(f"PyTorch output shape: {pytorch_output.shape}")
    print(f"PyTorch output range: [{pytorch_output.min():.2f}, {pytorch_output.max():.2f}]")
    
    # Run TFLite model
    print("\nRunning TFLite 9-patch model with 5D aggregation...")
    tflite_output = run_tflite_9patch_model(interpreter, test_patches)
    print(f"TFLite output shape: {tflite_output.shape}")
    print(f"TFLite output range: [{tflite_output.min():.2f}, {tflite_output.max():.2f}]")
    
    # Compare outputs (use 5e-4 tolerance for TFLite numerical precision)
    success = compare_outputs(pytorch_output, tflite_output, tolerance=5e-4)
    
    # Also compare with batch-9 model for reference
    print("\n" + "=" * 70)
    print("Comparing with batch-9 model (for reference)")
    print("=" * 70)
    
    batch9_model_path = Path.home() / "work" / "UVQ" / "uvq" / "models" / "tflite_models" / "uvq1.5" / "distortion_net.tflite"
    if batch9_model_path.exists():
        print(f"\nBatch-9 model: {batch9_model_path}")
        batch9_interpreter = load_tflite_model(str(batch9_model_path))
        
        input_details = batch9_interpreter.get_input_details()
        output_details = batch9_interpreter.get_output_details()
        
        batch9_interpreter.set_tensor(input_details[0]['index'], test_patches.astype(np.float32))
        batch9_interpreter.invoke()
        batch9_output = batch9_interpreter.get_tensor(output_details[0]['index'])
        
        print(f"Batch-9 output shape: {batch9_output.shape}")
        print(f"Batch-9 output range: [{batch9_output.min():.2f}, {batch9_output.max():.2f}]")
        
        # Compare 9-patch with batch-9
        diff = np.abs(tflite_output - batch9_output)
        max_diff = np.max(diff)
        correlation = np.corrcoef(tflite_output.flatten(), batch9_output.flatten())[0, 1]
        
        print(f"\n9-patch vs batch-9 comparison:")
        print(f"  Max difference: {max_diff:.6e}")
        print(f"  Correlation:    {correlation:.10f}")
        
        if max_diff < 1e-4:
            print(f"  ✅ 9-patch matches batch-9 perfectly!")
        else:
            print(f"  ⚠️  9-patch differs from batch-9 (expected if aggregation logic differs)")
    
    print("\n" + "=" * 70)
    if success:
        print("✅ Overall verification: PASSED")
    else:
        print("❌ Overall verification: FAILED")
    print("=" * 70)
    
    return success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

