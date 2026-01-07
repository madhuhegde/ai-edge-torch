#!/usr/bin/env python3
"""Verify DistortionNet 3-patch TFLite model against PyTorch.

This script compares the output of the 3-patch DistortionNet TFLite model
with the PyTorch implementation to ensure they match.
"""

import numpy as np
import torch
import tensorflow as tf
from pathlib import Path
import sys

# Add UVQ source path
UVQ_SOURCE_PATH = Path.home() / "work" / "UVQ" / "uvq"
sys.path.insert(0, str(UVQ_SOURCE_PATH))

from uvq_models import DistortionNet3PatchWrapper


def verify_3patch_model():
    """Verify the 3-patch DistortionNet model."""
    print("="*70)
    print("Verifying DistortionNet 3-Patch Model")
    print("="*70)
    
    # 1. Load PyTorch model
    print("\n1. Loading PyTorch model...")
    pytorch_model = DistortionNet3PatchWrapper(eval_mode=True)
    pytorch_model.eval()
    
    # 2. Load TFLite model
    print("\n2. Loading TFLite model...")
    tflite_model_path = Path.home() / "work" / "UVQ" / "uvq" / "models" / "tflite_models" / "uvq1.5" / "distortion_net_3patch.tflite"
    
    if not tflite_model_path.exists():
        print(f"✗ TFLite model not found: {tflite_model_path}")
        return False
    
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"   Input shape:  {input_details[0]['shape']}")
    print(f"   Output shape: {output_details[0]['shape']}")
    
    # 3. Create test input (3 patches)
    print("\n3. Creating test input (3 patches)...")
    # Use a fixed seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    test_input_np = np.random.randn(3, 360, 640, 3).astype(np.float32) * 2  # Range approx [-6, 6]
    test_input_torch = torch.from_numpy(test_input_np)
    
    print(f"   Input shape: {test_input_np.shape}")
    print(f"   Input range: [{test_input_np.min():.2f}, {test_input_np.max():.2f}]")
    
    # 4. Run PyTorch inference
    print("\n4. Running PyTorch inference...")
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input_torch).numpy()
    print(f"   Output shape: {pytorch_output.shape}")
    print(f"   Output range: [{pytorch_output.min():.2f}, {pytorch_output.max():.2f}]")
    
    # 5. Run TFLite inference
    print("\n5. Running TFLite inference...")
    interpreter.set_tensor(input_details[0]['index'], test_input_np)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])
    print(f"   Output shape: {tflite_output.shape}")
    print(f"   Output range: [{tflite_output.min():.2f}, {tflite_output.max():.2f}]")
    
    # 6. Compare outputs
    print("\n6. Comparing outputs...")
    
    if pytorch_output.shape != tflite_output.shape:
        print(f"   ✗ Shape mismatch! PyTorch: {pytorch_output.shape}, TFLite: {tflite_output.shape}")
        return False
    
    print(f"   ✓ Shapes match: {pytorch_output.shape}")
    
    abs_diff = np.abs(pytorch_output - tflite_output)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)
    
    # Calculate relative error
    rel_diff = abs_diff / (np.abs(pytorch_output) + 1e-8)
    max_rel_diff = np.max(rel_diff)
    mean_rel_diff = np.mean(rel_diff)
    
    # Calculate correlation
    correlation = np.corrcoef(pytorch_output.flatten(), tflite_output.flatten())[0, 1]
    
    print(f"\n   Absolute differences:")
    print(f"     Max:  {max_abs_diff:.6f}")
    print(f"     Mean: {mean_abs_diff:.6f}")
    
    print(f"\n   Relative differences:")
    print(f"     Max:  {max_rel_diff:.6f}")
    print(f"     Mean: {mean_rel_diff:.6f}")
    
    print(f"\n   Correlation: {correlation:.8f}")
    
    # Check if results match
    if correlation >= 0.9999 and max_abs_diff < 0.1:
        print(f"\n   ✅ Excellent match (correlation >= 0.9999 and max_diff < 0.1)")
        return True
    elif correlation >= 0.999 and max_abs_diff < 1.0:
        print(f"\n   ✓ Good match (correlation >= 0.999 and max_diff < 1.0)")
        return True
    else:
        print(f"\n   ⚠️  Results differ more than expected")
        return False


def main():
    print("\n" + "="*70)
    print("DistortionNet 3-Patch Model Verification")
    print("="*70)
    
    success = verify_3patch_model()
    
    print("\n" + "="*70)
    if success:
        print("✅ Verification PASSED")
    else:
        print("❌ Verification FAILED")
    print("="*70)
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())

