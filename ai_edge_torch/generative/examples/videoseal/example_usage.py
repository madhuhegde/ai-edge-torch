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
Example usage of VideoSeal TFLite models.

This script demonstrates how to use the converted TFLite models for
watermark embedding and detection.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path


def load_tflite_model(model_path):
    """Load a TFLite model and return the interpreter."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


def embed_watermark(embedder, image, message):
    """
    Embed a watermark into an image.
    
    Args:
        embedder: TFLite interpreter for embedder model
        image: numpy array of shape (1, 3, H, W) in [0, 1] range
        message: numpy array of shape (1, 256) with binary values
    
    Returns:
        Watermarked image of shape (1, 3, H, W)
    """
    input_details = embedder.get_input_details()
    output_details = embedder.get_output_details()
    
    # Set inputs
    embedder.set_tensor(input_details[0]['index'], image.astype(np.float32))
    embedder.set_tensor(input_details[1]['index'], message.astype(np.float32))
    
    # Run inference
    embedder.invoke()
    
    # Get output
    watermarked = embedder.get_tensor(output_details[0]['index'])
    return watermarked


def detect_watermark(detector, image):
    """
    Detect and extract watermark from an image.
    
    Args:
        detector: TFLite interpreter for detector model
        image: numpy array of shape (1, 3, H, W) in [0, 1] range
    
    Returns:
        Tuple of (detection_mask, message)
        - detection_mask: Confidence that watermark is present
        - message: Detected binary message of shape (256,)
    """
    input_details = detector.get_input_details()
    output_details = detector.get_output_details()
    
    # Set input
    detector.set_tensor(input_details[0]['index'], image.astype(np.float32))
    
    # Run inference
    detector.invoke()
    
    # Get output
    predictions = detector.get_tensor(output_details[0]['index'])
    
    # Extract detection mask and message
    detection_mask = predictions[0, 0]  # First channel
    message_logits = predictions[0, 1:]  # Remaining 256 channels
    
    # Apply threshold to get binary message
    # Average over spatial dimensions if needed
    if len(message_logits.shape) > 1:
        message_logits = message_logits.mean(axis=(1, 2))
    
    message = (message_logits > 0).astype(np.float32)
    
    return detection_mask, message


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


def calculate_bit_accuracy(msg1, msg2):
    """Calculate bit accuracy between two messages."""
    return np.mean(msg1 == msg2) * 100


def main():
    print("="*70)
    print("VideoSeal TFLite Example Usage")
    print("="*70)
    
    # Configuration
    tflite_dir = Path("./videoseal_tflite")
    embedder_path = tflite_dir / "videoseal_embedder_videoseal_256.tflite"
    detector_path = tflite_dir / "videoseal_detector_videoseal_256.tflite"
    image_size = 256
    
    print(f"\nConfiguration:")
    print(f"  TFLite directory: {tflite_dir}")
    print(f"  Embedder model: {embedder_path.name}")
    print(f"  Detector model: {detector_path.name}")
    print(f"  Image size: {image_size}×{image_size}")
    
    # Check if models exist
    if not embedder_path.exists() or not detector_path.exists():
        print("\n✗ Error: TFLite models not found!")
        print("\nPlease run the conversion script first:")
        print("  python convert_to_tflite.py --output_dir ./videoseal_tflite")
        return 1
    
    # Load models
    print("\n" + "-"*70)
    print("Loading Models")
    print("-"*70)
    
    print("Loading embedder...")
    embedder = load_tflite_model(embedder_path)
    print(f"✓ Embedder loaded ({embedder_path.stat().st_size / (1024**2):.1f} MB)")
    
    print("Loading detector...")
    detector = load_tflite_model(detector_path)
    print(f"✓ Detector loaded ({detector_path.stat().st_size / (1024**2):.1f} MB)")
    
    # Create test data
    print("\n" + "-"*70)
    print("Creating Test Data")
    print("-"*70)
    
    # Generate random image
    original_image = np.random.rand(1, 3, image_size, image_size).astype(np.float32)
    print(f"Original image shape: {original_image.shape}")
    print(f"Original image range: [{original_image.min():.3f}, {original_image.max():.3f}]")
    
    # Generate random message
    original_message = np.random.randint(0, 2, (1, 256)).astype(np.float32)
    num_ones = int(original_message.sum())
    print(f"Original message shape: {original_message.shape}")
    print(f"Original message: {num_ones} ones, {256 - num_ones} zeros")
    print(f"Message (first 32 bits): {original_message[0, :32].astype(int)}")
    
    # Embed watermark
    print("\n" + "-"*70)
    print("Embedding Watermark")
    print("-"*70)
    
    print("Running embedder...")
    watermarked_image = embed_watermark(embedder, original_image, original_message)
    
    print(f"✓ Watermarked image shape: {watermarked_image.shape}")
    print(f"  Watermarked image range: [{watermarked_image.min():.3f}, {watermarked_image.max():.3f}]")
    
    # Calculate image quality
    psnr = calculate_psnr(original_image, watermarked_image)
    mse = np.mean((original_image - watermarked_image) ** 2)
    max_diff = np.max(np.abs(original_image - watermarked_image))
    
    print(f"  PSNR: {psnr:.2f} dB (higher is better)")
    print(f"  MSE: {mse:.6f}")
    print(f"  Max difference: {max_diff:.6f}")
    
    if psnr > 40:
        print("  ✓ Excellent imperceptibility (PSNR > 40 dB)")
    elif psnr > 30:
        print("  ✓ Good imperceptibility (PSNR > 30 dB)")
    else:
        print("  ⚠ Watermark may be visible (PSNR < 30 dB)")
    
    # Detect watermark
    print("\n" + "-"*70)
    print("Detecting Watermark")
    print("-"*70)
    
    print("Running detector...")
    detection_mask, detected_message = detect_watermark(detector, watermarked_image)
    
    print(f"✓ Detection mask mean: {detection_mask.mean():.3f}")
    print(f"  Detection mask range: [{detection_mask.min():.3f}, {detection_mask.max():.3f}]")
    
    num_detected_ones = int(detected_message.sum())
    print(f"  Detected message: {num_detected_ones} ones, {256 - num_detected_ones} zeros")
    print(f"  Message (first 32 bits): {detected_message[:32].astype(int)}")
    
    # Calculate bit accuracy
    bit_accuracy = calculate_bit_accuracy(original_message[0], detected_message)
    num_correct = int(np.sum(original_message[0] == detected_message))
    num_errors = 256 - num_correct
    
    print(f"\n  Bit accuracy: {bit_accuracy:.2f}% ({num_correct}/256 bits correct)")
    print(f"  Bit errors: {num_errors}")
    
    if bit_accuracy == 100:
        print("  ✓ Perfect detection!")
    elif bit_accuracy > 95:
        print("  ✓ Excellent detection (>95%)")
    elif bit_accuracy > 90:
        print("  ✓ Good detection (>90%)")
    else:
        print("  ⚠ Poor detection (<90%)")
    
    # Test with unmodified image
    print("\n" + "-"*70)
    print("Testing with Original (Unwatermarked) Image")
    print("-"*70)
    
    print("Running detector on original image...")
    detection_mask_orig, detected_message_orig = detect_watermark(detector, original_image)
    
    print(f"Detection mask mean: {detection_mask_orig.mean():.3f}")
    
    # Compare with watermarked detection
    mask_diff = abs(detection_mask.mean() - detection_mask_orig.mean())
    print(f"Mask difference: {mask_diff:.3f}")
    
    if mask_diff > 0.5:
        print("✓ Good separation between watermarked and unwatermarked images")
    else:
        print("⚠ Weak separation between watermarked and unwatermarked images")
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    print(f"\n✓ Embedding:")
    print(f"  - PSNR: {psnr:.2f} dB")
    print(f"  - Imperceptibility: {'Excellent' if psnr > 40 else 'Good' if psnr > 30 else 'Fair'}")
    
    print(f"\n✓ Detection:")
    print(f"  - Bit accuracy: {bit_accuracy:.2f}%")
    print(f"  - Bit errors: {num_errors}/256")
    print(f"  - Quality: {'Perfect' if bit_accuracy == 100 else 'Excellent' if bit_accuracy > 95 else 'Good' if bit_accuracy > 90 else 'Fair'}")
    
    print(f"\n✓ Separation:")
    print(f"  - Mask difference: {mask_diff:.3f}")
    print(f"  - Quality: {'Good' if mask_diff > 0.5 else 'Weak'}")
    
    print("\n" + "="*70)
    print("✓ Example completed successfully!")
    print("="*70)
    
    return 0


if __name__ == '__main__':
    exit(main())

