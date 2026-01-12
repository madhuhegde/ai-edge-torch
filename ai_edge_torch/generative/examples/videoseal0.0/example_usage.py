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
Example usage of VideoSeal 0.0 TFLite models.

This script demonstrates how to use the converted TFLite models for
watermark embedding and detection.

Usage:
    python example_usage.py
"""

import numpy as np
from PIL import Image
import tensorflow as tf


def example_detector():
    """Example: Detect watermark using TFLite detector."""
    print("\n" + "="*70)
    print("Example 1: Watermark Detection with TFLite")
    print("="*70)
    
    # Load TFLite detector
    tflite_path = "./videoseal00_tflite/videoseal00_detector_256.tflite"
    
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        print(f"✓ Loaded TFLite detector: {tflite_path}")
    except Exception as e:
        print(f"✗ Failed to load detector: {e}")
        print(f"\nPlease run conversion first:")
        print(f"  python convert_detector_to_tflite.py --output_dir ./videoseal00_tflite")
        return
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"\nModel Info:")
    print(f"  Input shape: {input_details[0]['shape']}")
    print(f"  Output shape: {output_details[0]['shape']}")
    
    # Create sample watermarked image (in practice, load a real image)
    print(f"\nCreating sample image...")
    img = np.random.rand(1, 3, 256, 256).astype(np.float32)
    
    # Run detection
    print(f"Running detection...")
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    # Extract results
    confidence = output[0, 0]
    message_logits = output[0, 1:]
    message = (message_logits > 0).astype(int)
    
    print(f"\n✓ Detection Results:")
    print(f"  Confidence: {confidence:.4f}")
    print(f"  Message bits: {message.sum()}/96 are 1")
    print(f"  First 32 bits: {message[:32].tolist()}")
    
    # Interpret confidence
    if confidence > 0.5:
        print(f"\n✓ Watermark DETECTED (confidence: {confidence:.2%})")
    else:
        print(f"\n✗ No watermark detected (confidence: {confidence:.2%})")


def example_embedder():
    """Example: Embed watermark using TFLite embedder."""
    print("\n" + "="*70)
    print("Example 2: Watermark Embedding with TFLite")
    print("="*70)
    
    # Load TFLite embedder
    tflite_path = "./videoseal00_tflite/videoseal00_embedder_256.tflite"
    
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        print(f"✓ Loaded TFLite embedder: {tflite_path}")
    except Exception as e:
        print(f"✗ Failed to load embedder: {e}")
        print(f"\nPlease run conversion first:")
        print(f"  python convert_embedder_to_tflite.py --output_dir ./videoseal00_tflite")
        return
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"\nModel Info:")
    print(f"  Input 0 (image) shape: {input_details[0]['shape']}")
    print(f"  Input 1 (message) shape: {input_details[1]['shape']}")
    print(f"  Output shape: {output_details[0]['shape']}")
    
    # Create sample image and message
    print(f"\nPreparing inputs...")
    img = np.random.rand(1, 3, 256, 256).astype(np.float32)
    message = np.random.randint(0, 2, (1, 96)).astype(np.float32)
    
    print(f"  Image shape: {img.shape}")
    print(f"  Message: {message.sum().astype(int)}/96 bits are 1")
    print(f"  First 32 bits: {message[0, :32].astype(int).tolist()}")
    
    # Run embedding
    print(f"\nEmbedding watermark...")
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.set_tensor(input_details[1]['index'], message)
    interpreter.invoke()
    img_w = interpreter.get_tensor(output_details[0]['index'])
    
    # Calculate PSNR
    mse = np.mean((img_w - img) ** 2)
    psnr = 10 * np.log10(1.0 / mse)
    
    print(f"\n✓ Embedding Results:")
    print(f"  Output shape: {img_w.shape}")
    print(f"  Output range: [{img_w.min():.3f}, {img_w.max():.3f}]")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  Quality: {'Excellent' if psnr > 40 else 'Good' if psnr > 35 else 'Fair'}")


def example_end_to_end():
    """Example: Complete watermarking pipeline."""
    print("\n" + "="*70)
    print("Example 3: End-to-End Watermarking Pipeline")
    print("="*70)
    
    # Load both models
    embedder_path = "./videoseal00_tflite/videoseal00_embedder_256.tflite"
    detector_path = "./videoseal00_tflite/videoseal00_detector_256.tflite"
    
    try:
        embedder = tf.lite.Interpreter(model_path=embedder_path)
        embedder.allocate_tensors()
        print(f"✓ Loaded embedder")
        
        detector = tf.lite.Interpreter(model_path=detector_path)
        detector.allocate_tensors()
        print(f"✓ Loaded detector")
    except Exception as e:
        print(f"✗ Failed to load models: {e}")
        print(f"\nPlease run conversions first:")
        print(f"  python convert_embedder_to_tflite.py --output_dir ./videoseal00_tflite")
        print(f"  python convert_detector_to_tflite.py --output_dir ./videoseal00_tflite")
        return
    
    # Prepare inputs
    print(f"\n1. Preparing original image and message...")
    img = np.random.rand(1, 3, 256, 256).astype(np.float32)
    message = np.random.randint(0, 2, (1, 96)).astype(np.float32)
    
    print(f"   Original message: {message.sum().astype(int)}/96 bits are 1")
    
    # Embed watermark
    print(f"\n2. Embedding watermark...")
    emb_input = embedder.get_input_details()
    emb_output = embedder.get_output_details()
    
    embedder.set_tensor(emb_input[0]['index'], img)
    embedder.set_tensor(emb_input[1]['index'], message)
    embedder.invoke()
    img_w = embedder.get_tensor(emb_output[0]['index'])
    
    # Calculate PSNR
    mse = np.mean((img_w - img) ** 2)
    psnr = 10 * np.log10(1.0 / mse)
    print(f"   PSNR: {psnr:.2f} dB")
    
    # Detect watermark
    print(f"\n3. Detecting watermark...")
    det_input = detector.get_input_details()
    det_output = detector.get_output_details()
    
    detector.set_tensor(det_input[0]['index'], img_w)
    detector.invoke()
    output = detector.get_tensor(det_output[0]['index'])
    
    confidence = output[0, 0]
    detected_msg = (output[0, 1:] > 0).astype(int)
    
    # Compare messages
    bit_accuracy = np.mean(message[0].astype(int) == detected_msg) * 100
    
    print(f"\n✓ Pipeline Results:")
    print(f"  Detection confidence: {confidence:.4f}")
    print(f"  Bit accuracy: {bit_accuracy:.2f}%")
    print(f"  Correctly detected: {int(bit_accuracy * 96 / 100)}/96 bits")
    
    if bit_accuracy > 95:
        print(f"\n✓ SUCCESS - Watermark embedded and detected correctly!")
    else:
        print(f"\n⚠️  WARNING - Low bit accuracy, check model quality")


def example_with_real_image():
    """Example: Use with real image file."""
    print("\n" + "="*70)
    print("Example 4: Watermarking Real Images")
    print("="*70)
    
    # This example shows the structure for real images
    # You'll need to provide actual image files
    
    print("\nTo use with real images:")
    print("\n1. Load and preprocess image:")
    print("""
    from PIL import Image
    import numpy as np
    
    # Load image
    img = Image.open("input.jpg").convert("RGB")
    img = img.resize((256, 256))
    
    # Convert to tensor format
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
    img_tensor = np.expand_dims(img_tensor, axis=0)  # Add batch dim
    """)
    
    print("\n2. Embed watermark:")
    print("""
    # Create message (96 bits)
    message = np.random.randint(0, 2, (1, 96)).astype(np.float32)
    
    # Embed using TFLite embedder
    embedder.set_tensor(input_details[0]['index'], img_tensor)
    embedder.set_tensor(input_details[1]['index'], message)
    embedder.invoke()
    img_w_tensor = embedder.get_tensor(output_details[0]['index'])
    """)
    
    print("\n3. Save watermarked image:")
    print("""
    # Convert back to image format
    img_w_array = img_w_tensor[0]  # Remove batch dim
    img_w_array = np.transpose(img_w_array, (1, 2, 0))  # CHW -> HWC
    img_w_array = np.clip(img_w_array * 255, 0, 255).astype(np.uint8)
    
    # Save
    img_w = Image.fromarray(img_w_array)
    img_w.save("watermarked.jpg")
    """)
    
    print("\n4. Detect watermark:")
    print("""
    # Load watermarked image
    img_w = Image.open("watermarked.jpg").convert("RGB")
    img_w = img_w.resize((256, 256))
    img_w_tensor = preprocess_image(img_w)
    
    # Detect using TFLite detector
    detector.set_tensor(input_details[0]['index'], img_w_tensor)
    detector.invoke()
    output = detector.get_tensor(output_details[0]['index'])
    
    # Extract message
    confidence = output[0, 0]
    message = (output[0, 1:] > 0).astype(int)
    """)


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("VideoSeal 0.0 TFLite Usage Examples")
    print("="*70)
    print("\nThese examples demonstrate how to use VideoSeal 0.0 TFLite models")
    print("for watermark embedding and detection.")
    
    # Run examples
    example_detector()
    example_embedder()
    example_end_to_end()
    example_with_real_image()
    
    print("\n" + "="*70)
    print("Examples Complete")
    print("="*70)
    print("\nFor more information, see:")
    print("  - README.md: Complete documentation")
    print("  - convert_detector_to_tflite.py: Convert detector")
    print("  - convert_embedder_to_tflite.py: Convert embedder")
    print("  - verify_detector_tflite.py: Verify accuracy")


if __name__ == '__main__':
    main()
