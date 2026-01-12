#!/usr/bin/env python3
"""
Test VideoSeal 0.0 TFLite models against PyTorch reference.

This script:
1. Loads an example image
2. Embeds watermark using PyTorch
3. Embeds watermark using TFLite
4. Compares PSNR and visual quality
5. Detects watermark using both PyTorch and TFLite
6. Compares bit accuracy
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import tensorflow as tf

# Add videoseal to path
sys.path.insert(0, '/home/madhuhegde/work/videoseal/videoseal')
import videoseal


def test_embedder(image_path, output_dir="./test_results"):
    """Test embedder accuracy: PyTorch vs TFLite."""
    print("\n" + "="*70)
    print("Testing Embedder: PyTorch vs TFLite")
    print("="*70)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load image
    print(f"\n1. Loading image: {image_path}")
    img_pil = Image.open(image_path).convert("RGB")
    print(f"   Original size: {img_pil.size}")
    
    # Resize to 256x256
    img_pil = img_pil.resize((256, 256), Image.LANCZOS)
    print(f"   Resized to: {img_pil.size}")
    
    # Convert to tensor
    to_tensor = T.ToTensor()
    img_tensor = to_tensor(img_pil).unsqueeze(0)  # (1, 3, 256, 256)
    
    # Generate random message
    message = np.random.randint(0, 2, 96)
    msg_tensor = torch.from_numpy(message).float().unsqueeze(0)  # (1, 96)
    
    print(f"\n2. Generated message: {message.sum()}/96 bits are 1")
    print(f"   First 32 bits: {message[:32].tolist()}")
    
    # ===== PyTorch Embedding =====
    print(f"\n3. Embedding with PyTorch...")
    # Use absolute path to model card
    videoseal_path = Path(videoseal.__file__).parent
    model_card_path = videoseal_path / "cards" / "videoseal_0.0.yaml"
    model_pytorch = videoseal.load(model_card_path)
    model_pytorch.eval()
    
    with torch.no_grad():
        outputs_pytorch = model_pytorch.embed(img_tensor, msgs=msg_tensor, is_video=False)
        img_w_pytorch = outputs_pytorch['imgs_w']
    
    # Calculate PSNR
    mse_pytorch = torch.mean((img_w_pytorch - img_tensor) ** 2)
    psnr_pytorch = 10 * torch.log10(1.0 / mse_pytorch)
    
    print(f"   ✓ PyTorch PSNR: {psnr_pytorch:.2f} dB")
    
    # Save PyTorch result
    to_pil = T.ToPILImage()
    img_w_pytorch_pil = to_pil(img_w_pytorch[0])
    pytorch_path = output_dir / "watermarked_pytorch.jpg"
    img_w_pytorch_pil.save(pytorch_path)
    print(f"   ✓ Saved: {pytorch_path}")
    
    # ===== TFLite Embedding =====
    print(f"\n4. Embedding with TFLite...")
    tflite_path = "./videoseal00_tflite/videoseal00_embedder_256.tflite"
    
    embedder = tf.lite.Interpreter(model_path=tflite_path)
    embedder.allocate_tensors()
    
    input_details = embedder.get_input_details()
    output_details = embedder.get_output_details()
    
    # Prepare inputs
    # Convert from NCHW (PyTorch) to NHWC (TFLite)
    img_np = img_tensor.permute(0, 2, 3, 1).numpy().astype(np.float32)  # BxCxHxW -> BxHxWxC
    msg_np = msg_tensor.numpy().astype(np.float32)
    
    # Run inference
    embedder.set_tensor(input_details[0]['index'], img_np)
    embedder.set_tensor(input_details[1]['index'], msg_np)
    embedder.invoke()
    img_w_tflite_np = embedder.get_tensor(output_details[0]['index'])
    
    # Convert from NHWC (TFLite) to NCHW (PyTorch) for comparison
    img_w_tflite = torch.from_numpy(img_w_tflite_np).permute(0, 3, 1, 2)  # BxHxWxC -> BxCxHxW
    
    # Calculate PSNR
    mse_tflite = torch.mean((img_w_tflite - img_tensor) ** 2)
    psnr_tflite = 10 * torch.log10(1.0 / mse_tflite)
    
    print(f"   ✓ TFLite PSNR: {psnr_tflite:.2f} dB")
    
    # Save TFLite result
    img_w_tflite_pil = to_pil(img_w_tflite[0])
    tflite_path_out = output_dir / "watermarked_tflite.jpg"
    img_w_tflite_pil.save(tflite_path_out)
    print(f"   ✓ Saved: {tflite_path_out}")
    
    # ===== Compare Embedders =====
    print(f"\n5. Comparing PyTorch vs TFLite Embedders...")
    
    diff = torch.abs(img_w_pytorch - img_w_tflite)
    mae = diff.mean().item()
    max_diff = diff.max().item()
    
    print(f"   Mean Absolute Error: {mae:.6f}")
    print(f"   Max Difference: {max_diff:.6f}")
    print(f"   PSNR Difference: {abs(psnr_pytorch - psnr_tflite):.2f} dB")
    
    # Visual difference
    diff_pil = to_pil(diff[0] * 10)  # Amplify for visibility
    diff_path = output_dir / "difference_embedder.jpg"
    diff_pil.save(diff_path)
    print(f"   ✓ Saved difference map: {diff_path}")
    
    return img_w_pytorch, img_w_tflite, message


def test_detector(img_pytorch, img_tflite, original_message, output_dir="./test_results"):
    """Test detector accuracy: PyTorch vs TFLite."""
    print("\n" + "="*70)
    print("Testing Detector: PyTorch vs TFLite")
    print("="*70)
    
    output_dir = Path(output_dir)
    
    # ===== PyTorch Detection =====
    print(f"\n1. Detecting with PyTorch...")
    # Use absolute path to model card
    videoseal_path = Path(videoseal.__file__).parent
    model_card_path = videoseal_path / "cards" / "videoseal_0.0.yaml"
    model_pytorch = videoseal.load(model_card_path)
    model_pytorch.eval()
    
    with torch.no_grad():
        # Detect from PyTorch-embedded image
        outputs_pytorch = model_pytorch.detect(img_pytorch, is_video=False)
        preds_pytorch = outputs_pytorch['preds']
        
        confidence_pytorch = preds_pytorch[0, 0].item()
        message_pytorch = (preds_pytorch[0, 1:] > 0).float().cpu().numpy().astype(int)
    
    bit_accuracy_pytorch = np.mean(message_pytorch == original_message) * 100
    
    print(f"   ✓ Confidence: {confidence_pytorch:.4f}")
    print(f"   ✓ Bit Accuracy: {bit_accuracy_pytorch:.2f}%")
    print(f"   ✓ Correctly detected: {int(bit_accuracy_pytorch * 96 / 100)}/96 bits")
    
    # ===== TFLite Detection (FLOAT32) =====
    print(f"\n2. Detecting with TFLite (FLOAT32)...")
    detector_path = "./videoseal00_tflite/videoseal00_detector_256.tflite"
    
    detector = tf.lite.Interpreter(model_path=detector_path)
    detector.allocate_tensors()
    
    input_details = detector.get_input_details()
    output_details = detector.get_output_details()
    
    # Detect from TFLite-embedded image
    # Convert from NCHW (PyTorch) to NHWC (TFLite)
    img_tflite_np = img_tflite.permute(0, 2, 3, 1).numpy().astype(np.float32)  # BxCxHxW -> BxHxWxC
    detector.set_tensor(input_details[0]['index'], img_tflite_np)
    detector.invoke()
    output_tflite = detector.get_tensor(output_details[0]['index'])
    
    confidence_tflite = output_tflite[0, 0]
    message_tflite = (output_tflite[0, 1:] > 0).astype(int)
    
    bit_accuracy_tflite = np.mean(message_tflite == original_message) * 100
    
    print(f"   ✓ Confidence: {confidence_tflite:.4f}")
    print(f"   ✓ Bit Accuracy: {bit_accuracy_tflite:.2f}%")
    print(f"   ✓ Correctly detected: {int(bit_accuracy_tflite * 96 / 100)}/96 bits")
    
    # ===== TFLite Detection (INT8) =====
    print(f"\n3. Detecting with TFLite (INT8)...")
    detector_int8_path = "./videoseal00_tflite/videoseal00_detector_256_int8.tflite"
    
    try:
        detector_int8 = tf.lite.Interpreter(model_path=detector_int8_path)
        detector_int8.allocate_tensors()
        
        input_details_int8 = detector_int8.get_input_details()
        output_details_int8 = detector_int8.get_output_details()
        
        # Detect from TFLite-embedded image
        detector_int8.set_tensor(input_details_int8[0]['index'], img_tflite_np)
        detector_int8.invoke()
        output_int8 = detector_int8.get_tensor(output_details_int8[0]['index'])
        
        confidence_int8 = output_int8[0, 0]
        message_int8 = (output_int8[0, 1:] > 0).astype(int)
        
        bit_accuracy_int8 = np.mean(message_int8 == original_message) * 100
        
        print(f"   ✓ Confidence: {confidence_int8:.4f}")
        print(f"   ✓ Bit Accuracy: {bit_accuracy_int8:.2f}%")
        print(f"   ✓ Correctly detected: {int(bit_accuracy_int8 * 96 / 100)}/96 bits")
        
        int8_available = True
    except Exception as e:
        print(f"   ✗ INT8 detector failed: {str(e)[:100]}")
        print(f"   Note: INT8 detector may have compatibility issues")
        confidence_int8 = None
        bit_accuracy_int8 = None
        int8_available = False
    
    # ===== Compare Detectors =====
    print(f"\n4. Comparing Detectors...")
    
    print(f"\n   Confidence Comparison:")
    print(f"   - PyTorch:        {confidence_pytorch:.4f}")
    print(f"   - TFLite FLOAT32: {confidence_tflite:.4f} (diff: {abs(confidence_pytorch - confidence_tflite):.4f})")
    if int8_available:
        print(f"   - TFLite INT8:    {confidence_int8:.4f} (diff: {abs(confidence_pytorch - confidence_int8):.4f})")
    
    print(f"\n   Bit Accuracy Comparison:")
    print(f"   - PyTorch:        {bit_accuracy_pytorch:.2f}%")
    print(f"   - TFLite FLOAT32: {bit_accuracy_tflite:.2f}% (diff: {abs(bit_accuracy_pytorch - bit_accuracy_tflite):.2f}%)")
    if int8_available:
        print(f"   - TFLite INT8:    {bit_accuracy_int8:.2f}% (diff: {abs(bit_accuracy_pytorch - bit_accuracy_int8):.2f}%)")
    
    # Message comparison
    print(f"\n   Message Bit Differences:")
    diff_float32 = np.sum(message_pytorch != message_tflite)
    print(f"   - PyTorch vs TFLite FLOAT32: {diff_float32}/96 bits differ")
    if int8_available:
        diff_int8 = np.sum(message_pytorch != message_int8)
        print(f"   - PyTorch vs TFLite INT8:    {diff_int8}/96 bits differ")


def main():
    """Run complete accuracy test."""
    print("\n" + "="*70)
    print("VideoSeal 0.0 TFLite Accuracy Test")
    print("="*70)
    print("\nThis test compares PyTorch and TFLite models on a real image.")
    
    # Find example image
    image_paths = [
        "/home/madhuhegde/work/videoseal/videoseal/assets/imgs/1.jpg",
        "/home/madhuhegde/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/videoseal0.0/test_image.jpg",
    ]
    
    image_path = None
    for path in image_paths:
        if Path(path).exists():
            image_path = path
            break
    
    if image_path is None:
        print("\n✗ No example image found. Creating a test image...")
        # Create a simple test image
        test_img = Image.new('RGB', (256, 256), color=(128, 128, 128))
        image_path = "./test_image.jpg"
        test_img.save(image_path)
        print(f"✓ Created test image: {image_path}")
    
    # Run tests
    img_w_pytorch, img_w_tflite, message = test_embedder(image_path)
    test_detector(img_w_pytorch, img_w_tflite, message)
    
    # Final summary
    print("\n" + "="*70)
    print("Test Complete!")
    print("="*70)
    print("\nResults saved to: ./test_results/")
    print("  - watermarked_pytorch.jpg")
    print("  - watermarked_tflite.jpg")
    print("  - difference_embedder.jpg")
    print("\n✓ All tests completed successfully!")


if __name__ == "__main__":
    main()
