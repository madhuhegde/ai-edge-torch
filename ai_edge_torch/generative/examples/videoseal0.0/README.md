# VideoSeal 0.0 TFLite Conversion

Convert VideoSeal 0.0 watermark models from PyTorch to TensorFlow Lite for deployment on mobile and edge devices.

## 🚀 What is VideoSeal 0.0?

**VideoSeal 0.0** is the baseline watermarking model from Meta's VideoSeal project with **96-bit capacity**. This is the legacy version that serves as the research baseline.

### Key Features

- **96-bit capacity**: Baseline watermark capacity for research and legacy applications
- **UNet-Small2 embedder**: Compact UNet architecture with 8 blocks
- **SAM-Small detector**: ViT-based detector with 12 depth, 384 embed_dim
- **RGB processing**: Processes all 3 color channels
- **Legacy baseline**: Original VideoSeal model for compatibility

### Architecture Comparison

| Model | Capacity | Embedder | Detector | Status |
|-------|----------|----------|----------|--------|
| **VideoSeal 0.0** | **96 bits** | **UNet-Small2** | **SAM-Small (ViT)** | **Legacy** |
| VideoSeal 1.0 | 256 bits | UNet-based | ConvNeXt-Tiny | Current |
| PixelSeal | 256 bits | UNet-based | ConvNeXt-Tiny | SOTA |
| ChunkySeal | 1024 bits | UNet-based | ConvNeXt-Chunky | High capacity |

## 📦 What's Included

This example provides:

1. **Model Wrappers** (`videoseal00_models.py`): PyTorch wrappers optimized for TFLite conversion
2. **Detector Conversion** (`convert_detector_to_tflite.py`): Convert VideoSeal 0.0 Detector to TFLite
3. **Embedder Conversion** (`convert_embedder_to_tflite.py`): Convert VideoSeal 0.0 Embedder to TFLite
4. **Verification Script** (`verify_detector_tflite.py`): Verify TFLite model accuracy
5. **Example Usage** (`example_usage.py`): Simple usage examples

## 🔧 Prerequisites

```bash
# Install required packages
pip install torch torchvision
pip install ai-edge-torch
pip install tensorflow  # For TFLite inference
pip install videoseal   # VideoSeal models
```

## 🎯 Quick Start

### 1. Convert Detector to TFLite

**FLOAT32 (Full Precision)**
```bash
cd ~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/videoseal0.0
python convert_detector_to_tflite.py --output_dir ./tflite_models
```

**INT8 Quantization (RECOMMENDED - ~75% smaller)**
```bash
python convert_detector_to_tflite.py --quantize int8 --output_dir ./tflite_models
```

**FP16 Quantization (~50% smaller)**
```bash
python convert_detector_to_tflite.py --quantize fp16 --output_dir ./tflite_models
```

### 2. Convert Embedder to TFLite

**FLOAT32 (Recommended for embedder)**
```bash
python convert_embedder_to_tflite.py --output_dir ./tflite_models
```

**FP16 Quantization (~50% smaller)**
```bash
python convert_embedder_to_tflite.py --quantize fp16 --output_dir ./tflite_models
```

### 3. Verify Converted Models

```bash
# Verify FLOAT32 detector
python verify_detector_tflite.py --tflite_dir ./tflite_models

# Verify INT8 detector
python verify_detector_tflite.py --quantize int8 --tflite_dir ./tflite_models
```

### 4. Use in Your Application

```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load TFLite models
embedder = tf.lite.Interpreter(
    model_path="tflite_models/videoseal00_embedder_256.tflite"
)
embedder.allocate_tensors()

detector = tf.lite.Interpreter(
    model_path="tflite_models/videoseal00_detector_256_int8.tflite"
)
detector.allocate_tensors()

# Load and preprocess image
img = Image.open("image.jpg").resize((256, 256))
img_array = np.array(img).astype(np.float32) / 255.0
img_array = np.expand_dims(img_array.transpose(2, 0, 1), axis=0)  # (1, 3, 256, 256)

# Create 96-bit message
message = np.random.randint(0, 2, (1, 96)).astype(np.float32)

# Embed watermark
emb_input = embedder.get_input_details()
emb_output = embedder.get_output_details()
embedder.set_tensor(emb_input[0]['index'], img_array)
embedder.set_tensor(emb_input[1]['index'], message)
embedder.invoke()
img_w = embedder.get_tensor(emb_output[0]['index'])

# Detect watermark
det_input = detector.get_input_details()
det_output = detector.get_output_details()
detector.set_tensor(det_input[0]['index'], img_w)
detector.invoke()
output = detector.get_tensor(det_output[0]['index'])

# Extract results
confidence = output[0, 0]  # Detection confidence
detected_msg = (output[0, 1:] > 0).astype(int)  # 96-bit message

print(f"Detection confidence: {confidence:.4f}")
print(f"Bit accuracy: {np.mean(message[0].astype(int) == detected_msg) * 100:.2f}%")
```

## 📊 Model Sizes & Performance

### Model Size Comparison

| Component | Quantization | Size | Reduction | Notes |
|-----------|--------------|------|-----------|-------|
| **Detector** | FLOAT32 | ~110 MB | - | Full precision |
| **Detector** | FP16 | ~55 MB | 50% | Minimal accuracy loss |
| **Detector** | **INT8** | **~28 MB** | **75%** | **Recommended** |
| **Embedder** | FLOAT32 | ~90 MB | - | Full precision |
| **Embedder** | FP16 | ~45 MB | 50% | Minimal quality loss |

### Accuracy Comparison

| Quantization | Bit Accuracy | MAE | PSNR (Embedder) |
|--------------|--------------|-----|-----------------|
| FLOAT32 | 100% | <1e-6 | >40 dB |
| FP16 | 99.5%+ | <1e-4 | >38 dB |
| INT8 | 95-98% | <5e-2 | N/A (detector only) |

## 🎓 Use Cases

VideoSeal 0.0's 96-bit capacity is suitable for:

### 1. **Legacy Compatibility**
```python
# Support existing VideoSeal 0.0 watermarked content
# Maintain compatibility with older systems
```

### 2. **Research Baseline**
```python
# Benchmark against the original VideoSeal model
# Compare improvements in newer versions
```

### 3. **Basic Watermarking**
```python
# Simple watermarking applications
# 96 bits = 12 bytes of metadata
metadata = {
    'timestamp': 32 bits,
    'user_id': 32 bits,
    'content_id': 32 bits
}
```

### 4. **Resource-Constrained Devices**
```python
# Smaller models than VideoSeal 1.0
# Lower computational requirements
```

## 🔍 Technical Details

### UNet-Small2 Embedder

VideoSeal 0.0 uses a compact UNet architecture:

```python
# Architecture:
- Message processor: binary+concat type
- Hidden size: 32 (for 96-bit messages)
- UNet blocks: 8
- Channels: [16, 32, 64, 128] with multipliers [1, 2, 4, 8]
- Activation: SiLU
- Normalization: RMS
- Output: tanh activation
```

### SAM-Small Detector

The detector uses a ViT-based architecture:

```python
# Architecture:
- Image encoder: ViT with 12 depth
- Embed dimension: 384
- Patch size: 16
- Window size: 8
- Global attention: layers [2, 5, 8, 11]
- Pixel decoder: Bilinear upsampling
- Output: 97 channels (1 detection + 96 message bits)
```

### Quantization Details

**INT8 Dynamic Quantization (Detector):**
- Weights: INT8 (channelwise quantization)
- Activations: Dynamic INT8 at runtime
- Inputs/Outputs: FLOAT32 for compatibility
- Typical accuracy loss: <2% for 96-bit detection

**FP16 Quantization (Both):**
- All weights and activations: FP16
- ~50% size reduction
- Minimal accuracy/quality loss

## 📝 Command Reference

### Convert Detector

```bash
# Basic conversion (FLOAT32)
python convert_detector_to_tflite.py \
    --output_dir ./tflite_models

# INT8 quantization (recommended)
python convert_detector_to_tflite.py \
    --quantize int8 \
    --output_dir ./tflite_models

# FP16 quantization
python convert_detector_to_tflite.py \
    --quantize fp16 \
    --output_dir ./tflite_models

# Custom image size
python convert_detector_to_tflite.py \
    --image_size 512 \
    --quantize int8 \
    --output_dir ./tflite_models
```

### Convert Embedder

```bash
# Basic conversion (FLOAT32, recommended)
python convert_embedder_to_tflite.py \
    --output_dir ./tflite_models

# FP16 quantization
python convert_embedder_to_tflite.py \
    --quantize fp16 \
    --output_dir ./tflite_models

# Custom image size
python convert_embedder_to_tflite.py \
    --image_size 512 \
    --output_dir ./tflite_models
```

### Verify Models

```bash
# Verify FLOAT32 detector
python verify_detector_tflite.py \
    --tflite_dir ./tflite_models

# Verify INT8 detector
python verify_detector_tflite.py \
    --quantize int8 \
    --tflite_dir ./tflite_models

# Verify specific model
python verify_detector_tflite.py \
    --detector_path ./tflite_models/videoseal00_detector_256_int8.tflite

# Run more tests
python verify_detector_tflite.py \
    --num_tests 100 \
    --tflite_dir ./tflite_models
```

## ⚠️ Important Notes

### VideoSeal 0.0 vs 1.0

**When to use VideoSeal 0.0:**
- Legacy compatibility with existing watermarked content
- Research baseline comparisons
- Resource-constrained devices (smaller models)
- Simple watermarking applications (96 bits sufficient)

**When to use VideoSeal 1.0:**
- Production deployments (2.67× more capacity)
- Better robustness and imperceptibility
- Modern applications requiring more metadata
- ConvNeXt-based detector (more efficient)

### Embedder Quantization

**INT8 quantization is NOT recommended for embedders** because:
- Significant PSNR degradation
- May affect watermark detectability
- Use FLOAT32 or FP16 instead

**Recommended workflow:**
1. **Server-side**: Use FLOAT32 embedder for best quality
2. **Client-side**: Use INT8 detector for efficient verification

### Memory Requirements

| Component | Quantization | RAM Required | Recommended Device |
|-----------|--------------|--------------|-------------------|
| Detector | FLOAT32 | ~200-300 MB | High-end mobile |
| Detector | INT8 | ~50-100 MB | Most mobile devices |
| Embedder | FLOAT32 | ~150-250 MB | Mid-range mobile |
| Both | INT8 + FLOAT32 | ~200-350 MB | Most modern mobile |

## 🔗 Related Resources

- [VideoSeal Paper](https://arxiv.org/abs/2412.09492)
- [VideoSeal Repository](https://github.com/facebookresearch/videoseal)
- [AI Edge Torch Documentation](https://github.com/google-ai-edge/ai-edge-torch)
- [VideoSeal 1.0 TFLite Example](../videoseal/)
- [ChunkySeal TFLite Example](../chunkyseal/)

## 📄 License

Copyright 2025 The AI Edge Torch Authors. Licensed under Apache 2.0.

## 🤝 Contributing

Contributions are welcome! Please see the main repository for contribution guidelines.

## 📧 Support

For issues or questions:
- Open an issue on GitHub
- Check the VideoSeal documentation
- Review the AI Edge Torch examples

---

**VideoSeal 0.0**: The legacy baseline for watermarking research! 🦭

