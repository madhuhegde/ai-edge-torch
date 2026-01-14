# VideoSeal Detector TFLite Conversion

Convert VideoSeal Detector (watermark detection and extraction) to TFLite format for mobile and edge deployment.

## Overview

This module provides tools to convert the **VideoSeal Detector** from PyTorch to TFLite. The Detector can:
- ✅ Detect if an image contains a watermark
- ✅ Extract the embedded 256-bit (or 1024-bit) message
- ✅ Run efficiently on mobile and edge devices

**Note**: The Embedder (watermark insertion) is not yet supported for TFLite conversion due to ai-edge-torch limitations with VideoSeal's dynamic tensor operations. Use PyTorch for embedding watermarks on the server side.

## Quick Start

### 1. Convert Detector to TFLite

```bash
cd ~/work/videoseal/videoseal  # Important: run from VideoSeal directory
eval "$(micromamba shell hook --shell bash)"
micromamba activate local_tf_env

python ~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/videoseal/convert_detector_to_tflite.py \
    --output_dir ~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/videoseal/videoseal_tflite
```

### 2. Verify the Converted Model

```bash
python ~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/videoseal/verify_detector_tflite.py \
    --tflite_dir ~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/videoseal/videoseal_tflite
```

## Model Details

### VideoSeal Detector

**Input:**
- Image: `(1, 3, 256, 256)` - RGB image in [0, 1] range

**Output:**
- Predictions: `(1, 257)` - Detection results
  - Channel 0: Detection confidence (higher = watermark present)
  - Channels 1-256: Binary message bits (threshold at 0)

**Size:** ~128 MB

## Usage Examples

### Python with TFLite

```python
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model
detector = tf.lite.Interpreter(
    model_path="videoseal_tflite/videoseal_detector_videoseal_256.tflite"
)
detector.allocate_tensors()

# Load and preprocess image
img = Image.open("test_image.jpg").convert("RGB")
img = img.resize((256, 256))
img_array = np.array(img).astype(np.float32) / 255.0
img_array = np.transpose(img_array, (2, 0, 1))  # HWC to CHW
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Run detection
detector.set_tensor(detector.get_input_details()[0]['index'], img_array)
detector.invoke()
predictions = detector.get_tensor(detector.get_output_details()[0]['index'])

# Extract results
detection_confidence = predictions[0, 0]
message_bits = (predictions[0, 1:] > 0).astype(int)

print(f"Detection confidence: {detection_confidence:.3f}")
print(f"Detected message (first 32 bits): {message_bits[:32]}")
print(f"Total bits detected: {message_bits.sum()}/256")
```

### Workflow: Server-Side Embedding + Edge Detection

```
Server (PyTorch):
  1. Load image
  2. Embed watermark using PyTorch VideoSeal
  3. Distribute watermarked images

Mobile/Edge Device (TFLite):
  1. Receive image
  2. Detect watermark using TFLite Detector
  3. Verify authenticity
  4. Extract embedded information
```

## Command-Line Options

### convert_detector_to_tflite.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output_dir` | str | `./videoseal_tflite` | Output directory for TFLite model |
| `--model_name` | str | `videoseal` | Model variant: `videoseal`, `pixelseal`, `chunkyseal` |
| `--image_size` | int | `256` | Input image size (height and width) |
| `--no_simple` | flag | False | Use dynamic version (not recommended) |

### verify_detector_tflite.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--detector_path` | str | Auto-detected | Path to TFLite detector model |
| `--model_name` | str | `videoseal` | Model variant used |
| `--image_size` | int | `256` | Image size used during conversion |
| `--tflite_dir` | str | `./videoseal_tflite` | Directory with TFLite models |

## Available Models

| Model | Capacity | Best For | Status |
|-------|----------|----------|--------|
| **VideoSeal v1.0** | 256 bits | Stable, recommended | ✅ Supported |
| **PixelSeal** | 256 bits | SOTA quality | ✅ Supported |
| **ChunkySeal** | 1024 bits | High capacity | ✅ Supported |

## Examples

### Convert Different Variants

```bash
# VideoSeal (default, recommended)
python convert_detector_to_tflite.py --model_name videoseal

# PixelSeal (best quality)
python convert_detector_to_tflite.py --model_name pixelseal

# ChunkySeal (1024-bit capacity)
python convert_detector_to_tflite.py --model_name chunkyseal
```

### Convert Different Image Sizes

```bash
# 512x512 images
python convert_detector_to_tflite.py --image_size 512

# 1024x1024 images
python convert_detector_to_tflite.py --image_size 1024
```

## Performance

| Image Size | Model Size | Inference Time (CPU) |
|------------|------------|---------------------|
| 256×256 | ~128 MB | ~200 ms |
| 512×512 | ~128 MB | ~800 ms |
| 1024×1024 | ~128 MB | ~3.2 s |

*Times are approximate and vary by hardware*

## Troubleshooting

### "Model card not found" Error

**Problem**: Script can't find VideoSeal model cards

**Solution**: Run from VideoSeal directory
```bash
cd ~/work/videoseal/videoseal
python /path/to/convert_detector_to_tflite.py --output_dir /path/to/output
```

### Missing Dependencies

**Problem**: ModuleNotFoundError for videoseal, omegaconf, etc.

**Solution**: Install dependencies
```bash
micromamba activate local_tf_env
pip install omegaconf einops timm==0.9.16 av opencv-python
```

### Low Detection Accuracy

**Problem**: TFLite detector gives poor results

**Solution**: 
1. Verify model with `verify_detector_tflite.py`
2. Ensure input images are preprocessed correctly (normalized to [0, 1])
3. Check that watermarks were embedded with compatible VideoSeal version

## Limitations

### Embedder Not Supported

The VideoSeal Embedder cannot be converted to TFLite due to:
- Dynamic tensor concatenation operations
- Complex message processing in UNet architecture
- ai-edge-torch limitations with certain PyTorch operations

**Workaround**: Use PyTorch embedder on server, TFLite detector on edge devices.

### Fixed Image Size

The converted model expects fixed-size inputs (e.g., 256×256). For different sizes:
- Convert a new model with `--image_size`
- Or resize images to match the model's expected size

## Integration Examples

### Android (Kotlin)

```kotlin
// Load model
val detector = Interpreter(File("videoseal_detector_videoseal_256.tflite"))

// Prepare input
val inputBuffer = ByteBuffer.allocateDirect(1 * 3 * 256 * 256 * 4)
inputBuffer.order(ByteOrder.nativeOrder())
// ... fill with image data ...

// Run inference
val outputBuffer = ByteBuffer.allocateDirect(1 * 257 * 4)
outputBuffer.order(ByteOrder.nativeOrder())
detector.run(inputBuffer, outputBuffer)

// Extract results
outputBuffer.rewind()
val detectionConfidence = outputBuffer.float
val messageBits = FloatArray(256)
outputBuffer.asFloatBuffer().get(messageBits)
```

### iOS (Swift)

```swift
// Load model
let interpreter = try Interpreter(modelPath: "videoseal_detector_videoseal_256.tflite")
try interpreter.allocateTensors()

// Prepare input
var inputData = Data()
// ... fill with image data ...

// Run inference
try interpreter.copy(inputData, toInputAt: 0)
try interpreter.invoke()

// Extract results
let outputTensor = try interpreter.output(at: 0)
let predictions = [Float32](unsafeData: outputTensor.data) ?? []
let detectionConfidence = predictions[0]
let messageBits = Array(predictions[1...256])
```

## References

- **VideoSeal Paper**: [arXiv:2412.09492](https://arxiv.org/abs/2412.09492)
- **VideoSeal GitHub**: [github.com/facebookresearch/videoseal](https://github.com/facebookresearch/videoseal)
- **AI Edge Torch**: [github.com/google-ai-edge/ai-edge-torch](https://github.com/google-ai-edge/ai-edge-torch)

## Support

For issues:
- **VideoSeal models**: See [VideoSeal repository](https://github.com/facebookresearch/videoseal)
- **TFLite conversion**: Open issue in AI Edge Torch repository
- **This script**: Check IMPLEMENTATION_SUMMARY.md for technical details

## License

Apache License 2.0. VideoSeal models are licensed under MIT License by Meta Platforms, Inc.

