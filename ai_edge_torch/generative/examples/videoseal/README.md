# VideoSeal TFLite Conversion

Convert VideoSeal watermarking models to TFLite format for deployment on mobile and edge devices.

## Overview

**VideoSeal** is a state-of-the-art invisible watermarking model for images and videos developed by Meta AI Research. This module provides tools to convert VideoSeal PyTorch models to TFLite format for efficient on-device inference.

### Key Features

- 🔒 **Invisible Watermarks**: Embed 256-bit (or 1024-bit with ChunkySeal) watermarks imperceptibly
- 🎯 **High Accuracy**: State-of-the-art detection accuracy
- 📱 **Mobile-Ready**: Optimized TFLite models for edge deployment
- 🚀 **Multiple Variants**: Support for VideoSeal, PixelSeal, and ChunkySeal
- ⚡ **INT8 Quantization**: ~75% model size reduction with minimal accuracy loss

## Model Architecture

VideoSeal consists of two main components:

### 1. Embedder
- **Input**: 
  - Image: `(batch, 3, height, width)` in [0, 1] range
  - Message: `(batch, 256)` binary vector
- **Output**: 
  - Watermarked image: `(batch, 3, height, width)` in [0, 1] range
- **Function**: Embeds an invisible watermark into the image

### 2. Detector
- **Input**: 
  - Image: `(batch, 3, height, width)` in [0, 1] range
- **Output**: 
  - Predictions: `(batch, 257, height_out, width_out)`
    - Channel 0: Detection mask (watermark presence confidence)
    - Channels 1-256: Binary message bits
- **Function**: Detects and extracts the watermark from the image

## Available Models

| Model | Capacity | Best For | Paper |
|-------|----------|----------|-------|
| **VideoSeal v1.0** | 256 bits | Stable, recommended default | [arXiv:2412.09492](https://arxiv.org/abs/2412.09492) |
| **PixelSeal** | 256 bits | SOTA imperceptibility & robustness | [arXiv:2512.16874](https://arxiv.org/abs/2512.16874) |
| **ChunkySeal** | 1024 bits | High capacity | [arXiv:2510.12812](https://arxiv.org/abs/2510.12812) |

## Prerequisites

### 1. Install VideoSeal

```bash
cd ~/work/videoseal/videoseal
source ~/work/UVQ/uvq_env/bin/activate
pip install -r requirements.txt
```

### 2. Install AI Edge Torch

```bash
pip install ai-edge-torch
```

### 3. Download VideoSeal Models

Models are automatically downloaded on first use. The checkpoint will be cached in `~/work/videoseal/videoseal/ckpts/`.

## Usage

### Basic Conversion

Convert detector to TFLite (FLOAT32):

```bash
cd ~/work/videoseal/videoseal
eval "$(micromamba shell hook --shell bash)"
micromamba activate local_tf_env

python ~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/videoseal/convert_detector_to_tflite.py \
  --output_dir ~/work/models/videoseal_tflite
```

### INT8 Quantization (Recommended for Mobile)

Convert detector with INT8 quantization for ~75% size reduction:

```bash
python ~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/videoseal/convert_detector_to_tflite.py \
  --quantize int8 \
  --output_dir ~/work/models/videoseal_tflite
```

**Output**: `videoseal_detector_videoseal_256_int8.tflite` (~11 MB vs ~45 MB for FLOAT32)

See [INT8_QUANTIZATION.md](INT8_QUANTIZATION.md) for detailed guide.

### Convert Specific Model

```bash
# Convert only embedder
python convert_to_tflite.py --model embedder --output_dir ./tflite_models

# Convert only detector
python convert_to_tflite.py --model detector --output_dir ./tflite_models
```

### Quantization Options

```bash
# FP16 quantization (~50% size reduction)
python ~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/videoseal/convert_detector_to_tflite.py \
  --quantize fp16 \
  --output_dir ~/work/models/videoseal_tflite
```

| Quantization | Size | Speed | Accuracy | Best For |
|--------------|------|-------|----------|----------|
| FLOAT32 (default) | 45 MB | 1x | 100% | Development, high accuracy |
| FP16 | ~22 MB | 1.2x | >99.5% | Balanced size/accuracy |
| INT8 | ~11 MB | 1.5-2x | >99% | Mobile, edge devices |

### Use Different VideoSeal Variants

```bash
# Use PixelSeal (SOTA imperceptibility & robustness)
python ~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/videoseal/convert_detector_to_tflite.py \
  --model_name pixelseal \
  --quantize int8 \
  --output_dir ~/work/models/videoseal_tflite

# Use ChunkySeal (1024-bit capacity)
python ~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/videoseal/convert_detector_to_tflite.py \
  --model_name chunkyseal \
  --output_dir ~/work/models/videoseal_tflite
```

### Custom Image Size

```bash
# Convert for 512x512 images with INT8
python ~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/videoseal/convert_detector_to_tflite.py \
  --image_size 512 \
  --quantize int8 \
  --output_dir ~/work/models/videoseal_tflite
```

### Advanced Options

```bash
# Use dynamic version (supports variable image sizes)
python convert_to_tflite.py --no_simple --output_dir ./tflite_models

# Combine options
python convert_to_tflite.py \
    --model_name pixelseal \
    --image_size 512 \
    --output_dir ./tflite_models
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output_dir` | str | `./videoseal_tflite` | Directory to save TFLite models |
| `--model_name` | str | `videoseal` | VideoSeal variant: `videoseal`, `pixelseal`, or `chunkyseal` |
| `--quantize` | str | None | Quantization type: `int8` (~75% smaller), `fp16` (~50% smaller), or None (FLOAT32) |
| `--image_size` | int | `256` | Input image size (height and width) |
| `--no_simple` | flag | False | Use dynamic version instead of fixed-size version |

## Output Files

After conversion, you'll find the following files in the output directory:

```
videoseal_tflite/
├── videoseal_embedder_videoseal_256.tflite  # Embedder model
└── videoseal_detector_videoseal_256.tflite  # Detector model
```

File naming convention: `videoseal_{component}_{variant}_{size}.tflite`

## Model Sizes

Approximate TFLite model sizes:

| Model | Embedder | Detector | Total |
|-------|----------|----------|-------|
| VideoSeal v1.0 (256x256) | ~50 MB | ~150 MB | ~200 MB |
| PixelSeal (256x256) | ~50 MB | ~150 MB | ~200 MB |
| ChunkySeal (256x256) | ~100 MB | ~300 MB | ~400 MB |

*Note: Sizes are approximate and may vary based on conversion settings.*

## Using TFLite Models

### Python Example

```python
import numpy as np
import tensorflow as tf

# Load embedder
embedder = tf.lite.Interpreter(model_path="videoseal_embedder_videoseal_256.tflite")
embedder.allocate_tensors()

# Prepare inputs
img = np.random.rand(1, 3, 256, 256).astype(np.float32)
msg = np.random.randint(0, 2, (1, 256)).astype(np.float32)

# Set inputs
embedder.set_tensor(embedder.get_input_details()[0]['index'], img)
embedder.set_tensor(embedder.get_input_details()[1]['index'], msg)

# Run inference
embedder.invoke()

# Get output
watermarked_img = embedder.get_tensor(embedder.get_output_details()[0]['index'])
print(f"Watermarked image shape: {watermarked_img.shape}")
```

### C++ Example

```cpp
// Load model
std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile("videoseal_embedder_videoseal_256.tflite");

// Build interpreter
tflite::ops::builtin::BuiltinOpResolver resolver;
std::unique_ptr<tflite::Interpreter> interpreter;
tflite::InterpreterBuilder(*model, resolver)(&interpreter);

// Allocate tensors
interpreter->AllocateTensors();

// Set inputs and run inference
// ... (similar to Python example)
```

## Model Comparison

### VideoSeal v1.0 (Recommended)
- ✅ **Stable and well-tested**
- ✅ **Good balance of imperceptibility and robustness**
- ✅ **256-bit capacity**
- ✅ **Smallest model size**

### PixelSeal (SOTA)
- ✅ **Best imperceptibility**
- ✅ **Best robustness**
- ✅ **256-bit capacity**
- ⚠️ **Similar size to VideoSeal**

### ChunkySeal (High Capacity)
- ✅ **1024-bit capacity** (4× more than others)
- ✅ **Good for high-information watermarks**
- ⚠️ **Larger model size** (~2× VideoSeal)
- ⚠️ **Slightly more visible**

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError: No module named 'videoseal'`:

```bash
# Make sure VideoSeal is in your Python path
export PYTHONPATH=$PYTHONPATH:~/work/videoseal/videoseal
```

Or the script automatically adds it, so ensure the path in `videoseal_models.py` is correct.

### Memory Issues

For large models or high-resolution images:

```bash
# Use smaller image size
python convert_to_tflite.py --image_size 256 --output_dir ./tflite_models

# Convert models separately
python convert_to_tflite.py --model embedder --output_dir ./tflite_models
python convert_to_tflite.py --model detector --output_dir ./tflite_models
```

### Conversion Failures

If conversion fails:

1. **Check PyTorch version**: Ensure PyTorch >= 2.3
2. **Check ai-edge-torch version**: Update to latest version
3. **Check VideoSeal installation**: Verify models load correctly in PyTorch
4. **Try simple mode**: Use default settings without `--no_simple`

### Model Download Issues

If automatic model download fails:

```bash
# Manually download
cd ~/work/videoseal/videoseal
mkdir -p ckpts
wget https://dl.fbaipublicfiles.com/videoseal/y_256b_img.pth -O ckpts/videoseal_y_256b_img.pth
```

## Performance Considerations

### Image Size Trade-offs

| Size | Pros | Cons |
|------|------|------|
| 256×256 | Fast inference, small model | Lower resolution |
| 512×512 | Good balance | Moderate speed |
| 1024×1024 | High resolution | Slower inference |

### Simple vs Dynamic Mode

**Simple Mode (Default)**:
- ✅ Fixed image size (e.g., 256×256)
- ✅ Faster conversion
- ✅ Smaller model
- ✅ Optimized for TFLite

**Dynamic Mode (`--no_simple`)**:
- ✅ Supports variable image sizes
- ⚠️ Larger model
- ⚠️ Slower inference
- ⚠️ May have compatibility issues

## References

- **VideoSeal Paper**: [arXiv:2412.09492](https://arxiv.org/abs/2412.09492)
- **PixelSeal Paper**: [arXiv:2512.16874](https://arxiv.org/abs/2512.16874)
- **ChunkySeal Paper**: [arXiv:2510.12812](https://arxiv.org/abs/2510.12812)
- **VideoSeal GitHub**: [github.com/facebookresearch/videoseal](https://github.com/facebookresearch/videoseal)
- **AI Edge Torch**: [github.com/google-ai-edge/ai-edge-torch](https://github.com/google-ai-edge/ai-edge-torch)

## License

This code is licensed under the Apache License 2.0. VideoSeal models are licensed under the MIT License by Meta Platforms, Inc.

## Citation

If you use VideoSeal in your research, please cite:

```bibtex
@article{fernandez2024videoseal,
  title={Video Seal: Open and Efficient Video Watermarking},
  author={Fernandez, Pierre and Elsahar, Hady and Yalniz, I. Zeki and Mourachko, Alexandre},
  journal={arXiv preprint arXiv:2412.09492},
  year={2024}
}
```

## Support

For issues related to:
- **VideoSeal models**: See [VideoSeal repository](https://github.com/facebookresearch/videoseal)
- **TFLite conversion**: Open an issue in the AI Edge Torch repository
- **This conversion script**: Contact the maintainers

## Changelog

### Version 1.0 (2025-01-03)
- Initial release
- Support for VideoSeal v1.0, PixelSeal, and ChunkySeal
- Embedder and Detector conversion
- Simple and dynamic modes
- Configurable image sizes

