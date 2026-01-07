# UVQ 1.5 TFLite Conversion

This directory contains scripts to convert UVQ 1.5 (Universal Video Quality) PyTorch models to TFLite format for deployment on mobile and edge devices.

## Overview

UVQ 1.5 is a video quality assessment model that consists of three components:

1. **ContentNet** - Extracts semantic content features from video frames
2. **DistortionNet** - Detects visual distortions using patch-based processing
3. **AggregationNet** - Combines features to produce quality scores [1-5]

## Model Architecture

```
Video Frame (1920x1080)
    ↓
┌─────────────────────────────────────────────────────┐
│  ContentNet                                         │
│  Input: (1, 3, 256, 256)                           │
│  Output: (1, 8, 8, 128)                            │
│  Size: ~15 MB                                       │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│  DistortionNet (Patch-based)                        │
│  Input: (9, 3, 360, 640) - 9 patches per frame     │
│  Output: (1, 24, 24, 128)                          │
│  Size: ~15 MB                                       │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│  AggregationNet                                     │
│  Input: Content (1,8,8,128) + Distortion (1,24,24,128) │
│  Output: Quality Score (1,1) in range [1-5]        │
│  Size: ~293 KB                                      │
└─────────────────────────────────────────────────────┘
```

## Requirements

- Python 3.11+
- PyTorch 2.9.0+
- ai-edge-torch 0.7.0+
- UVQ 1.5 PyTorch models (from `~/work/models/UVQ/uvq1.5/`)

## Installation

1. Activate the ai_edge_torch environment:
```bash
eval "$(micromamba shell hook --shell bash)"
micromamba activate ai_edge_torch_env
```

2. Ensure UVQ models are available:
```bash
ls ~/work/models/UVQ/uvq1.5/
# Should show: content_net.pth, distortion_net.pth, aggregation_net.pth
```

## Usage

### Convert All Models

```bash
cd ~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/uvq1.5
python convert_to_tflite.py --output_dir ./tflite_models
```

### Convert Specific Model

```bash
# Convert only ContentNet
python convert_to_tflite.py --model content --output_dir ./tflite_models

# Convert only DistortionNet
python convert_to_tflite.py --model distortion --output_dir ./tflite_models

# Convert only AggregationNet
python convert_to_tflite.py --model aggregation --output_dir ./tflite_models
```

### Custom Model Paths

```bash
python convert_to_tflite.py \
    --content_path /path/to/content_net.pth \
    --distortion_path /path/to/distortion_net.pth \
    --aggregation_path /path/to/aggregation_net.pth \
    --output_dir ./tflite_models
```

## Model Inputs and Outputs

### ContentNet

**Input:**
- Shape: `(batch, 3, 256, 256)`
- Type: float32
- Range: [-1, 1]
- Description: RGB video frames resized to 256x256

**Output:**
- Shape: `(batch, 8, 8, 128)`
- Type: float32
- Description: Content feature maps

### DistortionNet

**Input:**
- Shape: `(batch * 9, 3, 360, 640)`
- Type: float32
- Range: [-1, 1]
- Description: 9 patches per frame (3x3 grid from 1920x1080 frame)
  - Each patch is 640x360 pixels
  - Patches are arranged: top-left to bottom-right

**Output:**
- Shape: `(batch, 24, 24, 128)`
- Type: float32
- Description: Distortion feature maps

### AggregationNet

**Inputs:**
1. Content features: `(batch, 8, 8, 128)`
2. Distortion features: `(batch, 24, 24, 128)`

**Output:**
- Shape: `(batch, 1)`
- Type: float32
- Range: [1, 5]
- Description: Video quality score
  - 1 = Poor quality
  - 3 = Average quality
  - 5 = Excellent quality

## File Structure

```
uvq1.5/
├── README.md                  # This file
├── __init__.py                # Package initialization
├── uvq_models.py              # Model wrapper classes
├── convert_to_tflite.py       # Main conversion script
└── verify_tflite.py           # Verification script (optional)
```

## Example: Complete Pipeline

```python
import torch
from uvq_models import create_content_net, create_distortion_net, create_aggregation_net

# Load models
content_net = create_content_net()
distortion_net = create_distortion_net()
aggregation_net = create_aggregation_net()

# Prepare inputs
video_frame_256 = torch.randn(1, 3, 256, 256)  # Resized for content
video_patches = torch.randn(9, 3, 360, 640)     # 9 patches for distortion

# Run inference
with torch.no_grad():
    content_features = content_net(video_frame_256)
    distortion_features = distortion_net(video_patches)
    quality_score = aggregation_net(content_features, distortion_features)

print(f"Quality Score: {quality_score.item():.3f}")
```

## Preprocessing Video Frames

Before running inference, video frames must be preprocessed:

1. **For ContentNet:**
   - Resize frame to 256x256 using bicubic interpolation
   - Convert to RGB format
   - Normalize to [-1, 1] range: `(pixel / 255.0 - 0.5) * 2`

2. **For DistortionNet:**
   - Keep frame at 1920x1080 (or resize if different)
   - Split into 3x3 grid of 640x360 patches
   - Convert to RGB format
   - Normalize to [-1, 1] range: `(pixel / 255.0 - 0.5) * 2`

## Memory Requirements

| Model | PyTorch Size | TFLite Size (est.) | Inference Memory |
|-------|--------------|-------------------|------------------|
| ContentNet | 15 MB | ~15 MB | ~50 MB |
| DistortionNet | 15 MB | ~15 MB | ~100 MB |
| AggregationNet | 293 KB | ~300 KB | ~10 MB |
| **Total** | **~30 MB** | **~30 MB** | **~160 MB** |

## Known Issues

1. **Large Model Size**: The TFLite models are relatively large (~30 MB total). Consider quantization for mobile deployment.
2. **Patch Processing**: DistortionNet requires splitting frames into patches, which adds preprocessing overhead.
3. **Memory Usage**: Processing 1080p frames requires significant memory. Consider processing at lower resolution if needed.

## Troubleshooting

### Import Error: Cannot find UVQ modules

Make sure the UVQ source path is correct in `uvq_models.py`:
```python
UVQ_SOURCE_PATH = Path.home() / "work" / "UVQ" / "uvq"
```

### Model Not Found Error

Verify model paths:
```bash
ls ~/work/models/UVQ/uvq1.5/
```

Should show:
- `content_net.pth`
- `distortion_net.pth`
- `aggregation_net.pth`

### Conversion Fails

Check that you're using the correct environment:
```bash
micromamba activate ai_edge_torch_env
python --version  # Should be 3.11+
pip list | grep ai-edge-torch  # Should show 0.7.0+
```

## Performance Optimization

### Quantization (Future Work)

Post-training quantization can reduce model size and improve inference speed:

```bash
python convert_to_tflite.py --quantize --output_dir ./tflite_models
```

**Note**: Quantization support is planned but not yet implemented.

### Batch Processing

For processing multiple frames, batch them together:
- ContentNet: Process multiple frames in one batch
- DistortionNet: Process patches from multiple frames together
- AggregationNet: Process multiple feature pairs together

## Citation

If you use UVQ 1.5 in your work, please cite:

```bibtex
@article{uvq,
  title={UVQ: Universal Video Quality Assessment},
  author={Google Research},
  year={2025}
}
```

## License

Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Contact

For questions or issues, please refer to:
- UVQ Source: `~/work/UVQ/uvq/`
- Model Documentation: `~/work/models/UVQ/README.md`
- ai-edge-torch: https://github.com/google-ai-edge/ai-edge-torch

