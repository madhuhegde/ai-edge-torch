# Minimal DistortionNet TFLite Conversion Summary

## Date
January 6, 2025

## Overview
Successfully converted three minimal DistortionNet models to TFLite format using the updated TensorFlow format `[B, H, W, C]` instead of PyTorch format `[B, C, H, W]`.

## Converted Models

### 1. Single Block Model
**File:** `distortion_net_single.tflite`
- **Size:** 38 KB (0.04 MB)
- **Parameters:** 6,004
- **Input Shape:** `[9, 180, 320, 16]` (TensorFlow format: B, H, W, C)
  - 9 patches (batch size)
  - 180×320 resolution
  - 16 channels (after initial convolution)
- **Output Shape:** `[9, 24, 180, 320]`
- **Purpose:** Isolate and debug depthwise convolution issues
- **Operators:** CONV_2D, DEPTHWISE_CONV_2D, LOGISTIC, MUL, RESHAPE, SUM, TRANSPOSE
- **Estimated Ops:** 11.509 G ops (5.754 G MACs)

### 2. Minimal Model (5 Layers)
**File:** `distortion_net_minimal.tflite`
- **Size:** 164 KB (0.16 MB)
- **Parameters:** 31,378
- **Input Shape:** `[9, 360, 640, 3]` (TensorFlow format: B, H, W, C)
  - 9 patches (3×3 grid from 1080p frame)
  - 360×640 resolution per patch
  - 3 RGB channels
- **Output Shape:** `[9, 86, 148, 128]`
- **Layers:**
  - Stage 1: Initial Conv (3→32)
  - Stage 2: MBConv1 (32→16)
  - Stage 3: MBConv6 (16→24, 2 blocks)
  - Final Conv (24→128)
  - MaxPool + Permute
- **Operators:** ADD, CONV_2D, DEPTHWISE_CONV_2D, LOGISTIC, MAX_POOL_2D, MUL, RESHAPE, SUM
- **Estimated Ops:** 19.246 G ops (9.623 G MACs)

### 3. Medium Model (10 Layers)
**File:** `distortion_net_medium.tflite`
- **Size:** 1.5 MB (1.42 MB)
- **Parameters:** 349,620
- **Input Shape:** `[9, 360, 640, 3]` (TensorFlow format: B, H, W, C)
  - 9 patches (3×3 grid from 1080p frame)
  - 360×640 resolution per patch
  - 3 RGB channels
- **Output Shape:** `[9, 19, 28, 128]`
- **Layers:**
  - Stages 1-3 (as in Minimal)
  - Stage 4: MBConv6 (24→40, 2 blocks)
  - Stage 5: MBConv6 (40→80, 3 blocks)
  - Final Conv (80→128)
  - MaxPool + Permute
- **Operators:** ADD, CONV_2D, DEPTHWISE_CONV_2D, LOGISTIC, MAX_POOL_2D, MUL, RESHAPE, SUM
- **Estimated Ops:** 26.899 G ops (13.449 G MACs)

## Conversion Details

### Environment
- **Tool:** ai-edge-torch
- **Python Environment:** local_tf_env (micromamba)
- **TensorFlow Lite:** Using XNNPACK delegate for CPU

### Input Format
All models now use **TensorFlow format: [B, H, W, C]**
- Batch dimension first
- Height and Width in the middle
- Channels last

This is consistent with:
- Main UVQ 1.5 conversion script (`convert_to_tflite.py`)
- TFLite runtime expectations
- Standard TensorFlow conventions

### Wrapper Implementation
Models use `DistortionNetMinimalWrapper` class that:
1. Accepts input in TensorFlow format `[B, H, W, C]`
2. Converts to PyTorch format `[B, C, H, W]` using `.permute(0, 3, 1, 2).contiguous()`
3. Processes through the PyTorch model
4. Output is already in NHWC format (handled by PermuteLayerNHWC)

**Key Feature:** Uses `.contiguous()` after permute to prevent GATHER_ND operators

## Verification Results

All models passed verification:
- ✅ TFLite models load successfully
- ✅ Input/output shapes match expectations
- ✅ Inference runs without errors
- ✅ No problematic operators (GATHER_ND, GATHER, SCATTER_ND)
- ✅ Uses efficient operators (CONV_2D, DEPTHWISE_CONV_2D, TRANSPOSE)

## Comparison with Full DistortionNet

| Model | Parameters | Size | Layers | Purpose |
|-------|-----------|------|--------|---------|
| **Single** | 6,004 | 38 KB | 1 block | Debug depthwise conv |
| **Minimal** | 31,378 | 164 KB | 5 layers | Quick debugging |
| **Medium** | 349,620 | 1.5 MB | 10 layers | More comprehensive test |
| **Full** | ~3.5M | 14.5 MB | 18 layers | Production model |

## Usage Example

### Python (TFLite Inference)
```python
import tensorflow as tf
import numpy as np

# Load model
interpreter = tf.lite.Interpreter(model_path="distortion_net_minimal.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Create input (9 patches of 360x640x3)
input_data = np.random.randn(9, 360, 640, 3).astype(np.float32)

# Run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

print(f"Output shape: {output.shape}")  # (9, 86, 148, 128)
```

## Files Location

**Conversion Script:**
`~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/uvq1.5/convert_minimal_distortionnet.py`

**TFLite Models:**
`~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/uvq1.5/debug_models/`
- `distortion_net_single.tflite`
- `distortion_net_minimal.tflite`
- `distortion_net_medium.tflite`

**Source Models:**
`~/work/UVQ/uvq/uvq1p5_pytorch/utils/distortionnet_minimal.py`

## Next Steps

These minimal models can be used for:
1. **Debugging:** Isolate TFLite conversion issues
2. **Testing:** Verify inference pipeline with smaller models
3. **Profiling:** Measure performance of different model sizes
4. **Development:** Quick iteration during development

## Status
✅ **COMPLETE** - All three models successfully converted and verified with BHWC format

