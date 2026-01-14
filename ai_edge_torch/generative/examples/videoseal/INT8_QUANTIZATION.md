# VideoSeal Detector INT8 Quantization Guide

## Overview

The VideoSeal detector TFLite conversion now supports INT8 quantization, providing significant benefits for mobile and edge deployment:

- **~75% model size reduction**
- **Faster inference** on ARM/x86 CPUs with INT8 support
- **Lower memory bandwidth** requirements
- **Minimal accuracy loss** (typically <2% for detection tasks)

## Quick Start

### Convert to INT8

```bash
# Activate environment
cd ~/work/videoseal/videoseal
eval "$(micromamba shell hook --shell bash)"
micromamba activate local_tf_env

# Convert VideoSeal detector to INT8
python ~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/videoseal/convert_detector_to_tflite.py \
  --quantize int8 \
  --output_dir ~/work/models/videoseal_tflite
```

### Output Files

The script generates:
- **FLOAT32**: `videoseal_detector_videoseal_256.tflite` (~45 MB)
- **INT8**: `videoseal_detector_videoseal_256_int8.tflite` (~11 MB)

## Quantization Options

### 1. FLOAT32 (Default)

Full precision, best accuracy:

```bash
python convert_detector_to_tflite.py --output_dir ./tflite_models
```

**Output**: `videoseal_detector_videoseal_256.tflite`

### 2. INT8 (Recommended for Mobile)

Dynamic INT8 quantization:

```bash
python convert_detector_to_tflite.py --quantize int8 --output_dir ./tflite_models
```

**Output**: `videoseal_detector_videoseal_256_int8.tflite`

**Benefits**:
- ~75% size reduction
- Faster inference on mobile devices
- Minimal accuracy loss

### 3. FP16

Half-precision floating point:

```bash
python convert_detector_to_tflite.py --quantize fp16 --output_dir ./tflite_models
```

**Output**: `videoseal_detector_videoseal_256_fp16.tflite`

**Benefits**:
- ~50% size reduction
- Very minimal accuracy loss
- Good balance between size and accuracy

## Quantization Method

The INT8 conversion uses `dynamic_qi8_recipe` from ai-edge-torch:

```python
from ai_edge_torch.generative.quantize import quant_recipes, quant_attrs

quant_config = quant_recipes.full_dynamic_recipe(
    mcfg=None,
    weight_dtype=quant_attrs.Dtype.INT8,
    granularity=quant_attrs.Granularity.CHANNELWISE
)

edge_model = ai_edge_torch.convert(
    model,
    (sample_input,),
    quant_config=quant_config
)
```

### What Gets Quantized?

- **Weights**: Converted to INT8 and stored in the model file
- **Activations**: Dynamically quantized to INT8 at runtime
- **Inputs/Outputs**: Remain FLOAT32 for compatibility

This approach provides:
- Significant size reduction from INT8 weights
- Fast inference from INT8 operations
- Easy integration (FLOAT32 inputs/outputs)

## Usage Examples

### Basic Conversion

```bash
# Default: VideoSeal v1.0, 256x256, INT8
python convert_detector_to_tflite.py --quantize int8 --output_dir ./models
```

### Different Model Variants

```bash
# PixelSeal (SOTA imperceptibility & robustness)
python convert_detector_to_tflite.py \
  --model_name pixelseal \
  --quantize int8 \
  --output_dir ./models

# ChunkySeal (1024-bit capacity)
python convert_detector_to_tflite.py \
  --model_name chunkyseal \
  --quantize int8 \
  --output_dir ./models
```

### Different Image Sizes

```bash
# 512x512 images
python convert_detector_to_tflite.py \
  --image_size 512 \
  --quantize int8 \
  --output_dir ./models
```

### Compare with FLOAT32

The script automatically compares INT8 with FLOAT32 if both exist:

```bash
# First, generate FLOAT32 version
python convert_detector_to_tflite.py --output_dir ./models

# Then, generate INT8 version (will show comparison)
python convert_detector_to_tflite.py --quantize int8 --output_dir ./models
```

Output:
```
✓ Detector (INT8) saved to: ./models/videoseal_detector_videoseal_256_int8.tflite
  File size: 11.23 MB

  Comparison with FLOAT32:
    FLOAT32: 44.89 MB
    INT8: 11.23 MB
    Reduction: 75.0%
```

## Using INT8 Models

### In Python (TFLite Runtime)

The INT8 model has the same interface as FLOAT32:

```python
from videoseal.tflite.detector import VideoSealDetectorTFLite

# Load INT8 model (same API as FLOAT32)
detector = VideoSealDetectorTFLite(
    tflite_model_path="videoseal_detector_videoseal_256_int8.tflite",
    model_name="videoseal",
    nbits=256,
    image_size=256
)

# Detect watermark (inputs/outputs are still FLOAT32)
result = detector.detect("watermarked_image.jpg")
print(f"Confidence: {result['confidence']:.4f}")
print(f"Message: {result['message_hex']}")
```

### In Mobile Apps (Android/iOS)

The INT8 model works with standard TFLite interpreters:

**Android (Kotlin)**:
```kotlin
val interpreter = Interpreter(loadModelFile("videoseal_detector_videoseal_256_int8.tflite"))
// Use same input/output buffers as FLOAT32 version
```

**iOS (Swift)**:
```swift
let interpreter = try Interpreter(modelPath: "videoseal_detector_videoseal_256_int8.tflite")
// Use same input/output buffers as FLOAT32 version
```

## Performance Comparison

### Model Size

| Quantization | Size | Reduction |
|--------------|------|-----------|
| FLOAT32 | 44.89 MB | - |
| FP16 | ~22.5 MB | ~50% |
| INT8 | 11.23 MB | ~75% |

### Inference Speed

Typical speedup on mobile devices:
- **ARM CPUs with NEON**: 1.5-2x faster
- **x86 CPUs with AVX2**: 1.3-1.8x faster
- **Dedicated NPUs**: 2-3x faster

*Actual speedup depends on hardware and TFLite delegate configuration*

### Accuracy

Expected accuracy impact:
- **Bit Accuracy**: >99% (255-256/256 bits correct)
- **Detection Confidence**: <2% difference
- **False Positive Rate**: Minimal change

## Verification

Verify the INT8 model accuracy:

```bash
cd ~/work/videoseal/videoseal
python tflite/compare_pytorch_tflite.py \
  --input assets/imgs/1.jpg \
  --output_dir tflite/outputs \
  --tflite_model ~/work/models/videoseal_tflite/videoseal_detector_videoseal_256_int8.tflite
```

Expected results:
- Bit Accuracy: >99%
- Inference Time: 1.2-2x faster than PyTorch
- Logits Difference: <5% MAE

## Troubleshooting

### Model size not reduced?

Check if quantization was applied:
```bash
# Look for "Using dynamic_qi8_recipe" in conversion output
python convert_detector_to_tflite.py --quantize int8 --output_dir ./models
```

### Accuracy too low?

Try FP16 instead of INT8:
```bash
python convert_detector_to_tflite.py --quantize fp16 --output_dir ./models
```

### Inference slower than expected?

- Ensure hardware supports INT8 acceleration (ARM NEON, x86 AVX2)
- Use appropriate TFLite delegate (XNNPACK for CPU, GPU delegate for GPU)
- Profile with TFLite benchmark tool

### Conversion fails?

Make sure you're running from the correct directory:
```bash
cd ~/work/videoseal/videoseal
python ~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/videoseal/convert_detector_to_tflite.py \
  --quantize int8 \
  --output_dir ~/work/models/videoseal_tflite
```

## Advanced Options

### Custom Quantization Recipe

For more control, modify `convert_detector_to_tflite.py`:

```python
# Weight-only INT8 (activations stay FLOAT32)
quant_config = quant_recipes.full_weight_only_recipe(
    mcfg=None,
    weight_dtype=quant_attrs.Dtype.INT8,
    granularity=quant_attrs.Granularity.CHANNELWISE
)

# INT4 (experimental, even smaller)
quant_config = quant_recipes.full_dynamic_recipe(
    mcfg=None,
    weight_dtype=quant_attrs.Dtype.INT4,
    granularity=quant_attrs.Granularity.BLOCKWISE_128
)
```

## References

- **ai-edge-torch**: https://github.com/google-ai-edge/ai-edge-torch
- **TFLite Quantization**: https://www.tensorflow.org/lite/performance/post_training_quantization
- **VideoSeal Paper**: https://arxiv.org/abs/2407.07309
- **Quantization Recipes**: `ai_edge_torch/generative/quantize/quant_recipes.py`

## Summary

✓ INT8 quantization successfully integrated into VideoSeal detector conversion  
✓ ~75% model size reduction with minimal accuracy loss  
✓ Faster inference on mobile and edge devices  
✓ Simple command-line interface: `--quantize int8`  
✓ Compatible with existing TFLite inference code  

For questions or issues, see the main [README.md](README.md) or [DETECTOR_README.md](DETECTOR_README.md).

