# NHWC Format Changes for VideoSeal TFLite Models

**Date:** January 12, 2026  
**Change:** Updated all VideoSeal TFLite models to use NHWC (batch, height, width, channels) format instead of NCHW (batch, channels, height, width)

---

## Summary

All VideoSeal 0.0 and VideoSeal 1.0 TFLite models have been updated to use **NHWC format** `[N, H, W, C]`, which is the standard TensorFlow/TFLite format.

**Previous format:** `[N, C, H, W]` (PyTorch/NCHW)  
**New format:** `[N, H, W, C]` (TensorFlow/NHWC)

---

## Changes Made

### VideoSeal 0.0 (96-bit)

#### Conversion Scripts
**Location:** `~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/videoseal0.0/`

1. **`videoseal00_models.py`**
   - `VideoSeal00DetectorWrapper.forward()`: Added NHWC → NCHW conversion at input, processes in NCHW
   - `VideoSeal00EmbedderWrapper.forward()`: Added NHWC → NCHW conversion at input, NCHW → NHWC at output
   
2. **`convert_embedder_to_tflite.py`**
   - Sample input changed from `torch.rand(1, 3, 256, 256)` to `torch.rand(1, 256, 256, 3)`
   - Documentation updated to reflect NHWC format
   
3. **`convert_detector_to_tflite.py`**
   - Sample input changed from `torch.rand(1, 3, 256, 256)` to `torch.rand(1, 256, 256, 3)`
   - Documentation updated to reflect NHWC format

#### Inference Wrappers
**Location:** `~/work/videoseal/videoseal/tflite/`

4. **`embedder00.py`**
   - Expected input shape: `(1, 256, 256, 3)` instead of `(1, 3, 256, 256)`
   - `preprocess_image()`: Removed CHW transpose, keeps HWC format
   - `postprocess_image()`: Removed HWC transpose, already in correct format
   
5. **`detector00.py`**
   - Expected input shape: `(1, 256, 256, 3)` instead of `(1, 3, 256, 256)`
   - `preprocess_image()`: Removed CHW transpose, keeps HWC format

---

### VideoSeal 1.0 (256-bit)

#### Conversion Scripts
**Location:** `~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/videoseal/`

6. **`videoseal_models.py`**
   - `VideoSealDetectorWrapper.forward()`: Added NHWC → NCHW conversion at input
   - `VideoSealEmbedderWrapper.forward()`: Added NHWC → NCHW conversion at input, NCHW → NHWC at output
   - Updated docstrings to reflect NHWC format

#### Inference Wrappers
**Location:** `~/work/videoseal/videoseal/tflite/`

7. **`embedder.py`**
   - Expected input shape: `(1, 256, 256, 3)` instead of `(1, 3, 256, 256)`
   - `preprocess_image()`: Removed CHW transpose, keeps HWC format
   - `postprocess_image()`: Removed HWC transpose, already in correct format
   
8. **`detector.py`**
   - Expected input shape: `(1, 256, 256, 3)` instead of `(1, 3, 256, 256)`
   - `preprocess_image()`: Removed CHW transpose, keeps HWC format

---

## Technical Details

### Model Wrappers (Conversion)

The model wrappers now handle format conversion internally with `.contiguous()` optimization:

```python
def forward(self, imgs, msgs):
    # Input: imgs in NHWC format [N, H, W, C]
    
    # Convert to NCHW for PyTorch model
    # Use .contiguous() to avoid GATHER_ND operations in TFLite
    imgs_nchw = imgs.permute(0, 3, 1, 2).contiguous()
    
    # Process with PyTorch model (NCHW)
    output_nchw = self.model(imgs_nchw, msgs)
    
    # Convert back to NHWC for TFLite
    # Use .contiguous() to avoid GATHER_ND operations in TFLite
    output_nhwc = output_nchw.permute(0, 2, 3, 1).contiguous()
    
    return output_nhwc
```

**Key Optimization:** `.contiguous()` ensures tensors are stored in contiguous memory layout, which prevents TFLite from generating GATHER_ND operations during conversion. This results in:
- Faster inference (no memory scatter/gather)
- Smaller model size (fewer operations)
- Better TFLite optimization

### Inference Wrappers (Runtime)

The inference wrappers now expect and return NHWC format:

```python
def preprocess_image(self, image):
    # Load and resize image
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Already in HWC format, just add batch dimension
    img_array = np.expand_dims(img_array, axis=0)  # [1, H, W, C]
    
    return img_array  # NHWC format
```

---

## Impact

### ✅ Benefits

1. **Standard TensorFlow Format**: NHWC is the native TensorFlow/TFLite format
2. **Better Performance**: Some TFLite operations are optimized for NHWC
3. **Consistency**: Matches TensorFlow ecosystem conventions
4. **Easier Integration**: Works seamlessly with TensorFlow preprocessing pipelines
5. **No GATHER_ND Operations**: Using `.contiguous()` avoids expensive scatter/gather ops
6. **Optimized Memory Layout**: Contiguous memory enables better TFLite optimization

### ⚠️ Breaking Changes

**Models must be regenerated** with the new format:

```bash
cd ~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/videoseal0.0

# Regenerate VideoSeal 0.0 models
python convert_embedder_to_tflite.py --output_dir ./videoseal00_tflite
python convert_detector_to_tflite.py --output_dir ./videoseal00_tflite
```

```bash
cd ~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/videoseal

# Regenerate VideoSeal 1.0 models (if needed)
python convert_to_tflite.py --output_dir ./videoseal_tflite
```

### 🔄 Migration Guide

**Old code (NCHW):**
```python
# Old format: [1, 3, 256, 256]
img_array = np.transpose(image, (2, 0, 1))  # HWC → CHW
img_array = np.expand_dims(img_array, axis=0)
```

**New code (NHWC):**
```python
# New format: [1, 256, 256, 3]
img_array = np.expand_dims(image, axis=0)  # Already HWC, just add batch
```

---

## Validation

### Before Regenerating Models

**Old models (NCHW):**
- Input shape: `[1, 3, 256, 256]`
- Output shape: `[1, 3, 256, 256]` (embedder) or `[1, 97]` (detector)

### After Regenerating Models

**New models (NHWC):**
- Input shape: `[1, 256, 256, 3]`
- Output shape: `[1, 256, 256, 3]` (embedder) or `[1, 97]` (detector)

### Test Script

```python
import tensorflow as tf
import numpy as np

# Load model
interpreter = tf.lite.Interpreter("videoseal00_embedder_256.tflite")
interpreter.allocate_tensors()

# Check input shape
input_details = interpreter.get_input_details()
print(f"Input shape: {input_details[0]['shape']}")
# Expected: [1, 256, 256, 3]

# Test with NHWC input
img = np.random.rand(1, 256, 256, 3).astype(np.float32)
msg = np.random.randint(0, 2, (1, 96)).astype(np.float32)

interpreter.set_tensor(input_details[0]['index'], img)
interpreter.set_tensor(input_details[1]['index'], msg)
interpreter.invoke()

output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
print(f"Output shape: {output.shape}")
# Expected: [1, 256, 256, 3]
```

---

## Files Modified

### VideoSeal 0.0
```
ai_edge_torch/generative/examples/videoseal0.0/
├── videoseal00_models.py              ✅ Updated
├── convert_embedder_to_tflite.py      ✅ Updated
└── convert_detector_to_tflite.py      ✅ Updated

videoseal/tflite/
├── embedder00.py                       ✅ Updated
└── detector00.py                       ✅ Updated
```

### VideoSeal 1.0
```
ai_edge_torch/generative/examples/videoseal/
└── videoseal_models.py                 ✅ Updated

videoseal/tflite/
├── embedder.py                         ✅ Updated
└── detector.py                         ✅ Updated
```

---

## Next Steps

1. **Regenerate Models**: Run conversion scripts to create new NHWC models
2. **Test Models**: Verify input/output shapes match NHWC format
3. **Update Documentation**: Update any external documentation referencing model formats
4. **Validate Accuracy**: Run accuracy tests to ensure no regression

---

## Compatibility

### ✅ Compatible With
- TensorFlow Lite runtime
- TensorFlow preprocessing pipelines
- Standard image libraries (PIL, OpenCV) via numpy arrays
- Mobile deployment (Android, iOS)

### ⚠️ Requires Updates
- Existing TFLite models (must be regenerated)
- Custom preprocessing code (remove CHW transposes)
- Integration code expecting NCHW format

---

## References

- **TensorFlow Format**: https://www.tensorflow.org/api_docs/python/tf/keras/backend/image_data_format
- **TFLite Best Practices**: https://www.tensorflow.org/lite/performance/best_practices
- **NHWC vs NCHW**: https://www.tensorflow.org/xla/shapes#data_layout

---

**Status:** ✅ Changes Complete  
**Models Regenerated:** ⏳ Pending (user action required)  
**Tested:** ⏳ Pending model regeneration

