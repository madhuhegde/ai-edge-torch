# VideoSeal 0.0 TFLite Conversion Guide

## Quick Reference

This guide provides step-by-step instructions for converting VideoSeal 0.0 models to TFLite format.

## 📋 Prerequisites

### 1. Install Dependencies

```bash
# Core dependencies
pip install torch torchvision
pip install ai-edge-torch
pip install tensorflow
pip install videoseal

# Verify installations
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import ai_edge_torch; print('AI Edge Torch: OK')"
python -c "import videoseal; print('VideoSeal: OK')"
```

### 2. Navigate to Directory

```bash
cd ~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/videoseal0.0
```

### 3. Set Python Path (if needed)

```bash
export PYTHONPATH=/home/madhuhegde/work/videoseal/videoseal:$PYTHONPATH
```

## 🚀 Conversion Steps

### Step 1: Test Model Wrappers

Before conversion, verify the model wrappers work correctly:

```bash
python videoseal00_models.py
```

**Expected output:**
```
Testing VideoSeal 0.0 Detector Wrapper...
✓ Loaded VideoSeal 0.0 detector
✓ Created test input: torch.Size([1, 3, 256, 256])
✓ Output shape: torch.Size([1, 97])
  Expected: (1, 97) - 1 detection + 96 message bits

✓ Detection confidence: 0.xxxx
✓ Message bits: xx/96 are 1
✓ First 32 bits: [...]

Testing VideoSeal 0.0 Embedder Wrapper...
✓ Loaded VideoSeal 0.0 embedder
✓ Created test inputs: img=torch.Size([1, 3, 256, 256]), msg=torch.Size([1, 96])
✓ Output shape: torch.Size([1, 3, 256, 256])
  Expected: (1, 3, 256, 256)

✓ PSNR: xx.xx dB
✓ Output range: [0.xxx, 0.xxx]

✓ VideoSeal 0.0 wrappers are working correctly!
```

### Step 2: Convert Detector

#### Option A: FLOAT32 (Full Precision)

```bash
python convert_detector_to_tflite.py --output_dir ./videoseal00_tflite
```

**Output:**
- File: `videoseal00_detector_256.tflite`
- Size: ~110 MB
- Accuracy: Reference (100%)

#### Option B: INT8 (Recommended for Production)

```bash
python convert_detector_to_tflite.py --quantize int8 --output_dir ./videoseal00_tflite
```

**Output:**
- File: `videoseal00_detector_256_int8.tflite`
- Size: ~28 MB (75% reduction)
- Accuracy: 95-98%

#### Option C: FP16 (Balanced)

```bash
python convert_detector_to_tflite.py --quantize fp16 --output_dir ./videoseal00_tflite
```

**Output:**
- File: `videoseal00_detector_256_fp16.tflite`
- Size: ~55 MB (50% reduction)
- Accuracy: 99.5%+

### Step 3: Convert Embedder

#### Option A: FLOAT32 (Recommended)

```bash
python convert_embedder_to_tflite.py --output_dir ./videoseal00_tflite
```

**Output:**
- File: `videoseal00_embedder_256.tflite`
- Size: ~90 MB
- PSNR: >40 dB

#### Option B: FP16 (Smaller)

```bash
python convert_embedder_to_tflite.py --quantize fp16 --output_dir ./videoseal00_tflite
```

**Output:**
- File: `videoseal00_embedder_256_fp16.tflite`
- Size: ~45 MB (50% reduction)
- PSNR: >38 dB

⚠️ **Note**: INT8 quantization is NOT recommended for embedders due to quality degradation.

### Step 4: Verify Detector

Verify the converted detector matches PyTorch accuracy:

```bash
# Verify FLOAT32
python verify_detector_tflite.py --tflite_dir ./videoseal00_tflite

# Verify INT8
python verify_detector_tflite.py --quantize int8 --tflite_dir ./videoseal00_tflite
```

**Expected results:**

**FLOAT32:**
```
MAE Check: ✓ PASS
  Threshold: <1e-05
  Actual: 0.000001

Bit Accuracy Check: ✓ PASS
  Threshold: ≥99.5%
  Actual: 100.00%

✓ Verification PASSED - TFLite model is accurate!
```

**INT8:**
```
MAE Check: ✓ PASS
  Threshold: <0.05
  Actual: 0.012345

Bit Accuracy Check: ✓ PASS
  Threshold: ≥95.0%
  Actual: 97.34%

✓ Verification PASSED - TFLite model is accurate!
```

### Step 5: Test Usage

Run the example usage script to see how to use the models:

```bash
python example_usage.py
```

## 📊 Conversion Results

After successful conversion, you should have:

```
videoseal00_tflite/
├── videoseal00_detector_256.tflite          # ~110 MB (FLOAT32)
├── videoseal00_detector_256_int8.tflite     # ~28 MB (INT8) ✅ Recommended
├── videoseal00_detector_256_fp16.tflite     # ~55 MB (FP16)
├── videoseal00_embedder_256.tflite          # ~90 MB (FLOAT32) ✅ Recommended
└── videoseal00_embedder_256_fp16.tflite     # ~45 MB (FP16)
```

## 🎯 Recommended Configurations

### For Mobile Deployment

**Best balance of size and quality:**
```bash
# Detector: INT8 (small, fast)
python convert_detector_to_tflite.py --quantize int8 --output_dir ./videoseal00_tflite

# Embedder: FLOAT32 (best quality)
python convert_embedder_to_tflite.py --output_dir ./videoseal00_tflite
```

**Total size:** ~118 MB (28 MB + 90 MB)

### For Server Deployment

**Best accuracy:**
```bash
# Both FLOAT32
python convert_detector_to_tflite.py --output_dir ./videoseal00_tflite
python convert_embedder_to_tflite.py --output_dir ./videoseal00_tflite
```

**Total size:** ~200 MB (110 MB + 90 MB)

### For Ultra-Compact Deployment

**Smallest size:**
```bash
# Detector: INT8
python convert_detector_to_tflite.py --quantize int8 --output_dir ./videoseal00_tflite

# Embedder: FP16
python convert_embedder_to_tflite.py --quantize fp16 --output_dir ./videoseal00_tflite
```

**Total size:** ~73 MB (28 MB + 45 MB)

## 🔧 Troubleshooting

### Issue: Model not found

**Error:**
```
Checkpoint not found in any of these locations:
  - /mnt/shared/shared/VideoSeal/rgb_96b.pth
  - ~/.cache/videoseal/rgb_96b.pth
Loading VideoSeal 0.0 from model card (will download if needed)...
```

**Solution:** The model will be downloaded automatically. Wait for download to complete.

### Issue: Conversion fails

**Error:**
```
✗ Conversion failed: ...
```

**Solutions:**
1. Check ai-edge-torch installation: `pip install --upgrade ai-edge-torch`
2. Check TensorFlow installation: `pip install --upgrade tensorflow`
3. Verify PyTorch model loads: `python videoseal00_models.py`
4. Check available memory (conversion requires ~4-8 GB RAM)

### Issue: Verification fails

**Error:**
```
✗ Verification FAILED - TFLite model has accuracy issues
```

**Solutions:**
1. Re-run conversion with different quantization
2. Check TensorFlow version compatibility
3. Verify PyTorch model works correctly
4. Try with fewer test samples: `--num_tests 5`

### Issue: Low PSNR in embedder

**Problem:** PSNR < 35 dB

**Causes:**
- Using INT8 quantization (not recommended)
- Model conversion issue

**Solutions:**
1. Use FLOAT32: `python convert_embedder_to_tflite.py --output_dir ./videoseal00_tflite`
2. Use FP16 if size is critical: `--quantize fp16`
3. Never use INT8 for embedder

## 📝 Next Steps

After successful conversion:

1. **Integrate into your application** - See `example_usage.py` for code examples
2. **Test with real images** - Verify watermarking quality on your data
3. **Benchmark performance** - Measure inference time on target device
4. **Deploy to production** - Use TFLite models in your mobile/edge app

## 🔗 Additional Resources

- **README.md** - Complete documentation
- **example_usage.py** - Usage examples
- **VideoSeal Repository** - https://github.com/facebookresearch/videoseal
- **AI Edge Torch** - https://github.com/google-ai-edge/ai-edge-torch

## 📧 Support

If you encounter issues:
1. Check this guide thoroughly
2. Review error messages carefully
3. Consult the main README.md
4. Open an issue on GitHub with:
   - Error message
   - Python/TensorFlow/PyTorch versions
   - System information
   - Steps to reproduce

---

**Happy Converting!** 🚀
