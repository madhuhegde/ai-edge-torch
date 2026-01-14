# VideoSeal TFLite Conversion - Quick Start Guide

Get started with converting VideoSeal models to TFLite in 5 minutes!

## Prerequisites

```bash
# Activate your environment
source ~/work/UVQ/uvq_env/bin/activate

# Navigate to the conversion directory
cd ~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/videoseal
```

## Step 1: Convert Models (2-3 minutes)

```bash
# Convert both embedder and detector to TFLite
python convert_to_tflite.py --output_dir ./tflite_models
```

**Expected output:**
```
======================================================================
VideoSeal PyTorch to TFLite Conversion
======================================================================
Model variant: videoseal
Image size: 256x256
Output directory: ./tflite_models

Converting VideoSeal Embedder (videoseal) to TFLite...
✓ Embedder saved to: ./tflite_models/videoseal_embedder_videoseal_256.tflite
  File size: ~50 MB

Converting VideoSeal Detector (videoseal) to TFLite...
✓ Detector saved to: ./tflite_models/videoseal_detector_videoseal_256.tflite
  File size: ~150 MB

✓ Successfully converted 2/2 models
```

## Step 2: Verify Models (1 minute)

```bash
# Verify the converted models match PyTorch reference
python verify_tflite.py --tflite_dir ./tflite_models
```

**Expected output:**
```
Verifying Embedder...
✓ VERIFICATION PASSED
  TFLite model produces similar results to PyTorch reference

Verifying Detector...
✓ VERIFICATION PASSED
  TFLite model produces similar results to PyTorch reference

✓ All models verified successfully!
```

## Step 3: Use the Models

### Python Example

```python
import numpy as np
import tensorflow as tf

# Load embedder
embedder = tf.lite.Interpreter(
    model_path="tflite_models/videoseal_embedder_videoseal_256.tflite"
)
embedder.allocate_tensors()

# Prepare inputs
img = np.random.rand(1, 3, 256, 256).astype(np.float32)  # Image in [0, 1]
msg = np.random.randint(0, 2, (1, 256)).astype(np.float32)  # 256-bit message

# Embed watermark
embedder.set_tensor(embedder.get_input_details()[0]['index'], img)
embedder.set_tensor(embedder.get_input_details()[1]['index'], msg)
embedder.invoke()
watermarked = embedder.get_tensor(embedder.get_output_details()[0]['index'])

print(f"Original image shape: {img.shape}")
print(f"Watermarked image shape: {watermarked.shape}")
print(f"Embedded {msg.sum():.0f} bits of '1's")

# Load detector
detector = tf.lite.Interpreter(
    model_path="tflite_models/videoseal_detector_videoseal_256.tflite"
)
detector.allocate_tensors()

# Detect watermark
detector.set_tensor(detector.get_input_details()[0]['index'], watermarked)
detector.invoke()
predictions = detector.get_tensor(detector.get_output_details()[0]['index'])

# Extract message (threshold at 0)
detected_msg = (predictions[0, 1:] > 0).astype(np.float32)
bit_accuracy = np.mean(detected_msg == msg[0]) * 100

print(f"Detected message shape: {detected_msg.shape}")
print(f"Bit accuracy: {bit_accuracy:.1f}%")
```

## Common Use Cases

### 1. Convert PixelSeal (Best Quality)

```bash
python convert_to_tflite.py \
    --model_name pixelseal \
    --output_dir ./tflite_models
```

### 2. Convert for Higher Resolution

```bash
python convert_to_tflite.py \
    --image_size 512 \
    --output_dir ./tflite_models
```

### 3. Convert Only Embedder

```bash
python convert_to_tflite.py \
    --model embedder \
    --output_dir ./tflite_models
```

### 4. Convert ChunkySeal (High Capacity)

```bash
python convert_to_tflite.py \
    --model_name chunkyseal \
    --output_dir ./tflite_models
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'videoseal'"

**Solution:**
```bash
# Ensure VideoSeal is installed
cd ~/work/videoseal/videoseal
pip install -r requirements.txt

# Or add to Python path
export PYTHONPATH=$PYTHONPATH:~/work/videoseal/videoseal
```

### Issue: "Model download failed"

**Solution:**
```bash
# Manually download the model
cd ~/work/videoseal/videoseal
mkdir -p ckpts
wget https://dl.fbaipublicfiles.com/videoseal/y_256b_img.pth \
    -O ckpts/videoseal_y_256b_img.pth
```

### Issue: "Out of memory"

**Solution:**
```bash
# Use smaller image size
python convert_to_tflite.py --image_size 256 --output_dir ./tflite_models

# Or convert models one at a time
python convert_to_tflite.py --model embedder --output_dir ./tflite_models
python convert_to_tflite.py --model detector --output_dir ./tflite_models
```

## Next Steps

- Read the [full README](README.md) for detailed documentation
- Check the [verification script](verify_tflite.py) for testing
- Explore [VideoSeal examples](~/work/videoseal/videoseal/notebooks/) for more use cases

## Model Comparison

| Model | Capacity | Quality | Speed | Size |
|-------|----------|---------|-------|------|
| **VideoSeal** | 256 bits | Good | Fast | ~200 MB |
| **PixelSeal** | 256 bits | **Best** | Fast | ~200 MB |
| **ChunkySeal** | **1024 bits** | Good | Moderate | ~400 MB |

**Recommendation**: Start with **VideoSeal** for a good balance, use **PixelSeal** for best quality, or **ChunkySeal** for high-capacity watermarks.

## Help

For more options:
```bash
python convert_to_tflite.py --help
python verify_tflite.py --help
```

## References

- **VideoSeal Paper**: https://arxiv.org/abs/2412.09492
- **VideoSeal GitHub**: https://github.com/facebookresearch/videoseal
- **AI Edge Torch**: https://github.com/google-ai-edge/ai-edge-torch

