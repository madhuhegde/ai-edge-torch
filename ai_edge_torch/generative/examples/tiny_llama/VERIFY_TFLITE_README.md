# TinyLlama TFLite Verification

## Overview

The `verify_tflite.py` script verifies that a converted TinyLlama TFLite model produces outputs consistent with the original PyTorch model. It uses the proper TFLite inference workflow with prefill and decode signatures.

## Usage

```bash
python verify_tflite.py \
  --tflite_path=/path/to/model.tflite \
  --checkpoint_path=/path/to/pytorch/checkpoint \
  --max_new_tokens=50 \
  --prompts="What is AI?" \
  --prompts="Hello, how are you?"
```

### Arguments

- `--tflite_path`: Path to the converted TFLite model (required)
- `--checkpoint_path`: Path to the original PyTorch checkpoint (required)
- `--max_new_tokens`: Maximum number of tokens to generate (default: 30)
- `--prompts`: One or more test prompts for text generation verification (default: ["What is AI?", "Hello, how are you?"])
- `--tolerance`: Maximum allowed difference for numerical comparisons (default: 0.01, not used in current implementation)

## TFLite Inference Workflow

The script implements the correct TFLite inference workflow for generative models:

1. **Initialize KV Cache**: All KV caches are initialized to zeros before inference
2. **Prefill Stage**: Process the entire prompt using the `prefill` signature
   - Input: Prompt tokens + initial KV cache (all zeros)
   - Output: Updated KV cache (and optionally logits, depending on conversion settings)
3. **Decode Stage**: Generate tokens one at a time using the `decode` signature
   - Input: Previous token + current KV cache + position
   - Output: Logits for next token + updated KV cache
   - Uses greedy decoding (argmax) to select next token
4. **Termination**: Stop when EOS token is generated or max tokens reached

### Model Conversion Settings

The verification script works with models converted using **default settings** (`output_logits_on_prefill=False`). This is the recommended setting for production models as it optimizes size and performance.

When `output_logits_on_prefill=False`:
- The `prefill` signature only outputs KV caches (no logits)
- The `decode` signature outputs both KV caches and logits
- The script automatically adapts by using an initial decode call to get the first predicted token

## Critical Implementation Detail: Input Position Padding

When the prompt length is less than the expected prefill sequence length, the script pads both tokens and positions:

```python
# Correct padding approach:
padded_tokens = [token1, token2, ..., tokenN, 0, 0, 0, ...]  # Pad tokens with zeros
padded_input_pos = [0, 1, 2, ..., N-1, N, N+1, N+2, ...]     # Continue position sequence
```

**⚠️ Important**: The `input_pos` array must use **continuing position values**, not zeros. Padding `input_pos` with zeros will corrupt the model's positional encodings and produce garbage output.

### Why This Matters

The model uses positional embeddings (RoPE - Rotary Position Embeddings) that depend on the position indices. If you pad with zeros:
- Position 0 appears multiple times in the sequence
- This confuses the attention mechanism
- The model generates nonsensical, repetitive output

## Expected Results

With correct implementation, the TFLite model should produce outputs nearly identical to PyTorch:

**Example Output:**
```
Prompt: "What is AI?"

PyTorch:  "Artificial Intelligence (AI) is a field of computer science that involves..."
TFLite:   "Artificial Intelligence (AI) is a field of computer science that involves..."

First 10 tokens match: 10/10 ✓
```

Minor divergence may occur after many tokens due to quantization effects, which is normal and acceptable.

## Verification Process

The script performs end-to-end text generation comparison:

1. **TFLite Generation**: Runs the full prefill + decode loop to generate text
2. **PyTorch Generation**: Generates the same text using the PyTorch model
3. **Comparison**: Checks if the first 10 tokens match (allowing for minor divergence due to quantization)

## Example: Basic Verification

```bash
# Verify a quantized TFLite model
python verify_tflite.py \
  --tflite_path=/tmp/tinyllama_q8_ekv1280.tflite \
  --checkpoint_path=/path/to/TinyLlama-1.1B-Chat-v1.0 \
  --max_new_tokens=50 \
  --prompts="What is AI?" \
  --prompts="Explain machine learning"

# Expected output:
# ✓ First 10 tokens match: 10/10
# ✓ ALL TESTS PASSED
```

## Example: Manual TFLite Inference

Here's how to run TFLite inference manually (this is what the script does internally):

```python
import tensorflow as tf
import numpy as np

# Load model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get signatures
prefill_runner = interpreter.get_signature_runner("prefill")
decode_runner = interpreter.get_signature_runner("decode")

# Model configuration for TinyLlama
num_layers = 22
num_kv_heads = 4
head_dim = 64
kv_max_len = 1280

# 1. Initialize KV cache to zeros
kv_cache = {}
for i in range(num_layers):
    kv_cache[f'kv_cache_k_{i}'] = np.zeros((1, kv_max_len, num_kv_heads, head_dim), dtype=np.float32)
    kv_cache[f'kv_cache_v_{i}'] = np.zeros((1, kv_max_len, num_kv_heads, head_dim), dtype=np.float32)

# 2. Prepare prompt (must be padded to expected length, e.g., 64)
prompt_tokens = np.array([1, 529, 29989, ...], dtype=np.int32)  # Your tokens
seq_len = len(prompt_tokens)
expected_len = 64

# Pad tokens and positions
padded_tokens = np.pad(prompt_tokens, (0, expected_len - seq_len), constant_values=0)
padded_input_pos = np.arange(expected_len, dtype=np.int32)  # CRITICAL: Use continuing positions!

# 3. Run prefill
prefill_inputs = {
    'tokens': padded_tokens.reshape(1, -1),
    'input_pos': padded_input_pos,
    **kv_cache
}
prefill_outputs = prefill_runner(**prefill_inputs)

# 4. Update KV cache from prefill
for i in range(num_layers):
    kv_cache[f'kv_cache_k_{i}'] = prefill_outputs[f'kv_cache_k_{i}']
    kv_cache[f'kv_cache_v_{i}'] = prefill_outputs[f'kv_cache_v_{i}']

# 5. Generate tokens with decode loop
generated = []
next_token = prompt_tokens[-1]
current_pos = seq_len - 1

for step in range(50):  # Generate up to 50 tokens
    # First iteration: get prediction for last prompt token
    if step == 0:
        decode_inputs = {
            'tokens': np.array([[prompt_tokens[-1]]], dtype=np.int32),
            'input_pos': np.array([seq_len - 1], dtype=np.int32),
            **kv_cache
        }
    else:
        # Subsequent iterations: use previously generated token
        current_pos += 1
        decode_inputs = {
            'tokens': np.array([[next_token]], dtype=np.int32),
            'input_pos': np.array([current_pos], dtype=np.int32),
            **kv_cache
        }
    
    # Run decode
    decode_outputs = decode_runner(**decode_inputs)
    
    # Update KV cache
    for i in range(num_layers):
        kv_cache[f'kv_cache_k_{i}'] = decode_outputs[f'kv_cache_k_{i}']
        kv_cache[f'kv_cache_v_{i}'] = decode_outputs[f'kv_cache_v_{i}']
    
    # Get next token (greedy decoding)
    logits = decode_outputs['logits']
    next_token = np.argmax(logits[0, 0, :])
    generated.append(int(next_token))
    
    # Stop at EOS
    if next_token == eos_token_id:
        break

print(f"Generated tokens: {generated}")
```

## Summary

### Key Takeaways

1. **Proper Workflow**: The script implements the correct prefill + decode workflow for TFLite generative models
2. **Works with Default Models**: No special conversion flags needed - works with production models
3. **Critical Detail**: Input position padding must use continuing sequence values, not zeros
4. **Quantization Effects**: Minor divergence in outputs is expected and acceptable
5. **End-to-End Testing**: Compares full text generation, not just individual logits

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Model generates nonsense/repetitive tokens | `input_pos` padded with zeros | Use `np.arange(expected_len)` for padded positions |
| "File format not supported" error | Wrong checkpoint format | Use `.safetensors`, `.bin`, or `.pt` checkpoint files |
| First token differs significantly | KV cache not initialized properly | Ensure all KV caches are initialized to zeros before prefill |
| Output diverges after many tokens | Normal quantization effects | Expected behavior - compare first 10 tokens only |

### Performance Notes

- TFLite quantized models (int8) are typically 4x smaller than float32 PyTorch models
- Inference speed depends on hardware delegate (CPU/GPU/NPU)
- XNNPACK delegate provides optimized CPU inference on mobile devices
- For production deployment, see the C++ examples in `ai_edge_torch/generative/examples/cpp/`

## Related Documentation

- [TinyLlama Model Documentation](tiny_llama.py)
- [TFLite Conversion Guide](convert_to_tflite.py)
- [PyTorch Model Verification](verify.py)
- [Generative API Overview](../../README.md)
