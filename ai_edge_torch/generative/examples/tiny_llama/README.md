# TinyLlama Model Examples

This directory contains the TinyLlama model implementation with support for Flash Attention and TFLite conversion.

## Features

- ✅ **Full TinyLlama 1.1B model** support
- ✅ **Flash Attention** integration (production-ready)
- ✅ **TFLite conversion** with multi-signature support
- ✅ **Grouped Query Attention** (GQA: 32 query heads, 4 KV heads)
- ✅ **RoPE embeddings** with 100% coverage
- ✅ **KV cache** for efficient autoregressive generation
- ✅ **Quantization** support (int8)

## Quick Start

### 1. Verify the Model

```bash
# Standard attention (default)
python verify.py \
  --checkpoint_path=/path/to/TinyLlama-1.1B-Chat-v1.0 \
  --prompts="What is a turboprop engine?" \
  --max_new_tokens=50

# With Flash Attention
python verify_flash.py \
  --checkpoint_path=/path/to/TinyLlama-1.1B-Chat-v1.0 \
  --prompts="What is a turboprop engine?" \
  --max_new_tokens=50
```

### 2. Convert to TFLite

```bash
# Standard attention
python convert_to_tflite.py \
  --checkpoint_path=/path/to/TinyLlama-1.1B-Chat-v1.0 \
  --prefill_seq_lens=64,128,256 \
  --kv_cache_max_len=1024 \
  --quantize=true

# With Flash Attention (for long sequences)
python convert_to_tflite.py \
  --checkpoint_path=/path/to/TinyLlama-1.1B-Chat-v1.0 \
  --use_flash_attention=true \
  --prefill_seq_lens=64,128,256,512 \
  --kv_cache_max_len=2048 \
  --quantize=true
```

### 3. Verify TFLite Model

```bash
python verify_tflite.py \
  --tflite_path=/tmp/tinyllama_q8_ekv1024.tflite \
  --checkpoint_path=/path/to/TinyLlama-1.1B-Chat-v1.0
```

## Files

### Core Files
- **`tiny_llama.py`** - Model definition and configuration
- **`convert_to_tflite.py`** - TFLite conversion script
- **`verify.py`** - Verification against original Hugging Face model
- **`verify_flash.py`** - Verification with Flash Attention enabled
- **`verify_tflite.py`** - TFLite model verification

### Test Files
- **`test_flash_attention.py`** - Flash Attention integration tests
- **`debug_flash_generation.py`** - Debugging utilities

### Documentation
- **`README.md`** - This file
- **`FLASH_ATTENTION_INTEGRATION.md`** - Complete Flash Attention integration guide
- **`FLASH_INTEGRATION_SUMMARY.md`** - Quick reference summary
- **`VERIFY_TFLITE_README.md`** - TFLite verification guide

## Flash Attention

### What is Flash Attention?

Flash Attention is a memory-efficient attention algorithm that:
- Reduces memory from O(N²) to O(N)
- Enables 128x memory reduction for long sequences
- Produces **identical outputs** to standard attention
- Fully compatible with KV cache, RoPE, GQA, and HLFB

### When to Use Flash Attention

✅ **Use Flash Attention for:**
- Long sequences (> 512 tokens)
- Prefill with large context windows
- Memory-constrained environments
- Training or inference with many attention layers

❌ **Standard attention is fine for:**
- Short sequences (< 128 tokens)
- Single-token decode (minimal memory anyway)
- When memory is not a constraint

### Usage

```python
from ai_edge_torch.generative.examples.tiny_llama import tiny_llama

# Enable Flash Attention
model = tiny_llama.build_model(
    checkpoint_path="/path/to/checkpoint",
    kv_cache_max_len=2048,
    use_flash_attention=True  # <-- Enable here
)

# Use normally - interface is unchanged
output = model(tokens, input_pos, kv_cache)
```

### Performance

**Memory Comparison** (TinyLlama, seq_len=2048):
- Standard attention: 2.15 GB
- Flash Attention: 16.8 MB
- **Reduction: 128x**

**Numerical Accuracy**:
- Prefill: max diff < 3e-5
- Decode: max diff < 1e-4
- Generated text: **identical**

See [`FLASH_ATTENTION_INTEGRATION.md`](FLASH_ATTENTION_INTEGRATION.md) for details.

## Model Configuration

### TinyLlama 1.1B Specifications

- **Parameters**: 1.1 Billion
- **Layers**: 22
- **Embedding dimension**: 2048
- **Attention heads**: 32 (query), 4 (KV) - Grouped Query Attention
- **Head dimension**: 64
- **Vocabulary size**: 32,000
- **Context length**: 2048 tokens
- **RoPE**: 100% coverage, base 10000

### Attention Architecture

**Grouped Query Attention (GQA)**:
- 32 query heads
- 4 key/value heads
- 8:1 ratio (8x memory savings vs MHA)

**Rotary Position Embedding (RoPE)**:
- Applied to 100% of head dimensions
- Base frequency: 10,000
- Enables strong position awareness

## TFLite Conversion

### Multi-Signature Support

The converted TFLite model has two signatures:

1. **`prefill`** - Process initial prompt tokens
   - Input: `tokens` [1, seq_len], `input_pos` [seq_len], KV cache (zeros)
   - Output: KV cache (updated)
   - Note: By default, `output_logits_on_prefill=False` for efficiency

2. **`decode`** - Generate tokens one at a time
   - Input: `tokens` [1, 1], `input_pos` [1], KV cache (from prefill)
   - Output: `logits` [1, 1, 32000], KV cache (updated)

### Quantization

```bash
# int8 quantization (recommended)
python convert_to_tflite.py \
  --checkpoint_path=/path/to/checkpoint \
  --quantize=true

# float32 (no quantization)
python convert_to_tflite.py \
  --checkpoint_path=/path/to/checkpoint \
  --quantize=false
```

Quantization typically:
- Reduces model size by 4x
- Minimal accuracy loss (< 1% perplexity increase)
- Faster inference on mobile devices

## Testing

### Run All Tests

```bash
# Flash Attention integration tests
python test_flash_attention.py

# Expected output:
# ✅ Forward pass with KV cache
# ✅ Prefill and decode stages
# ✅ Autoregressive generation
# ✅ Flash vs Standard attention comparison
# ✅ All Tests Passed!
```

### Manual Testing

```python
import torch
from ai_edge_torch.generative.examples.tiny_llama import tiny_llama

# Load model
model = tiny_llama.build_model(
    "/path/to/TinyLlama-1.1B-Chat-v1.0",
    use_flash_attention=True
)
model.eval()

# Generate text
from ai_edge_torch.generative.layers import kv_cache as kv_utils
import transformers

tokenizer = transformers.AutoTokenizer.from_pretrained(
    "/path/to/TinyLlama-1.1B-Chat-v1.0"
)

prompt = "What is a turboprop engine?"
tokens = tokenizer.encode(prompt, return_tensors="pt")
input_pos = torch.arange(tokens.shape[1])

kv_cache = kv_utils.KVCache.from_model_config(model.config)

with torch.no_grad():
    output = model(tokens, input_pos, kv_cache)
    logits = output['logits']
    next_token = torch.argmax(logits[0, -1, :])

print(f"Next token: {tokenizer.decode([next_token])}")
```

## Troubleshooting

### Import Error: `ai_edge_litert`

**Solution**: Install the nightly builds:
```bash
pip install ai-edge-litert-nightly ai-edge-quantizer-nightly
```

### Flash Attention produces different outputs

**Check**:
1. Are both models using the same weights?
2. Are both in eval mode (`model.eval()`)?
3. Is dropout disabled?

Differences should be < 1e-4. If larger, please file an issue.

### TFLite conversion fails

**Common issues**:
1. Wrong checkpoint path - verify file exists
2. Insufficient memory - reduce `prefill_seq_lens` or `kv_cache_max_len`
3. Missing dependencies - run `pip install -r requirements.txt`

### TFLite model generates nonsense

**Check**:
1. Are you initializing KV cache to zeros before prefill?
2. Are you using the correct signature (prefill vs decode)?
3. Are `input_pos` values correct and continuous?

See [`VERIFY_TFLITE_README.md`](VERIFY_TFLITE_README.md) for detailed TFLite usage.

## Contributing

When modifying the TinyLlama implementation:

1. **Run tests**: Ensure `test_flash_attention.py` passes
2. **Verify**: Run `verify.py` and `verify_flash.py`
3. **Test TFLite**: Convert and verify TFLite model
4. **Check lints**: Ensure no new linter errors
5. **Update docs**: Update relevant documentation

## References

- **TinyLlama**: https://github.com/jzhang38/TinyLlama
- **Flash Attention Paper**: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022)
- **Grouped Query Attention**: Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models" (2023)
- **RoPE**: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)

## License

See [LICENSE](../../../../LICENSE) in the repository root.

