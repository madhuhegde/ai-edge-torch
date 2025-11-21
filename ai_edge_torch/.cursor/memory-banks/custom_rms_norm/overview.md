# Custom RMS Normalization - Overview

## What is RMS Normalization?

RMS (Root Mean Square) Normalization is a simplified variant of Layer Normalization that normalizes inputs using only the root mean square statistic, without mean centering.

### Formula

```
RMS(x) = sqrt(mean(x²) + ε)
output = (x / RMS(x)) * weight
```

Where:
- `x`: Input tensor
- `weight`: Learnable scale parameter
- `ε`: Small constant for numerical stability (typically 1e-6)

### Key Characteristics

- **No mean centering**: Unlike LayerNorm, RMS norm doesn't subtract the mean
- **Simpler computation**: Fewer operations than full LayerNorm
- **Equivalent performance**: Often achieves similar results to LayerNorm
- **Popular in transformers**: Used in models like Gemma, LLaMA

## Custom Op Implementation

This implementation defines RMS norm as a **custom operator** using:

1. **PyTorch Custom Op**: `torch.library.custom_op` for PyTorch integration
2. **StableHLO Lowering**: `stablehlo.custom_call` for TFLite deployment
3. **C++ Kernel**: External kernel implementation for TFLite runtime

### Why Custom Op?

Unlike `bmm_4d` or `dynamic_update_slice` which decompose to standard StableHLO operations, this implementation uses `custom_call` to enable:

- **Custom C++ kernels**: Direct control over implementation
- **Hardware optimization**: Target-specific optimizations
- **Novel operations**: Operations not in standard TFLite op set

## Architecture

```
PyTorch Model
    ↓
torch.ops.ai_edge_torch.custom_rms_norm
    ↓ (lowering)
stablehlo.custom_call @ai_edge_torch.rms_norm
    ↓ (VHLO with fallback handler)
TFLite STABLEHLO_CUSTOM_CALL (op 173)
    ↓ (TFLite runtime)
C++ Kernel: ai_edge_torch.rms_norm
```

## Comparison

| Aspect | Custom RMS Norm | Standard Ops (bmm_4d) |
|--------|----------------|----------------------|
| PyTorch definition | ✅ `custom_op` | ✅ `custom_op` |
| StableHLO lowering | `custom_call` | Standard ops |
| VHLO support | ✅ With fallback handler | ✅ Supported |
| TFLite conversion | ✅ With modified TensorFlow | ✅ Works |
| C++ kernel | Required | Not needed |
| Use case | Novel ops, optimization | Standard functionality |

## Use Cases

1. **Hardware-specific optimization**: Implement RMS norm for specific accelerators
2. **Research**: Experiment with custom normalization variants
3. **Performance**: Optimize critical operations with hand-tuned kernels
4. **Deployment**: Deploy models with operations not in standard TFLite

## Related

- [Usage Guide](./usage.md) - How to use custom RMS norm
- [Implementation](./implementation.md) - Technical details
- [Limitations](./limitations.md) - Current constraints

---

**Type**: Custom Operator  
**Category**: Normalization  
**Status**: Working with modified TensorFlow (VHLO fallback handler)  
**TFLite Op Code**: 173 (STABLEHLO_CUSTOM_CALL)  
**Kernel Name**: ai_edge_torch.rms_norm

