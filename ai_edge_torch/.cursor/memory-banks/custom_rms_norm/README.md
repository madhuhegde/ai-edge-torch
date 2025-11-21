# Custom RMS Normalization - Memory Bank

## Quick Links

- **[Quick Start](./quickstart.md)** - 5-minute getting started guide ⭐
- **[Overview](./overview.md)** - Introduction to custom RMS norm
- **[Usage Guide](./usage.md)** - How to use the operator
- **[Implementation](./implementation.md)** - Technical details
- **[Testing Guide](./testing.md)** - Test suite and verification
- **[Status](./status.md)** - Implementation progress and next steps
- **[Limitations](./limitations.md)** - Current limitations and workarounds

---

## What is Custom RMS Norm?

A custom operator implementation of RMS (Root Mean Square) Normalization for AI Edge Torch that uses `stablehlo.custom_call` for TFLite deployment with external C++ kernels.

## Quick Example

```python
from ai_edge_torch.generative.custom_ops.custom_rms_norm import custom_rms_norm
import torch

x = torch.randn(2, 128, 768)
weight = torch.ones(768)
output = custom_rms_norm(x, weight, epsilon=1e-6)
```

## Status

✅ **Working with Modified TensorFlow**

This custom op successfully uses `stablehlo.custom_call` and converts to TFLite with the VHLO custom_call fallback handler.

**Requirements**:
- Modified TensorFlow (with fallback handler) - See `VHLO_CUSTOM_CALL_FIX.md`
- Custom C++ TFLite kernel for runtime execution (to be implemented)
- TFLite op code: 173 (STABLEHLO_CUSTOM_CALL)
- Kernel name: `ai_edge_torch.rms_norm`

**Test Results**:
- ✅ All 15 PyTorch tests passing
- ✅ TFLite conversion successful
- ✅ Standalone test script available

## Files

- **Implementation**: `generative/custom_ops/custom_rms_norm.py`
- **Tests**: 
  - `generative/test/test_custom_rms_norm.py` - Comprehensive test suite
  - `generative/test/test_tflite_conversion.py` - Standalone conversion test
- **Documentation**: `.cursor/memory-banks/custom_rms_norm/`
- **TensorFlow Modification**: `~/work/tensorflow/tensorflow/tensorflow/compiler/mlir/lite/flatbuffer_export.cc`

---

**Location**: `.cursor/memory-banks/custom_rms_norm/`  
**Purpose**: Custom operator for RMS normalization using `stablehlo.custom_call`

