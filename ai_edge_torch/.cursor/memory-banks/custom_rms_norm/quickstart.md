# Custom RMS Normalization - Quick Start

## 🚀 5-Minute Getting Started

### 1. Import and Use

```python
from ai_edge_torch.generative.custom_ops.custom_rms_norm import custom_rms_norm
import torch

# Create tensors
x = torch.randn(2, 128, 768)        # [batch, seq_len, hidden_dim]
weight = torch.ones(768)             # [hidden_dim]

# Apply RMS normalization
output = custom_rms_norm(x, weight, 1e-6)
```

### 2. In a PyTorch Module

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
    
    def forward(self, x):
        return custom_rms_norm(x, self.weight, 1e-6)
```

### 3. Convert to TFLite

```bash
# Using standalone test script (easiest)
python generative/test/test_tflite_conversion.py --output_path model.tflite
```

**Or manually**:

```python
import ai_edge_torch

model = MyModel()
sample_input = (torch.randn(2, 128, 768),)

tflite_model = ai_edge_torch.convert(model, sample_input)
tflite_model.export('model.tflite')
```

### 4. Run Tests

```bash
# Run comprehensive test suite
python generative/test/test_custom_rms_norm.py

# Run standalone conversion test
python generative/test/test_tflite_conversion.py
```

---

## ⚙️ Environment Setup

### Prerequisites

1. **Modified TensorFlow Required**
   ```bash
   # Must have TensorFlow with VHLO fallback handler
   # See VHLO_CUSTOM_CALL_FIX.md for build instructions
   ```

2. **Activate Environment**
   ```bash
   micromamba activate local_tf_env
   ```

3. **Verify Installation**
   ```bash
   python -c "from ai_edge_torch.generative.custom_ops.custom_rms_norm import custom_rms_norm"
   ```

---

## 📋 Common Commands

### Generate TFLite Model

```bash
# Default location (/tmp)
python generative/test/test_tflite_conversion.py

# Custom location
python generative/test/test_tflite_conversion.py --output_path ~/models/my_model.tflite
```

### Run Tests

```bash
# All tests
python generative/test/test_custom_rms_norm.py

# Specific test
python generative/test/test_custom_rms_norm.py TestCustomRMSNorm.test_tflite_conversion
```

### Verify TFLite Model

```python
from tensorflow.lite.python import schema_py_generated as schema_fb

with open('model.tflite', 'rb') as f:
    buf = bytearray(f.read())

model = schema_fb.Model.GetRootAsModel(buf, 0)
subgraph = model.Subgraphs(0)
op = subgraph.Operators(0)
opcode = model.OperatorCodes(op.OpcodeIndex())

print(f"Op Code: {opcode.BuiltinCode()}")  # Should be 173
```

---

## 🔍 Key Information

### Operation Details

| Property | Value |
|----------|-------|
| **Namespace** | `ai_edge_torch::custom_rms_norm` |
| **StableHLO Op** | `stablehlo.custom_call` |
| **TFLite Op Code** | 173 (STABLEHLO_CUSTOM_CALL) |
| **Kernel Name** | `ai_edge_torch.rms_norm` |
| **Status** | ✅ Working (PyTorch + TFLite conversion) |

### Input/Output Shapes

```python
# Input
x:      [batch, seq_len, hidden_dim] or [..., hidden_dim]
weight: [hidden_dim]

# Output
output: same shape as x
```

### Parameters

```python
custom_rms_norm(
    x: torch.Tensor,       # Input tensor
    weight: torch.Tensor,  # Scale weights (1D)
    epsilon: float         # Numerical stability (e.g., 1e-6)
)
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [README](./README.md) | Entry point with quick links |
| [Overview](./overview.md) | What is custom RMS norm |
| [Usage](./usage.md) | Detailed usage examples |
| [Implementation](./implementation.md) | Technical details |
| [Testing](./testing.md) | Test suite guide |
| [Status](./status.md) | Implementation progress |
| [Limitations](./limitations.md) | Current constraints |

---

## ✅ Quick Validation

### Test PyTorch Functionality

```python
import torch
from ai_edge_torch.generative.custom_ops.custom_rms_norm import custom_rms_norm

x = torch.randn(2, 128, 768)
weight = torch.ones(768)
output = custom_rms_norm(x, weight, 1e-6)

print(f"Input shape:  {x.shape}")
print(f"Output shape: {output.shape}")
print(f"✅ Basic functionality works!" if output.shape == x.shape else "❌ Error")
```

### Test TFLite Conversion

```bash
python generative/test/test_tflite_conversion.py
# Look for: ✅ ALL TESTS PASSED!
```

### Test Full Suite

```bash
python generative/test/test_custom_rms_norm.py
# Look for: Ran 15 tests ... OK
```

---

## ⚠️ Important Notes

1. **Epsilon Parameter**
   - No default value (must be specified)
   - Typical value: `1e-6`
   - Must be positive

2. **Weight Dimension**
   - Must be 1D
   - Size must match last dimension of input

3. **Modified TensorFlow**
   - Standard TensorFlow won't work
   - Custom build required for VHLO support
   - See `VHLO_CUSTOM_CALL_FIX.md`

4. **C++ Kernel**
   - TFLite conversion works ✅
   - Runtime execution requires C++ kernel (pending)

---

## 🆘 Troubleshooting

### Import Error

```bash
# Error: ModuleNotFoundError
# Fix: Reinstall in editable mode
pip install -e . --no-deps
```

### VHLO Error

```bash
# Error: vhlo.custom_call_v1 not supported
# Fix: Use modified TensorFlow
# See: VHLO_CUSTOM_CALL_FIX.md
```

### Shape Mismatch

```python
# Error: Weight dimension mismatch
# Fix: Ensure weight.shape[0] == x.shape[-1]
x = torch.randn(2, 128, 768)
weight = torch.ones(768)  # Not 512!
```

---

## 🎯 Next Steps

1. ✅ **Use in PyTorch** - Ready to use
2. ✅ **Convert to TFLite** - Working
3. ⏳ **Implement C++ Kernel** - Next phase
4. ⏳ **Test Runtime** - Depends on kernel

---

## 📞 References

- **Implementation**: `generative/custom_ops/custom_rms_norm.py`
- **Tests**: `generative/test/test_custom_rms_norm.py`
- **Standalone Test**: `generative/test/test_tflite_conversion.py`
- **TensorFlow Mod**: `~/work/tensorflow/tensorflow/tensorflow/compiler/mlir/lite/flatbuffer_export.cc`

---

**Quick Reference Card** | Last Updated: November 21, 2024

