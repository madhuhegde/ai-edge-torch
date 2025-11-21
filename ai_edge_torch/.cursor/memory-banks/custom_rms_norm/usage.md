# Custom RMS Normalization - Usage Guide

## Basic Usage

### Import

```python
from ai_edge_torch.generative.custom_ops.custom_rms_norm import custom_rms_norm
import torch
```

### Simple Example

```python
# Create input tensor [batch, seq_len, hidden_dim]
x = torch.randn(2, 128, 768)

# Create weight parameter
weight = torch.ones(768)

# Apply RMS normalization
output = custom_rms_norm(x, weight, epsilon=1e-6)

print(output.shape)  # torch.Size([2, 128, 768])
```

## In a PyTorch Module

```python
import torch.nn as nn
from ai_edge_torch.generative.custom_ops.custom_rms_norm import custom_rms_norm

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.norm_weight = nn.Parameter(torch.ones(hidden_dim))
        self.linear = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        # Apply attention or other operations
        x = self.linear(x)
        
        # Apply custom RMS normalization
        x = custom_rms_norm(x, self.norm_weight, epsilon=1e-6)
        
        return x

# Create model
model = TransformerBlock(768)

# Forward pass
x = torch.randn(2, 128, 768)
output = model(x)
```

## Export to StableHLO

```python
import torch
from ai_edge_torch.generative.custom_ops.custom_rms_norm import custom_rms_norm

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(768))
    
    def forward(self, x):
        return custom_rms_norm(x, self.weight, 1e-6)

# Export
model = SimpleModel()
x = torch.randn(2, 128, 768)

exported_program = torch.export.export(model, (x,))
print("✅ Exported to FX graph with custom_rms_norm")
```

## Parameters

### `custom_rms_norm(x, weight, epsilon=1e-6)`

**Arguments**:

- **x** (`torch.Tensor`): Input tensor of shape `[..., hidden_dim]`
  - Must have at least 1 dimension
  - Last dimension is the feature dimension to normalize

- **weight** (`torch.Tensor`): Scale parameter of shape `[hidden_dim]`
  - Must be 1-dimensional
  - Size must match last dimension of input

- **epsilon** (`float`, optional): Numerical stability constant
  - Default: `1e-6`
  - Must be positive
  - Prevents division by zero

**Returns**:
- `torch.Tensor`: Normalized tensor with same shape as input

## Input Shapes

Supported input shapes:

```python
# 2D: [seq_len, hidden_dim]
x = torch.randn(128, 768)
out = custom_rms_norm(x, weight, 1e-6)

# 3D: [batch, seq_len, hidden_dim]
x = torch.randn(2, 128, 768)
out = custom_rms_norm(x, weight, 1e-6)

# 4D: [batch, heads, seq_len, hidden_dim]
x = torch.randn(2, 12, 128, 64)
weight = torch.ones(64)
out = custom_rms_norm(x, weight, 1e-6)
```

## Validation

The operator validates inputs:

```python
# ❌ Wrong weight dimension
x = torch.randn(2, 128, 768)
weight = torch.ones(512)  # Should be 768
# Raises: ValueError

# ❌ Weight not 1D
weight = torch.ones(768, 1)  # Should be [768]
# Raises: ValueError

# ❌ Negative epsilon
custom_rms_norm(x, weight, epsilon=-1e-6)
# Raises: ValueError
```

## Testing

Run tests:

```bash
cd /home/madhuhegde/work/ai_edge_torch/ai-edge-torch/ai_edge_torch
micromamba run -n ai_edge_torch_env python -m pytest generative/test/test_custom_rms_norm.py -v
```

## TFLite Conversion

✅ **Now Working**: With modified TensorFlow (VHLO fallback handler)

### Prerequisites

1. **Modified TensorFlow Required**: 
   - Built from source with VHLO custom_call fallback handler
   - See `VHLO_CUSTOM_CALL_FIX.md` for build instructions
   - Location: `~/work/tensorflow/tensorflow`
   - Install in separate environment (e.g., `local_tf_env`)

2. **Environment Setup**:
```bash
# Use environment with modified TensorFlow
micromamba activate local_tf_env
```

### Conversion (Works Now!)

```python
import ai_edge_torch
import torch
from ai_edge_torch.generative.custom_ops.custom_rms_norm import custom_rms_norm

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(768))
    
    def forward(self, x):
        return torch.ops.ai_edge_torch.custom_rms_norm(x, self.weight, 1e-6)

# Convert to TFLite
model = SimpleModel()
sample_input = (torch.randn(2, 128, 768),)

edge_model = ai_edge_torch.convert(model, sample_input)
print("✅ Conversion succeeded!")

# Save
edge_model.export('/tmp/custom_rms_norm.tflite')
```

### Verify TFLite Model

```python
from tensorflow.lite.python import schema_py_generated as schema_fb

with open('/tmp/custom_rms_norm.tflite', 'rb') as f:
    buf = bytearray(f.read())

model = schema_fb.Model.GetRootAsModel(buf, 0)
subgraph = model.Subgraphs(0)
op = subgraph.Operators(0)
opcode = model.OperatorCodes(op.OpcodeIndex())

print(f"Builtin Code: {opcode.BuiltinCode()}")  # Should be 173 (STABLEHLO_CUSTOM_CALL)

# Extract custom call options
if op.BuiltinOptions2Type() == schema_fb.BuiltinOptions2.StablehloCustomCallOptions:
    options = schema_fb.StablehloCustomCallOptions()
    options.Init(op.BuiltinOptions2().Bytes, op.BuiltinOptions2().Pos)
    print(f"Call Target: {options.CallTargetName().decode('utf-8')}")
    # Should print: ai_edge_torch.rms_norm
```

### Next Steps: C++ Kernel Implementation

See [Limitations](./limitations.md) for details on implementing the C++ TFLite kernel.

## Best Practices

### 1. Use Learned Weights

```python
# ✅ Good: Learnable weights
class RMSNormLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        return custom_rms_norm(x, self.weight, 1e-6)
```

### 2. Consistent Epsilon

```python
# ✅ Good: Use consistent epsilon value
EPSILON = 1e-6

class Model(nn.Module):
    def forward(self, x):
        return custom_rms_norm(x, self.weight, EPSILON)
```

### 3. Validate Shapes

```python
# ✅ Good: Validate before calling
def apply_rms_norm(x, weight):
    assert x.shape[-1] == weight.shape[0], "Dimension mismatch"
    return custom_rms_norm(x, weight, 1e-6)
```

## Examples

### Example 1: Transformer Layer Norm

```python
class TransformerLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=8)
        self.norm1 = nn.Parameter(torch.ones(dim))
        self.norm2 = nn.Parameter(torch.ones(dim))
        self.ffn = nn.Linear(dim, dim)
    
    def forward(self, x):
        # Pre-norm for attention
        normed = custom_rms_norm(x, self.norm1, 1e-6)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        
        # Pre-norm for FFN
        normed = custom_rms_norm(x, self.norm2, 1e-6)
        ffn_out = self.ffn(normed)
        x = x + ffn_out
        
        return x
```

### Example 2: Multiple Normalization Layers

```python
class DeepModel(nn.Module):
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_layers)
        ])
        self.norms = nn.ParameterList([
            nn.Parameter(torch.ones(dim)) for _ in range(num_layers)
        ])
    
    def forward(self, x):
        for layer, norm_weight in zip(self.layers, self.norms):
            x = layer(x)
            x = custom_rms_norm(x, norm_weight, 1e-6)
        return x
```

## See Also

- [Overview](./overview.md) - What is custom RMS norm
- [Implementation](./implementation.md) - Technical details
- [Limitations](./limitations.md) - Current constraints

---

**Quick Start**: Copy the basic example above  
**Tests**: Run `pytest generative/test/test_custom_rms_norm.py`

