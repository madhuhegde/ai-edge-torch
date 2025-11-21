# Custom RMS Normalization - Implementation Details

## File Structure

```
generative/custom_ops/custom_rms_norm.py    # Custom op implementation
generative/test/test_custom_rms_norm.py     # Test suite
.cursor/memory-banks/custom_rms_norm/       # Documentation
```

## Implementation Overview

The custom RMS norm consists of three main components:

1. **PyTorch Implementation** - Eager execution
2. **Fake Implementation** - Shape inference during tracing
3. **StableHLO Lowering** - Conversion to MLIR

---

## 1. PyTorch Custom Op Definition

```python
@torch.library.custom_op("ai_edge_torch::custom_rms_norm", mutates_args=())
def custom_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
```

**Note**: `epsilon` does not have a default value to avoid issues with PyTorch's fake tensor system during `torch.export.export`.

### Key Features

- **Namespace**: `ai_edge_torch::custom_rms_norm`
- **Immutable**: `mutates_args=()` indicates no in-place modifications
- **Type annotations**: Required for torch.export

### Implementation

```python
# Validate inputs
if x.dim() < 1:
    raise ValueError("Input must have at least 1 dimension")

if weight.dim() != 1:
    raise ValueError("Weight must be 1-dimensional")

if x.shape[-1] != weight.shape[0]:
    raise ValueError("Dimension mismatch")

if epsilon <= 0:
    raise ValueError("Epsilon must be positive")

# Compute RMS normalization
variance = x.pow(2).mean(dim=-1, keepdim=True)
x_normalized = x * torch.rsqrt(variance + epsilon)
return x_normalized * weight
```

### RMS Norm Algorithm

1. **Compute variance**: `variance = mean(x²)` along last dimension
2. **Normalize**: `x_norm = x / sqrt(variance + ε)`
3. **Scale**: `output = x_norm * weight`

---

## 2. Fake Implementation

```python
@custom_rms_norm.register_fake
def _(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    # Validate constraints
    if x.dim() < 1:
        raise ValueError(...)
    
    # Return same shape as input
    return x.clone()
```

### Purpose

- **Shape inference**: PyTorch tracing needs output shape without execution
- **Validation**: Checks constraints during graph tracing
- **Performance**: Avoids actual computation during export

### Why Clone?

`x.clone()` ensures:
- No aliasing with input tensor
- Same shape and dtype as input
- Separate tensor for graph analysis

---

## 3. StableHLO Lowering

```python
@lowerings.lower(torch.ops.ai_edge_torch.custom_rms_norm)
def _custom_rms_norm_lower(
    lctx,
    x: ir.Value,
    weight: ir.Value,
    epsilon: float,
):
    # Create custom_call operation with proper attribute types
    result = stablehlo.custom_call(
        [x.type],  # result types (same shape as input)
        [x, weight],  # operands
        call_target_name=ir.StringAttr.get("ai_edge_torch.rms_norm"),
        has_side_effect=ir.BoolAttr.get(False),
        backend_config=ir.StringAttr.get(""),
        api_version=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 1),
    )
    return result
```

### StableHLO Custom Call

**Components**:

1. **Output types**: `[x.type]` - Result has same type as input
2. **Inputs**: `[x, weight]` - Two input tensors
3. **Target name**: `"ai_edge_torch.rms_norm"` - Must match C++ kernel registration
4. **Backend config**: Empty string (epsilon passed as function parameter)
5. **API version**: 1 (API_VERSION_ORIGINAL)
6. **Has side effect**: False

### Generated MLIR

```mlir
%result = stablehlo.custom_call @ai_edge_torch.rms_norm(%input, %weight) {
    backend_config = "",
    has_side_effect = false,
    api_version = 1 : i32
} : (tensor<2x128x768xf32>, tensor<768xf32>) -> tensor<2x128x768xf32>
```

---

## Key Design Decisions

### 1. Why `custom_call`?

- **Direct kernel control**: C++ implementation for optimization
- **Hardware targeting**: Can optimize for specific accelerators
- **Novel operations**: Not constrained by standard ops

### 2. Why Empty Backend Config?

- **Simplicity**: Epsilon is passed as function parameter, not serialized
- **Extensible**: Can add backend_config later if needed for additional metadata
- **Standard**: MLIR attribute format allows empty strings

### 3. Why Separate Fake Implementation?

- **Performance**: Tracing doesn't execute actual computation
- **Flexibility**: Can have different eager vs. traced behavior
- **Required**: PyTorch export requires fake implementation

---

## Comparison with Other Custom Ops

### vs. `dynamic_update_slice`

| Aspect | custom_rms_norm | dynamic_update_slice |
|--------|-----------------|---------------------|
| StableHLO op | `custom_call` | `dynamic_update_slice` (built-in) |
| VHLO support | ❌ Not supported | ✅ Supported |
| C++ kernel | Required | Not needed (built-in) |
| TFLite flag | Blocked | `_experimental_enable_dynamic_update_slice` |

### vs. `bmm_4d`

| Aspect | custom_rms_norm | bmm_4d |
|--------|-----------------|---------|
| StableHLO op | `custom_call` | `dot_general` (standard) |
| Decomposition | External kernel | Standard ops |
| TFLite support | Requires workaround | ✅ Works |

---

## MLIR IR Flow

```
PyTorch Graph
    ↓ torch.export
FX Graph with custom_rms_norm node
    ↓ AI Edge Torch lowering
StableHLO MLIR
    func @main(%x, %weight) {
        %result = stablehlo.custom_call @ai_edge_torch.rms_norm(%x, %weight) {
            backend_config = "",
            has_side_effect = false,
            api_version = 1 : i32
        }
        return %result
    }
    ↓ VHLO serialization
VHLO MLIR (with modified TensorFlow fallback handler)
    ↓ TFLite conversion
TFLite FlatBuffer with STABLEHLO_CUSTOM_CALL (op code 173)
    ↓ TFLite runtime
C++ Kernel: ai_edge_torch.rms_norm
```

**Note**: Requires modified TensorFlow with VHLO custom_call fallback handler.
See `VHLO_CUSTOM_CALL_FIX.md` for details on the TensorFlow modifications.

---

## Testing

### Test Structure

```python
class TestCustomRMSNorm(parameterized.TestCase):
    def test_opcheck_custom_rms_norm(self, x, weight, epsilon):
        # Validate op contract
        torch.library.opcheck(custom_rms_norm, (x, weight, epsilon))
        
        # Verify correctness
        out = custom_rms_norm(x, weight, epsilon)
        expected = reference_implementation(x, weight, epsilon)
        assert torch.allclose(out, expected)
```

### Test Coverage

1. **Different input shapes**: 2D, 3D, 4D tensors
2. **Different epsilon values**: 1e-5, 1e-6, etc.
3. **Learned weights**: Non-uniform weight tensors
4. **Export verification**: Check op appears in FX graph
5. **Numerical accuracy**: Compare with reference implementation
6. **Input validation**: Test error handling

### Running Tests

```bash
cd ai_edge_torch
micromamba run -n ai_edge_torch_env python -m pytest \
    generative/test/test_custom_rms_norm.py -v
```

---

## Implementation Patterns

### Pattern 1: Custom Op with Standard Lowering

```python
# bmm_4d style - lowers to standard StableHLO
@lowerings.lower(torch.ops.ai_edge_torch.bmm_4d)
def _lower(lctx, lhs, rhs):
    return stablehlo.dot_general(...)  # Standard op
```

### Pattern 2: Custom Op with Custom Call (This Implementation)

```python
# custom_rms_norm style - lowers to custom_call
@lowerings.lower(torch.ops.ai_edge_torch.custom_rms_norm)
def _lower(lctx, x, weight, epsilon):
    return stablehlo.custom_call(
        call_target_name="CustomRmsNorm",
        ...
    )
```

---

## Future Enhancements

### 1. Support Multiple Dimensions

Currently normalizes over last dimension only. Could add `dim` parameter:

```python
def custom_rms_norm(x, weight, epsilon, dim=-1):
    variance = x.pow(2).mean(dim=dim, keepdim=True)
    ...
```

### 2. Fused Operations

Combine RMS norm with other operations:

```python
def fused_rms_norm_gelu(x, weight, epsilon):
    x = custom_rms_norm(x, weight, epsilon)
    return gelu(x)
```

### 3. Quantization Support

Add quantized version:

```python
def quantized_rms_norm(x, weight, epsilon, scale, zero_point):
    ...
```

---

## Related Files

- **Implementation**: `generative/custom_ops/custom_rms_norm.py`
- **Tests**: `generative/test/test_custom_rms_norm.py`
- **Similar ops**: `generative/custom_ops/dynamic_update_slice.py`

## See Also

- [Overview](./overview.md) - What is custom RMS norm
- [Usage Guide](./usage.md) - How to use
- [Limitations](./limitations.md) - Current constraints

---

**Implementation Style**: Custom operator with `stablehlo.custom_call`  
**Lines of Code**: ~160 (implementation + tests)  
**Dependencies**: torch, ai_edge_torch.odml_torch, JAX MLIR

