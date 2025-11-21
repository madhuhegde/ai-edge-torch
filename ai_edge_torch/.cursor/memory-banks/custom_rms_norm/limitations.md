# Custom RMS Normalization - Limitations and Workarounds

## Current Status

### ✅ TFLite Conversion Working (with Modified TensorFlow)

**Solution Implemented**: Added VHLO custom_call fallback handler to TensorFlow

**Status**: Successfully converts to TFLite with `STABLEHLO_CUSTOM_CALL` operation

**Requirement**: Modified TensorFlow build with fallback handler (see `VHLO_CUSTOM_CALL_FIX.md`)

**Previous Issue** (now resolved): 
```
Error: 'vhlo.custom_call_v1' op is not part of the vhlo support yet.
```

**Root Cause** (fixed): The typed cast `llvm::dyn_cast<mlir::vhlo::CustomCallOpV1>` was failing, causing the operation to be unhandled. Added a fallback handler that detects the operation by name and extracts attributes generically.

---

## How It Works Now

### Conversion Pipeline (With Modified TensorFlow)

```
PyTorch Model
    ↓
torch.export ✅
    ↓
FX Graph with custom_rms_norm ✅
    ↓
StableHLO MLIR with custom_call ✅
    ↓
AI Edge Torch (module_bytecode_vhlo) ✅
    ↓
TensorFlow SavedModel ✅
    ↓
TFLite Converter (Python) ✅
    ↓
TFLite Converter (C++ MLIR Pipeline)
    ├─ StableHLO → VHLO conversion ✅
    ├─ vhlo.custom_call_v1 detected ✅
    ├─ Fallback handler triggered ✅
    └─ Creates STABLEHLO_CUSTOM_CALL ✅
    ↓
TFLite Flatbuffer (op code 173) ✅
```

### TensorFlow Modification Required

| Component | Modification | Purpose |
|-----------|-------------|---------|
| flatbuffer_export.cc | Added fallback handler | Handles vhlo.custom_call_v1 by name |
| flatbuffer_export.cc | Generic attribute extraction | Extracts call_target_name without typed cast |
| flatbuffer_export.cc | Debug logging (optional) | Trace conversion flow |

---

## Workarounds

### Option 1: Direct TFLite Flatbuffer Generation (Recommended)

**Approach**: Implement custom TFLite converter in AI Edge Torch that bypasses TensorFlow's converter.

**Steps**:
1. Parse StableHLO MLIR directly
2. Map operations to TFLite ops manually
3. Generate TFLite flatbuffer directly
4. Insert CUSTOM op definitions

**Timeline**: 3-4 weeks of development

**Pros**:
- ✅ Complete solution within AI Edge Torch
- ✅ Supports all custom ops
- ✅ No external dependencies

**Cons**:
- ⚠️ Complex implementation
- ⚠️ Maintenance burden
- ⚠️ May miss TFLite optimizations

**Status**: Not yet implemented

---

### Option 2: Upstream VHLO Support (Long-term)

**Approach**: Contribute `custom_call_v1` support to VHLO dialect.

**Steps**:
1. Engage with MLIR-HLO maintainers
2. Propose design for `custom_call_v1`
3. Implement and test
4. Wait for TensorFlow adoption

**Timeline**: 3-6 months

**Pros**:
- ✅ Proper long-term solution
- ✅ Benefits entire community
- ✅ Maintains TFLite optimizations

**Cons**:
- ⚠️ Outside AI Edge Torch control
- ⚠️ Long timeline
- ⚠️ Requires coordination

**Status**: Not started

---

### Option 3: Use Decomposed Implementation (Alternative)

**Approach**: Implement RMS norm using standard StableHLO ops instead of `custom_call`.

**Example**:

```python
@lowerings.lower(torch.ops.ai_edge_torch.custom_rms_norm)
def _lower(lctx, x, weight, epsilon):
    # Decompose into standard ops
    x_squared = stablehlo.multiply(x, x)
    variance = stablehlo.reduce(x_squared, ...)  # mean over last dim
    rsqrt_var = stablehlo.rsqrt(...)
    normalized = stablehlo.multiply(x, rsqrt_var)
    return stablehlo.multiply(normalized, weight)
```

**Pros**:
- ✅ Works with standard TFLite converter
- ✅ No VHLO issues
- ✅ Immediate solution

**Cons**:
- ❌ Not a true custom op
- ❌ No custom C++ kernel
- ❌ Limited optimization control

**Note**: This defeats the purpose of having a custom op with external kernel.

---

## Current Status by Component

| Component | Status | Notes |
|-----------|--------|-------|
| PyTorch custom op | ✅ Working | Full functionality |
| Shape inference | ✅ Working | Fake implementation |
| StableHLO lowering | ✅ Working | Generates custom_call |
| torch.export | ✅ Working | FX graph generation |
| AI Edge Torch | ✅ Working | MLIR generation |
| TensorFlow | ✅ Working | SavedModel creation |
| **TFLite converter** | ❌ **Blocked** | **VHLO serialization fails** |
| TFLite runtime | ⏸️ Untested | Needs TFLite model first |
| C++ kernel | ⏸️ Not implemented | Awaiting TFLite support |

---

## What Works

### ✅ PyTorch Execution

```python
x = torch.randn(2, 128, 768)
weight = torch.ones(768)

# Works perfectly in PyTorch
output = custom_rms_norm(x, weight, 1e-6)
```

### ✅ Export to FX Graph

```python
model = SimpleModel()
exported_program = torch.export.export(model, (x,))

# Custom op appears in FX graph
for node in exported_program.graph.nodes:
    if 'custom_rms_norm' in node.target.__name__:
        print("✅ Found custom_rms_norm in graph")
```

### ✅ StableHLO MLIR Generation

```python
from ai_edge_torch.odml_torch import export

lowered = export.exported_program_to_mlir(exported_program)
mlir_text = lowered.get_text()

# custom_call appears in MLIR
assert "stablehlo.custom_call @CustomRmsNorm" in mlir_text
```

---

## What Doesn't Work

### ❌ TFLite Conversion

```python
import ai_edge_torch

model = SimpleModel()
x = torch.randn(2, 128, 768)

try:
    edge_model = ai_edge_torch.convert(model, (x,))
except Exception as e:
    print(f"❌ Error: {e}")
    # Error: 'vhlo.custom_call_v1' op is not part of the vhlo support yet
```

### ❌ TFLite Execution

Cannot test TFLite runtime because model generation fails.

---

## Recommendations

### For Research/Development

**Use PyTorch execution**: The custom op works perfectly in PyTorch for research and development.

```python
# ✅ Works for development
model = MyModel()
output = model(input)
```

### For Deployment

**Choose one**:

1. **Wait for Option 1**: Direct TFLite generation (recommended, 3-4 weeks)
2. **Contribute Option 2**: Upstream VHLO support (long-term, 3-6 months)
3. **Use Option 3**: Decomposed implementation (immediate, but not true custom op)

### For Production

**Use HLFB Composites** (like Gemma3 does):

```python
# Works today with standard TFLite converter
from ai_edge_torch.generative.layers.normalization import rms_norm_with_hlfb

# Uses composite operations instead of custom_call
output = rms_norm_with_hlfb(x, weight, epsilon)
```

---

## Testing Without TFLite

You can still test the custom op:

### 1. PyTorch Testing

```bash
pytest generative/test/test_custom_rms_norm.py -v
```

### 2. Export Testing

```python
# Verify op appears in exported graph
exported_program = torch.export.export(model, (x,))
# Check FX graph contains custom_rms_norm
```

### 3. MLIR Testing

```python
# Verify StableHLO IR generation
lowered = export.exported_program_to_mlir(exported_program)
mlir_text = lowered.get_text()
assert "stablehlo.custom_call" in mlir_text
```

---

## Future Outlook

### Short Term (1-2 months)

- Implement direct TFLite flatbuffer generation
- Enable end-to-end custom op support
- Add C++ kernel implementation examples

### Medium Term (3-6 months)

- Contribute VHLO custom_call support upstream
- Improve documentation and examples
- Add more custom op implementations

### Long Term (6-12 months)

- Standard TFLite converter supports custom_call
- Deprecate direct flatbuffer generation
- Full custom op ecosystem

---

## FAQ

**Q: Can I use this custom op in production?**  
A: Not yet. TFLite conversion is currently blocked. Use HLFB composites or wait for direct flatbuffer generation.

**Q: Will this ever work with standard TFLite converter?**  
A: Yes, once VHLO adds `custom_call_v1` support upstream.

**Q: Should I use this or HLFB composites?**  
A: For production today, use HLFB composites. For custom kernels, wait for direct flatbuffer generation.

**Q: How can I help?**  
A: Contribute to Option 1 (direct flatbuffer generation) or Option 2 (upstream VHLO support).

---

## See Also

- [Overview](./overview.md) - What is custom RMS norm
- [Usage Guide](./usage.md) - How to use (when working)
- [Implementation](./implementation.md) - Technical details

---

**Status**: ⚠️ Blocked by VHLO serialization  
**Workaround**: Awaiting direct TFLite flatbuffer generation  
**ETA**: 3-4 weeks for Option 1, 3-6 months for Option 2

