# Custom RMS Normalization - Testing Guide

## Test Suite Overview

The custom RMS norm implementation includes comprehensive tests covering PyTorch functionality, StableHLO lowering, and TFLite conversion.

## Test Files

### 1. `test_custom_rms_norm.py`

**Location**: `generative/test/test_custom_rms_norm.py`

Comprehensive test suite with 15 test cases using the `absltest` framework.

#### Test Categories

1. **OpCheck Tests** (5 tests)
   - 2D tensors
   - 3D tensors
   - 4D tensors
   - Different epsilon values
   - Learned weights

2. **Export Tests** (1 test)
   - Verifies custom op appears in exported program

3. **Numerical Tests** (1 test)
   - Validates numerical accuracy against reference implementation

4. **Shape Tests** (1 test)
   - Ensures output shape matches input shape

5. **Integration Tests** (1 test)
   - Tests custom op with other operations

6. **Validation Tests** (1 test)
   - Input dimension validation
   - Weight dimension validation
   - Epsilon validation

7. **Size Variation Tests** (4 tests)
   - Small hidden dimensions
   - Large hidden dimensions
   - Single batch
   - Large batch

8. **TFLite Conversion Test** (1 test)
   - Verifies TFLite conversion
   - Validates STABLEHLO_CUSTOM_CALL operation
   - Checks call_target_name attribute

#### Running the Tests

```bash
# Run all tests
cd ~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch
micromamba run -n local_tf_env python generative/test/test_custom_rms_norm.py

# Run specific test
micromamba run -n local_tf_env python generative/test/test_custom_rms_norm.py TestCustomRMSNorm.test_tflite_conversion
```

#### Expected Output

```
Running tests under Python 3.11.14: /path/to/python
[ RUN      ] TestCustomRMSNorm.test_different_sizes_large_batch
[       OK ] TestCustomRMSNorm.test_different_sizes_large_batch
...
[ RUN      ] TestCustomRMSNorm.test_tflite_conversion
[       OK ] TestCustomRMSNorm.test_tflite_conversion
...
----------------------------------------------------------------------
Ran 15 tests in 1.585s

OK
```

---

### 2. `test_tflite_conversion.py`

**Location**: `generative/test/test_tflite_conversion.py`

Standalone test script for TFLite conversion with detailed progress output.

#### Features

- ✅ Command-line argument support
- ✅ Detailed step-by-step output
- ✅ Explicit error messages
- ✅ Model verification
- ✅ Attribute validation
- ✅ Configurable output path

#### Usage

```bash
# Save to default location (/tmp/custom_rms_norm.tflite)
python generative/test/test_tflite_conversion.py

# Save to custom location
python generative/test/test_tflite_conversion.py --output_path ~/work/LLM_dir/custom_rms_norm.tflite

# Show help
python generative/test/test_tflite_conversion.py --help
```

#### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--output_path` | Path to save TFLite model | `/tmp/custom_rms_norm.tflite` |

#### Expected Output

```
================================================================================
Testing TFLite Conversion with custom_rms_norm
================================================================================

1. Creating model (hidden_dim=768)...
   ✓ Model created

2. Converting to TFLite...
   ✓ Conversion succeeded

3. Saving TFLite model...
   ✓ Saved to /tmp/custom_rms_norm.tflite

4. Verifying TFLite model structure...
   ✓ Found 1 subgraph(s)
   ✓ Found 1 operator(s)

5. Looking for STABLEHLO_CUSTOM_CALL operation...
   Op 0: builtin_code = 173 (STABLEHLO_CUSTOM_CALL) ✓

6. Verifying custom call attributes...
   call_target_name: ai_edge_torch.rms_norm
   api_version: 1
   has_side_effect: False
   backend_config: ''
   ✓ call_target_name is correct
   ✓ api_version is correct
   ✓ has_side_effect is correct

================================================================================
✅ ALL TESTS PASSED!
================================================================================

The TFLite model contains:
  - STABLEHLO_CUSTOM_CALL operation (op code 173)
  - call_target_name: ai_edge_torch.rms_norm
  - api_version: 1
  - has_side_effect: False

Next step: Implement C++ TFLite kernel for 'ai_edge_torch.rms_norm'
================================================================================

TFLite model saved at: /tmp/custom_rms_norm.tflite
```

---

## Test Environment Setup

### Prerequisites

1. **Modified TensorFlow**
   ```bash
   # TensorFlow with VHLO fallback handler
   # See VHLO_CUSTOM_CALL_FIX.md for build instructions
   ```

2. **Python Environment**
   ```bash
   # Create or activate environment
   micromamba activate local_tf_env
   
   # Install AI Edge Torch in editable mode
   cd ~/work/ai_edge_torch/ai-edge-torch
   pip install -e . --no-deps
   ```

3. **Verify Installation**
   ```bash
   python -c "import ai_edge_torch; print(ai_edge_torch.__version__)"
   python -c "from ai_edge_torch.generative.custom_ops.custom_rms_norm import custom_rms_norm"
   ```

---

## Reference Implementation

The tests validate against this reference implementation:

```python
def compute_rms_norm_reference(x, weight, epsilon=1e-6):
    """Reference implementation of RMS normalization."""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normalized = x * torch.rsqrt(variance + epsilon)
    return x_normalized * weight
```

---

## TFLite Model Verification

### Manual Verification

```python
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema_fb

# Load TFLite model
with open('custom_rms_norm.tflite', 'rb') as f:
    buf = bytearray(f.read())

model = schema_fb.Model.GetRootAsModel(buf, 0)
subgraph = model.Subgraphs(0)

# Check operations
for i in range(subgraph.OperatorsLength()):
    op = subgraph.Operators(i)
    opcode = model.OperatorCodes(op.OpcodeIndex())
    print(f"Op {i}: builtin_code = {opcode.BuiltinCode()}")
    
    if opcode.BuiltinCode() == 173:  # STABLEHLO_CUSTOM_CALL
        options = schema_fb.StablehloCustomCallOptions()
        options.Init(op.BuiltinOptions2().Bytes, op.BuiltinOptions2().Pos)
        print(f"  call_target_name: {options.CallTargetName().decode('utf-8')}")
        print(f"  api_version: {options.ApiVersion()}")
```

### Using TensorFlow Lite Interpreter

```python
# Load interpreter
interpreter = tf.lite.Interpreter(model_path='custom_rms_norm.tflite')
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input shape:", input_details[0]['shape'])
print("Output shape:", output_details[0]['shape'])
```

---

## Troubleshooting

### Test Failures

#### 1. ModuleNotFoundError: No module named 'ai_edge_torch.generative.custom_ops.custom_rms_norm'

**Solution**: Reinstall AI Edge Torch in editable mode
```bash
cd ~/work/ai_edge_torch/ai-edge-torch
micromamba run -n local_tf_env pip install -e . --no-deps
```

#### 2. VHLO custom_call_v1 not supported

**Error**: `'vhlo.custom_call_v1' op is not part of the vhlo support yet`

**Solution**: Use modified TensorFlow with fallback handler
- See `VHLO_CUSTOM_CALL_FIX.md` for build instructions
- Verify TensorFlow build: Check for debug logs showing "FALLBACK: Handling vhlo.custom_call_v1"

#### 3. TFLite model contains decomposed ops instead of STABLEHLO_CUSTOM_CALL

**Cause**: Using unmodified TensorFlow

**Solution**: Rebuild and install modified TensorFlow with fallback handler

### Debug Logging

The modified TensorFlow includes debug logging:

```
[DEBUG] VHLO dialect detected for op: vhlo.custom_call_v1
[DEBUG] FALLBACK: Handling vhlo.custom_call_v1 by name
[DEBUG] Found call_target_name attribute
[DEBUG] Extracted call_target_name: ai_edge_torch.rms_norm
[DEBUG] FALLBACK: Created StablehloCustomCallOptions
```

If you see these logs, the fallback handler is working correctly.

---

## Test Results Summary

### Current Status (as of Nov 21, 2024)

✅ **All 15 tests passing**
- PyTorch functionality: ✅
- StableHLO lowering: ✅
- TFLite conversion: ✅
- Model verification: ✅

### Performance

- Test execution time: ~1.6 seconds (all 15 tests)
- TFLite model size: 4.1 KB
- No memory leaks detected

### Compatibility

- **Python**: 3.11.14
- **PyTorch**: Compatible with torch.library.custom_op API
- **TensorFlow**: 2.21.0-dev (custom build with fallback handler)
- **AI Edge Torch**: 0.7.0

---

## Next Steps

1. **Implement C++ TFLite Kernel**
   - Create kernel for `ai_edge_torch.rms_norm`
   - Register kernel with TFLite runtime
   - Test end-to-end execution

2. **Add Runtime Tests**
   - Test actual TFLite model execution
   - Validate output against reference
   - Benchmark performance

3. **Integration Tests**
   - Test in Gemma3 model conversion
   - Verify with real model workloads
   - Performance profiling

---

## See Also

- [Overview](./overview.md) - Introduction to custom RMS norm
- [Usage Guide](./usage.md) - How to use the operator
- [Implementation](./implementation.md) - Technical details
- [Limitations](./limitations.md) - Current limitations

---

**Last Updated**: November 21, 2024  
**Test Environment**: Ubuntu 22.04, ARM64  
**Status**: All tests passing ✅

