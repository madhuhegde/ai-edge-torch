# Copyright 2024 The AI Edge Torch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Verify TFLite model output matches PyTorch model output.

This script loads both the PyTorch model and the converted TFLite model,
runs inference with the same inputs, and compares the outputs to verify
they match within acceptable tolerance.
"""

import os
from absl import app
from absl import flags
import numpy as np
from ai_edge_torch.generative.examples.amd_llama_small import amd_llama_attention
import tensorflow as tf
import torch

_TFLITE_PATH = flags.DEFINE_string(
    'tflite_path',
    '/home/madhuhegde/work/models/tflite_models/amd_llama_attention_decode.tflite',
    'Path to the TFLite model file.',
)

_KV_LEN = flags.DEFINE_integer(
    'kv_len',
    512,
    'KV cache length for testing.',
)

_NUM_HEADS = flags.DEFINE_integer(
    'num_heads',
    12,
    'Number of attention heads.',
)

_HEAD_DIM = flags.DEFINE_integer(
    'head_dim',
    64,
    'Head dimension.',
)

_TOLERANCE = flags.DEFINE_float(
    'tolerance',
    1e-5,
    'Tolerance for comparing outputs (default: 1e-5).',
)

_SAVE_TEST_VECTORS = flags.DEFINE_string(
    'save_test_vectors',
    None,
    'Path to save test vectors in C-friendly format (optional).',
)


def create_test_inputs(kv_len: int = 512, num_heads: int = 12, head_dim: int = 64, seed: int = 42):
  """Create test inputs with fixed seed for reproducibility.

  Args:
    kv_len: KV cache length
    num_heads: Number of attention heads
    head_dim: Head dimension
    seed: Random seed for reproducibility

  Returns:
    Tuple of (q, k, v, mask) tensors
  """
  torch.manual_seed(seed)
  np.random.seed(seed)

  # Query: [batch=1, seq_len=1, num_heads, head_dim]
  q = torch.randn(1, 1, num_heads, head_dim, dtype=torch.float32)

  # Key: [batch=1, kv_len, num_heads, head_dim]
  k = torch.randn(1, kv_len, num_heads, head_dim, dtype=torch.float32)

  # Value: [batch=1, kv_len, num_heads, head_dim]
  v = torch.randn(1, kv_len, num_heads, head_dim, dtype=torch.float32)

  # Mask: [batch=1, 1, 1, kv_len] (causal mask or zeros)
  mask = torch.zeros(1, 1, 1, kv_len, dtype=torch.float32)

  return q, k, v, mask


def run_pytorch_inference(model, q, k, v, mask):
  """Run inference on PyTorch model.

  Args:
    model: PyTorch model
    q, k, v, mask: Input tensors

  Returns:
    Output tensor
  """
  model.eval()
  with torch.no_grad():
    output = model(q, k, v, mask)
  return output


def run_tflite_inference(tflite_path, q, k, v, mask):
  """Run inference on TFLite model.

  Args:
    tflite_path: Path to TFLite model
    q, k, v, mask: Input tensors (PyTorch)

  Returns:
    Output as numpy array
  """
  # Load TFLite model
  interpreter = tf.lite.Interpreter(model_path=tflite_path)
  interpreter.allocate_tensors()

  # Get input and output details
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # Convert PyTorch tensors to numpy
  inputs = {
      'decode_args_0': q.numpy(),
      'decode_args_1': k.numpy(),
      'decode_args_2': v.numpy(),
      'decode_args_3': mask.numpy(),
  }

  # Set input tensors
  for detail in input_details:
    input_name = detail['name']
    # Map input names to our inputs
    if 'args_0' in input_name:
      interpreter.set_tensor(detail['index'], inputs['decode_args_0'])
    elif 'args_1' in input_name:
      interpreter.set_tensor(detail['index'], inputs['decode_args_1'])
    elif 'args_2' in input_name:
      interpreter.set_tensor(detail['index'], inputs['decode_args_2'])
    elif 'args_3' in input_name:
      interpreter.set_tensor(detail['index'], inputs['decode_args_3'])

  # Run inference
  interpreter.invoke()

  # Get output
  output = interpreter.get_tensor(output_details[0]['index'])
  return output


def compare_outputs(pytorch_output, tflite_output, tolerance=1e-5):
  """Compare PyTorch and TFLite outputs.

  Args:
    pytorch_output: PyTorch tensor output
    tflite_output: TFLite numpy array output
    tolerance: Maximum allowed difference

  Returns:
    Tuple of (max_diff, mean_diff, all_close)
  """
  # Convert PyTorch to numpy
  pytorch_np = pytorch_output.numpy()

  # Ensure shapes match
  if pytorch_np.shape != tflite_output.shape:
    return None, None, False, f"Shape mismatch: PyTorch {pytorch_np.shape} vs TFLite {tflite_output.shape}"

  # Calculate differences
  diff = np.abs(pytorch_np - tflite_output)
  max_diff = np.max(diff)
  mean_diff = np.mean(diff)
  all_close = np.allclose(pytorch_np, tflite_output, atol=tolerance, rtol=tolerance)

  return max_diff, mean_diff, all_close, None


def save_test_vectors_c_format(q, k, v, mask, output_path, pytorch_output=None, tflite_output=None):
  """Save test vectors in C-friendly format.

  Args:
    q, k, v, mask: Input tensors (PyTorch)
    output_path: Path to save the test vectors file
    pytorch_output: PyTorch output tensor (optional)
    tflite_output: TFLite output numpy array (optional)
  """
  with open(output_path, 'w') as f:
    f.write('/* Test vectors for AMD Llama Attention Model\n')
    f.write(' * Generated automatically by verify_attention_tflite.py\n')
    f.write(' *\n')
    f.write(' * Input shapes:\n')
    f.write(f' *   Q:    {list(q.shape)}\n')
    f.write(f' *   K:    {list(k.shape)}\n')
    f.write(f' *   V:    {list(v.shape)}\n')
    f.write(f' *   Mask: {list(mask.shape)}\n')
    if pytorch_output is not None:
      f.write(f' *   Output (PyTorch): {list(pytorch_output.shape)}\n')
    if tflite_output is not None:
      f.write(f' *   Output (TFLite):  {list(tflite_output.shape)}\n')
    f.write(' */\n\n')
    f.write('#include <stdint.h>\n')
    f.write('#include <stddef.h>\n\n')

    # Save Q
    q_np = q.numpy().flatten()
    f.write(f'// Q tensor: shape {list(q.shape)}, total elements: {q_np.size}\n')
    f.write(f'static const float q_data[] = {{\n')
    for i, val in enumerate(q_np):
      if i < len(q_np) - 1:
        f.write(f'  {val:.9f}f,\n')
      else:
        f.write(f'  {val:.9f}f\n')
    f.write('};\n\n')
    f.write(f'static const size_t q_size = {q_np.size};\n')
    f.write(f'static const int q_shape[] = {{ {", ".join(map(str, q.shape))} }};\n\n')

    # Save K
    k_np = k.numpy().flatten()
    f.write(f'// K tensor: shape {list(k.shape)}, total elements: {k_np.size}\n')
    f.write(f'static const float k_data[] = {{\n')
    for i, val in enumerate(k_np):
      if i < len(k_np) - 1:
        f.write(f'  {val:.9f}f,\n')
      else:
        f.write(f'  {val:.9f}f\n')
    f.write('};\n\n')
    f.write(f'static const size_t k_size = {k_np.size};\n')
    f.write(f'static const int k_shape[] = {{ {", ".join(map(str, k.shape))} }};\n\n')

    # Save V
    v_np = v.numpy().flatten()
    f.write(f'// V tensor: shape {list(v.shape)}, total elements: {v_np.size}\n')
    f.write(f'static const float v_data[] = {{\n')
    for i, val in enumerate(v_np):
      if i < len(v_np) - 1:
        f.write(f'  {val:.9f}f,\n')
      else:
        f.write(f'  {val:.9f}f\n')
    f.write('};\n\n')
    f.write(f'static const size_t v_size = {v_np.size};\n')
    f.write(f'static const int v_shape[] = {{ {", ".join(map(str, v.shape))} }};\n\n')

    # Save Mask
    mask_np = mask.numpy().flatten()
    f.write(f'// Mask tensor: shape {list(mask.shape)}, total elements: {mask_np.size}\n')
    f.write(f'static const float mask_data[] = {{\n')
    for i, val in enumerate(mask_np):
      if i < len(mask_np) - 1:
        f.write(f'  {val:.9f}f,\n')
      else:
        f.write(f'  {val:.9f}f\n')
    f.write('};\n\n')
    f.write(f'static const size_t mask_size = {mask_np.size};\n')
    f.write(f'static const int mask_shape[] = {{ {", ".join(map(str, mask.shape))} }};\n\n')

    # Save outputs if provided
    if pytorch_output is not None:
      output_np = pytorch_output.numpy().flatten()
      f.write(f'// Expected output (PyTorch): shape {list(pytorch_output.shape)}, total elements: {output_np.size}\n')
      f.write(f'static const float expected_output_pytorch[] = {{\n')
      for i, val in enumerate(output_np):
        if i < len(output_np) - 1:
          f.write(f'  {val:.9f}f,\n')
        else:
          f.write(f'  {val:.9f}f\n')
      f.write('};\n\n')
      f.write(f'static const size_t expected_output_pytorch_size = {output_np.size};\n')
      f.write(f'static const int expected_output_pytorch_shape[] = {{ {", ".join(map(str, pytorch_output.shape))} }};\n\n')

    if tflite_output is not None:
      output_np = tflite_output.flatten()
      f.write(f'// Expected output (TFLite): shape {list(tflite_output.shape)}, total elements: {output_np.size}\n')
      f.write(f'static const float expected_output_tflite[] = {{\n')
      for i, val in enumerate(output_np):
        if i < len(output_np) - 1:
          f.write(f'  {val:.9f}f,\n')
        else:
          f.write(f'  {val:.9f}f\n')
      f.write('};\n\n')
      f.write(f'static const size_t expected_output_tflite_size = {output_np.size};\n')
      f.write(f'static const int expected_output_tflite_shape[] = {{ {", ".join(map(str, tflite_output.shape))} }};\n\n')


def main(_):
  """Main verification function."""
  print('=' * 80)
  print('Verifying TFLite Model Output Against PyTorch Model')
  print('=' * 80)
  print(f'TFLite path: {_TFLITE_PATH.value}')
  print(f'KV length: {_KV_LEN.value}')
  print(f'Tolerance: {_TOLERANCE.value}')
  print()

  # Check if TFLite file exists
  if not os.path.exists(_TFLITE_PATH.value):
    print(f'❌ Error: TFLite file not found: {_TFLITE_PATH.value}')
    return

  # Create PyTorch model
  print('1. Creating PyTorch model...')
  pytorch_model = amd_llama_attention.AttentionOnlyModel(
      head_dim=_HEAD_DIM.value, num_heads=_NUM_HEADS.value
  )
  pytorch_model.eval()
  print(f'   ✓ PyTorch model created (head_dim={_HEAD_DIM.value}, num_heads={_NUM_HEADS.value})')
  print()

  # Create test inputs
  print('2. Creating test inputs...')
  q, k, v, mask = create_test_inputs(
      kv_len=_KV_LEN.value, num_heads=_NUM_HEADS.value, head_dim=_HEAD_DIM.value, seed=42
  )
  print(f'   Q shape: {q.shape}')
  print(f'   K shape: {k.shape}')
  print(f'   V shape: {v.shape}')
  print(f'   Mask shape: {mask.shape}')
  print()

  # Run PyTorch inference
  print('3. Running PyTorch inference...')
  pytorch_output = run_pytorch_inference(pytorch_model, q, k, v, mask)
  print(f'   Output shape: {pytorch_output.shape}')
  print(f'   Output range: [{pytorch_output.min():.6f}, {pytorch_output.max():.6f}]')
  print(f'   Output mean: {pytorch_output.mean():.6f}')
  print(f'   Output std: {pytorch_output.std():.6f}')
  print()

  # Run TFLite inference
  print('4. Running TFLite inference...')
  try:
    tflite_output = run_tflite_inference(_TFLITE_PATH.value, q, k, v, mask)
    print(f'   Output shape: {tflite_output.shape}')
    print(f'   Output range: [{tflite_output.min():.6f}, {tflite_output.max():.6f}]')
    print(f'   Output mean: {tflite_output.mean():.6f}')
    print(f'   Output std: {tflite_output.std():.6f}')
    print()
  except Exception as e:
    print(f'   ❌ Error running TFLite inference: {e}')
    import traceback
    traceback.print_exc()
    return

  # Compare outputs
  print('5. Comparing outputs...')
  max_diff, mean_diff, all_close, error = compare_outputs(
      pytorch_output, tflite_output, tolerance=_TOLERANCE.value
  )

  if error:
    print(f'   ❌ {error}')
    return

  print(f'   Max difference: {max_diff:.2e}')
  print(f'   Mean difference: {mean_diff:.2e}')
  print(f'   Tolerance: {_TOLERANCE.value:.2e}')
  print()

  # Final result
  print('=' * 80)
  if all_close:
    print('✓ VERIFICATION PASSED: Outputs match within tolerance!')
  else:
    print('❌ VERIFICATION FAILED: Outputs do not match within tolerance')
    print(f'   Max difference ({max_diff:.2e}) exceeds tolerance ({_TOLERANCE.value:.2e})')
  print('=' * 80)

  # Additional statistics
  if max_diff is not None:
    print()
    print('Additional Statistics:')
    pytorch_np = pytorch_output.numpy()
    diff = np.abs(pytorch_np - tflite_output)
    print(f'   Min difference: {np.min(diff):.2e}')
    print(f'   Max difference: {np.max(diff):.2e}')
    print(f'   Mean difference: {np.mean(diff):.2e}')
    print(f'   Median difference: {np.median(diff):.2e}')
    print(f'   95th percentile: {np.percentile(diff, 95):.2e}')
    print(f'   99th percentile: {np.percentile(diff, 99):.2e}')

    # Check relative error
    relative_error = diff / (np.abs(pytorch_np) + 1e-8)
    print(f'   Max relative error: {np.max(relative_error):.2e}')
    print(f'   Mean relative error: {np.mean(relative_error):.2e}')

  # Save test vectors if requested
  if _SAVE_TEST_VECTORS.value:
    print()
    print('6. Saving test vectors...')
    save_test_vectors_c_format(
        q, k, v, mask,
        _SAVE_TEST_VECTORS.value,
        pytorch_output=pytorch_output if 'pytorch_output' in locals() else None,
        tflite_output=tflite_output if 'tflite_output' in locals() else None
    )
    print(f'   ✓ Test vectors saved to: {_SAVE_TEST_VECTORS.value}')


if __name__ == '__main__':
  app.run(main)

