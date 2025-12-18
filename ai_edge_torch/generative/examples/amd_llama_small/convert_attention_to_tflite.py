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

"""Convert attention-only model to TFLite with single decode signature.

This script converts the AttentionOnlyModel to TFLite format with a single
"decode" signature that takes Q, K, V, and mask as inputs.
"""

import os
from absl import app
from absl import flags
import ai_edge_torch
from ai_edge_torch.generative.examples.amd_llama_small import amd_llama_attention
import torch

_OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    '/home/madhuhegde/work/models/tflite_models/amd_llama_attention_decode.tflite',
    'Path to save the TFLite model.',
)

_KV_LEN = flags.DEFINE_integer(
    'kv_len',
    512,
    'KV cache length for decode signature.',
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


def create_sample_inputs(kv_len: int = 512, num_heads: int = 12, head_dim: int = 64):
  """Create sample inputs for decode signature.

  Args:
    kv_len: KV cache length (default: 512)
    num_heads: Number of attention heads (default: 12)
    head_dim: Head dimension (default: 64)

  Returns:
    Tuple of (q, k, v, mask) tensors
  """
  # Query: [batch=1, seq_len=1, num_heads, head_dim]
  q = torch.randn(1, 1, num_heads, head_dim, dtype=torch.float32)

  # Key: [batch=1, kv_len, num_heads, head_dim]
  k = torch.randn(1, kv_len, num_heads, head_dim, dtype=torch.float32)

  # Value: [batch=1, kv_len, num_heads, head_dim]
  v = torch.randn(1, kv_len, num_heads, head_dim, dtype=torch.float32)

  # Mask: [batch=1, 1, 1, kv_len] (causal mask or zeros)
  mask = torch.zeros(1, 1, 1, kv_len, dtype=torch.float32)

  return q, k, v, mask


def main(_):
  """Main conversion function."""
  print('=' * 80)
  print('Converting Attention-Only Model to TFLite')
  print('=' * 80)
  print(f'Number of heads: {_NUM_HEADS.value}')
  print(f'Head dimension: {_HEAD_DIM.value}')
  print(f'KV length: {_KV_LEN.value}')
  print(f'Output path: {_OUTPUT_PATH.value}')
  print()

  # Create model
  print('1. Creating attention-only model...')
  model = amd_llama_attention.AttentionOnlyModel(
      head_dim=_HEAD_DIM.value, num_heads=_NUM_HEADS.value
  )
  model.eval()
  print(f'   ✓ Model created (head_dim={_HEAD_DIM.value}, num_heads={_NUM_HEADS.value})')
  print()

  # Create sample inputs
  print('2. Creating sample inputs for decode signature...')
  q, k, v, mask = create_sample_inputs(
      kv_len=_KV_LEN.value, num_heads=_NUM_HEADS.value, head_dim=_HEAD_DIM.value
  )
  print(f'   Q shape: {q.shape}')
  print(f'   K shape: {k.shape}')
  print(f'   V shape: {v.shape}')
  print(f'   Mask shape: {mask.shape}')
  print()

  # Test forward pass
  print('3. Testing forward pass...')
  with torch.no_grad():
    output = model(q, k, v, mask)
  print(f'   Output shape: {output.shape}')
  print(f'   ✓ Forward pass successful')
  print()

  # Convert to TFLite
  print('4. Converting to TFLite (decode signature)...')
  print('   Note: Using scaled_dot_product_attention (not _with_hlfb)')
  print('   This will decompose into standard TFLite ops (no STABLEHLO_COMPOSITE)')
  print()

  # Ensure output directory exists
  output_dir = os.path.dirname(_OUTPUT_PATH.value)
  if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

  # Convert using ai_edge_torch.signature API
  edge_model = ai_edge_torch.signature(
      'decode',
      model,
      (q, k, v, mask),
  ).convert()

  # Export to TFLite
  edge_model.export(_OUTPUT_PATH.value)
  print(f'   ✓ TFLite model saved to: {_OUTPUT_PATH.value}')
  print()

  # Verify the model
  print('5. Verifying TFLite model...')
  try:
    from tensorflow.lite.python import schema_py_generated as schema_fb

    with open(_OUTPUT_PATH.value, 'rb') as f:
      buf = bytearray(f.read())

    tflite_model = schema_fb.Model.GetRootAsModel(buf, 0)
    subgraph = tflite_model.Subgraphs(0)

    print(f'   Model version: {tflite_model.Version()}')
    print(f'   Subgraphs: {tflite_model.SubgraphsLength()}')
    print(f'   Operators: {subgraph.OperatorsLength()}')
    print(f'   Tensors: {subgraph.TensorsLength()}')
    print(f'   Inputs: {subgraph.InputsLength()}')
    print(f'   Outputs: {subgraph.OutputsLength()}')

    # Check for STABLEHLO_COMPOSITE ops
    composite_count = 0
    for i in range(subgraph.OperatorsLength()):
      op = subgraph.Operators(i)
      opcode_idx = op.OpcodeIndex()
      opcode = tflite_model.OperatorCodes(opcode_idx)
      if opcode.BuiltinCode() == 206:  # STABLEHLO_COMPOSITE
        composite_count += 1

    if composite_count == 0:
      print(f'   ✓ No STABLEHLO_COMPOSITE operations found (as expected)')
    else:
      print(f'   ⚠ Found {composite_count} STABLEHLO_COMPOSITE operation(s)')

    # Check signatures
    if tflite_model.SignatureDefsLength() > 0:
      print(f'   Signatures: {tflite_model.SignatureDefsLength()}')
      for i in range(tflite_model.SignatureDefsLength()):
        sig = tflite_model.SignatureDefs(i)
        sig_name = sig.SignatureKey().decode('utf-8') if sig.SignatureKey() else 'unknown'
        print(f'     - {sig_name}')

    file_size = os.path.getsize(_OUTPUT_PATH.value)
    print(f'   File size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)')
    print()

  except Exception as e:
    print(f'   ⚠ Could not verify model: {e}')
    print()

  print('=' * 80)
  print('Conversion complete!')
  print('=' * 80)


if __name__ == '__main__':
  app.run(main)

