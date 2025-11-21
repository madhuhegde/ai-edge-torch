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

"""A suite of tests to validate the Custom RMS Norm Op."""

from ai_edge_torch.generative.custom_ops.custom_rms_norm import custom_rms_norm
import torch
from torch import nn

from absl.testing import absltest as googletest, parameterized


def compute_rms_norm_reference(x, weight, epsilon=1e-6):
  """Reference implementation of RMS normalization."""
  variance = x.pow(2).mean(dim=-1, keepdim=True)
  x_normalized = x * torch.rsqrt(variance + epsilon)
  return x_normalized * weight


class CustomRMSNormMod(nn.Module):
  """Test module using custom RMS norm."""

  def __init__(self, hidden_dim: int):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(hidden_dim))

  def forward(self, x):
    out = custom_rms_norm(x, self.weight, 1e-6)
    return out


class CustomRMSNormWithOps(nn.Module):
  """Test module with custom RMS norm and other operations."""

  def __init__(self, hidden_dim: int):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(hidden_dim))

  def forward(self, x):
    # Apply custom RMS norm
    x = custom_rms_norm(x, self.weight, 1e-6)
    # Additional operations
    x = x * 2.0
    x = x + 1.0
    return x


class TestCustomRMSNorm(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'CustomRMSNorm_2d',
          torch.randn(128, 768),
          torch.ones(768),
          1e-6,
      ),
      (
          'CustomRMSNorm_3d',
          torch.randn(2, 128, 768),
          torch.ones(768),
          1e-6,
      ),
      (
          'CustomRMSNorm_4d',
          torch.randn(2, 4, 128, 768),
          torch.ones(768),
          1e-6,
      ),
      (
          'CustomRMSNorm_different_epsilon',
          torch.randn(2, 128, 512),
          torch.ones(512),
          1e-5,
      ),
      (
          'CustomRMSNorm_learned_weight',
          torch.randn(2, 128, 768),
          torch.randn(768),
          1e-6,
      ),
  )
  def test_opcheck_custom_rms_norm(self, x, weight, epsilon):
    """Test custom RMS norm with torch.library.opcheck."""
    torch.library.opcheck(custom_rms_norm, (x, weight, epsilon))
    
    # Verify correctness
    out = custom_rms_norm(x, weight, epsilon)
    expected = compute_rms_norm_reference(x, weight, epsilon)
    self.assertTrue(torch.allclose(out, expected, rtol=1e-5, atol=1e-6))

  def test_exported_program(self):
    """Test that custom RMS norm appears in exported program."""
    x = torch.randn(2, 128, 768)
    model = CustomRMSNormMod(768)
    
    ep = torch.export.export(model, (x,))
    
    custom_rms_norm_in_exported_program = False
    for node in ep.graph.nodes:
      if node.op == 'call_function':
        if 'custom_rms_norm' in node.target.__name__:
          custom_rms_norm_in_exported_program = True
          break
    
    self.assertTrue(custom_rms_norm_in_exported_program)

  def test_numerical_accuracy(self):
    """Test numerical accuracy of custom RMS norm."""
    x = torch.randn(2, 128, 768)
    weight = torch.randn(768)
    epsilon = 1e-6
    
    out = custom_rms_norm(x, weight, epsilon)
    expected = compute_rms_norm_reference(x, weight, epsilon)
    
    self.assertTrue(torch.allclose(out, expected, rtol=1e-5, atol=1e-6))

  def test_shape_preservation(self):
    """Test that output shape matches input shape."""
    shapes = [
        (128, 768),
        (2, 128, 768),
        (4, 2, 128, 768),
    ]
    
    for shape in shapes:
      x = torch.randn(*shape)
      weight = torch.ones(shape[-1])
      out = custom_rms_norm(x, weight, 1e-6)
      self.assertEqual(out.shape, x.shape)

  def test_with_multiple_operations(self):
    """Test custom RMS norm in a model with other operations."""
    x = torch.randn(2, 128, 768)
    model = CustomRMSNormWithOps(768)
    
    # Test forward pass
    out = model(x)
    
    # Verify shape
    self.assertEqual(out.shape, x.shape)
    
    # Export and verify
    ep = torch.export.export(model, (x,))
    
    custom_rms_norm_found = False
    for node in ep.graph.nodes:
      if node.op == 'call_function':
        if 'custom_rms_norm' in node.target.__name__:
          custom_rms_norm_found = True
          break
    
    self.assertTrue(custom_rms_norm_found)

  def test_input_validation(self):
    """Test input validation."""
    # Test dimension mismatch
    with self.assertRaises(ValueError):
      x = torch.randn(2, 128, 768)
      weight = torch.ones(512)  # Wrong dimension
      custom_rms_norm(x, weight, 1e-6)
    
    # Test weight not 1D
    with self.assertRaises(ValueError):
      x = torch.randn(2, 128, 768)
      weight = torch.ones(768, 1)  # Should be 1D
      custom_rms_norm(x, weight, 1e-6)
    
    # Test negative epsilon
    with self.assertRaises(ValueError):
      x = torch.randn(2, 128, 768)
      weight = torch.ones(768)
      custom_rms_norm(x, weight, -1e-6)

  @parameterized.named_parameters(
      ('small_hidden_dim', 2, 128, 64),
      ('large_hidden_dim', 2, 128, 2048),
      ('single_batch', 1, 256, 768),
      ('large_batch', 8, 128, 768),
  )
  def test_different_sizes(self, batch, seq_len, hidden_dim):
    """Test custom RMS norm with different tensor sizes."""
    x = torch.randn(batch, seq_len, hidden_dim)
    weight = torch.ones(hidden_dim)
    
    out = custom_rms_norm(x, weight, 1e-6)
    expected = compute_rms_norm_reference(x, weight, 1e-6)
    
    self.assertEqual(out.shape, x.shape)
    self.assertTrue(torch.allclose(out, expected, rtol=1e-5, atol=1e-6))

  def test_tflite_conversion(self):
    """Test converting a model with custom_rms_norm to TFLite.
    
    This test verifies that:
    1. The model converts successfully to TFLite
    2. The TFLite model contains STABLEHLO_CUSTOM_CALL operation
    3. The call_target_name is correctly set to "ai_edge_torch.rms_norm"
    
    Note: Requires modified TensorFlow with VHLO custom_call fallback handler.
    See VHLO_CUSTOM_CALL_FIX.md for details.
    """
    try:
      import ai_edge_torch
      from tensorflow.lite.python import schema_py_generated as schema_fb
    except ImportError:
      self.skipTest('ai_edge_torch or TensorFlow not available')
    
    # Define a simple model using custom_rms_norm
    class RMSNormModel(torch.nn.Module):
      def __init__(self, hidden_dim: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_dim))
      
      def forward(self, x):
        return torch.ops.ai_edge_torch.custom_rms_norm(x, self.weight, 1e-6)
    
    # Create model and sample input
    hidden_dim = 768
    model = RMSNormModel(hidden_dim)
    sample_input = (torch.randn(2, 128, hidden_dim),)
    
    # Convert to TFLite
    try:
      tflite_model = ai_edge_torch.convert(model, sample_input)
    except Exception as e:
      # If conversion fails, it might be due to missing VHLO fallback handler
      error_msg = str(e)
      if 'vhlo.custom_call_v1' in error_msg and 'not part of the vhlo support' in error_msg:
        self.skipTest(
            'Conversion failed: Requires modified TensorFlow with VHLO fallback handler. '
            'See VHLO_CUSTOM_CALL_FIX.md'
        )
      else:
        raise
    
    # Save to temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.tflite', delete=False) as f:
      tflite_path = f.name
      tflite_model.export(tflite_path)
    
    try:
      # Read and verify the TFLite model
      with open(tflite_path, 'rb') as f:
        buf = bytearray(f.read())
      
      model_fb = schema_fb.Model.GetRootAsModel(buf, 0)
      
      # Verify model structure
      self.assertGreater(model_fb.SubgraphsLength(), 0, 'Model should have at least one subgraph')
      
      subgraph = model_fb.Subgraphs(0)
      self.assertGreater(subgraph.OperatorsLength(), 0, 'Subgraph should have at least one operator')
      
      # Find the custom_call operation
      found_custom_call = False
      for i in range(subgraph.OperatorsLength()):
        op = subgraph.Operators(i)
        opcode_idx = op.OpcodeIndex()
        opcode = model_fb.OperatorCodes(opcode_idx)
        builtin_code = opcode.BuiltinCode()
        
        # Check if this is a STABLEHLO_CUSTOM_CALL (op code 173)
        if builtin_code == 173:  # STABLEHLO_CUSTOM_CALL
          found_custom_call = True
          
          # Verify BuiltinOptions2 contains StablehloCustomCallOptions
          self.assertEqual(
              op.BuiltinOptions2Type(),
              schema_fb.BuiltinOptions2.StablehloCustomCallOptions,
              'Expected StablehloCustomCallOptions'
          )
          
          # Extract and verify call_target_name
          options = schema_fb.StablehloCustomCallOptions()
          options.Init(op.BuiltinOptions2().Bytes, op.BuiltinOptions2().Pos)
          
          call_target_name = options.CallTargetName().decode('utf-8')
          self.assertEqual(
              call_target_name,
              'ai_edge_torch.rms_norm',
              f'Expected call_target_name to be "ai_edge_torch.rms_norm", got "{call_target_name}"'
          )
          
          # Verify other attributes
          self.assertEqual(options.ApiVersion(), 1, 'Expected API version 1')
          self.assertFalse(options.HasSideEffect(), 'Expected has_side_effect to be False')
          
          break
      
      self.assertTrue(
          found_custom_call,
          'Expected to find STABLEHLO_CUSTOM_CALL operation (op code 173) in TFLite model'
      )
      
    finally:
      # Clean up temporary file
      import os
      if os.path.exists(tflite_path):
        os.remove(tflite_path)


if __name__ == '__main__':
  googletest.main()

