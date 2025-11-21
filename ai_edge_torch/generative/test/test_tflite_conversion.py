#!/usr/bin/env python3
"""Standalone test for TFLite conversion of custom_rms_norm.

This test can be run independently to verify that:
1. The model converts successfully to TFLite
2. The TFLite model contains STABLEHLO_CUSTOM_CALL operation
3. The call_target_name is correctly set to "ai_edge_torch.rms_norm"

Requirements:
- Modified TensorFlow with VHLO custom_call fallback handler
- See VHLO_CUSTOM_CALL_FIX.md for details

Usage:
  python generative/test/test_tflite_conversion.py [--output_path PATH]
  
  Options:
    --output_path PATH    Path to save the TFLite model (default: /tmp/custom_rms_norm.tflite)
"""

import sys
import os
import argparse
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    import ai_edge_torch
    from tensorflow.lite.python import schema_py_generated as schema_fb
except ImportError as e:
    print(f"ERROR: Required module not found: {e}")
    print("Please ensure ai_edge_torch and TensorFlow are installed")
    sys.exit(1)

# Import the custom op
from generative.custom_ops.custom_rms_norm import custom_rms_norm


def test_tflite_conversion(output_path=None):
    """Test converting a model with custom_rms_norm to TFLite.
    
    Args:
        output_path: Path to save the TFLite model. If None, saves to /tmp/custom_rms_norm.tflite
    """
    if output_path is None:
        output_path = '/tmp/custom_rms_norm.tflite'
    
    print("=" * 80)
    print("Testing TFLite Conversion with custom_rms_norm")
    print("=" * 80)
    
    # Define a simple model using custom_rms_norm
    class RMSNormModel(torch.nn.Module):
        def __init__(self, hidden_dim: int):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(hidden_dim))
        
        def forward(self, x):
            return torch.ops.ai_edge_torch.custom_rms_norm(x, self.weight, 1e-6)
    
    # Create model and sample input
    hidden_dim = 768
    batch_size = 2
    seq_len = 128
    
    print(f"\n1. Creating model (hidden_dim={hidden_dim})...")
    model = RMSNormModel(hidden_dim)
    sample_input = (torch.randn(batch_size, seq_len, hidden_dim),)
    print("   ✓ Model created")
    
    # Convert to TFLite
    print("\n2. Converting to TFLite...")
    try:
        tflite_model = ai_edge_torch.convert(model, sample_input)
        print("   ✓ Conversion succeeded")
    except Exception as e:
        error_msg = str(e)
        if 'vhlo.custom_call_v1' in error_msg and 'not part of the vhlo support' in error_msg:
            print(f"   ✗ Conversion failed: {e}")
            print("\n" + "=" * 80)
            print("ERROR: VHLO custom_call not supported")
            print("=" * 80)
            print("This error indicates you need modified TensorFlow.")
            print("See VHLO_CUSTOM_CALL_FIX.md for details on building TensorFlow")
            print("with VHLO custom_call fallback handler.")
            sys.exit(1)
        else:
            print(f"   ✗ Unexpected error: {e}")
            raise
    
    # Save to file
    print("\n3. Saving TFLite model...")
    tflite_path = output_path
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(tflite_path) or '.', exist_ok=True)
    tflite_model.export(tflite_path)
    print(f"   ✓ Saved to {os.path.abspath(tflite_path)}")
    
    try:
        # Read and verify the TFLite model
        print("\n4. Verifying TFLite model structure...")
        with open(tflite_path, 'rb') as f:
            buf = bytearray(f.read())
        
        model_fb = schema_fb.Model.GetRootAsModel(buf, 0)
        
        # Verify model structure
        num_subgraphs = model_fb.SubgraphsLength()
        assert num_subgraphs > 0, "Model should have at least one subgraph"
        print(f"   ✓ Found {num_subgraphs} subgraph(s)")
        
        subgraph = model_fb.Subgraphs(0)
        num_ops = subgraph.OperatorsLength()
        assert num_ops > 0, "Subgraph should have at least one operator"
        print(f"   ✓ Found {num_ops} operator(s)")
        
        # Find the custom_call operation
        print("\n5. Looking for STABLEHLO_CUSTOM_CALL operation...")
        found_custom_call = False
        for i in range(num_ops):
            op = subgraph.Operators(i)
            opcode_idx = op.OpcodeIndex()
            opcode = model_fb.OperatorCodes(opcode_idx)
            builtin_code = opcode.BuiltinCode()
            
            print(f"   Op {i}: builtin_code = {builtin_code}", end="")
            
            # Check if this is a STABLEHLO_CUSTOM_CALL (op code 173)
            if builtin_code == 173:  # STABLEHLO_CUSTOM_CALL
                print(" (STABLEHLO_CUSTOM_CALL) ✓")
                found_custom_call = True
                
                # Verify BuiltinOptions2 contains StablehloCustomCallOptions
                options_type = op.BuiltinOptions2Type()
                assert options_type == schema_fb.BuiltinOptions2.StablehloCustomCallOptions, \
                    f"Expected StablehloCustomCallOptions, got {options_type}"
                
                # Extract and verify call_target_name
                options = schema_fb.StablehloCustomCallOptions()
                options.Init(op.BuiltinOptions2().Bytes, op.BuiltinOptions2().Pos)
                
                call_target_name = options.CallTargetName().decode('utf-8')
                api_version = options.ApiVersion()
                has_side_effect = options.HasSideEffect()
                backend_config = options.BackendConfig().decode('utf-8')
                
                print(f"\n6. Verifying custom call attributes...")
                print(f"   call_target_name: {call_target_name}")
                print(f"   api_version: {api_version}")
                print(f"   has_side_effect: {has_side_effect}")
                print(f"   backend_config: '{backend_config}'")
                
                assert call_target_name == 'ai_edge_torch.rms_norm', \
                    f'Expected "ai_edge_torch.rms_norm", got "{call_target_name}"'
                print("   ✓ call_target_name is correct")
                
                assert api_version == 1, f"Expected API version 1, got {api_version}"
                print("   ✓ api_version is correct")
                
                assert not has_side_effect, "Expected has_side_effect to be False"
                print("   ✓ has_side_effect is correct")
                
                break
            else:
                print()
        
        assert found_custom_call, \
            "Expected to find STABLEHLO_CUSTOM_CALL operation (op code 173)"
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nThe TFLite model contains:")
        print("  - STABLEHLO_CUSTOM_CALL operation (op code 173)")
        print("  - call_target_name: ai_edge_torch.rms_norm")
        print("  - api_version: 1")
        print("  - has_side_effect: False")
        print("\nNext step: Implement C++ TFLite kernel for 'ai_edge_torch.rms_norm'")
        print("=" * 80)
        
    finally:
        # Keep the TFLite file for inspection
        print(f"\nTFLite model saved at: {os.path.abspath(tflite_path)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test TFLite conversion of custom_rms_norm operation'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Path to save the TFLite model (default: /tmp/custom_rms_norm.tflite)'
    )
    
    args = parser.parse_args()
    test_tflite_conversion(output_path=args.output_path)

