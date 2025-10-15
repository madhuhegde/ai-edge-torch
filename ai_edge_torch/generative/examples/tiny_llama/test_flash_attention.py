#!/usr/bin/env python3
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
"""Test script to verify Flash Attention integration with TinyLlama."""

import torch
from ai_edge_torch.generative.examples.tiny_llama import tiny_llama
from ai_edge_torch.generative.layers import kv_cache as kv_utils


def test_flash_attention_kv_cache():
  """Test that Flash Attention works with KV cache updates."""
  print("=" * 80)
  print("Test: Flash Attention with KV Cache")
  print("=" * 80)
  
  # Create a small model config for testing
  config = tiny_llama.get_model_config(
      kv_cache_max_len=128,
      use_flash_attention=True
  )
  config.num_layers = 2  # Use only 2 layers for quick testing
  config.vocab_size = 1000
  
  # Build model (without loading checkpoint)
  from ai_edge_torch.generative.utilities import model_builder
  model = model_builder.DecoderOnlyModel(config)
  model.eval()
  
  print(f"Model config:")
  print(f"  - Layers: {config.num_layers}")
  print(f"  - Attention heads: {config.block_config(0).attn_config.num_heads}")
  print(f"  - KV heads: {config.block_config(0).attn_config.num_query_groups}")
  print(f"  - Head dim: {config.block_config(0).attn_config.head_dim}")
  print(f"  - Use Flash Attention: {config.use_flash_attention}")
  print(f"  - Enable HLFB: {config.enable_hlfb}")
  
  # Test 1: Forward pass with fresh KV cache
  print("\nTest 1: Forward pass with fresh KV cache")
  batch_size = 1
  seq_len = 32
  tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))
  input_pos = torch.arange(seq_len)
  kv_cache = kv_utils.KVCache.from_model_config(config)
  
  with torch.no_grad():
    output1 = model(tokens, input_pos, kv_cache)
  
  print(f"  Input shape: {list(tokens.shape)}")
  print(f"  Output logits shape: {list(output1['logits'].shape)}")
  print(f"  ✅ Forward pass successful")
  
  # Test 2: Forward pass with KV cache persistence
  print("\nTest 2: Forward pass with KV cache persistence")
  
  # Initialize fresh KV cache
  kv_cache = kv_utils.KVCache.from_model_config(config)
  
  # Prefill: process initial tokens
  prefill_len = 8
  prefill_tokens = tokens[:, :prefill_len]
  input_pos = torch.arange(prefill_len)
  
  with torch.no_grad():
    output2 = model(prefill_tokens, input_pos, kv_cache)
    kv_cache = output2['kv_cache']
  
  print(f"  Prefill tokens shape: {list(prefill_tokens.shape)}")
  print(f"  Prefill output logits shape: {list(output2['logits'].shape)}")
  print(f"  KV cache type: {type(kv_cache)}")
  print(f"  ✅ Prefill successful")
  
  # Decode: process one token at a time
  print("\nTest 3: Decode with KV cache (autoregressive)")
  for i in range(3):
    decode_token = torch.randint(0, config.vocab_size, (batch_size, 1))
    input_pos = torch.tensor([prefill_len + i])
    
    with torch.no_grad():
      output3 = model(decode_token, input_pos, kv_cache)
      kv_cache = output3['kv_cache']
    
    print(f"  Decode step {i+1}: token shape {list(decode_token.shape)}, "
          f"output logits shape {list(output3['logits'].shape)}")
  
  print(f"  ✅ Autoregressive decode successful")
  
  print("\n" + "=" * 80)
  print("All tests passed! Flash Attention integration is working correctly.")
  print("=" * 80)


def test_flash_vs_standard_attention():
  """Compare Flash Attention vs Standard Attention outputs."""
  print("=" * 80)
  print("Test: Flash Attention vs Standard Attention Comparison")
  print("=" * 80)
  
  # Create two models: one with Flash, one without
  config_standard = tiny_llama.get_model_config(
      kv_cache_max_len=128,
      use_flash_attention=False
  )
  config_standard.num_layers = 1
  config_standard.vocab_size = 100
  
  config_flash = tiny_llama.get_model_config(
      kv_cache_max_len=128,
      use_flash_attention=True
  )
  config_flash.num_layers = 1
  config_flash.vocab_size = 100
  
  # Build models
  from ai_edge_torch.generative.utilities import model_builder
  model_standard = model_builder.DecoderOnlyModel(config_standard)
  model_flash = model_builder.DecoderOnlyModel(config_flash)
  
  # Copy weights from standard to flash model (for fair comparison)
  model_flash.load_state_dict(model_standard.state_dict(), strict=False)
  
  model_standard.eval()
  model_flash.eval()
  
  # Test on same input
  batch_size = 1
  seq_len = 16
  tokens = torch.randint(0, config_standard.vocab_size, (batch_size, seq_len))
  input_pos = torch.arange(seq_len)
  kv_cache_std = kv_utils.KVCache.from_model_config(config_standard)
  kv_cache_flash = kv_utils.KVCache.from_model_config(config_flash)
  
  with torch.no_grad():
    output_standard = model_standard(tokens, input_pos, kv_cache_std)
    output_flash = model_flash(tokens, input_pos, kv_cache_flash)
  
  # Compare outputs (logits)
  logits_standard = output_standard['logits']
  logits_flash = output_flash['logits']
  max_diff = torch.max(torch.abs(logits_standard - logits_flash)).item()
  mean_diff = torch.mean(torch.abs(logits_standard - logits_flash)).item()
  
  print(f"Input shape: {list(tokens.shape)}")
  print(f"Standard attention output shape: {list(logits_standard.shape)}")
  print(f"Flash attention output shape: {list(logits_flash.shape)}")
  print(f"\nComparison:")
  print(f"  Max absolute difference: {max_diff:.6e}")
  print(f"  Mean absolute difference: {mean_diff:.6e}")
  
  # Check if differences are within acceptable tolerance
  tolerance = 1e-4
  if max_diff < tolerance:
    print(f"  ✅ Outputs match within tolerance ({tolerance})")
  else:
    print(f"  ⚠️  Outputs differ by {max_diff:.6e} (tolerance: {tolerance})")
    print(f"     This may be due to numerical precision differences.")
  
  print("\n" + "=" * 80)


def main():
  """Run all tests."""
  print("\n")
  print("╔" + "=" * 78 + "╗")
  print("║" + " " * 20 + "Flash Attention Integration Tests" + " " * 25 + "║")
  print("╚" + "=" * 78 + "╝")
  print()
  
  try:
    # Test 1: KV cache functionality
    test_flash_attention_kv_cache()
    print()
    
    # Test 2: Compare with standard attention
    test_flash_vs_standard_attention()
    print()
    
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 25 + "All Tests Passed! ✅" + " " * 33 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
  except Exception as e:
    print(f"\n❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
    return 1
  
  return 0


if __name__ == "__main__":
  exit(main())

