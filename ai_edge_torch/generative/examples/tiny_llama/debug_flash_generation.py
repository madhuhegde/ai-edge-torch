#!/usr/bin/env python3
"""Debug script to compare standard vs Flash attention during generation."""

import torch
from ai_edge_torch.generative.examples.tiny_llama import tiny_llama
from ai_edge_torch.generative.layers import kv_cache as kv_utils
from ai_edge_torch.generative.utilities import model_builder

def compare_generation_step_by_step():
    """Compare standard vs Flash attention token by token."""
    
    # Build both models
    config_std = tiny_llama.get_model_config(kv_cache_max_len=64, use_flash_attention=False)
    config_std.num_layers = 2  # Use 2 layers for faster testing
    config_std.vocab_size = 1000
    
    config_flash = tiny_llama.get_model_config(kv_cache_max_len=64, use_flash_attention=True)
    config_flash.num_layers = 2
    config_flash.vocab_size = 1000
    
    model_std = model_builder.DecoderOnlyModel(config_std)
    model_flash = model_builder.DecoderOnlyModel(config_flash)
    
    # Copy weights from standard to Flash model
    model_flash.load_state_dict(model_std.state_dict(), strict=False)
    
    model_std.eval()
    model_flash.eval()
    
    # Initial prompt
    prompt = torch.tensor([[10, 20, 30, 40, 50]])  # 5 tokens
    
    print("=" * 80)
    print("PREFILL PHASE")
    print("=" * 80)
    
    # Prefill - both models
    input_pos = torch.arange(prompt.shape[1])
    kv_cache_std = kv_utils.KVCache.from_model_config(config_std)
    kv_cache_flash = kv_utils.KVCache.from_model_config(config_flash)
    
    with torch.no_grad():
        output_std = model_std(prompt, input_pos, kv_cache_std)
        output_flash = model_flash(prompt, input_pos, kv_cache_flash)
    
    logits_std = output_std['logits']
    logits_flash = output_flash['logits']
    kv_cache_std = output_std['kv_cache']
    kv_cache_flash = output_flash['kv_cache']
    
    print(f"\nPrefill logits comparison:")
    print(f"  Standard shape: {logits_std.shape}")
    print(f"  Flash shape: {logits_flash.shape}")
    print(f"  Max diff: {torch.max(torch.abs(logits_std - logits_flash)).item():.6e}")
    print(f"  Mean diff: {torch.mean(torch.abs(logits_std - logits_flash)).item():.6e}")
    
    # Check last position
    print(f"\nLast position (idx=4) logits:")
    print(f"  Standard top-5: {torch.topk(logits_std[0, 4, :], 5).indices.tolist()}")
    print(f"  Flash top-5: {torch.topk(logits_flash[0, 4, :], 5).indices.tolist()}")
    
    # Get next token
    next_token_std = torch.argmax(logits_std[0, -1, :]).item()
    next_token_flash = torch.argmax(logits_flash[0, -1, :]).item()
    
    print(f"\nNext token prediction:")
    print(f"  Standard: {next_token_std}")
    print(f"  Flash: {next_token_flash}")
    print(f"  Match: {next_token_std == next_token_flash}")
    
    # DECODE PHASE - Generate 5 tokens
    print("\n" + "=" * 80)
    print("DECODE PHASE (Token by Token)")
    print("=" * 80)
    
    current_token_std = next_token_std
    current_token_flash = next_token_flash
    
    for step in range(5):
        print(f"\n--- Decode Step {step + 1} ---")
        
        input_token_std = torch.tensor([[current_token_std]])
        input_token_flash = torch.tensor([[current_token_flash]])
        input_pos = torch.tensor([5 + step])
        
        print(f"Input tokens: std={current_token_std}, flash={current_token_flash}")
        
        with torch.no_grad():
            output_std = model_std(input_token_std, input_pos, kv_cache_std)
            output_flash = model_flash(input_token_flash, input_pos, kv_cache_flash)
        
        logits_std = output_std['logits']
        logits_flash = output_flash['logits']
        kv_cache_std = output_std['kv_cache']
        kv_cache_flash = output_flash['kv_cache']
        
        print(f"Logits shapes: std={logits_std.shape}, flash={logits_flash.shape}")
        print(f"Logits max diff: {torch.max(torch.abs(logits_std - logits_flash)).item():.6e}")
        
        # Get top-5 predictions
        top5_std = torch.topk(logits_std[0, 0, :], 5)
        top5_flash = torch.topk(logits_flash[0, 0, :], 5)
        
        print(f"Standard top-5: indices={top5_std.indices.tolist()}, values={top5_std.values.tolist()[:5]}")
        print(f"Flash top-5: indices={top5_flash.indices.tolist()}, values={top5_flash.values.tolist()[:5]}")
        
        # Check if all logits are the same (the comma problem)
        logits_std_unique = torch.unique(logits_std[0, 0, :]).shape[0]
        logits_flash_unique = torch.unique(logits_flash[0, 0, :]).shape[0]
        
        print(f"Unique logit values: std={logits_std_unique}, flash={logits_flash_unique}")
        
        if logits_flash_unique < 10:
            print(f"⚠️  WARNING: Flash attention has very few unique logits!")
            print(f"   Flash logits sample: {logits_flash[0, 0, :10].tolist()}")
            print(f"   All same? {torch.all(logits_flash[0, 0, :] == logits_flash[0, 0, 0])}")
            
            # Check attention output before LM head
            print("\n   Debugging: Checking intermediate values...")
            # We need to trace through the model to see where it breaks
            break
        
        # Get next tokens
        current_token_std = torch.argmax(logits_std[0, 0, :]).item()
        current_token_flash = torch.argmax(logits_flash[0, 0, :]).item()
        
        print(f"Next tokens: std={current_token_std}, flash={current_token_flash}")
        print(f"Match: {current_token_std == current_token_flash}")


def trace_forward_pass():
    """Trace through forward pass to find where Flash attention diverges."""
    print("\n" + "=" * 80)
    print("DETAILED FORWARD PASS TRACING")
    print("=" * 80)
    
    config_flash = tiny_llama.get_model_config(kv_cache_max_len=64, use_flash_attention=True)
    config_flash.num_layers = 1  # Single layer for easier debugging
    config_flash.vocab_size = 100
    
    model = model_builder.DecoderOnlyModel(config_flash)
    model.eval()
    
    # Prefill
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    input_pos = torch.arange(5)
    kv_cache = kv_utils.KVCache.from_model_config(config_flash)
    
    with torch.no_grad():
        output = model(tokens, input_pos, kv_cache)
        kv_cache = output['kv_cache']
    
    print("\nPrefill completed successfully")
    
    # Decode - trace carefully
    next_token = torch.tensor([[42]])
    input_pos = torch.tensor([5])
    
    print("\nDecode step - tracing...")
    
    # Get inputs to model
    cos, sin = model.rope_cache
    rope = (cos.index_select(0, input_pos), sin.index_select(0, input_pos))
    mask = model.mask_cache.index_select(2, input_pos)
    mask = mask[:, :, :, : config_flash.kv_cache_max]
    
    print(f"Rope shapes: cos={rope[0].shape}, sin={rope[1].shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask unique values: {torch.unique(mask).tolist()}")
    print(f"First -inf position: {torch.where(torch.isinf(mask[0,0,0,:]))[0][0].item() if torch.any(torch.isinf(mask[0,0,0,:])) else 'None'}")
    
    # Embedding
    x = model.tok_embedding(next_token)
    print(f"\nEmbedding output shape: {x.shape}")
    print(f"Embedding output sample: {x[0, 0, :10].tolist()}")
    
    # First transformer block
    block = model.transformer_blocks[0]
    
    # Pre-attention norm
    x_norm = block.pre_atten_norm(x)
    print(f"\nPre-attention norm output: {x_norm[0, 0, :10].tolist()}")
    
    # Attention
    kv_entry = kv_cache.caches[0]
    print(f"\nKV cache shapes before attention:")
    print(f"  k_cache: {kv_entry.k_cache.shape}")
    print(f"  v_cache: {kv_entry.v_cache.shape}")
    
    # Check attention function
    print(f"\nAttention function type: {type(block.atten_func)}")
    print(f"SDPA function: {block.atten_func.sdpa_func}")
    
    # Call attention
    with torch.no_grad():
        atten_out, kv_entry_updated = block.atten_func(x_norm, rope, mask, input_pos, kv_entry)
    
    print(f"\nAttention output shape: {atten_out.shape}")
    print(f"Attention output sample: {atten_out[0, 0, :10].tolist()}")
    print(f"Attention output all same? {torch.all(atten_out[0, 0, :] == atten_out[0, 0, 0])}")
    
    if torch.all(atten_out[0, 0, :] == atten_out[0, 0, 0]):
        print("\n⚠️  PROBLEM FOUND: Attention output is constant!")
        print("   This means the attention mechanism is producing the same value for all dimensions.")
        return
    
    # Continue through the rest of the block
    x_with_residual = x + atten_out
    x_norm2 = block.post_atten_norm(x_with_residual)
    ff_out = block.ff(x_norm2)
    x_final = x_with_residual + ff_out
    
    print(f"\nFinal block output shape: {x_final.shape}")
    print(f"Final block output sample: {x_final[0, 0, :10].tolist()}")
    
    # Final norm and LM head
    x_normed = model.final_norm(x_final)
    logits = model.lm_head(x_normed)
    
    print(f"\nFinal logits shape: {logits.shape}")
    print(f"Final logits sample: {logits[0, 0, :10].tolist()}")
    print(f"Logits unique values: {torch.unique(logits[0, 0, :]).shape[0]}")


if __name__ == "__main__":
    # Run comparison
    compare_generation_step_by_step()
    
    # If problem found, trace in detail
    print("\n" * 3)
    trace_forward_pass()

