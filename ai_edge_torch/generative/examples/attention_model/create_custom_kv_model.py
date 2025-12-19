#!/usr/bin/env python3
"""
Create a custom attention model with specified KV length
Usage: python create_custom_kv_model.py --kv_len 160
"""

import os
import argparse

import torch
import ai_edge_torch

# Try to use TFLite batch_matmul directly if available
try:
    from ai_edge_torch.odml_torch.experimental.torch_tfl import _ops as tfl_ops
    USE_TFL_OPS = True
except ImportError:
    USE_TFL_OPS = False

class AttentionOnly(torch.nn.Module):
    def __init__(self, num_heads=12, head_dim=64, kv_len=160):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.kv_len = kv_len
        self.scale = head_dim ** -0.5

    def forward(self, q, k, v, mask):
        """
        Args:
            q: [batch, num_heads, 1, head_dim]
            k: [batch, num_heads, kv_len, head_dim]
            v: [batch, num_heads, kv_len, head_dim]
            mask: [batch, 1, 1, kv_len]
        Returns:
            output: [batch, num_heads, 1, head_dim]
        """
        # Inputs are already in the correct shape for attention
        # No transpose needed!
        
        # Scale Q before matmul
        q = q * self.scale
        
        if USE_TFL_OPS:
            # Use TFLite batch_matmul directly to avoid StableHLO
            # For Q@K^T: [B,N,1,H] @ [B,N,KV,H]^T -> [B,N,1,KV]
            scores = tfl_ops.tfl_batch_matmul(q, k, adj_x=False, adj_y=True)
        else:
            # Fallback to standard torch.matmul
            scores = torch.matmul(q, k.transpose(-2, -1))
        
        # Apply mask (broadcast from [batch, 1, 1, kv_len] to [batch, num_heads, 1, kv_len])
        scores = scores + mask
        
        # Softmax
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)  # [batch, num_heads, 1, kv_len]
        
        if USE_TFL_OPS:
            # Use TFLite batch_matmul directly
            # For attn@v: [B,N,1,KV] @ [B,N,KV,H] -> [B,N,1,H]
            output = tfl_ops.tfl_batch_matmul(attn_weights, v, adj_x=False, adj_y=False)
        else:
            # Fallback to standard torch.matmul
            output = torch.matmul(attn_weights, v)
        
        # Output is already in the correct shape [batch, num_heads, 1, head_dim]
        # No transpose needed!
        
        return output

def create_model(kv_len=160, num_heads=12, head_dim=64, output_dir=None):
    """Create and convert attention model to TFLite
    
    Args:
        kv_len: KV cache length
        num_heads: Number of attention heads
        head_dim: Head dimension
        output_dir: Directory to save the TFLite model (default: same as script directory)
    
    Returns:
        Path to the saved TFLite model
    """
    # Determine output directory
    if output_dir is None:
        # Save in the same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = script_dir
    else:
        output_dir = os.path.abspath(output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating attention model:")
    print(f"  num_heads: {num_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  kv_len: {kv_len}")
    
    # Calculate memory
    q_mem = 1 * 1 * num_heads * head_dim * 4
    k_mem = 1 * kv_len * num_heads * head_dim * 4
    v_mem = 1 * kv_len * num_heads * head_dim * 4
    mask_mem = 1 * 1 * 1 * kv_len * 4
    total_mem = q_mem + k_mem + v_mem + mask_mem
    
    print(f"\nMemory usage:")
    print(f"  Q: {q_mem:,} bytes")
    print(f"  K: {k_mem:,} bytes")
    print(f"  V: {v_mem:,} bytes")
    print(f"  Mask: {mask_mem:,} bytes")
    print(f"  Total: {total_mem:,} bytes ({total_mem/1024/1024:.2f} MB)")
    
    if total_mem > 1048576:
        print(f"\n⚠️  WARNING: Total memory ({total_mem:,} bytes) exceeds 1MB!")
    else:
        print(f"\n✓ Total memory is under 1MB")
    
    # Create model
    model = AttentionOnly(num_heads=num_heads, head_dim=head_dim, kv_len=kv_len)
    model.eval()
    
    # Create sample inputs with correct dimensions
    batch = 1
    q = torch.randn(batch, num_heads, 1, head_dim)  # [1, 12, 1, 64]
    k = torch.randn(batch, num_heads, kv_len, head_dim)  # [1, 12, kv_len, 64]
    v = torch.randn(batch, num_heads, kv_len, head_dim)  # [1, 12, kv_len, 64]
    mask = torch.zeros(batch, 1, 1, kv_len)  # [1, 1, 1, kv_len]
    
    sample_args = (q, k, v, mask)
    
    print("\nConverting to TFLite...")
    
    # Convert to TFLite
    edge_model = ai_edge_torch.convert(
        model,
        sample_args
    )
    
    # Save model
    output_file = os.path.join(output_dir, f"amd_llama_attention_kv{kv_len}.tflite")
    edge_model.export(output_file)
    
    print(f"\n✓ Model saved to: {output_file}")
    
    # Verify the model
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=output_file)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    print("\nModel input shapes:")
    for i, detail in enumerate(input_details):
        print(f"  Input {i}: {detail['shape'].tolist()}")
    
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create custom KV length attention model')
    parser.add_argument('--kv_len', type=int, default=160,
                        help='KV cache length (default: 160)')
    parser.add_argument('--num_heads', type=int, default=12,
                        help='Number of attention heads (default: 12)')
    parser.add_argument('--head_dim', type=int, default=64,
                        help='Head dimension (default: 64)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save the TFLite model (default: same as script directory)')
    
    args = parser.parse_args()
    
    create_model(kv_len=args.kv_len, num_heads=args.num_heads, head_dim=args.head_dim, output_dir=args.output_dir)

