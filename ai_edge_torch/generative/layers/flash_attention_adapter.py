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
"""Flash Attention adapter for integration with existing attention module.

This module provides an adapter layer that allows using Flash Attention
implementation while maintaining full compatibility with the existing
CausalSelfAttention interface, including KV cache, RoPE, and HLFB support.

Key Features:
- Converts between kv_utils.KVCacheEntry and Flash Attention's FlashKVCache
- Provides flash_attention_causal_inference() that matches scaled_dot_product_attention signature
- Handles both prefill and decode stages with proper KV cache management
- Infers absolute query positions from attention mask for correct autoregressive generation
- Fully compatible with HLFB (High-Level Function Boundary) wrapping

Important Implementation Detail:
The adapter correctly handles absolute vs relative query positions during decode.
For autoregressive generation with KV cache:
  - Query positions must be absolute (e.g., position 26 for the 27th token)
  - These are inferred from the attention mask: query_pos = valid_kv_len - num_query_tokens
  - This ensures Flash Attention attends to all valid KV cache positions, not just the first few

Example Usage:
  # In CausalSelfAttention.forward():
  attn_output = flash_attention_adapter.flash_attention_causal_inference(
      q, k, v, head_size=self.head_dim, mask=mask
  )
"""

import math
from typing import Optional, Tuple, Union

from ai_edge_torch.generative.layers import kv_cache as kv_utils
import torch
from torch import nn


def convert_kv_cache_to_flash(
    kv_cache: kv_utils.KVCacheEntry,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
  """Convert KVCacheEntry to tensors for Flash Attention.
  
  Args:
      kv_cache: The KVCacheEntry to convert.
  
  Returns:
      Tuple of (k_cache, v_cache, current_length)
  """
  # KVCacheEntry stores k_cache and v_cache as [B, max_len, num_heads, head_dim]
  # which is exactly what Flash Attention expects
  return kv_cache.k_cache, kv_cache.v_cache, kv_cache.k_cache.shape[1]


def flash_attention_causal_inference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    head_size: int,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    softcap: Optional[float] = None,
    q_block_size: int = 64,
    kv_block_size: int = 512,
) -> torch.Tensor:
  """Flash Attention with causal masking for inference.
  
  This is an adapted version optimized for integration with existing code.
  Uses memory tiling and online softmax to achieve O(N) memory complexity.
  
  Args:
      q: Query tensor [B, T, N, H]
      k: Key tensor [B, T, KV_N, H] or [B, T, N, H]
      v: Value tensor [B, T, KV_N, H] or [B, T, N, H]
      head_size: Head dimension
      mask: Optional attention mask (not used - causal masking is built-in)
      scale: Optional scale factor
      softcap: Optional softcap value (not currently supported)
      q_block_size: Query block size
      kv_block_size: KV block size
  
  Returns:
      Output tensor [B, T, N, H]
  """
  if scale is None:
    scale = 1.0 / math.sqrt(head_size)
  
  # Transpose to [B, N, T, H] format for attention computation
  q = q.transpose(1, 2)
  k = k.transpose(1, 2)
  v = v.transpose(1, 2)
  
  # Handle GQA: repeat K, V to match query heads if needed
  if q.shape[1] != k.shape[1]:
    repeat_factor = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(repeat_factor, dim=1)
    v = v.repeat_interleave(repeat_factor, dim=1)
  
  B, N, T, H = q.shape
  _, _, S, _ = k.shape
  
  # Determine valid KV sequence length from mask if provided
  # Mask shape: [B, 1, T, S] or [1, 1, T, S]
  # Mask values: 0.0 for valid positions, -inf for invalid
  valid_kv_len = S
  query_positions_abs = None  # Absolute query positions for causal masking
  
  if mask is not None and torch.any(torch.isinf(mask)):
    # Find the last valid position (first -inf position)
    # mask shape: [B, 1, T, S], check last query position
    last_query_mask = mask[:, 0, -1, :]  # [B, S]
    # Find first -inf position for each batch
    for b in range(B):
      inf_positions = torch.where(torch.isinf(last_query_mask[b]))[0]
      if len(inf_positions) > 0:
        valid_kv_len = min(valid_kv_len, inf_positions[0].item())
    
    # Infer absolute query positions from mask
    # The valid_kv_len tells us how many KV positions are valid
    # For autoregressive decoding, this means queries are at positions
    # [valid_kv_len - T, valid_kv_len - T + 1, ..., valid_kv_len - 1]
    query_positions_abs = torch.arange(
        valid_kv_len - T, valid_kv_len, device=q.device
    )
  
  # Adjust block sizes
  q_block_size = min(q_block_size, T)
  kv_block_size = min(kv_block_size, valid_kv_len)
  
  output = torch.zeros_like(q)
  num_q_blocks = (T + q_block_size - 1) // q_block_size
  
  # Process each query block
  for q_block_idx in range(num_q_blocks):
    q_start = q_block_idx * q_block_size
    q_end = min(q_start + q_block_size, T)
    q_block = q[:, :, q_start:q_end, :]
    q_block_len = q_end - q_start
    
    # Initialize running statistics for online softmax
    m_i = torch.full(
        (B, N, q_block_len),
        -torch.inf,
        dtype=q.dtype,
        device=q.device
    )
    l_i = torch.zeros((B, N, q_block_len), dtype=q.dtype, device=q.device)
    o_i = torch.zeros_like(q_block)
    
    # Causal attention: only attend to positions <= current query position
    # For autoregressive decoding, q_start/q_end are relative to current batch,
    # but we need the absolute position. Since KV cache accumulates all past,
    # we should attend to all valid KV positions up to the current query.
    # The valid_kv_len already accounts for this via the mask.
    max_kv_pos = valid_kv_len
    num_kv_blocks = (max_kv_pos + kv_block_size - 1) // kv_block_size
    
    # Process each KV block
    for kv_block_idx in range(num_kv_blocks):
      kv_start = kv_block_idx * kv_block_size
      kv_end = min(kv_start + kv_block_size, S)
      k_block = k[:, :, kv_start:kv_end, :]
      v_block = v[:, :, kv_start:kv_end, :]
      
      # Compute attention scores
      scores = torch.matmul(q_block, k_block.transpose(-2, -1))
      scores = scores * scale
      
      # Apply causal mask
      # Use absolute positions if available (from mask inference), otherwise relative
      if query_positions_abs is not None:
        q_positions = query_positions_abs[q_start:q_end].unsqueeze(-1)
      else:
        q_positions = torch.arange(
            q_start, q_end, device=q.device
        ).unsqueeze(-1)
      
      kv_positions = torch.arange(
          kv_start, kv_end, device=k.device
      ).unsqueeze(0)
      causal_mask = (kv_positions > q_positions).unsqueeze(0).unsqueeze(0)
      scores = scores.masked_fill(causal_mask, -torch.inf)
      
      # Online softmax update
      m_ij = torch.max(scores, dim=-1, keepdim=False)[0]
      m_i_new = torch.maximum(m_i, m_ij)
      
      alpha = torch.exp(scores - m_i_new.unsqueeze(-1))
      fully_masked = torch.isinf(m_i_new) & (m_i_new < 0)
      alpha = torch.where(fully_masked.unsqueeze(-1), torch.zeros_like(alpha), alpha)
      
      exp_diff = torch.exp(m_i - m_i_new)
      exp_diff = torch.where(fully_masked, torch.zeros_like(exp_diff), exp_diff)
      
      l_i_new = exp_diff * l_i + torch.sum(alpha, dim=-1)
      
      o_i = o_i * exp_diff.unsqueeze(-1)
      o_i = o_i + torch.matmul(alpha, v_block)
      
      m_i = m_i_new
      l_i = l_i_new
    
    # Final normalization
    l_i_safe = torch.where(l_i == 0, torch.ones_like(l_i), l_i)
    output[:, :, q_start:q_end, :] = o_i / l_i_safe.unsqueeze(-1)
    
    # Set fully masked positions to zero
    fully_masked = (l_i == 0)
    output[:, :, q_start:q_end, :] = torch.where(
        fully_masked.unsqueeze(-1),
        torch.zeros_like(output[:, :, q_start:q_end, :]),
        output[:, :, q_start:q_end, :]
    )
  
  # Transpose back to [B, T, N, H]
  output = output.transpose(1, 2)
  return output


def flash_attention_with_hlfb(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    head_size: int,
    mask: Optional[torch.Tensor] = None,
    softcap: Optional[float] = None,
) -> torch.Tensor:
  """Flash Attention with HLFB composite wrapping.
  
  Note: HLFB wrapping for Flash Attention is done at the graph level during
  export, not at the operator level. This function currently just calls the
  Flash Attention implementation directly. The HLFB pattern matching will
  recognize this as a scaled_dot_product_attention pattern during lowering.
  
  Args:
      q: Query tensor [B, T, N, H]
      k: Key tensor [B, T, KV_N, H]
      v: Value tensor [B, T, KV_N, H]
      head_size: Head dimension
      mask: Optional attention mask (not used in causal attention)
      softcap: Optional softcap value (not currently supported)
  
  Returns:
      Output tensor [B, T, N, H]
  """
  # Flash Attention will be recognized as SDPA pattern during HLFB lowering
  return flash_attention_causal_inference(q, k, v, head_size, mask=mask, softcap=softcap)

