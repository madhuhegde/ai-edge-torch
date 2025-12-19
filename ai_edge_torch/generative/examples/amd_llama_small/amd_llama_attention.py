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

"""Attention-only model for AMD-Llama (no embedding layer).

This module provides a minimal PyTorch model containing only scaled dot product
attention. It takes Q, K, V, and mask as inputs and outputs the attention result.
This model is designed to be exported to TFLite without StableHLO composite operations.
"""

import torch
from torch import nn
import torch.nn.functional as F


class AttentionOnlyModel(nn.Module):
  """Minimal attention-only model for decode signature.

  This model contains only scaled dot product attention with:
  - Configurable num_heads and head_dim
  - No embedding layer
  - No weights (stateless)

  Input shapes for decode:
    - q: [1, num_heads, 1, head_dim] (batch=1, num_heads, seq_len=1, head_dim)
    - k: [1, num_heads, kv_len, head_dim] (batch=1, num_heads, kv_seq_len, head_dim)
    - v: [1, num_heads, kv_len, head_dim] (batch=1, num_heads, kv_seq_len, head_dim)
    - mask: [1, 1, 1, kv_len] (batch=1, 1, 1, kv_seq_len)

  Output shape:
    - [1, num_heads, 1, head_dim] (batch=1, num_heads, seq_len=1, head_dim)
  """

  def __init__(self, head_dim: int = 64, num_heads: int = 12):
    """Initialize the attention-only model.

    Args:
      head_dim: Head dimension (default: 64)
      num_heads: Number of attention heads (default: 12)
    """
    super().__init__()
    self.head_dim = head_dim
    self.num_heads = num_heads
    self.scale = head_dim ** -0.5

  def forward(
      self,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      mask: torch.Tensor,
  ) -> torch.Tensor:
    """Forward pass of attention-only model.

    Args:
      q: Query tensor with shape [B, N, T, H] where B=batch, N=num_heads, T=seq_len, H=head_dim
      k: Key tensor with shape [B, N, KV_LEN, H]
      v: Value tensor with shape [B, N, KV_LEN, H]
      mask: Attention mask tensor with shape [B, 1, 1, KV_LEN]

    Returns:
      Attention output tensor with shape [B, N, T, H]
    """
    # Scale Q before matmul
    q = q * self.scale
    
    # Compute attention scores: Q @ K^T
    scores = torch.matmul(q, k.transpose(-2, -1))  # [B, N, T, KV_LEN]
    
    # Apply mask
    scores = scores + mask
    
    # Softmax
    attn_weights = F.softmax(scores, dim=-1)  # [B, N, T, KV_LEN]
    
    # Apply attention to values
    output = torch.matmul(attn_weights, v)  # [B, N, T, H]
    
    return output

