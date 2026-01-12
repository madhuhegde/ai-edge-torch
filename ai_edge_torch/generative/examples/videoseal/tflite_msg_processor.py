#!/usr/bin/env python3
# Copyright 2025 The AI Edge Torch Authors.
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

"""
TFLite-Friendly Message Processor for VideoSeal

This module provides a message processor that uses fixed-size operations
compatible with TFLite conversion, replacing dynamic tensor operations
from the original implementation.

Key differences from original:
1. Pre-computed indices as buffers (no torch.arange at runtime)
2. Uses expand() instead of repeat() for spatial broadcasting
3. Hardcoded spatial dimensions
"""

import torch
import torch.nn as nn


class TFLiteFriendlyMsgProcessor(nn.Module):
    """
    TFLite-compatible message processor with fixed-size operations.
    
    This processor replaces dynamic tensor operations with static equivalents
    while maintaining mathematical equivalence to the original implementation.
    
    Args:
        nbits: Number of message bits (e.g., 256 for VideoSeal 1.0)
        hidden_size: Embedding dimension (e.g., 256)
        spatial_size: Spatial dimension at bottleneck (e.g., 32 for 256px images)
        msg_processor_type: Type of message processing (default: 'binary+concat')
        msg_mult: Multiplier for message embeddings (default: 1.0)
    
    Example:
        >>> msg_proc = TFLiteFriendlyMsgProcessor(
        ...     nbits=256,
        ...     hidden_size=256,
        ...     spatial_size=32
        ... )
        >>> latents = torch.rand(1, 128, 32, 32)
        >>> msg = torch.randint(0, 2, (1, 256)).float()
        >>> output = msg_proc(latents, msg)
        >>> print(output.shape)  # [1, 384, 32, 32]
    """
    
    def __init__(
        self,
        nbits: int = 256,
        hidden_size: int = 256,
        spatial_size: int = 32,
        msg_processor_type: str = "binary+concat",
        msg_mult: float = 1.0,
    ):
        super().__init__()
        self.nbits = nbits
        self.hidden_size = hidden_size
        self.spatial_size = spatial_size
        self.msg_mult = msg_mult
        
        # Parse message processor type
        self.msg_processor_type = msg_processor_type if nbits > 0 else "none+_"
        self.msg_type = self.msg_processor_type.split("+")[0]
        self.msg_agg = self.msg_processor_type.split("+")[1]
        
        # Create embedding table (same as original)
        if self.msg_type.startswith("no"):  # no message
            self.msg_embeddings = torch.tensor([])
        elif self.msg_type.startswith("bin"):  # binary
            self.msg_embeddings = nn.Embedding(2 * nbits, hidden_size)
        elif self.msg_type.startswith("gau"):  # gaussian
            self.msg_embeddings = nn.Embedding(nbits, hidden_size)
        else:
            raise ValueError(f"Invalid msg_processor_type: {self.msg_processor_type}")
        
        # Pre-compute base indices (NEW - eliminates torch.arange at runtime)
        if self.msg_type.startswith("bin"):
            base_indices = 2 * torch.arange(nbits)
            self.register_buffer('base_indices', base_indices)
        elif self.msg_type.startswith("gau"):
            base_indices = torch.arange(nbits)
            self.register_buffer('base_indices', base_indices)
    
    def forward(
        self, 
        latents: torch.Tensor, 
        msg: torch.Tensor,
        verbose: bool = False
    ) -> torch.Tensor:
        """
        Apply message embeddings to latents.
        
        Args:
            latents: Feature maps [B, C, H, W]
            msg: Binary message [B, nbits]
            verbose: Print intermediate shapes for debugging
        
        Returns:
            Latents with message embeddings [B, C+hidden_size, H, W]
        """
        if self.nbits == 0:
            return latents
        
        # Create message embeddings
        if self.msg_type.startswith("bin"):
            # Use pre-computed indices (no torch.arange)
            indices = self.base_indices.unsqueeze(0).expand(msg.shape[0], -1)
            indices = (indices + msg).long()
            
            # Embedding lookup
            msg_aux = self.msg_embeddings(indices)  # [B, nbits, hidden_size]
            msg_aux = msg_aux.sum(dim=1)            # [B, hidden_size]
            
        elif self.msg_type.startswith("gau"):
            # Gaussian message processing
            msg_aux = self.msg_embeddings(self.base_indices)  # [nbits, hidden_size]
            msg_aux = torch.einsum("kd, bk -> bd", msg_aux, msg)  # [B, hidden_size]
        else:
            raise ValueError(f"Invalid msg_type: {self.msg_type}")
        
        # Spatial broadcast using explicit concatenation (TFLite-friendly)
        # This avoids tile/expand/repeat which can cause issues during export
        msg_aux = msg_aux.view(-1, self.hidden_size, 1, 1)
        
        # Concatenate along height dimension
        msg_list_h = [msg_aux for _ in range(self.spatial_size)]
        msg_aux_h = torch.cat(msg_list_h, dim=2)  # [B, C, H, 1]
        
        # Concatenate along width dimension  
        msg_list_w = [msg_aux_h for _ in range(self.spatial_size)]
        msg_aux = torch.cat(msg_list_w, dim=3)  # [B, C, H, W]
        
        if verbose:
            print(f"TFLiteFriendlyMsgProcessor:")
            print(f"  latents: {latents.shape}")
            print(f"  msg: {msg.shape}")
            print(f"  msg_aux: {msg_aux.shape}")
        
        # Apply to latents
        if self.msg_agg == "concat":
            latents = torch.cat([latents, self.msg_mult * msg_aux], dim=1)
        elif self.msg_agg == "add":
            latents = latents + self.msg_mult * msg_aux
        else:
            raise ValueError(f"Invalid msg_agg: {self.msg_agg}")
        
        if verbose:
            print(f"  output: {latents.shape}")
        
        return latents
    
    def get_random_msg(self, bsz: int = 1) -> torch.Tensor:
        """
        Generate random message (same as original).
        
        Args:
            bsz: Batch size
        
        Returns:
            Random message tensor [bsz, nbits]
        """
        if self.msg_type.startswith("bin"):
            return torch.randint(0, 2, (bsz, self.nbits))
        elif self.msg_type.startswith("gau"):
            gauss_vecs = torch.randn(bsz, self.nbits)
            return gauss_vecs / torch.norm(gauss_vecs, dim=-1, keepdim=True)
        return torch.tensor([])


def transfer_weights(original_msg_processor, tflite_msg_processor):
    """
    Transfer weights from original to TFLite-friendly processor.
    
    Args:
        original_msg_processor: Original MsgProcessor instance
        tflite_msg_processor: TFLiteFriendlyMsgProcessor instance
    
    Returns:
        None (modifies tflite_msg_processor in-place)
    """
    # Transfer embedding weights
    if hasattr(original_msg_processor, 'msg_embeddings') and \
       hasattr(tflite_msg_processor, 'msg_embeddings'):
        tflite_msg_processor.msg_embeddings.weight.data = \
            original_msg_processor.msg_embeddings.weight.data.clone()
        print("✓ Embedding weights transferred")
    else:
        print("⚠ No embeddings to transfer")


if __name__ == "__main__":
    # Test the TFLite-friendly message processor
    print("="*70)
    print("Testing TFLiteFriendlyMsgProcessor")
    print("="*70)
    
    # Create processor
    print("\n1. Creating message processor...")
    msg_proc = TFLiteFriendlyMsgProcessor(
        nbits=256,
        hidden_size=256,
        spatial_size=32,
        msg_processor_type='binary+concat',
        msg_mult=1.0
    )
    print(f"✓ Processor created")
    print(f"  nbits: {msg_proc.nbits}")
    print(f"  hidden_size: {msg_proc.hidden_size}")
    print(f"  spatial_size: {msg_proc.spatial_size}")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    latents = torch.rand(1, 128, 32, 32)
    msg = torch.randint(0, 2, (1, 256)).float()
    
    with torch.no_grad():
        output = msg_proc(latents, msg, verbose=True)
    
    print(f"\n✓ Forward pass successful")
    print(f"  Expected output shape: [1, 384, 32, 32]")
    print(f"  Actual output shape: {output.shape}")
    
    assert output.shape == (1, 384, 32, 32), f"Wrong output shape: {output.shape}"
    print("\n✓ All tests passed!")

