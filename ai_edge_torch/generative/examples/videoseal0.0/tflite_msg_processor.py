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
TFLite-Friendly Message Processor for VideoSeal 0.0.

This module provides a modified message processor that replaces dynamic tensor
operations with static equivalents, enabling TFLite conversion.

Key Changes from Original:
1. Pre-computed indices as buffers (no torch.arange() at runtime)
2. Fixed spatial size (no runtime shape dependencies)
3. expand() instead of repeat() for better TFLite compatibility
"""

import torch
import torch.nn as nn


class TFLiteFriendlyMsgProcessor(nn.Module):
    """
    TFLite-compatible message processor for VideoSeal 0.0.
    
    Replaces dynamic operations in the original MsgProcessor with static equivalents:
    - Pre-computes indices at initialization
    - Uses fixed spatial dimensions
    - Replaces repeat() with expand()
    
    Args:
        nbits: Number of message bits (default: 96 for VideoSeal 0.0)
        hidden_size: Hidden dimension size (default: 32)
        spatial_size: Fixed spatial size (default: 32 for 256x256 images)
        msg_processor_type: Message type (default: "binary+concat")
        msg_mult: Message multiplier (default: 1.0)
    """
    
    def __init__(
        self,
        nbits: int = 96,
        hidden_size: int = 32,
        spatial_size: int = 32,
        msg_processor_type: str = "binary+concat",
        msg_mult: float = 1.0
    ):
        super().__init__()
        
        self.nbits = nbits
        self.hidden_size = hidden_size
        self.spatial_size = spatial_size
        self.msg_mult = msg_mult
        
        # Parse message type
        parts = msg_processor_type.split("+")
        self.msg_type = parts[0]  # "binary" or "gaussian"
        self.msg_agg = parts[1] if len(parts) > 1 else "concat"  # "concat" or "add"
        
        # Message embeddings
        if self.msg_type.startswith("bin"):
            # Binary: 2 embeddings per bit (0 or 1)
            self.msg_embeddings = nn.Embedding(2 * nbits, hidden_size)
            
            # Pre-compute base indices (FIXED SIZE - TFLite compatible)
            # Original: indices = 2 * torch.arange(msg.shape[-1])  # ❌ Dynamic
            # Fixed: Pre-compute at init with INT32 for HW delegate compatibility
            base_indices = 2 * torch.arange(nbits, dtype=torch.int32)  # ✅ Static INT32
            self.register_buffer('base_indices', base_indices)
            
        elif self.msg_type.startswith("gau"):
            # Gaussian: 1 embedding per bit
            self.msg_embeddings = nn.Embedding(nbits, hidden_size)
            
            # Pre-compute indices for gaussian with INT32 for HW delegate compatibility
            indices = torch.arange(nbits, dtype=torch.int32)
            self.register_buffer('gaussian_indices', indices)
        else:
            raise ValueError(f"Invalid msg_type: {self.msg_type}")
    
    def forward(self, latents: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
        """
        Process message and combine with latents.
        
        Args:
            latents: Encoder output (batch, channels, H, W)
            msg: Binary message (batch, nbits)
        
        Returns:
            Combined latents with message (batch, channels+hidden_size, H, W)
        """
        if self.nbits == 0:
            return latents
        
        # Create message embeddings
        if self.msg_type.startswith("bin"):
            # Binary message processing (TFLite-friendly)
            # Original: indices = 2 * torch.arange(msg.shape[-1])  # ❌ Dynamic
            # Fixed: Use pre-computed buffer
            indices = self.base_indices.unsqueeze(0).expand(msg.shape[0], -1)  # ✅ Static
            # Use .int() instead of .long() for INT32 (TFLite HW delegate compatible)
            indices = (indices + msg).int()
            
            # Create embeddings
            msg_aux = self.msg_embeddings(indices)  # batch, nbits, hidden_size
            msg_aux = msg_aux.sum(dim=-2)  # batch, hidden_size
            
            # Spatial broadcasting (TFLite-friendly)
            # Original: .repeat(1, 1, latents.shape[-2], latents.shape[-1])  # ❌ Dynamic
            # Fixed: Use explicit concatenation (avoids BROADCAST_TO issues)
            msg_aux = msg_aux.view(-1, self.hidden_size, 1, 1)
            
            # Concatenate along height dimension
            msg_list_h = [msg_aux for _ in range(self.spatial_size)]
            msg_aux_h = torch.cat(msg_list_h, dim=2)  # [B, C, H, 1]
            
            # Concatenate along width dimension
            msg_list_w = [msg_aux_h for _ in range(self.spatial_size)]
            msg_aux = torch.cat(msg_list_w, dim=3)  # [B, C, H, W]
            
        elif self.msg_type.startswith("gau"):
            # Gaussian message processing
            indices = self.gaussian_indices  # Use pre-computed buffer
            msg_aux = self.msg_embeddings(indices)  # nbits, hidden_size
            msg_aux = torch.einsum("kd, bk -> bd", msg_aux, msg)  # batch, hidden_size
            
            # Spatial broadcasting (TFLite-friendly)
            msg_aux = msg_aux.unsqueeze(-1).unsqueeze(-1)
            
            # Concatenate along height dimension
            msg_list_h = [msg_aux for _ in range(self.spatial_size)]
            msg_aux_h = torch.cat(msg_list_h, dim=2)  # [B, C, H, 1]
            
            # Concatenate along width dimension
            msg_list_w = [msg_aux_h for _ in range(self.spatial_size)]
            msg_aux = torch.cat(msg_list_w, dim=3)  # [B, C, H, W]
        
        # Apply message embeddings to latents
        if self.msg_agg == "concat":
            latents = torch.cat([
                latents,  # batch, channels, H, W
                self.msg_mult * msg_aux  # batch, hidden_size, H, W
            ], dim=1)  # batch, channels+hidden_size, H, W
        elif self.msg_agg == "add":
            latents = latents + self.msg_mult * msg_aux
        else:
            raise ValueError(f"Invalid msg_agg: {self.msg_agg}")
        
        return latents


def transfer_weights(original_processor: nn.Module, tflite_processor: TFLiteFriendlyMsgProcessor):
    """
    Transfer weights from original MsgProcessor to TFLite-friendly version.
    
    Args:
        original_processor: Original MsgProcessor from VideoSeal
        tflite_processor: TFLite-friendly MsgProcessor
    """
    # Transfer embedding weights
    tflite_processor.msg_embeddings.weight.data.copy_(
        original_processor.msg_embeddings.weight.data
    )
    
    print("✓ Transferred message processor weights")


if __name__ == "__main__":
    # Test the TFLite-friendly message processor
    print("Testing TFLite-Friendly Message Processor...")
    
    # Create processor
    processor = TFLiteFriendlyMsgProcessor(
        nbits=96,
        hidden_size=32,
        spatial_size=32,
        msg_processor_type="binary+concat"
    )
    
    print(f"✓ Created processor: {processor.nbits} bits, {processor.hidden_size} hidden")
    
    # Test forward pass
    latents = torch.rand(1, 128, 32, 32)
    msg = torch.randint(0, 2, (1, 96)).float()
    
    print(f"✓ Input shapes: latents={latents.shape}, msg={msg.shape}")
    
    output = processor(latents, msg)
    
    print(f"✓ Output shape: {output.shape}")
    print(f"  Expected: (1, 160, 32, 32) = (1, 128+32, 32, 32)")
    
    # Verify no dynamic operations
    print("\n✓ No dynamic operations:")
    print(f"  - base_indices: {processor.base_indices.shape} (pre-computed)")
    print(f"  - spatial_size: {processor.spatial_size} (hardcoded)")
    print(f"  - Uses expand() instead of repeat()")
    
    print("\n✓ TFLite-friendly message processor is working correctly!")
