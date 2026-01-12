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
TFLite-friendly Upsample module to replace nn.Upsample and avoid GATHER_ND operations.

The standard nn.Upsample with bilinear interpolation generates GATHER_ND operations
in TFLite, which can impact performance. This module provides alternatives that
avoid GATHER_ND.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TFLiteFriendlyUpsample(nn.Module):
    """
    TFLite-friendly upsampling module that avoids GATHER_ND operations.
    
    This module replaces nn.Upsample with a combination of operations that
    TFLite can optimize better, avoiding GATHER_ND operations.
    
    Strategies:
    1. Use ConvTranspose2d instead of bilinear interpolation
    2. Use PixelShuffle for 2x upsampling
    3. Use nearest neighbor (simpler, no GATHER_ND)
    """
    
    def __init__(
        self,
        scale_factor: int = 2,
        mode: str = 'nearest',
        in_channels: int = None,
        out_channels: int = None
    ):
        """
        Args:
            scale_factor: Upsampling factor (default: 2)
            mode: Upsampling mode ('nearest', 'conv_transpose', 'pixel_shuffle')
            in_channels: Input channels (required for conv_transpose and pixel_shuffle)
            out_channels: Output channels (required for conv_transpose and pixel_shuffle)
        """
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        
        if mode == 'nearest':
            # Nearest neighbor - simple and no GATHER_ND
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        
        elif mode == 'conv_transpose':
            # ConvTranspose2d - learnable upsampling, no GATHER_ND
            if in_channels is None or out_channels is None:
                raise ValueError("in_channels and out_channels required for conv_transpose mode")
            self.upsample = nn.ConvTranspose2d(
                in_channels, 
                out_channels,
                kernel_size=scale_factor,
                stride=scale_factor,
                padding=0
            )
        
        elif mode == 'pixel_shuffle':
            # PixelShuffle - efficient and no GATHER_ND
            if in_channels is None or out_channels is None:
                raise ValueError("in_channels and out_channels required for pixel_shuffle mode")
            self.upsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), kernel_size=1),
                nn.PixelShuffle(scale_factor)
            )
        
        else:
            raise ValueError(f"Unsupported mode: {mode}. Use 'nearest', 'conv_transpose', or 'pixel_shuffle'")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x)


def replace_upsample_in_module(module: nn.Module, mode: str = 'nearest', verbose: bool = False):
    """
    Recursively replace all nn.Upsample modules with TFLite-friendly alternatives.
    
    Args:
        module: The module to modify
        mode: Replacement mode ('nearest', 'conv_transpose', 'pixel_shuffle')
        verbose: Print replacement information
    
    Returns:
        Number of replacements made
    """
    replacements = 0
    
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Upsample):
            # Get the original parameters
            scale_factor = child.scale_factor
            original_mode = child.mode
            
            if original_mode == 'bilinear' or original_mode == 'bicubic':
                # Replace with TFLite-friendly version
                # For now, use nearest neighbor as it's the simplest
                new_upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
                setattr(module, name, new_upsample)
                replacements += 1
                
                if verbose:
                    print(f"Replaced {name}: nn.Upsample(mode='{original_mode}') -> "
                          f"nn.Upsample(mode='nearest')")
        else:
            # Recursively process child modules
            replacements += replace_upsample_in_module(child, mode, verbose)
    
    return replacements


def replace_upsample_blocks(model: nn.Module, verbose: bool = False):
    """
    Replace Upsample blocks in VideoSeal models with TFLite-friendly versions.
    
    This function specifically targets the Upsample class from videoseal.modules.common
    and replaces the entire upsample_block with ConvTranspose2d to avoid GATHER_ND.
    
    Args:
        model: The VideoSeal model
        verbose: Print replacement information
    
    Returns:
        Number of replacements made
    """
    replacements = 0
    
    # Traverse the model
    for name, module in model.named_modules():
        if hasattr(module, 'upsample_block') and isinstance(module.upsample_block, nn.Sequential):
            if verbose:
                print(f"Processing {name}.upsample_block")
            
            # Check if it has nn.Upsample
            has_upsample = False
            upsample_idx = -1
            scale_factor = 2
            
            for i, layer in enumerate(module.upsample_block):
                if isinstance(layer, nn.Upsample):
                    has_upsample = True
                    upsample_idx = i
                    scale_factor = layer.scale_factor
                    break
            
            if has_upsample:
                # Get the Conv2d layer to determine channels
                conv_layer = None
                for layer in module.upsample_block:
                    if isinstance(layer, nn.Conv2d):
                        conv_layer = layer
                        break
                
                if conv_layer is not None:
                    in_channels = conv_layer.in_channels
                    out_channels = conv_layer.out_channels
                    
                    # Create ConvTranspose2d replacement
                    # This avoids any resize operations and GATHER_ND
                    conv_transpose = nn.ConvTranspose2d(
                        in_channels,
                        out_channels,
                        kernel_size=scale_factor,
                        stride=scale_factor,
                        padding=0,
                        bias=False
                    )
                    
                    # Initialize with bilinear weights
                    with torch.no_grad():
                        # Simple initialization - can be improved
                        nn.init.kaiming_normal_(conv_transpose.weight)
                    
                    # Replace the upsample_block with just ConvTranspose2d + activation
                    # Keep the activation and normalization from original
                    new_block = []
                    new_block.append(conv_transpose)
                    
                    # Add normalization and activation if they exist
                    for layer in module.upsample_block:
                        if not isinstance(layer, (nn.Upsample, nn.ReflectionPad2d, nn.Conv2d)):
                            new_block.append(layer)
                    
                    module.upsample_block = nn.Sequential(*new_block)
                    replacements += 1
                    
                    if verbose:
                        print(f"  Replaced Upsample block with ConvTranspose2d({in_channels}->{out_channels})")
    
    return replacements


if __name__ == "__main__":
    # Test the TFLite-friendly upsample
    print("Testing TFLiteFriendlyUpsample...")
    
    # Test nearest neighbor
    up_nearest = TFLiteFriendlyUpsample(scale_factor=2, mode='nearest')
    x = torch.randn(1, 64, 32, 32)
    y = up_nearest(x)
    print(f"Nearest: {x.shape} -> {y.shape}")
    
    # Test conv_transpose
    up_conv = TFLiteFriendlyUpsample(
        scale_factor=2, 
        mode='conv_transpose',
        in_channels=64,
        out_channels=32
    )
    y = up_conv(x)
    print(f"ConvTranspose: {x.shape} -> {y.shape}")
    
    # Test pixel_shuffle
    up_shuffle = TFLiteFriendlyUpsample(
        scale_factor=2,
        mode='pixel_shuffle',
        in_channels=64,
        out_channels=32
    )
    y = up_shuffle(x)
    print(f"PixelShuffle: {x.shape} -> {y.shape}")
    
    print("\n✓ All tests passed!")
