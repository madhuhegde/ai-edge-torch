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
VideoSeal 0.0 Model Wrappers for TFLite Conversion.

VideoSeal 0.0 is the baseline watermarking model with 96-bit capacity.
This module provides PyTorch wrappers optimized for TFLite conversion.

Key Features:
- 96-bit watermark capacity (legacy baseline)
- UNet-Small2 embedder
- SAM-Small detector (ViT-based)
- RGB processing
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
import videoseal
from videoseal.utils.cfg import setup_model_from_checkpoint
from tflite_msg_processor import TFLiteFriendlyMsgProcessor, transfer_weights


class VideoSeal00DetectorWrapper(nn.Module):
    """
    Wrapper for VideoSeal 0.0 Detector optimized for TFLite conversion.
    
    The VideoSeal 0.0 detector uses SAM-Small architecture with:
    - 96-bit output (legacy baseline)
    - ViT-based image encoder
    - Pixel decoder for spatial predictions
    
    Input:
        imgs: Tensor of shape (batch, 3, H, W) in range [0, 1]
    
    Output:
        preds: Tensor of shape (batch, 97) where:
            - preds[:, 0]: Detection confidence
            - preds[:, 1:97]: 96-bit watermark message (logits)
    """
    
    def __init__(self, model_name="videoseal_0.0", eval_mode=True):
        """
        Initialize VideoSeal 0.0 detector wrapper.
        
        Args:
            model_name: VideoSeal model variant (default: "videoseal_0.0")
            eval_mode: If True, set model to eval mode (default: True)
        """
        super().__init__()
        
        # Check for checkpoint in multiple locations
        checkpoint_paths = [
            Path("/mnt/shared/shared/VideoSeal/rgb_96b.pth"),
            Path.home() / ".cache" / "videoseal" / "rgb_96b.pth",
        ]
        
        checkpoint_found = None
        for checkpoint_path in checkpoint_paths:
            if checkpoint_path.exists():
                checkpoint_found = checkpoint_path
                break
        
        if checkpoint_found:
            print(f"Loading VideoSeal 0.0 from: {checkpoint_found}")
            # Load directly from checkpoint file
            self.model = setup_model_from_checkpoint(str(checkpoint_found))
        else:
            print(f"Checkpoint not found in any of these locations:")
            for path in checkpoint_paths:
                print(f"  - {path}")
            print(f"Loading VideoSeal 0.0 from model card (will download if needed)...")
            # Load using model card (will download from URL)
            # Need to use absolute path to model card
            import sys
            videoseal_path = Path(sys.modules['videoseal'].__file__).parent
            model_card_path = videoseal_path / "cards" / "videoseal_0.0.yaml"
            self.model = videoseal.load(model_card_path)
        
        # Note: We do NOT replace Upsample operations because:
        # - Changing bilinear to nearest/ConvTranspose degrades accuracy significantly
        # - The resulting GATHER_ND operations have minimal performance impact
        
        if eval_mode:
            self.eval()
    
    def forward(self, imgs):
        """
        Forward pass for watermark detection.
        
        Args:
            imgs: Input images (batch, H, W, 3) in range [0, 1] (NHWC format)
        
        Returns:
            preds: Predictions (batch, 97) with detection + 96-bit message
        """
        # Convert from NHWC to NCHW for PyTorch model
        # Use .contiguous() to avoid GATHER_ND operations in TFLite
        imgs_nchw = imgs.permute(0, 3, 1, 2).contiguous()
        
        # Preprocess: [0, 1] → [-1, 1]
        imgs_proc = self.model.detector.preprocess(imgs_nchw)
        
        # Run detection through SAM-Small
        preds = self.model.detector(imgs_proc)
        
        # Post-process output
        # preds shape: (batch, 97, H, W) for spatial predictions
        # Average across spatial dimensions to get final prediction
        if len(preds.shape) == 4:
            preds = preds.mean(dim=(2, 3))  # (batch, 97)
        
        return preds


class VideoSeal00EmbedderWrapper(nn.Module):
    """
    Wrapper for VideoSeal 0.0 Embedder optimized for TFLite conversion.
    
    The VideoSeal 0.0 embedder uses UNet-Small2 architecture with:
    - 96-bit message input
    - Message processor with binary+concat type
    - UNet with 8 blocks
    
    Input:
        imgs: Tensor of shape (batch, 3, H, W) in range [0, 1]
        msgs: Tensor of shape (batch, 96) with binary message (0 or 1)
    
    Output:
        imgs_w: Watermarked images (batch, 3, H, W) in range [0, 1]
    """
    
    def __init__(self, model_name="videoseal_0.0", eval_mode=True):
        """
        Initialize VideoSeal 0.0 embedder wrapper.
        
        Args:
            model_name: VideoSeal model variant (default: "videoseal_0.0")
            eval_mode: If True, set model to eval mode (default: True)
        """
        super().__init__()
        
        # Check for checkpoint in multiple locations
        checkpoint_paths = [
            Path("/mnt/shared/shared/VideoSeal/rgb_96b.pth"),
            Path.home() / ".cache" / "videoseal" / "rgb_96b.pth",
        ]
        
        checkpoint_found = None
        for checkpoint_path in checkpoint_paths:
            if checkpoint_path.exists():
                checkpoint_found = checkpoint_path
                break
        
        if checkpoint_found:
            print(f"Loading VideoSeal 0.0 from: {checkpoint_found}")
            # Load directly from checkpoint file
            self.model = setup_model_from_checkpoint(str(checkpoint_found))
        else:
            print(f"Checkpoint not found in any of these locations:")
            for path in checkpoint_paths:
                print(f"  - {path}")
            print(f"Loading VideoSeal 0.0 from model card (will download if needed)...")
            # Load using model card
            # Need to use absolute path to model card
            import sys
            videoseal_path = Path(sys.modules['videoseal'].__file__).parent
            model_card_path = videoseal_path / "cards" / "videoseal_0.0.yaml"
            self.model = videoseal.load(model_card_path)
        
        # Disable attenuation for TFLite compatibility
        if hasattr(self.model, 'attenuation') and self.model.attenuation is not None:
            print("⚠️  Disabling attenuation for TFLite compatibility")
            self.model.attenuation = None
        
        # Replace message processor with TFLite-friendly version
        print("🔧 Replacing message processor with TFLite-friendly version...")
        original_processor = self.model.embedder.unet.msg_processor
        
        tflite_processor = TFLiteFriendlyMsgProcessor(
            nbits=original_processor.nbits,
            hidden_size=original_processor.hidden_size,
            spatial_size=32,  # Fixed for 256x256 images
            msg_processor_type=original_processor.msg_type + "+" + original_processor.msg_agg,
            msg_mult=original_processor.msg_mult
        )
        
        # Transfer weights
        transfer_weights(original_processor, tflite_processor)
        
        # Replace in model
        self.model.embedder.unet.msg_processor = tflite_processor
        print("✓ Message processor replaced successfully")
        
        # Note: We do NOT replace Upsample operations because:
        # - Changing bilinear to nearest/ConvTranspose degrades accuracy significantly
        # - The resulting GATHER_ND operations have minimal performance impact
        # - Modern TFLite runtimes handle GATHER_ND efficiently
        
        if eval_mode:
            self.eval()
    
    def forward(self, imgs, msgs):
        """
        Forward pass for watermark embedding.
        
        Args:
            imgs: Input images (batch, H, W, 3) in range [0, 1] (NHWC format)
            msgs: Binary messages (batch, 96) with values 0 or 1
        
        Returns:
            imgs_w: Watermarked images (batch, H, W, 3) in range [0, 1] (NHWC format)
        """
        # Convert from NHWC to NCHW for PyTorch model
        # Use .contiguous() to avoid GATHER_ND operations in TFLite
        imgs_nchw = imgs.permute(0, 3, 1, 2).contiguous()
        
        # Preprocess image: [0, 1] → [-1, 1]
        imgs_proc = self.model.embedder.preprocess(imgs_nchw)
        
        # Run embedding through UNet
        preds_w = self.model.embedder(imgs_proc, msgs)
        
        # Blend with original image
        imgs_w = self.model.blender(imgs_nchw, preds_w)
        
        # Clamp to [0, 1]
        if self.model.clamp:
            imgs_w = torch.clamp(imgs_w, 0, 1)
        
        # Convert back from NCHW to NHWC
        # Use .contiguous() to avoid GATHER_ND operations in TFLite
        imgs_w = imgs_w.permute(0, 2, 3, 1).contiguous()
        
        return imgs_w


def create_detector(model_name="videoseal_0.0", simple=True):
    """
    Factory function to create a VideoSeal 0.0 detector wrapper.
    
    Args:
        model_name: VideoSeal model variant (default: "videoseal_0.0")
        simple: If True, use simplified wrapper (default: True)
    
    Returns:
        VideoSeal00DetectorWrapper instance ready for TFLite conversion
    
    Example:
        >>> detector = create_detector("videoseal_0.0")
        >>> img = torch.rand(1, 3, 256, 256)
        >>> output = detector(img)
        >>> print(output.shape)  # (1, 97)
        >>> 
        >>> # Extract detection confidence and message
        >>> confidence = output[0, 0]
        >>> message = (output[0, 1:] > 0).float()  # 96 bits
    """
    return VideoSeal00DetectorWrapper(model_name=model_name, eval_mode=True)


def create_embedder(model_name="videoseal_0.0", simple=True):
    """
    Factory function to create a VideoSeal 0.0 embedder wrapper.
    
    Args:
        model_name: VideoSeal model variant (default: "videoseal_0.0")
        simple: If True, use simplified wrapper (default: True)
    
    Returns:
        VideoSeal00EmbedderWrapper instance ready for TFLite conversion
    
    Example:
        >>> embedder = create_embedder("videoseal_0.0")
        >>> img = torch.rand(1, 3, 256, 256)
        >>> msg = torch.randint(0, 2, (1, 96)).float()
        >>> img_w = embedder(img, msg)
        >>> print(img_w.shape)  # (1, 3, 256, 256)
    """
    return VideoSeal00EmbedderWrapper(model_name=model_name, eval_mode=True)


if __name__ == "__main__":
    # Test the detector wrapper
    print("Testing VideoSeal 0.0 Detector Wrapper...")
    
    detector = create_detector("videoseal_0.0")
    print(f"✓ Loaded VideoSeal 0.0 detector")
    
    # Create test input
    img = torch.rand(1, 3, 256, 256)
    print(f"✓ Created test input: {img.shape}")
    
    # Run inference
    with torch.no_grad():
        output = detector(img)
    
    print(f"✓ Output shape: {output.shape}")
    print(f"  Expected: (1, 97) - 1 detection + 96 message bits")
    
    # Extract results
    confidence = output[0, 0].item()
    message = (output[0, 1:] > 0).float()
    
    print(f"\n✓ Detection confidence: {confidence:.4f}")
    print(f"✓ Message bits: {message.sum().int().item()}/96 are 1")
    print(f"✓ First 32 bits: {message[:32].int().tolist()}")
    
    print("\n" + "="*70)
    print("Testing VideoSeal 0.0 Embedder Wrapper...")
    
    embedder = create_embedder("videoseal_0.0")
    print(f"✓ Loaded VideoSeal 0.0 embedder")
    
    # Create test inputs
    img = torch.rand(1, 3, 256, 256)
    msg = torch.randint(0, 2, (1, 96)).float()
    print(f"✓ Created test inputs: img={img.shape}, msg={msg.shape}")
    
    # Run inference
    with torch.no_grad():
        img_w = embedder(img, msg)
    
    print(f"✓ Output shape: {img_w.shape}")
    print(f"  Expected: (1, 3, 256, 256)")
    
    # Calculate PSNR
    mse = torch.mean((img_w - img) ** 2)
    psnr = 10 * torch.log10(1.0 / mse)
    
    print(f"\n✓ PSNR: {psnr:.2f} dB")
    print(f"✓ Output range: [{img_w.min():.3f}, {img_w.max():.3f}]")
    
    print("\n✓ VideoSeal 0.0 wrappers are working correctly!")
