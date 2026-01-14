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
VideoSeal Model Definitions for TFLite Conversion

This module provides wrapper classes for VideoSeal models that are compatible
with ai-edge-torch conversion to TFLite format.

VideoSeal is a state-of-the-art invisible watermarking model for images and videos.
The model consists of two main components:
1. Embedder: Embeds a 256-bit watermark into images
2. Detector: Detects and extracts the watermark from images
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add VideoSeal source path
VIDEOSEAL_SOURCE_PATH = Path.home() / "work" / "videoseal" / "videoseal"
sys.path.insert(0, str(VIDEOSEAL_SOURCE_PATH))

import videoseal
from videoseal.data.transforms import RGB2YUV
from tflite_msg_processor import TFLiteFriendlyMsgProcessor, transfer_weights


class TFLiteFriendlyRGB2YUV(nn.Module):
    """TFLite-friendly RGB to YUV conversion for NHWC format.
    
    This version handles NHWC input directly without permutations,
    avoiding shape mismatch issues during TFLite conversion.
    """
    def __init__(self):
        super().__init__()
        # RGB to YUV conversion matrix
        self.register_buffer('M', torch.tensor([
            [0.299, 0.587, 0.114],
            [-0.14713, -0.28886, 0.436],
            [0.615, -0.51499, -0.10001]
        ], dtype=torch.float32))
    
    def forward(self, x):
        """
        Convert RGB to YUV for NHWC input.
        
        Args:
            x: Tensor of shape (batch, height, width, 3) in NHWC format
        
        Returns:
            yuv: Tensor of shape (batch, 3, height, width) in NCHW format
        """
        # x is already in NHWC format (batch, height, width, 3)
        # Apply matrix multiplication: (B, H, W, 3) @ (3, 3) -> (B, H, W, 3)
        yuv_nhwc = torch.matmul(x, self.M.T)
        
        # Convert to NCHW for the embedder
        yuv = yuv_nhwc.permute(0, 3, 1, 2).contiguous()
        
        return yuv


class VideoSealEmbedderWrapper(nn.Module):
    """Wrapper for VideoSeal Embedder that's optimized for TFLite conversion.
    
    The embedder takes an image and a binary message and produces a watermarked image.
    
    Input: 
        - Image: (batch, height, width, channels=3) in [0, 1] range (NHWC format)
        - Message: (batch, 256) binary vector {0, 1}
    
    Output:
        - Watermarked image: (batch, height, width, channels=3) in [0, 1] range (NHWC format)
    """
    
    def __init__(self, model_name="videoseal", eval_mode=True):
        """
        Args:
            model_name: Name of the VideoSeal model to load ('videoseal', 'pixelseal', 'chunkyseal')
            eval_mode: Whether to set model to eval mode
        """
        super().__init__()
        
        print(f"Loading VideoSeal model: {model_name}")
        # Use absolute path to model card to avoid path resolution issues
        import sys
        videoseal_path = Path(sys.modules['videoseal'].__file__).parent
        if model_name == "videoseal":
            model_name = "videoseal_1.0"  # Default to VideoSeal 1.0
        model_card_path = videoseal_path / "cards" / f"{model_name}.yaml"
        self.model = videoseal.load(model_card_path)
        
        # Extract components
        self.embedder = self.model.embedder
        self.blender = self.model.blender
        self.attenuation = self.model.attenuation
        self.img_size = self.model.img_size
        self.rgb2yuv = RGB2YUV()
        self.yuv_mode = self.embedder.yuv
        
        if eval_mode:
            self.eval()
    
    def forward(self, imgs, msgs):
        """
        Embed watermark into images.
        
        Args:
            imgs: Tensor of shape (batch, height, width, 3) in [0, 1] range (NHWC format)
            msgs: Tensor of shape (batch, 256) with binary values {0, 1}
        
        Returns:
            imgs_w: Watermarked images of shape (batch, height, width, 3) in [0, 1] range (NHWC format)
        """
        # Convert from NHWC to NCHW for PyTorch model
        # Use .contiguous() to avoid GATHER_ND operations in TFLite
        imgs_nchw = imgs.permute(0, 3, 1, 2).contiguous()
        
        original_size = imgs_nchw.shape[-2:]
        
        # Resize to processing size if needed
        imgs_res = imgs_nchw
        if imgs_nchw.shape[-2:] != (self.img_size, self.img_size):
            imgs_res = F.interpolate(
                imgs_nchw, 
                size=(self.img_size, self.img_size),
                mode='bilinear',
                align_corners=False,
                antialias=True
            )
        
        # Generate watermark
        if self.yuv_mode:
            # Convert to YUV and use only Y channel
            imgs_yuv = self.rgb2yuv(imgs_res)
            preds_w = self.embedder(imgs_yuv[:, 0:1], msgs)
        else:
            preds_w = self.embedder(imgs_res, msgs)
        
        # Resize watermark back to original size if needed
        if original_size != (self.img_size, self.img_size):
            preds_w = F.interpolate(
                preds_w,
                size=original_size,
                mode='bilinear',
                align_corners=False,
                antialias=True
            )
        
        # Blend watermark with original image
        imgs_w = self.blender(imgs_nchw, preds_w)
        
        # Apply attenuation if available
        if self.attenuation is not None:
            imgs_w = self.attenuation(imgs_nchw, imgs_w)
        
        # Clamp to valid range
        imgs_w = torch.clamp(imgs_w, 0, 1)
        
        # Convert back from NCHW to NHWC
        # Use .contiguous() to avoid GATHER_ND operations in TFLite
        imgs_w = imgs_w.permute(0, 2, 3, 1).contiguous()
        
        return imgs_w


class VideoSealDetectorWrapper(nn.Module):
    """Wrapper for VideoSeal Detector that's optimized for TFLite conversion.
    
    The detector takes a potentially watermarked image and extracts the embedded message.
    
    Input:
        - Image: (batch, height, width, channels=3) in [0, 1] range (NHWC format)
    
    Output:
        - Predictions: (batch, 257, height_out, width_out)
          - Channel 0: Detection mask (confidence that watermark is present)
          - Channels 1-256: Binary message bits
    """
    
    def __init__(self, model_name="videoseal", eval_mode=True):
        """
        Args:
            model_name: Name of the VideoSeal model to load ('videoseal', 'pixelseal', 'chunkyseal')
            eval_mode: Whether to set model to eval mode
        """
        super().__init__()
        
        print(f"Loading VideoSeal model: {model_name}")
        # Use absolute path to model card to avoid path resolution issues
        import sys
        videoseal_path = Path(sys.modules['videoseal'].__file__).parent
        if model_name == "videoseal":
            model_name = "videoseal_1.0"  # Default to VideoSeal 1.0
        model_card_path = videoseal_path / "cards" / f"{model_name}.yaml"
        self.model = videoseal.load(model_card_path)
        
        # Extract detector
        self.detector = self.model.detector
        self.img_size = self.model.img_size
        
        if eval_mode:
            self.eval()
    
    def forward(self, imgs):
        """
        Detect watermark from images.
        
        Args:
            imgs: Tensor of shape (batch, height, width, 3) in [0, 1] range (NHWC format)
        
        Returns:
            preds: Tensor of shape (batch, 257, height_out, width_out)
                   Channel 0: Detection mask
                   Channels 1-256: Message bits (apply sigmoid + threshold at 0.5 for binary)
        """
        # Convert from NHWC to NCHW for PyTorch model
        # Use .contiguous() to avoid GATHER_ND operations in TFLite
        imgs_nchw = imgs.permute(0, 3, 1, 2).contiguous()
        
        # Resize to processing size if needed
        imgs_res = imgs_nchw
        if imgs_nchw.shape[-2:] != (self.img_size, self.img_size):
            imgs_res = F.interpolate(
                imgs_nchw,
                size=(self.img_size, self.img_size),
                mode='bilinear',
                align_corners=False,
                antialias=True
            )
        
        # Detect watermark
        preds = self.detector(imgs_res)
        
        return preds


class VideoSealEmbedderSimple(nn.Module):
    """Simplified embedder for fixed-size images (256x256).
    
    This version is optimized for TFLite conversion by removing dynamic resizing.
    Use this for fixed-size image processing.
    """
    
    def __init__(self, model_name="videoseal", eval_mode=True):
        super().__init__()
        
        print(f"Loading VideoSeal model: {model_name}")
        # Use absolute path to model card to avoid path resolution issues
        import sys
        videoseal_path = Path(sys.modules['videoseal'].__file__).parent
        if model_name == "videoseal":
            model_name = "videoseal_1.0"  # Default to VideoSeal 1.0
        model_card_path = videoseal_path / "cards" / f"{model_name}.yaml"
        self.model = videoseal.load(model_card_path)
        
        self.embedder = self.model.embedder
        self.blender = self.model.blender
        self.attenuation = self.model.attenuation
        self.rgb2yuv = TFLiteFriendlyRGB2YUV()  # Use TFLite-friendly version for NHWC input
        self.yuv_mode = self.embedder.yuv
        
        # Replace message processor with TFLite-friendly version
        # Calculate spatial size at bottleneck (for 256x256 images)
        num_downs = len(self.embedder.unet.downs)
        spatial_size = 256 // (2 ** num_downs)
        
        original_msg_proc = self.embedder.unet.msg_processor
        self.tflite_msg_processor = TFLiteFriendlyMsgProcessor(
            nbits=original_msg_proc.nbits,
            hidden_size=original_msg_proc.hidden_size,
            spatial_size=spatial_size,
            msg_processor_type=original_msg_proc.msg_processor_type,
            msg_mult=original_msg_proc.msg_mult
        )
        
        # Transfer weights from original to TFLite processor
        transfer_weights(original_msg_proc, self.tflite_msg_processor)
        
        # Replace message processor in UNet
        self.embedder.unet.msg_processor = self.tflite_msg_processor
        print(f"✓ TFLite-friendly message processor installed (spatial_size={spatial_size})")
        
        if eval_mode:
            self.eval()
    
    def forward(self, imgs, msgs):
        """
        Args:
            imgs: (batch, 256, 256, 3) in [0, 1] (NHWC format)
            msgs: (batch, 256) binary
        Returns:
            imgs_w: (batch, 256, 256, 3) in [0, 1] (NHWC format)
        """
        # Convert from NHWC to NCHW for PyTorch model
        # Use .contiguous() to avoid GATHER_ND operations in TFLite
        imgs_nchw = imgs.permute(0, 3, 1, 2).contiguous()
        
        # Generate watermark
        if self.yuv_mode:
            # rgb2yuv expects NHWC input and returns NCHW
            imgs_yuv = self.rgb2yuv(imgs)
            preds_w = self.embedder(imgs_yuv[:, 0:1], msgs)
        else:
            preds_w = self.embedder(imgs_nchw, msgs)
        
        # Blend
        imgs_w = self.blender(imgs_nchw, preds_w)
        
        # Attenuate (disable for TFLite - uses boolean indexing)
        # if self.attenuation is not None:
        #     imgs_w = self.attenuation(imgs_nchw, imgs_w)
        
        # Clamp
        imgs_w = torch.clamp(imgs_w, 0, 1)
        
        # Convert back from NCHW to NHWC
        # Use .contiguous() to avoid GATHER_ND operations in TFLite
        imgs_w = imgs_w.permute(0, 2, 3, 1).contiguous()
        
        return imgs_w


class VideoSealDetectorSimple(nn.Module):
    """Simplified detector for fixed-size images (256x256).
    
    This version is optimized for TFLite conversion by removing dynamic resizing.
    Use this for fixed-size image processing with NHWC input/output format.
    """
    
    def __init__(self, model_name="videoseal", eval_mode=True):
        super().__init__()
        
        print(f"Loading VideoSeal model: {model_name}")
        # Use absolute path to model card to avoid path resolution issues
        import sys
        videoseal_path = Path(sys.modules['videoseal'].__file__).parent
        if model_name == "videoseal":
            model_name = "videoseal_1.0"  # Default to VideoSeal 1.0
        model_card_path = videoseal_path / "cards" / f"{model_name}.yaml"
        self.model = videoseal.load(model_card_path)
        self.detector = self.model.detector
        
        if eval_mode:
            self.eval()
    
    def forward(self, imgs):
        """
        Args:
            imgs: (batch, height, width, channels=3) in [0, 1] (NHWC format)
        Returns:
            preds: (batch, 257) - Detection confidence + 256-bit message
                   Note: For spatial outputs, this averages across spatial dimensions
        """
        # Convert from NHWC to NCHW for PyTorch model
        # Use .contiguous() to avoid GATHER_ND operations in TFLite
        imgs_nchw = imgs.permute(0, 3, 1, 2).contiguous()
        
        # Run detector (expects NCHW input)
        preds = self.detector(imgs_nchw)  # (batch, 257, H, W) or (batch, 257)
        
        # If output is spatial (4D), average across spatial dimensions for TFLite compatibility
        if len(preds.shape) == 4:
            preds = preds.mean(dim=(2, 3))  # (batch, 257)
        
        return preds


def create_embedder(model_name="videoseal", simple=True):
    """Create VideoSeal embedder model for TFLite conversion.
    
    Args:
        model_name: Name of the model ('videoseal', 'pixelseal', 'chunkyseal')
        simple: If True, use simplified version for fixed 256x256 images
    
    Returns:
        Embedder model ready for conversion
    """
    if simple:
        model = VideoSealEmbedderSimple(model_name=model_name, eval_mode=True)
    else:
        model = VideoSealEmbedderWrapper(model_name=model_name, eval_mode=True)
    
    model.eval()
    return model


def create_detector(model_name="videoseal", simple=True):
    """Create VideoSeal detector model for TFLite conversion.
    
    Args:
        model_name: Name of the model ('videoseal', 'pixelseal', 'chunkyseal')
        simple: If True, use simplified version for fixed 256x256 images
    
    Returns:
        Detector model ready for conversion
    """
    if simple:
        model = VideoSealDetectorSimple(model_name=model_name, eval_mode=True)
    else:
        model = VideoSealDetectorWrapper(model_name=model_name, eval_mode=True)
    
    model.eval()
    return model


class VideoSealEmbedderTFLite(nn.Module):
    """
    TFLite-compatible VideoSeal embedder with fixed-size message processor.
    
    This embedder replaces the dynamic message processor with a TFLite-friendly
    version that uses fixed-size operations while maintaining mathematical
    equivalence to the original implementation.
    
    Args:
        model_name: VideoSeal model variant ('videoseal', 'pixelseal', 'chunkyseal')
        image_size: Input image size (default: 256)
        eval_mode: Whether to set model to eval mode
    
    Example:
        >>> embedder = VideoSealEmbedderTFLite('videoseal', 256)
        >>> imgs = torch.rand(1, 3, 256, 256)
        >>> msgs = torch.randint(0, 2, (1, 256)).float()
        >>> imgs_w = embedder(imgs, msgs)
        >>> print(imgs_w.shape)  # [1, 3, 256, 256]
    """
    
    def __init__(self, model_name="videoseal", image_size=256, eval_mode=True):
        super().__init__()
        
        print(f"Loading VideoSeal model: {model_name}")
        
        # Use embedder-only checkpoint for ChunkySeal to reduce memory usage
        if model_name == "chunkyseal":
            embedder_checkpoint = "/mnt/shared/shared/ChunkySeal/chunkyseal_embedder_only.pth"
            import os
            if os.path.exists(embedder_checkpoint):
                print(f"  Using embedder-only checkpoint: {embedder_checkpoint}")
                from videoseal.utils.cfg import setup_model_from_checkpoint
                self.model = setup_model_from_checkpoint(embedder_checkpoint)
            else:
                print(f"  Embedder-only checkpoint not found, using full model")
                self.model = videoseal.load(model_name)
        else:
            self.model = videoseal.load(model_name)
        
        # Extract components
        self.embedder = self.model.embedder
        self.blender = self.model.blender
        self.attenuation = self.model.attenuation
        self.rgb2yuv = RGB2YUV()
        self.yuv_mode = self.embedder.yuv
        self.image_size = image_size
        
        # Calculate spatial size at bottleneck
        num_downs = len(self.embedder.unet.downs)
        spatial_size = image_size // (2 ** num_downs)
        
        print(f"UNet architecture:")
        print(f"  Downsample layers: {num_downs}")
        print(f"  Spatial size at bottleneck: {spatial_size}x{spatial_size}")
        
        # Create TFLite-friendly message processor
        original_msg_proc = self.embedder.unet.msg_processor
        self.tflite_msg_processor = TFLiteFriendlyMsgProcessor(
            nbits=original_msg_proc.nbits,
            hidden_size=original_msg_proc.hidden_size,
            spatial_size=spatial_size,
            msg_processor_type=original_msg_proc.msg_processor_type,
            msg_mult=original_msg_proc.msg_mult
        )
        
        # Transfer weights from original to TFLite processor
        transfer_weights(original_msg_proc, self.tflite_msg_processor)
        
        # Replace message processor in UNet
        self.embedder.unet.msg_processor = self.tflite_msg_processor
        
        print(f"✓ TFLite-friendly message processor installed")
        print(f"  nbits: {original_msg_proc.nbits}")
        print(f"  hidden_size: {original_msg_proc.hidden_size}")
        print(f"  spatial_size: {spatial_size}")
        print(f"  msg_type: {self.tflite_msg_processor.msg_type}")
        print(f"  msg_agg: {self.tflite_msg_processor.msg_agg}")
        
        if eval_mode:
            self.eval()
    
    def forward(self, imgs, msgs):
        """
        Embed watermark into images.
        
        Args:
            imgs: Tensor [B, 3, H, W] in range [0, 1]
            msgs: Tensor [B, nbits] with binary values {0, 1}
        
        Returns:
            imgs_w: Watermarked images [B, 3, H, W] in range [0, 1]
        """
        # Generate watermark
        if self.yuv_mode:
            imgs_yuv = self.rgb2yuv(imgs)
            preds_w = self.embedder(imgs_yuv[:, 0:1], msgs)
        else:
            preds_w = self.embedder(imgs, msgs)
        
        # Apply fixed attenuation to match PyTorch behavior
        # The original attenuation (JND) module uses boolean indexing which is not supported by TFLite
        # We use a fixed attenuation factor based on the average heatmap value (~0.11)
        # This reduces watermark strength to match PyTorch output
        attenuation_factor = 0.11
        preds_w = preds_w * attenuation_factor
        
        # Blend watermark with original image
        imgs_w = self.blender(imgs, preds_w)
        
        # Clamp to valid range
        imgs_w = torch.clamp(imgs_w, 0, 1)
        
        return imgs_w


def create_embedder_tflite(model_name="videoseal", image_size=256):
    """
    Create TFLite-compatible VideoSeal embedder.
    
    This function creates an embedder with a fixed-size message processor
    that is compatible with TFLite conversion.
    
    Args:
        model_name: VideoSeal model variant ('videoseal', 'pixelseal', 'chunkyseal')
        image_size: Input image size (default: 256)
    
    Returns:
        VideoSealEmbedderTFLite instance ready for conversion
    
    Example:
        >>> embedder = create_embedder_tflite('videoseal', 256)
        >>> imgs = torch.rand(1, 3, 256, 256)
        >>> msgs = torch.randint(0, 2, (1, 256)).float()
        >>> imgs_w = embedder(imgs, msgs)
    """
    model = VideoSealEmbedderTFLite(model_name=model_name, image_size=image_size, eval_mode=True)
    model.eval()
    return model

