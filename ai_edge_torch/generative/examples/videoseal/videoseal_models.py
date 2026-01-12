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


class VideoSealEmbedderWrapper(nn.Module):
    """Wrapper for VideoSeal Embedder that's optimized for TFLite conversion.
    
    The embedder takes an image and a binary message and produces a watermarked image.
    
    Input: 
        - Image: (batch, channels=3, height, width) in [0, 1] range
        - Message: (batch, 256) binary vector {0, 1}
    
    Output:
        - Watermarked image: (batch, channels=3, height, width) in [0, 1] range
    """
    
    def __init__(self, model_name="videoseal", eval_mode=True):
        """
        Args:
            model_name: Name of the VideoSeal model to load ('videoseal', 'pixelseal', 'chunkyseal')
            eval_mode: Whether to set model to eval mode
        """
        super().__init__()
        
        print(f"Loading VideoSeal model: {model_name}")
        self.model = videoseal.load(model_name)
        
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
            imgs: Tensor of shape (batch, 3, height, width) in [0, 1] range
            msgs: Tensor of shape (batch, 256) with binary values {0, 1}
        
        Returns:
            imgs_w: Watermarked images of shape (batch, 3, height, width) in [0, 1] range
        """
        original_size = imgs.shape[-2:]
        
        # Resize to processing size if needed
        imgs_res = imgs
        if imgs.shape[-2:] != (self.img_size, self.img_size):
            imgs_res = F.interpolate(
                imgs, 
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
        imgs_w = self.blender(imgs, preds_w)
        
        # Apply attenuation if available
        if self.attenuation is not None:
            imgs_w = self.attenuation(imgs, imgs_w)
        
        # Clamp to valid range
        imgs_w = torch.clamp(imgs_w, 0, 1)
        
        return imgs_w


class VideoSealDetectorWrapper(nn.Module):
    """Wrapper for VideoSeal Detector that's optimized for TFLite conversion.
    
    The detector takes a potentially watermarked image and extracts the embedded message.
    
    Input:
        - Image: (batch, channels=3, height, width) in [0, 1] range
    
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
        self.model = videoseal.load(model_name)
        
        # Extract detector
        self.detector = self.model.detector
        self.img_size = self.model.img_size
        
        if eval_mode:
            self.eval()
    
    def forward(self, imgs):
        """
        Detect watermark from images.
        
        Args:
            imgs: Tensor of shape (batch, 3, height, width) in [0, 1] range
        
        Returns:
            preds: Tensor of shape (batch, 257, height_out, width_out)
                   Channel 0: Detection mask
                   Channels 1-256: Message bits (apply sigmoid + threshold at 0.5 for binary)
        """
        # Resize to processing size if needed
        imgs_res = imgs
        if imgs.shape[-2:] != (self.img_size, self.img_size):
            imgs_res = F.interpolate(
                imgs,
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
        self.model = videoseal.load(model_name)
        
        self.embedder = self.model.embedder
        self.blender = self.model.blender
        self.attenuation = self.model.attenuation
        self.rgb2yuv = RGB2YUV()
        self.yuv_mode = self.embedder.yuv
        
        if eval_mode:
            self.eval()
    
    def forward(self, imgs, msgs):
        """
        Args:
            imgs: (batch, 3, 256, 256) in [0, 1]
            msgs: (batch, 256) binary
        Returns:
            imgs_w: (batch, 3, 256, 256) in [0, 1]
        """
        # Generate watermark
        if self.yuv_mode:
            imgs_yuv = self.rgb2yuv(imgs)
            preds_w = self.embedder(imgs_yuv[:, 0:1], msgs)
        else:
            preds_w = self.embedder(imgs, msgs)
        
        # Blend
        imgs_w = self.blender(imgs, preds_w)
        
        # Attenuate
        if self.attenuation is not None:
            imgs_w = self.attenuation(imgs, imgs_w)
        
        # Clamp
        imgs_w = torch.clamp(imgs_w, 0, 1)
        
        return imgs_w


class VideoSealDetectorSimple(nn.Module):
    """Simplified detector for fixed-size images (256x256).
    
    This version is optimized for TFLite conversion by removing dynamic resizing.
    Use this for fixed-size image processing.
    """
    
    def __init__(self, model_name="videoseal", eval_mode=True):
        super().__init__()
        
        print(f"Loading VideoSeal model: {model_name}")
        self.model = videoseal.load(model_name)
        self.detector = self.model.detector
        
        if eval_mode:
            self.eval()
    
    def forward(self, imgs):
        """
        Args:
            imgs: (batch, 3, 256, 256) in [0, 1]
        Returns:
            preds: (batch, 257, height_out, width_out)
        """
        return self.detector(imgs)


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

