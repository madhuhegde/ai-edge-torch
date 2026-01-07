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
UVQ 1.5 Model Definitions for TFLite Conversion

This module provides wrapper classes for UVQ 1.5 models that are compatible
with ai-edge-torch conversion to TFLite format.
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add UVQ source path
UVQ_SOURCE_PATH = Path.home() / "work" / "UVQ" / "uvq"
sys.path.insert(0, str(UVQ_SOURCE_PATH))

from uvq1p5_pytorch.utils import contentnet
from uvq1p5_pytorch.utils import distortionnet
from uvq1p5_pytorch.utils import aggregationnet


class ContentNetWrapper(nn.Module):
    """Wrapper for ContentNet that's optimized for TFLite conversion.
    
    ContentNet extracts semantic content features from video frames.
    Input: Video frames resized to 256x256
    Output: Feature maps of shape (batch, 8, 8, 128)
    """
    
    def __init__(self, model_path=None, eval_mode=True):
        super().__init__()
        if model_path is None:
            model_path = Path.home() / "work" / "models" / "UVQ" / "uvq1.5" / "content_net.pth"
        
        self.content_net = contentnet.ContentNet(
            model_path=str(model_path),
            eval_mode=eval_mode,
            pretrained=True,
        )
        
        if eval_mode:
            self.eval()
    
    def forward(self, video_frame):
        """
        Args:
            video_frame: Tensor of shape (batch, height=256, width=256, channels=3)
                        Values should be in [-1, 1] range
        
        Returns:
            features: Tensor of shape (batch, 8, 8, 128)
        """
        # Convert from TensorFlow format (B, H, W, C) to PyTorch format (B, C, H, W)
        # Use contiguous() to ensure memory layout is optimal for TFLite conversion
        video_frame = video_frame.permute(0, 3, 1, 2).contiguous()
        
        # ContentNet model expects (batch, 3, 256, 256) and outputs (batch, 128, 8, 8)
        features = self.content_net.model(video_frame)
        
        # Permute to (batch, 8, 8, 128) format
        features = features.permute(0, 2, 3, 1).contiguous()
        return features


class DistortionNetWrapper(nn.Module):
    """Wrapper for DistortionNet that's optimized for TFLite conversion.
    
    DistortionNet detects visual distortions using patch-based processing.
    Input: Video frames at 1920x1080, split into 3x3 patches (640x360 each)
    Output: Feature maps of shape (batch, 24, 24, 128)
    """
    
    def __init__(self, model_path=None, eval_mode=True):
        super().__init__()
        if model_path is None:
            model_path = Path.home() / "work" / "models" / "UVQ" / "uvq1.5" / "distortion_net.pth"
        
        self.distortion_net = distortionnet.DistortionNet(
            model_path=str(model_path),
            eval_mode=eval_mode,
            pretrained=True,
        )
        
        if eval_mode:
            self.eval()
    
    def forward(self, video_patches):
        """
        Args:
            video_patches: Tensor of shape (batch * 9, height=360, width=640, channels=3)
                          9 patches per frame (3x3 grid)
                          Values should be in [-1, 1] range
        
        Returns:
            features: Tensor of shape (batch, 24, 24, 128)
        """
        # Convert from TensorFlow format (B, H, W, C) to PyTorch format (B, C, H, W)
        # Use contiguous() to ensure memory layout is optimal for TFLite conversion
        video_patches = video_patches.permute(0, 3, 1, 2).contiguous()
        
        # Process patches through DistortionNet core
        # Output is [9, 8, 8, 128] (NHWC) due to PermuteLayerNHWC in the model
        patch_features = self.distortion_net.model(video_patches)
        
        # Match the exact reshape logic from PyTorch DistortionNet.forward()
        # batch_features shape: [9, 8, 8, 128] (NHWC)
        batch_size = patch_features.shape[0] // 9
        num_patches_y = 3
        num_patches_x = 3
        patch_feature_height = 8
        patch_feature_width = 8
        feature_channels = 128
        
        # Reshape to [batch, num_patches_y, num_patches_x, patch_feature_height, patch_feature_width, feature_channels]
        # For batch=1: [1, 3, 3, 8, 8, 128]
        features = patch_features.reshape(
            batch_size,
            num_patches_y,
            num_patches_x,
            patch_feature_height,
            patch_feature_width,
            feature_channels,
        )
        
        # Permute to [batch, num_patches_y, patch_feature_height, num_patches_x, patch_feature_width, feature_channels]
        # For batch=1: [1, 3, 8, 3, 8, 128]
        features = features.permute(0, 1, 3, 2, 4, 5).contiguous()
        
        # Reshape to [batch, num_patches_y * patch_feature_height, num_patches_x * patch_feature_width, feature_channels]
        # For batch=1: [1, 24, 24, 128]
        features = features.reshape(
            batch_size,
            num_patches_y * patch_feature_height,
            num_patches_x * patch_feature_width,
            feature_channels,
        )
        
        return features


class AggregationNetWrapper(nn.Module):
    """Wrapper for AggregationNet that's optimized for TFLite conversion.
    
    AggregationNet combines content and distortion features to produce quality scores.
    Input: Content features (batch, 8, 8, 128) + Distortion features (batch, 24, 24, 128)
    Output: Quality score in range [1, 5]
    """
    
    def __init__(self, model_path=None, eval_mode=True):
        super().__init__()
        if model_path is None:
            model_path = Path.home() / "work" / "models" / "UVQ" / "uvq1.5" / "aggregation_net.pth"
        
        self.aggregation_net = aggregationnet.AggregationNet(
            model_path=str(model_path),
            eval_mode=eval_mode,
            pretrained=True,
        )
        
        if eval_mode:
            self.eval()
    
    def forward(self, content_features, distortion_features):
        """
        Args:
            content_features: Tensor of shape (batch, 8, 8, 128)
            distortion_features: Tensor of shape (batch, 24, 24, 128)
        
        Returns:
            quality_score: Tensor of shape (batch, 1) in range [1, 5]
        """
        # AggregationNet.forward() expects NHWC format and permutes to NCHW internally
        # It returns a dict with 'uvq_1p5_features' key containing the quality score
        # The result is torch.mean(torch.stack(results), dim=0) which collapses the batch
        result = self.aggregation_net(content_features, distortion_features)
        
        # Extract the quality score from the dict
        quality_score = result['uvq_1p5_features']
        
        return quality_score


class DistortionNet3PatchWrapper(nn.Module):
    """Wrapper for DistortionNet that processes 3 patches and aggregates them horizontally.
    
    This model takes 3 patches as input and outputs a single row of aggregated features.
    Used for processing one row of a 3x3 patch grid at a time.
    """
    
    def __init__(self, model_path=None, eval_mode=True):
        super().__init__()
        if model_path is None:
            model_path = Path.home() / "work" / "models" / "UVQ" / "uvq1.5" / "distortion_net.pth"
        
        self.distortion_net = distortionnet.DistortionNet(
            model_path=str(model_path),
            eval_mode=eval_mode,
            pretrained=True,
        )
        
        if eval_mode:
            self.eval()
    
    def forward(self, video_patches_3):
        """
        Args:
            video_patches_3: Tensor of shape (3, height=360, width=640, channels=3)
                             3 patches in TensorFlow format [B, H, W, C]
        
        Returns:
            features: Tensor of shape (1, 8, 24, 128) - 3 patches aggregated horizontally
        """
        # Convert from TensorFlow format (B, H, W, C) to PyTorch format (B, C, H, W)
        video_patches_3 = video_patches_3.permute(0, 3, 1, 2).contiguous()
        
        # Process patches through DistortionNet core
        # Output is [3, 8, 8, 128] (NHWC) due to PermuteLayerNHWC in the model
        patch_features = self.distortion_net.model(video_patches_3)
        
        # Aggregate 3 patches horizontally using the correct PyTorch logic
        # Input: [3, 8, 8, 128]
        # We want to arrange them as: [1, 8, 24, 128] (one row of 3 patches)
        
        batch_size = 1
        num_patches_x = 3  # 3 patches horizontally
        patch_feature_height = 8
        patch_feature_width = 8
        feature_channels = 128
        
        # Reshape to [1, 1, 3, 8, 8, 128] to match PyTorch's structure
        # (batch=1, num_patches_y=1, num_patches_x=3, patch_h=8, patch_w=8, channels=128)
        features = patch_features.reshape(
            batch_size,
            1,  # num_patches_y = 1 (single row)
            num_patches_x,
            patch_feature_height,
            patch_feature_width,
            feature_channels,
        )
        
        # Permute to [1, 1, 8, 3, 8, 128]
        features = features.permute(0, 1, 3, 2, 4, 5).contiguous()
        
        # Reshape to [1, 8, 24, 128]
        features = features.reshape(
            batch_size,
            1 * patch_feature_height,  # 8
            num_patches_x * patch_feature_width,  # 24
            feature_channels,
        )
        
        return features


class UVQ1p5Core(nn.Module):
    """Complete UVQ 1.5 model combining all three networks.
    
    This is the full pipeline for video quality assessment.
    """
    
    def __init__(self, content_model_path=None, distortion_model_path=None, 
                 aggregation_model_path=None, eval_mode=True):
        super().__init__()
        
        self.content_net = ContentNetWrapper(content_model_path, eval_mode)
        self.distortion_net = DistortionNetWrapper(distortion_model_path, eval_mode)
        self.aggregation_net = AggregationNetWrapper(aggregation_model_path, eval_mode)
        
        if eval_mode:
            self.eval()
    
    def forward(self, video_frame_256, video_patches_1080):
        """
        Args:
            video_frame_256: Tensor of shape (batch, 256, 256, 3) for content analysis
            video_patches_1080: Tensor of shape (batch * 9, 360, 640, 3) for distortion analysis
        
        Returns:
            quality_score: Tensor of shape (batch, 1) in range [1, 5]
        """
        # Extract features
        content_features = self.content_net(video_frame_256)
        distortion_features = self.distortion_net(video_patches_1080)
        
        # Aggregate to quality score
        quality_score = self.aggregation_net(content_features, distortion_features)
        
        return quality_score


def create_content_net(model_path=None):
    """Create ContentNet model for TFLite conversion."""
    model = ContentNetWrapper(model_path=model_path, eval_mode=True)
    model.eval()
    return model


def create_distortion_net(model_path=None):
    """Create DistortionNet model for TFLite conversion."""
    model = DistortionNetWrapper(model_path=model_path, eval_mode=True)
    model.eval()
    return model


def create_aggregation_net(model_path=None):
    """Create AggregationNet model for TFLite conversion."""
    model = AggregationNetWrapper(model_path=model_path, eval_mode=True)
    model.eval()
    return model


def create_distortion_net_3patch(model_path=None):
    """Create DistortionNet 3-patch model for TFLite conversion."""
    model = DistortionNet3PatchWrapper(model_path=model_path, eval_mode=True)
    model.eval()
    return model


def create_uvq1p5_full(content_path=None, distortion_path=None, aggregation_path=None):
    """Create full UVQ 1.5 model for TFLite conversion."""
    model = UVQ1p5Core(content_path, distortion_path, aggregation_path, eval_mode=True)
    model.eval()
    return model

