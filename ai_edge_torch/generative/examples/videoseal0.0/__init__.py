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
VideoSeal 0.0 TFLite Conversion Module.

This module provides tools to convert VideoSeal 0.0 models from PyTorch to
TensorFlow Lite format for deployment on mobile and edge devices.

VideoSeal 0.0 is the baseline watermarking model with 96-bit capacity.

Main Components:
- videoseal00_models: Model wrappers for TFLite conversion
- convert_detector_to_tflite: Convert detector to TFLite
- convert_embedder_to_tflite: Convert embedder to TFLite
- verify_detector_tflite: Verify TFLite model accuracy
- example_usage: Usage examples

Example:
    >>> from videoseal00_models import create_detector, create_embedder
    >>> 
    >>> # Create PyTorch wrappers
    >>> detector = create_detector("videoseal_0.0")
    >>> embedder = create_embedder("videoseal_0.0")
    >>> 
    >>> # Convert to TFLite using the conversion scripts
    >>> # python convert_detector_to_tflite.py --output_dir ./tflite_models
    >>> # python convert_embedder_to_tflite.py --output_dir ./tflite_models

For more information, see README.md.
"""

__version__ = "1.0.0"
__author__ = "The AI Edge Torch Authors"

from videoseal00_models import (
    VideoSeal00DetectorWrapper,
    VideoSeal00EmbedderWrapper,
    create_detector,
    create_embedder,
)

__all__ = [
    "VideoSeal00DetectorWrapper",
    "VideoSeal00EmbedderWrapper",
    "create_detector",
    "create_embedder",
]
