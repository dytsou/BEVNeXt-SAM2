# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from hydra import initialize_config_module
from hydra.core.global_hydra import GlobalHydra

if not GlobalHydra.instance().is_initialized():
    initialize_config_module("sam2", version_base="1.2")

"""SAM 2 module for segmentation"""

from .sam2_image_predictor import SAM2ImagePredictor
from .sam2_video_predictor import SAM2VideoPredictor
from .build_sam import build_sam2, build_sam2_video_predictor
from .automatic_mask_generator import SAM2AutomaticMaskGenerator

__version__ = '1.0.0'
