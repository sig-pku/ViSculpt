# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import os

# Let unsupported MPS operators run on CPU. Users can explicitly set this to 0
# before importing sam3 when they prefer an immediate error over fallback.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from .device import resolve_device
from .inference import Sam3TextSegmenter, SegmentationResult
from .model_builder import build_sam3_image_model, resolve_sam3_checkpoint

__version__ = "1.0.0"

__all__ = [
    "Sam3TextSegmenter",
    "SegmentationResult",
    "build_sam3_image_model",
    "resolve_device",
    "resolve_sam3_checkpoint",
]
