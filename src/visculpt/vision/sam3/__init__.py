"""Client boundary for the local SAM3 Gradio inference service."""

from .client import Sam3GradioClient, Sam3SegmentationResult
from .config import Sam3GradioConfig
from .errors import (
    Sam3ClientError,
    Sam3InputError,
    Sam3OutputError,
    Sam3ResponseError,
    Sam3ServiceError,
    Sam3TransportError,
)

__all__ = [
    "Sam3ClientError",
    "Sam3GradioClient",
    "Sam3GradioConfig",
    "Sam3InputError",
    "Sam3OutputError",
    "Sam3ResponseError",
    "Sam3SegmentationResult",
    "Sam3ServiceError",
    "Sam3TransportError",
]
