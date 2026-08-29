"""Device selection and mixed-precision policy for SAM 3 inference."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

import torch


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    """Resolve and validate CUDA, Apple MPS, or CPU inference devices."""
    requested = "auto" if device is None else str(device).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    if resolved.type not in {"cuda", "mps", "cpu"}:
        raise ValueError(
            f"Unsupported device '{requested}'; choose auto, cuda, mps, or cpu"
        )
    return resolved


def autocast_dtype(device: torch.device) -> torch.dtype | None:
    """Return the inference autocast dtype selected for a resolved device."""
    if device.type == "cuda":
        with torch.cuda.device(device):
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps" and torch.amp.is_autocast_available("mps"):
        return torch.float16
    return None


def inference_autocast(device: torch.device) -> AbstractContextManager:
    """Use backend-appropriate mixed precision, keeping CPU inference in FP32."""
    dtype = autocast_dtype(device)
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def precision_name(device: torch.device) -> str:
    """Return a short user-facing description of the inference precision."""
    dtype = autocast_dtype(device)
    return "float32" if dtype is None else str(dtype).removeprefix("torch.")


__all__ = [
    "autocast_dtype",
    "inference_autocast",
    "precision_name",
    "resolve_device",
]
