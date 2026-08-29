"""Client for the local SAM3 Gradio text-segmentation endpoint."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast
from uuid import uuid4

import numpy as np
from gradio_client import Client, handle_file
from gradio_client.exceptions import AppError
from PIL import Image, UnidentifiedImageError

from visculpt.bridge import JsonValue

from .config import Sam3GradioConfig
from .errors import (
    Sam3InputError,
    Sam3OutputError,
    Sam3ResponseError,
    Sam3ServiceError,
    Sam3TransportError,
)

SEGMENT_API_NAME = "/segment"


class GradioClientProtocol(Protocol):
    """Small Gradio client surface used by the integration."""

    def predict(self, *args: object, **kwargs: object) -> object:
        """Call one named Gradio endpoint."""

    def close(self) -> None:
        """Release client workers and heartbeat resources."""


type GradioClientFactory = Callable[..., GradioClientProtocol]
type FileHandler = Callable[[str | Path], object]


@dataclass(frozen=True, slots=True)
class Sam3SegmentationResult:
    """Normalized result returned by the SAM3 Gradio service."""

    service_url: str
    image_path: str
    prompt: str
    confidence_threshold: float
    overlay_opacity: float
    overlay_path: str
    mask_path: str
    instance_masks_path: str
    metadata: dict[str, JsonValue]
    metadata_path: str | None = None

    def as_payload(self) -> dict[str, JsonValue]:
        """Return a JSON-compatible Tool result."""
        return {
            "service": {
                "url": self.service_url,
                "api_name": SEGMENT_API_NAME,
            },
            "input": {
                "image_path": self.image_path,
                "prompt": self.prompt,
                "confidence_threshold": self.confidence_threshold,
                "overlay_opacity": self.overlay_opacity,
            },
            "result": {
                "overlay_path": self.overlay_path,
                "mask_path": self.mask_path,
                "instance_masks_path": self.instance_masks_path,
                "metadata": self.metadata,
                "metadata_path": self.metadata_path,
            },
        }


class Sam3GradioClient:
    """Call the loopback SAM3 Gradio service and normalize its outputs."""

    def __init__(
        self,
        config: Sam3GradioConfig | None = None,
        *,
        client_factory: GradioClientFactory | None = None,
        file_handler: FileHandler | None = None,
    ) -> None:
        self.config = config or Sam3GradioConfig.from_env()
        self._client_factory = client_factory or Client
        self._file_handler = file_handler or handle_file

    def segment(
        self,
        *,
        image_path: str | Path,
        prompt: str,
        confidence_threshold: float = 0.5,
        overlay_opacity: float = 0.45,
        output_dir: str | Path | None = None,
    ) -> Sam3SegmentationResult:
        """Segment an image using a short text prompt."""
        resolved_image = _resolve_input_image(image_path)
        normalized_prompt = _validate_prompt(prompt)
        confidence = _bounded_number(
            confidence_threshold,
            name="confidence_threshold",
            minimum=0.05,
            maximum=0.95,
        )
        opacity = _bounded_number(
            overlay_opacity,
            name="overlay_opacity",
            minimum=0.0,
            maximum=1.0,
        )
        resolved_output_dir = _resolve_output_dir(output_dir)
        metadata_path: Path | None = None
        if resolved_output_dir is None:
            raw_result = self._predict(
                image_path=resolved_image,
                prompt=normalized_prompt,
                confidence_threshold=confidence,
                overlay_opacity=opacity,
                download_dir=None,
            )
            (
                overlay_path,
                mask_path,
                instance_masks_path,
                metadata,
            ) = _parse_response(raw_result)
        else:
            with tempfile.TemporaryDirectory(
                prefix="agentic-geometry-sam3-"
            ) as temporary_directory:
                raw_result = self._predict(
                    image_path=resolved_image,
                    prompt=normalized_prompt,
                    confidence_threshold=confidence,
                    overlay_opacity=opacity,
                    download_dir=Path(temporary_directory),
                )
                (
                    overlay_path,
                    mask_path,
                    instance_masks_path,
                    metadata,
                ) = _parse_response(raw_result)
                (
                    overlay_path,
                    mask_path,
                    instance_masks_path,
                    metadata_path,
                ) = _persist_outputs(
                    overlay_path=overlay_path,
                    mask_path=mask_path,
                    instance_masks_path=instance_masks_path,
                    metadata=metadata,
                    output_dir=resolved_output_dir,
                    image_stem=resolved_image.stem,
                    prompt=normalized_prompt,
                )

        return Sam3SegmentationResult(
            service_url=self.config.service_url,
            image_path=str(resolved_image),
            prompt=normalized_prompt,
            confidence_threshold=confidence,
            overlay_opacity=opacity,
            overlay_path=str(overlay_path),
            mask_path=str(mask_path),
            instance_masks_path=str(instance_masks_path),
            metadata=metadata,
            metadata_path=(
                str(metadata_path) if metadata_path is not None else None
            ),
        )

    def _predict(
        self,
        *,
        image_path: Path,
        prompt: str,
        confidence_threshold: float,
        overlay_opacity: float,
        download_dir: Path | None,
    ) -> object:
        client = self._create_client(download_dir)
        try:
            return client.predict(
                image=self._file_handler(image_path),
                prompt=prompt,
                confidence_threshold=confidence_threshold,
                overlay_opacity=overlay_opacity,
                api_name=SEGMENT_API_NAME,
            )
        except AppError as error:
            raise Sam3ServiceError(
                f"SAM3 inference failed: {error}"
            ) from error
        except Exception as error:
            raise Sam3ServiceError(
                f"SAM3 Gradio request failed: {error}"
            ) from error
        finally:
            _close_client(client)

    def _create_client(
        self,
        download_dir: Path | None,
    ) -> GradioClientProtocol:
        kwargs: dict[str, object] = {
            "verbose": False,
            "analytics_enabled": False,
            "httpx_kwargs": {"timeout": self.config.timeout},
        }
        if download_dir is not None:
            kwargs["download_files"] = download_dir
        try:
            return self._client_factory(
                self.config.service_url,
                **kwargs,
            )
        except Exception as error:
            raise Sam3TransportError(
                "Cannot connect to SAM3 Gradio service at "
                f"{self.config.service_url}: {error}"
            ) from error


def _resolve_input_image(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)):
        raise Sam3InputError("image_path must be a string or Path")
    raw_value = str(value).strip()
    if not raw_value:
        raise Sam3InputError("image_path must not be empty")
    path = Path(os.path.expandvars(os.path.expanduser(raw_value)))
    if not path.is_absolute():
        raise Sam3InputError("image_path must be absolute")
    resolved = path.resolve()
    if not resolved.is_file():
        raise Sam3InputError(f"image_path is not a file: {resolved}")
    return resolved


def _resolve_output_dir(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, (str, Path)):
        raise Sam3InputError("output_dir must be a string or Path")
    raw_value = str(value).strip()
    if not raw_value:
        raise Sam3InputError("output_dir must not be empty")
    path = Path(os.path.expandvars(os.path.expanduser(raw_value)))
    if not path.is_absolute():
        raise Sam3InputError("output_dir must be absolute")
    resolved = path.resolve()
    if resolved.exists() and not resolved.is_dir():
        raise Sam3InputError(f"output_dir is not a directory: {resolved}")
    try:
        resolved.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise Sam3OutputError(
            f"Cannot create output directory {resolved}: {error}"
        ) from error
    return resolved


def _validate_prompt(value: str) -> str:
    if not isinstance(value, str):
        raise Sam3InputError("prompt must be a string")
    normalized = value.strip()
    if not normalized:
        raise Sam3InputError("prompt must not be empty")
    return normalized


def _bounded_number(
    value: float,
    *,
    name: str,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Sam3InputError(f"{name} must be a number")
    number = float(value)
    if not minimum <= number <= maximum:
        raise Sam3InputError(
            f"{name} must be between {minimum:g} and {maximum:g}"
        )
    return number


def _parse_response(
    value: object,
) -> tuple[Path, Path, Path, dict[str, JsonValue]]:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise Sam3ResponseError(
            "SAM3 service response must contain overlay, semantic mask, "
            "per-instance mask archive, and metadata. Restart the SAM3 "
            "service after updating sam3-inference-service."
        )
    overlay_path = _response_file(value[0], "overlay")
    mask_path = _response_file(value[1], "mask")
    instance_masks_path = _response_file(
        value[2],
        "per-instance mask archive",
    )
    metadata = value[3]
    if not isinstance(metadata, dict):
        raise Sam3ResponseError("SAM3 metadata must be a JSON object")
    try:
        json.dumps(metadata, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise Sam3ResponseError(
            f"SAM3 metadata is not valid JSON: {error}"
        ) from error
    typed_metadata = cast(dict[str, JsonValue], metadata)
    if (
        typed_metadata.get("protocol_version")
        != "sam3-gradio-segmentation/v2"
    ):
        raise Sam3ResponseError(
            "SAM3 service protocol must be sam3-gradio-segmentation/v2. "
            "Restart the updated SAM3 inference service."
        )
    _validate_instance_masks_archive(
        instance_masks_path,
        semantic_mask_path=mask_path,
        metadata=typed_metadata,
    )
    return overlay_path, mask_path, instance_masks_path, typed_metadata


def _response_file(value: object, label: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise Sam3ResponseError(f"SAM3 {label} output is not a file path")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise Sam3ResponseError(
            f"SAM3 {label} output file does not exist: {path}"
        )
    return path


def _persist_outputs(
    *,
    overlay_path: Path,
    mask_path: Path,
    instance_masks_path: Path,
    metadata: dict[str, JsonValue],
    output_dir: Path,
    image_stem: str,
    prompt: str,
) -> tuple[Path, Path, Path, Path]:
    tag = uuid4().hex[:8]
    prefix = (
        f"{_slug(image_stem, 'image')}-"
        f"{_slug(prompt, 'segment')}-{tag}"
    )
    overlay_destination = output_dir / (
        f"{prefix}-overlay{overlay_path.suffix or '.png'}"
    )
    mask_destination = output_dir / (
        f"{prefix}-mask{mask_path.suffix or '.png'}"
    )
    instance_masks_destination = output_dir / f"{prefix}-instances.npz"
    metadata_destination = output_dir / f"{prefix}-metadata.json"
    try:
        shutil.copy2(overlay_path, overlay_destination)
        shutil.copy2(mask_path, mask_destination)
        shutil.copy2(instance_masks_path, instance_masks_destination)
        metadata_destination.write_text(
            json.dumps(
                metadata,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise Sam3OutputError(
            f"Cannot persist SAM3 outputs in {output_dir}: {error}"
        ) from error
    return (
        overlay_destination,
        mask_destination,
        instance_masks_destination,
        metadata_destination,
    )


def _validate_instance_masks_archive(
    path: Path,
    *,
    semantic_mask_path: Path,
    metadata: dict[str, JsonValue],
) -> None:
    """Reject stale or inconsistent SAM3 instance-mask responses."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            if archive.files != ["masks"]:
                raise Sam3ResponseError(
                    "SAM3 instance archive must contain only the 'masks' array"
                )
            masks = np.asarray(archive["masks"])
    except Sam3ResponseError:
        raise
    except (OSError, ValueError, KeyError) as error:
        raise Sam3ResponseError(
            f"Cannot read SAM3 instance-mask archive: {error}"
        ) from error
    if masks.ndim != 3:
        raise Sam3ResponseError(
            "SAM3 instance masks must have shape (instance, height, width)"
        )
    if masks.dtype not in (np.dtype(np.bool_), np.dtype(np.uint8)):
        raise Sam3ResponseError(
            "SAM3 instance masks must use bool or uint8 values"
        )
    if masks.dtype == np.dtype(np.uint8):
        unique = np.unique(masks)
        if not np.all(np.isin(unique, (0, 1, 255))):
            raise Sam3ResponseError(
                "SAM3 uint8 instance masks contain non-binary values"
            )

    instance_count = metadata.get("instance_count")
    if (
        isinstance(instance_count, bool)
        or not isinstance(instance_count, int)
        or instance_count < 0
    ):
        raise Sam3ResponseError(
            "SAM3 metadata instance_count must be a non-negative integer"
        )
    instances = metadata.get("instances")
    if not isinstance(instances, list) or len(instances) != instance_count:
        raise Sam3ResponseError(
            "SAM3 metadata instances must match instance_count"
        )
    if masks.shape[0] != instance_count:
        raise Sam3ResponseError(
            "SAM3 instance archive count does not match metadata"
        )
    for expected_index, instance in enumerate(instances):
        if not isinstance(instance, dict):
            raise Sam3ResponseError(
                "SAM3 instance metadata entries must be objects"
            )
        if instance.get("instance_index") != expected_index:
            raise Sam3ResponseError(
                "SAM3 instance metadata indices must be contiguous and ordered"
            )
        foreground_pixels = instance.get("foreground_pixels")
        actual_foreground = int(np.count_nonzero(masks[expected_index]))
        if (
            isinstance(foreground_pixels, bool)
            or not isinstance(foreground_pixels, int)
            or foreground_pixels != actual_foreground
        ):
            raise Sam3ResponseError(
                "SAM3 instance foreground metadata does not match mask data"
            )

    try:
        with Image.open(semantic_mask_path) as image:
            semantic_mask = np.asarray(image.convert("L"), dtype=np.uint8)
    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ) as error:
        raise Sam3ResponseError(
            f"Cannot read SAM3 semantic mask: {semantic_mask_path}"
        ) from error
    if masks.shape[1:] != semantic_mask.shape:
        raise Sam3ResponseError(
            "SAM3 instance-mask dimensions do not match semantic mask"
        )
    instance_union = np.any(masks > 0, axis=0)
    semantic_union = semantic_mask >= 128
    if not np.array_equal(instance_union, semantic_union):
        raise Sam3ResponseError(
            "SAM3 semantic mask is inconsistent with per-instance masks"
        )


def _slug(value: str, fallback: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-_").lower()
    return slug[:48] or fallback


def _close_client(client: GradioClientProtocol) -> None:
    try:
        client.close()
    except Exception:
        pass
