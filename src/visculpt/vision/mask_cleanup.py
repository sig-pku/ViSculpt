"""Brush-independent cleanup for semantic segmentation masks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

from visculpt.bridge import JsonValue

type MaskArray = NDArray[np.uint8]

_MASK_THRESHOLD = 128
_MINIMUM_COMPONENT_RATIO = 0.00005


class MaskCleanupError(ValueError):
    """Raised when a semantic mask cannot be cleaned safely."""


@dataclass(frozen=True, slots=True)
class CleanedMaskResult:
    """Persisted cleaned mask and deterministic cleanup metadata."""

    cleaned_mask_path: str
    cleaning: dict[str, JsonValue]

    def as_payload(self) -> dict[str, JsonValue]:
        """Return the fragment added to a segmentation result."""
        return {
            "cleaned_mask_path": self.cleaned_mask_path,
            "cleaning": self.cleaning,
        }


@dataclass(frozen=True, slots=True)
class CleanedInstanceMaskResult:
    """Cleanup result for one SAM3 instance, including rejected noise."""

    instance_index: int
    cleaned_mask_path: str | None
    cleaning: dict[str, JsonValue]

    def as_payload(self) -> dict[str, JsonValue]:
        """Return one JSON-compatible instance cleanup record."""
        return {
            "instance_index": self.instance_index,
            "cleanup_status": (
                "accepted" if self.cleaned_mask_path is not None else "rejected"
            ),
            "cleaned_mask_path": self.cleaned_mask_path,
            "cleaning": self.cleaning,
        }


def clean_segmentation_mask(
    mask_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> CleanedMaskResult:
    """Clean one SAM3 mask without depending on Sculpt settings."""
    resolved_mask = _resolve_mask_path(mask_path)
    grayscale = _load_grayscale_mask(resolved_mask)
    binary = np.where(
        grayscale >= _MASK_THRESHOLD,
        255,
        0,
    ).astype(np.uint8)
    try:
        cleaned, metadata = _clean_mask(binary)
    except cv2.error as error:
        raise MaskCleanupError(f"OpenCV mask cleanup failed: {error}") from error
    if not np.any(cleaned):
        raise MaskCleanupError(
            "SAM3 mask has no foreground after deterministic cleanup"
        )

    directory = _resolve_output_dir(output_dir, resolved_mask)
    path = _cleaned_mask_path(resolved_mask, directory)
    _write_mask(path, cleaned)
    return CleanedMaskResult(
        cleaned_mask_path=str(path),
        cleaning=metadata,
    )


def render_cleaned_mask_overlay(
    image_path: str | Path,
    cleaned_mask_path: str | Path,
    *,
    opacity: float,
    output_dir: str | Path | None = None,
) -> str:
    """Overlay one cleaned semantic mask on its source screenshot."""
    if not 0.0 <= opacity <= 1.0:
        raise MaskCleanupError("overlay opacity must be between 0 and 1")
    resolved_image = Path(
        os.path.expandvars(os.path.expanduser(str(image_path)))
    ).resolve()
    if not resolved_image.is_file():
        raise MaskCleanupError(f"source image is not a file: {resolved_image}")
    resolved_mask = _resolve_mask_path(cleaned_mask_path)
    mask = _load_grayscale_mask(resolved_mask)
    try:
        with Image.open(resolved_image) as source:
            base = source.convert("RGBA")
    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ) as error:
        raise MaskCleanupError(
            f"cannot read source image for cleaned-mask overlay: {resolved_image}"
        ) from error
    if base.size != (mask.shape[1], mask.shape[0]):
        raise MaskCleanupError(
            "cleaned mask dimensions do not match the source image"
        )

    alpha = np.rint(mask.astype(np.float32) * opacity).astype(np.uint8)
    tint = Image.new("RGBA", base.size, (46, 204, 113, 255))
    tint.putalpha(Image.fromarray(alpha, mode="L"))
    rendered = Image.alpha_composite(base, tint).convert("RGB")
    directory = _resolve_output_dir(output_dir, resolved_mask)
    output_path = directory / f"{resolved_mask.stem}-overlay.png"
    _write_png(output_path, rendered)
    return str(output_path)


def clean_instance_mask_archive(
    archive_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> tuple[CleanedInstanceMaskResult, ...]:
    """Clean every SAM3 instance independently without merging candidates."""
    resolved_archive = _resolve_mask_path(archive_path)
    try:
        with np.load(resolved_archive, allow_pickle=False) as archive:
            if archive.files != ["masks"]:
                raise MaskCleanupError(
                    "SAM3 instance archive must contain only the 'masks' array"
                )
            masks = np.asarray(archive["masks"])
    except MaskCleanupError:
        raise
    except (OSError, ValueError, KeyError) as error:
        raise MaskCleanupError(
            f"cannot read SAM3 instance-mask archive: {resolved_archive}"
        ) from error
    if masks.ndim != 3:
        raise MaskCleanupError(
            "SAM3 instance masks must have shape (instance, height, width)"
        )
    if masks.dtype not in (np.dtype(np.bool_), np.dtype(np.uint8)):
        raise MaskCleanupError(
            "SAM3 instance masks must use bool or uint8 values"
        )

    base_directory = _resolve_output_dir(output_dir, resolved_archive)
    directory = base_directory / f"{resolved_archive.stem}-cleaned-instances"
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise MaskCleanupError(
            f"cannot create instance-mask output directory: {directory}"
        ) from error

    results: list[CleanedInstanceMaskResult] = []
    for index, source in enumerate(masks):
        binary = np.where(source > 0, 255, 0).astype(np.uint8)
        try:
            cleaned, metadata = _clean_mask(binary)
        except cv2.error as error:
            raise MaskCleanupError(
                f"OpenCV cleanup failed for SAM3 instance {index}: {error}"
            ) from error
        path: Path | None = None
        if np.any(cleaned):
            path = directory / f"instance-{index:03d}-cleaned-mask.png"
            _write_mask(path, cleaned)
        results.append(
            CleanedInstanceMaskResult(
                instance_index=index,
                cleaned_mask_path=str(path) if path is not None else None,
                cleaning=metadata,
            )
        )
    return tuple(results)


def _clean_mask(mask: MaskArray) -> tuple[MaskArray, dict[str, JsonValue]]:
    before_pixels = int(np.count_nonzero(mask))
    before_regions = _foreground_component_count(mask)
    height, width = mask.shape
    close_radius = max(1, min(3, round(min(width, height) * 0.002)))
    minimum_area = max(
        8,
        round(mask.size * _MINIMUM_COMPONENT_RATIO),
    )
    kernel_size = close_radius * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    filtered = _remove_small_components(
        closed,
        minimum_area=minimum_area,
    )
    filled = _fill_small_holes(
        filtered,
        maximum_hole_area=minimum_area,
    )
    after_pixels = int(np.count_nonzero(filled))
    return filled, {
        "algorithm": "semantic-mask-cleanup/v2",
        "operations": [
            "grayscale_threshold",
            "tiny_elliptic_closing",
            "8_connected_area_opening",
            "bounded_hole_filling",
        ],
        "mask_threshold": _MASK_THRESHOLD,
        "close_radius_pixels": close_radius,
        "minimum_component_area_pixels": minimum_area,
        "maximum_filled_hole_area_pixels": minimum_area,
        "foreground_pixels_before": before_pixels,
        "foreground_pixels_after": after_pixels,
        "foreground_pixel_delta": after_pixels - before_pixels,
        "region_count_before": before_regions,
        "region_count_after": _foreground_component_count(filled),
        "width": width,
        "height": height,
    }


def _remove_small_components(
    mask: MaskArray,
    *,
    minimum_area: int,
) -> MaskArray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )
    keep = np.zeros(count, dtype=np.uint8)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area:
            keep[label] = 255
    return keep[labels]


def _fill_small_holes(
    mask: MaskArray,
    *,
    maximum_hole_area: int,
) -> MaskArray:
    inverse = cv2.bitwise_not(mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverse,
        connectivity=8,
    )
    border_labels = set(int(value) for value in labels[0, :])
    border_labels.update(int(value) for value in labels[-1, :])
    border_labels.update(int(value) for value in labels[:, 0])
    border_labels.update(int(value) for value in labels[:, -1])
    result = mask.copy()
    for label in range(1, count):
        if label in border_labels:
            continue
        if int(stats[label, cv2.CC_STAT_AREA]) <= maximum_hole_area:
            result[labels == label] = 255
    return result


def _foreground_component_count(mask: MaskArray) -> int:
    count, _ = cv2.connectedComponents(mask, connectivity=8)
    return max(0, int(count) - 1)


def _load_grayscale_mask(path: Path) -> MaskArray:
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("L"), dtype=np.uint8)
    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ) as error:
        raise MaskCleanupError(f"cannot read SAM3 mask image: {path}") from error


def _resolve_mask_path(value: str | Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()
    if not path.is_file():
        raise MaskCleanupError(f"SAM3 mask is not a file: {path}")
    return path


def _resolve_output_dir(
    output_dir: str | Path | None,
    mask_path: Path,
) -> Path:
    directory = (
        mask_path.parent
        if output_dir is None
        else Path(
            os.path.expandvars(os.path.expanduser(str(output_dir)))
        ).resolve()
    )
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise MaskCleanupError(
            f"cannot create mask-cleanup output directory: {directory}"
        ) from error
    return directory


def _cleaned_mask_path(mask_path: Path, output_dir: Path) -> Path:
    stem = mask_path.stem
    if stem.endswith("-mask"):
        stem = stem[: -len("-mask")]
    return output_dir / f"{stem}-cleaned-mask.png"


def _write_mask(path: Path, mask: MaskArray) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        Image.fromarray(mask).save(temporary, format="PNG")
        os.replace(temporary, path)
    except OSError as error:
        raise MaskCleanupError(f"cannot write cleaned mask: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _write_png(path: Path, image: Image.Image) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        image.save(temporary, format="PNG")
        os.replace(temporary, path)
    except OSError as error:
        raise MaskCleanupError(
            f"cannot write cleaned-mask overlay: {path}"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)
