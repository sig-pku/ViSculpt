"""Deterministic mask remapping for orthographic viewport ROI focus."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from .mask_cleanup import MaskCleanupError, render_cleaned_mask_overlay


class ViewportMaskTransformError(ValueError):
    """Raised when one focused mask cannot be produced safely."""


@dataclass(frozen=True, slots=True)
class FocusedMaskResult:
    """Persisted mask and metadata in the focused screenshot coordinates."""

    schema_version: str
    cleaned_mask_path: str
    cleaned_overlay_path: str
    metadata_path: str
    source_roi: dict[str, int]
    focused_roi: dict[str, int]
    foreground_pixels: int
    affine_matrix: list[list[float]]

    def as_payload(self) -> dict[str, object]:
        """Return a JSON-serializable payload."""
        return asdict(self)


def cleaned_mask_roi(mask_path: str | Path) -> dict[str, int]:
    """Return an exclusive foreground bounding box in screenshot pixels."""
    mask = _read_binary_mask(mask_path)
    rows, columns = np.nonzero(mask)
    if columns.size == 0:
        raise ViewportMaskTransformError("cleaned mask has no foreground")
    return {
        "x_min": int(columns.min()),
        "y_min": int(rows.min()),
        "x_max": int(columns.max()) + 1,
        "y_max": int(rows.max()) + 1,
    }


def warp_cleaned_mask_to_focused_view(
    *,
    source_mask_path: str | Path,
    focused_image_path: str | Path,
    focus_result: Mapping[str, object],
    output_dir: str | Path,
    overlay_opacity: float,
) -> FocusedMaskResult:
    """Affine-warp an existing cleaned mask without another SAM3 call."""
    mask = _read_binary_mask(source_mask_path)
    image_path = Path(focused_image_path).expanduser().resolve()
    if not image_path.is_file():
        raise ViewportMaskTransformError(
            f"focused screenshot is not a file: {image_path}"
        )
    try:
        with Image.open(image_path) as image:
            target_size = image.size
    except (OSError, UnidentifiedImageError) as error:
        raise ViewportMaskTransformError(
            f"cannot read focused screenshot: {image_path}"
        ) from error

    transform = focus_result.get("image_transform")
    if not isinstance(transform, Mapping):
        raise ViewportMaskTransformError(
            "ROI focus result is missing image_transform"
        )
    source = _dimensions(transform.get("source"), label="source")
    target = _dimensions(transform.get("target"), label="target")
    if (mask.shape[1], mask.shape[0]) != source:
        raise ViewportMaskTransformError(
            "cleaned mask dimensions do not match focus source dimensions"
        )
    if target_size != target:
        raise ViewportMaskTransformError(
            "focused screenshot dimensions do not match focus target dimensions"
        )
    matrix = _affine_matrix(transform.get("matrix"))
    try:
        focused = cv2.warpAffine(
            mask,
            np.asarray(matrix, dtype=np.float64),
            target,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    except cv2.error as error:
        raise ViewportMaskTransformError(
            f"OpenCV could not transform the cleaned mask: {error}"
        ) from error
    focused = np.where(focused > 0, 255, 0).astype(np.uint8)
    foreground = int(np.count_nonzero(focused))
    if foreground == 0:
        raise ViewportMaskTransformError(
            "focused cleaned mask has no foreground after affine transform"
        )

    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    mask_path = directory / "focused-cleaned-mask.png"
    if not cv2.imwrite(str(mask_path), focused):
        raise ViewportMaskTransformError(
            f"cannot write focused cleaned mask: {mask_path}"
        )
    try:
        overlay_path = render_cleaned_mask_overlay(
            image_path,
            mask_path,
            opacity=overlay_opacity,
            output_dir=directory,
        )
    except MaskCleanupError as error:
        raise ViewportMaskTransformError(str(error)) from error
    result = FocusedMaskResult(
        schema_version="focused-cleaned-mask/v1",
        cleaned_mask_path=str(mask_path),
        cleaned_overlay_path=overlay_path,
        metadata_path=str(directory / "focused-mask-transform.json"),
        source_roi=cleaned_mask_roi(source_mask_path),
        focused_roi=cleaned_mask_roi(mask_path),
        foreground_pixels=foreground,
        affine_matrix=matrix,
    )
    Path(result.metadata_path).write_text(
        json.dumps(result.as_payload(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def _read_binary_mask(path_value: str | Path) -> np.ndarray:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise ViewportMaskTransformError(f"mask is not a file: {path}")
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ViewportMaskTransformError(f"cannot read mask: {path}")
    return np.where(mask >= 128, 255, 0).astype(np.uint8)


def _dimensions(value: object, *, label: str) -> tuple[int, int]:
    if not isinstance(value, Mapping):
        raise ViewportMaskTransformError(
            f"image_transform.{label} must be an object"
        )
    width = value.get("width")
    height = value.get("height")
    origin = value.get("origin")
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or width < 2
        or isinstance(height, bool)
        or not isinstance(height, int)
        or height < 2
        or origin != "TOP_LEFT"
    ):
        raise ViewportMaskTransformError(
            f"image_transform.{label} dimensions are invalid"
        )
    return width, height


def _affine_matrix(value: object) -> list[list[float]]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise ViewportMaskTransformError("affine matrix must be 2x3")
    matrix: list[list[float]] = []
    for row in value:
        if (
            not isinstance(row, Sequence)
            or isinstance(row, (str, bytes))
            or len(row) != 3
        ):
            raise ViewportMaskTransformError("affine matrix must be 2x3")
        parsed: list[float] = []
        for item in row:
            if (
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
            ):
                raise ViewportMaskTransformError(
                    "affine matrix must contain finite numbers"
                )
            parsed.append(float(item))
        matrix.append(parsed)
    determinant = matrix[0][0] * matrix[1][1] - (
        matrix[0][1] * matrix[1][0]
    )
    if abs(determinant) < 1e-12:
        raise ViewportMaskTransformError("affine matrix is singular")
    return matrix
