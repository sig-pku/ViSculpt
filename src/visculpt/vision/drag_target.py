"""Deterministic component binding for one localized Drag target."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

from visculpt.bridge import JsonValue

type MaskArray = NDArray[np.uint8]


class DragTargetBindingError(ValueError):
    """Raised when an anchor cannot be bound to one semantic component."""


@dataclass(frozen=True, slots=True)
class DragTargetBindingResult:
    """Persisted component-only mask and identity-validation overlay."""

    component_mask_path: str
    anchor_overlay_path: str
    component_label: int
    component_area_pixels: int
    bbox_xyxy: tuple[int, int, int, int]
    centroid_x: float
    centroid_y: float
    anchor_x: int
    anchor_y: int

    def as_payload(self) -> dict[str, JsonValue]:
        """Return a JSON-safe target binding stored in LangGraph State."""
        return {
            "algorithm": {
                "name": "anchor-connected-component-binding/v1",
                "deterministic": True,
                "identity_contract": (
                    "the exact connected semantic component containing "
                    "the validated Drag anchor"
                ),
            },
            "component_mask_path": self.component_mask_path,
            "anchor_overlay_path": self.anchor_overlay_path,
            "component_label": self.component_label,
            "component_area_pixels": self.component_area_pixels,
            "bbox_xyxy": list(self.bbox_xyxy),
            "centroid": {
                "x": round(self.centroid_x, 6),
                "y": round(self.centroid_y, 6),
            },
            "anchor": {"x": self.anchor_x, "y": self.anchor_y},
        }


def bind_drag_target_component(
    *,
    cleaned_mask_path: str | Path,
    image_path: str | Path,
    anchor_x: int,
    anchor_y: int,
    output_dir: str | Path,
    expected_width: int | None = None,
    expected_height: int | None = None,
) -> DragTargetBindingResult:
    """Bind an anchor to one connected target and persist review evidence."""
    source_mask = _resolved_file(cleaned_mask_path, label="target mask")
    source_image = _resolved_file(image_path, label="source image")
    mask = _load_mask(source_mask)
    image = _load_rgb(source_image)
    height, width = mask.shape
    if image.shape[:2] != mask.shape:
        raise DragTargetBindingError(
            "Drag target mask dimensions do not match the screenshot"
        )
    if expected_width is not None and width != expected_width:
        raise DragTargetBindingError(
            "Drag target mask width does not match the screenshot"
        )
    if expected_height is not None and height != expected_height:
        raise DragTargetBindingError(
            "Drag target mask height does not match the screenshot"
        )
    if not 0 <= anchor_x < width or not 0 <= anchor_y < height:
        raise DragTargetBindingError(
            "Drag anchor is outside the target mask image"
        )
    if mask[anchor_y, anchor_x] == 0:
        raise DragTargetBindingError(
            "Validated Drag anchor is outside the semantic target mask"
        )

    try:
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    except cv2.error as error:
        raise DragTargetBindingError(
            f"OpenCV connected-component binding failed: {error}"
        ) from error
    label = int(labels[anchor_y, anchor_x])
    if count <= 1 or label <= 0:
        raise DragTargetBindingError(
            "Drag anchor does not identify a foreground component"
        )
    component = np.where(labels == label, 255, 0).astype(np.uint8)
    x = int(stats[label, cv2.CC_STAT_LEFT])
    y = int(stats[label, cv2.CC_STAT_TOP])
    component_width = int(stats[label, cv2.CC_STAT_WIDTH])
    component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
    area = int(stats[label, cv2.CC_STAT_AREA])
    centroid_x = float(centroids[label, 0])
    centroid_y = float(centroids[label, 1])

    directory = _resolved_directory(output_dir)
    component_path = directory / "drag-target-component-mask.png"
    overlay_path = directory / "drag-target-anchor-overlay.png"
    _write_image(component_path, component)
    _write_anchor_overlay(
        overlay_path,
        image=image,
        component=component,
        anchor_x=anchor_x,
        anchor_y=anchor_y,
    )
    return DragTargetBindingResult(
        component_mask_path=str(component_path),
        anchor_overlay_path=str(overlay_path),
        component_label=label,
        component_area_pixels=area,
        bbox_xyxy=(x, y, x + component_width, y + component_height),
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        anchor_x=anchor_x,
        anchor_y=anchor_y,
    )


def _write_anchor_overlay(
    path: Path,
    *,
    image: NDArray[np.uint8],
    component: MaskArray,
    anchor_x: int,
    anchor_y: int,
) -> None:
    foreground = component > 0
    overlay = image.astype(np.float32)
    tint = np.zeros_like(overlay)
    tint[..., 0] = 16.0
    tint[..., 1] = 196.0
    tint[..., 2] = 255.0
    overlay[foreground] = (
        overlay[foreground] * 0.52 + tint[foreground] * 0.48
    )
    rendered = np.clip(overlay, 0.0, 255.0).astype(np.uint8)
    contours, _ = cv2.findContours(
        component,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(rendered, contours, -1, (255, 255, 255), 2)
    radius = max(6, round(min(component.shape) * 0.008))
    cv2.circle(
        rendered,
        (anchor_x, anchor_y),
        radius + 3,
        (255, 255, 255),
        thickness=-1,
        lineType=cv2.LINE_AA,
    )
    cv2.circle(
        rendered,
        (anchor_x, anchor_y),
        radius,
        (255, 48, 72),
        thickness=-1,
        lineType=cv2.LINE_AA,
    )
    _write_image(path, rendered)


def _load_mask(path: Path) -> MaskArray:
    try:
        with Image.open(path) as image:
            grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ) as error:
        raise DragTargetBindingError(
            f"Cannot read Drag target mask: {path}"
        ) from error
    return np.where(grayscale >= 128, 255, 0).astype(np.uint8)


def _load_rgb(path: Path) -> NDArray[np.uint8]:
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ) as error:
        raise DragTargetBindingError(
            f"Cannot read Drag source image: {path}"
        ) from error


def _write_image(path: Path, value: NDArray[np.uint8]) -> None:
    try:
        Image.fromarray(value).save(path)
    except OSError as error:
        raise DragTargetBindingError(
            f"Cannot write Drag target artifact: {path}"
        ) from error


def _resolved_file(value: str | Path, *, label: str) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()
    if not path.is_file():
        raise DragTargetBindingError(f"Drag {label} is not a file: {path}")
    return path


def _resolved_directory(value: str | Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise DragTargetBindingError(
            f"Cannot create Drag target output directory: {path}"
        ) from error
    if not path.is_dir():
        raise DragTargetBindingError(
            f"Drag target output path is not a directory: {path}"
        )
    return path
