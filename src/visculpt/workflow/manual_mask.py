"""Deterministic rasterization for user-painted semantic masks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ManualMaskError(ValueError):
    """Raised when a manual mask submission cannot be used safely."""


class ManualMaskPoint(BaseModel):
    """One point in original screenshot pixel coordinates."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    x: float = Field(ge=0.0, le=100_000.0)
    y: float = Field(ge=0.0, le=100_000.0)


class ManualMaskStroke(BaseModel):
    """One constant-radius paint stroke."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    brush_size: float = Field(
        ge=1.0,
        le=4096.0,
        description="Blender-style brush radius in source-image pixels.",
    )
    points: list[ManualMaskPoint] = Field(min_length=1, max_length=10_000)


class ManualMaskPaintResponse(BaseModel):
    """Resume payload accepted by the paint-stage LangGraph interrupt."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["finish", "skip"]
    image_width: int | None = Field(default=None, ge=1, le=100_000)
    image_height: int | None = Field(default=None, ge=1, le=100_000)
    strokes: list[ManualMaskStroke] | None = Field(
        default=None,
        max_length=256,
    )

    @model_validator(mode="after")
    def validate_finish_payload(self) -> ManualMaskPaintResponse:
        """Require bounded geometry only when the user finishes painting."""
        if self.decision == "skip":
            if any(
                value is not None
                for value in (self.image_width, self.image_height, self.strokes)
            ):
                raise ValueError("skip must not include mask geometry")
            return self
        if self.image_width is None or self.image_height is None:
            raise ValueError("finish requires image dimensions")
        if not self.strokes:
            raise ValueError("finish requires at least one paint stroke")
        point_count = sum(len(stroke.points) for stroke in self.strokes)
        if point_count > 100_000:
            raise ValueError("manual mask contains too many points")
        return self


class ManualMaskReviewResponse(BaseModel):
    """Resume payload accepted by the review-stage interrupt."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["confirm", "redraw", "skip"]


class ManualMaskRasterResult(BaseModel):
    """Files and metrics produced after intersecting with the model mask."""

    model_config = ConfigDict(extra="forbid")

    image_width: int
    image_height: int
    stroke_count: int
    point_count: int
    painted_foreground_pixels: int
    model_foreground_pixels: int
    intersection_foreground_pixels: int
    painted_mask_path: str
    cleaned_mask_path: str
    cleaned_overlay_path: str


def rasterize_manual_mask(
    *,
    image_path: str | Path,
    model_mask_path: str | Path,
    submission: ManualMaskPaintResponse,
    output_dir: str | Path,
    overlay_opacity: float = 0.48,
) -> ManualMaskRasterResult:
    """Rasterize paint strokes and clip them to the segmented model."""
    if submission.decision != "finish":
        raise ManualMaskError("only a finished paint response can be rasterized")
    if not 0.0 <= overlay_opacity <= 1.0:
        raise ManualMaskError("overlay_opacity must be between 0 and 1")
    image_file = _existing_file(image_path, label="source image")
    model_mask_file = _existing_file(model_mask_path, label="model mask")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    try:
        with Image.open(image_file) as source:
            image = source.convert("RGBA")
        with Image.open(model_mask_file) as source:
            model_mask = source.convert("L")
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        raise ManualMaskError(f"cannot read manual mask inputs: {error}") from error

    width, height = image.size
    if submission.image_width != width or submission.image_height != height:
        raise ManualMaskError(
            "manual mask dimensions do not match the source screenshot"
        )
    if model_mask.size != image.size:
        raise ManualMaskError(
            "whole-model mask dimensions do not match the source screenshot"
        )
    strokes = submission.strokes or []
    painted = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(painted)
    point_count = 0
    for stroke in strokes:
        radius = float(stroke.brush_size)
        diameter = max(1, int(round(radius * 2.0)))
        points = [
            _validated_point(point.x, point.y, width=width, height=height)
            for point in stroke.points
        ]
        point_count += len(points)
        if len(points) > 1:
            draw.line(points, fill=255, width=diameter, joint="curve")
        for point in (points if len(points) == 1 else (points[0], points[-1])):
            x, y = point
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=255,
            )

    painted_array = np.asarray(painted, dtype=np.uint8) > 0
    model_array = np.asarray(model_mask, dtype=np.uint8) > 0
    intersection = painted_array & model_array
    intersection_image = Image.fromarray(
        intersection.astype(np.uint8) * 255,
    )
    painted_path = destination / "painted-mask.png"
    cleaned_path = destination / "manual-cleaned-mask.png"
    overlay_path = destination / "manual-cleaned-overlay.png"
    painted.save(painted_path)
    intersection_image.save(cleaned_path)

    tint = Image.new("RGBA", image.size, (61, 214, 140, 0))
    alpha = Image.fromarray(
        intersection.astype(np.uint8) * int(round(255 * overlay_opacity)),
    )
    tint.putalpha(alpha)
    Image.alpha_composite(image, tint).convert("RGB").save(overlay_path)

    return ManualMaskRasterResult(
        image_width=width,
        image_height=height,
        stroke_count=len(strokes),
        point_count=point_count,
        painted_foreground_pixels=int(np.count_nonzero(painted_array)),
        model_foreground_pixels=int(np.count_nonzero(model_array)),
        intersection_foreground_pixels=int(np.count_nonzero(intersection)),
        painted_mask_path=str(painted_path),
        cleaned_mask_path=str(cleaned_path),
        cleaned_overlay_path=str(overlay_path),
    )


def _existing_file(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ManualMaskError(f"{label} is not a file: {path}")
    return path


def _validated_point(
    x: float,
    y: float,
    *,
    width: int,
    height: int,
) -> tuple[float, float]:
    if not math.isfinite(x) or not math.isfinite(y):
        raise ManualMaskError("manual mask points must be finite")
    if x < 0.0 or x > width - 1 or y < 0.0 or y > height - 1:
        raise ManualMaskError("manual mask point lies outside the screenshot")
    return x, y
