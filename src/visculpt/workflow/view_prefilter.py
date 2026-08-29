"""Deterministic validation for SAM3 standard-view probes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from PIL import Image, UnidentifiedImageError

from visculpt.bridge import JsonValue


@dataclass(frozen=True, slots=True)
class ViewSegmentationAssessment:
    """Small JSON-safe summary used to accept or reject one view."""

    valid: bool
    invalid_reason: str | None
    instance_count: int
    max_confidence: float | None
    foreground_pixels: int
    foreground_ratio: float
    mask_width: int
    mask_height: int

    def as_payload(self) -> dict[str, JsonValue]:
        """Serialize the assessment for LangGraph State."""
        return {
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
            "instance_count": self.instance_count,
            "max_confidence": self.max_confidence,
            "foreground_pixels": self.foreground_pixels,
            "foreground_ratio": self.foreground_ratio,
            "mask_width": self.mask_width,
            "mask_height": self.mask_height,
        }


def assess_view_segmentation(
    *,
    mask_path: str | Path,
    metadata: Mapping[str, JsonValue],
) -> ViewSegmentationAssessment:
    """Require both a reported SAM3 instance and non-empty mask pixels."""
    count = _instance_count(metadata)
    foreground_pixels, width, height = _mask_foreground(mask_path)
    pixel_count = width * height
    ratio = foreground_pixels / pixel_count

    reason: str | None = None
    if count is None:
        reason = "missing_instance_count"
        normalized_count = 0
    else:
        normalized_count = count
        if count <= 0:
            reason = "no_instances"
    if reason is None and foreground_pixels == 0:
        reason = "empty_mask"

    return ViewSegmentationAssessment(
        valid=reason is None,
        invalid_reason=reason,
        instance_count=normalized_count,
        max_confidence=_max_confidence(metadata),
        foreground_pixels=foreground_pixels,
        foreground_ratio=ratio,
        mask_width=width,
        mask_height=height,
    )


def _instance_count(metadata: Mapping[str, JsonValue]) -> int | None:
    value = metadata.get("instance_count")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _mask_foreground(mask_path: str | Path) -> tuple[int, int, int]:
    path = Path(mask_path).resolve()
    if not path.is_file():
        raise ValueError(f"SAM3 mask does not exist: {path}")
    try:
        with Image.open(path) as source:
            mask = source.convert("L")
            width, height = mask.size
            if width <= 0 or height <= 0:
                raise ValueError("SAM3 mask has invalid dimensions")
            histogram = mask.histogram()
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"Cannot read SAM3 mask: {path}") from error
    return sum(histogram[1:]), width, height


def _max_confidence(
    metadata: Mapping[str, JsonValue],
) -> float | None:
    scores: list[float] = []
    raw_scores = metadata.get("scores")
    if isinstance(raw_scores, list):
        scores.extend(_finite_scores(raw_scores))
    instances = metadata.get("instances")
    if isinstance(instances, list):
        for item in instances:
            if not isinstance(item, dict):
                continue
            score = item.get("score")
            scores.extend(_finite_scores([score]))
    return max(scores) if scores else None


def _finite_scores(values: list[object]) -> list[float]:
    scores: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        score = float(value)
        if math.isfinite(score):
            scores.append(score)
    return scores
