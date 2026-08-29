"""Small Python API for original SAM 3 text-prompted image segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from sam3.device import inference_autocast, precision_name, resolve_device
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


@dataclass(frozen=True)
class SegmentationResult:
    """CPU-side result for all instances matching one text concept."""

    masks: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray

    @property
    def semantic_mask(self) -> np.ndarray:
        if self.masks.shape[0] == 0:
            return np.zeros(self.masks.shape[1:], dtype=np.bool_)
        return self.masks.any(axis=0)

    def to_metadata(self) -> dict:
        return {
            "instance_count": int(self.masks.shape[0]),
            "instances": [
                {
                    "instance_index": index,
                    "score": round(float(score), 6),
                    "box_xyxy": [round(float(value), 2) for value in box],
                    "foreground_pixels": int(np.count_nonzero(mask)),
                }
                for index, (mask, score, box) in enumerate(
                    zip(self.masks, self.scores, self.boxes)
                )
            ],
        }


class Sam3TextSegmenter:
    """Load SAM 3 once and segment images with short text prompts."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        device: str | torch.device = "auto",
        allow_download: bool = True,
        confidence_threshold: float = 0.5,
        compile_model: bool = False,
    ) -> None:
        self.device = resolve_device(device)
        self.model = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            device=self.device,
            load_from_HF=allow_download,
            compile=compile_model,
        )
        self.checkpoint_path = Path(self.model.checkpoint_path)
        self.processor = Sam3Processor(
            self.model,
            device=self.device,
            confidence_threshold=confidence_threshold,
        )
        self.precision = precision_name(self.device)

    def segment(
        self,
        image: Image.Image | np.ndarray,
        prompt: str,
        *,
        confidence_threshold: float | None = None,
    ) -> SegmentationResult:
        with inference_autocast(self.device):
            state = self.processor.set_image(image)
            output = self.processor.set_text_prompt(
                prompt,
                state,
                confidence_threshold=confidence_threshold,
            )

        masks = (
            output["masks"]
            .squeeze(1)
            .detach()
            .to(device="cpu")
            .numpy()
            .astype(np.bool_, copy=False)
        )
        boxes = output["boxes"].detach().to(device="cpu").float().numpy()
        scores = output["scores"].detach().to(device="cpu").float().numpy()
        return SegmentationResult(masks=masks, boxes=boxes, scores=scores)


_COLORS = np.asarray(
    [
        (42, 157, 244),
        (255, 99, 132),
        (46, 204, 113),
        (255, 183, 77),
        (171, 71, 188),
        (38, 198, 218),
    ],
    dtype=np.float32,
)


def render_overlay(
    image: Image.Image | np.ndarray,
    result: SegmentationResult,
    *,
    opacity: float = 0.45,
) -> Image.Image:
    """Render instance masks, boxes, and confidence scores over the image."""
    opacity = float(opacity)
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("opacity must be between 0 and 1")

    rgb_image = (
        image.convert("RGB")
        if isinstance(image, Image.Image)
        else Image.fromarray(np.asarray(image)).convert("RGB")
    )
    canvas = np.asarray(rgb_image, dtype=np.float32).copy()
    for index, mask in enumerate(result.masks):
        color = _COLORS[index % len(_COLORS)]
        canvas[mask] = canvas[mask] * (1.0 - opacity) + color * opacity

    overlay = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(overlay)
    for index, (box, score) in enumerate(zip(result.boxes, result.scores)):
        color = tuple(int(value) for value in _COLORS[index % len(_COLORS)])
        xyxy = tuple(float(value) for value in box)
        draw.rectangle(xyxy, outline=color, width=3)
        draw.text((xyxy[0] + 3, xyxy[1] + 3), f"{score:.3f}", fill=color)
    return overlay


__all__ = [
    "Sam3TextSegmenter",
    "SegmentationResult",
    "render_overlay",
    "resolve_device",
]
