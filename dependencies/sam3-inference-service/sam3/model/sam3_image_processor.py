# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Pre/post-processing for text-prompted SAM 3 image segmentation."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from sam3.model import box_ops
from sam3.model.data_misc import FindStage, interpolate


class Sam3Processor:
    def __init__(
        self,
        model: torch.nn.Module,
        resolution: int = 1008,
        device: str | torch.device | None = None,
        confidence_threshold: float = 0.5,
    ) -> None:
        self.model = model
        self.resolution = resolution
        self.device = (
            next(model.parameters()).device if device is None else torch.device(device)
        )
        self.confidence_threshold = self._validate_threshold(confidence_threshold)
        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(resolution, resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            ]
        )
        self.find_stage = FindStage(
            img_ids=torch.tensor([0], device=self.device, dtype=torch.long),
            text_ids=torch.tensor([0], device=self.device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

    @staticmethod
    def _validate_threshold(value: float) -> float:
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        return value

    @staticmethod
    def _as_rgb_image(image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            if image.ndim not in (2, 3):
                raise ValueError("NumPy image must have shape HxW or HxWxC")
            return Image.fromarray(image).convert("RGB")
        raise ValueError("Image must be a PIL image or NumPy array")

    @torch.inference_mode()
    def set_image(self, image: Any, state: Dict | None = None) -> Dict:
        """Encode one image and return reusable inference state."""
        image = self._as_rgb_image(image)
        width, height = image.size
        image_tensor = v2.functional.to_image(image).to(self.device)
        image_tensor = self.transform(image_tensor).unsqueeze(0)

        state = {} if state is None else state
        state["original_height"] = height
        state["original_width"] = width
        state["backbone_out"] = self.model.backbone.forward_image(image_tensor)
        return state

    @torch.inference_mode()
    def set_text_prompt(
        self,
        prompt: str,
        state: Dict,
        confidence_threshold: float | None = None,
    ) -> Dict:
        """Run segmentation for a short text concept prompt."""
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Text prompt cannot be empty")
        if "backbone_out" not in state:
            raise ValueError("Call set_image before set_text_prompt")

        text_outputs = self.model.backbone.forward_text([prompt], device=self.device)
        state["backbone_out"].update(text_outputs)
        state["geometric_prompt"] = self.model._get_dummy_prompt()
        threshold = (
            self.confidence_threshold
            if confidence_threshold is None
            else self._validate_threshold(confidence_threshold)
        )
        return self._forward_grounding(state, threshold)

    @torch.inference_mode()
    def _forward_grounding(self, state: Dict, threshold: float) -> Dict:
        outputs = self.model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=self.find_stage,
            geometric_prompt=state["geometric_prompt"],
            find_target=None,
        )

        out_bbox = outputs["pred_boxes"]
        out_masks = outputs["pred_masks"]
        out_probs = outputs["pred_logits"].sigmoid()
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)

        keep = out_probs > threshold
        out_probs = out_probs[keep]
        out_masks = out_masks[keep]
        out_bbox = out_bbox[keep]

        image_height = state["original_height"]
        image_width = state["original_width"]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        scale = torch.tensor(
            [image_width, image_height, image_width, image_height],
            device=self.device,
        )
        boxes = boxes * scale[None, :]

        mask_logits = interpolate(
            out_masks.unsqueeze(1),
            (image_height, image_width),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()

        state["masks_logits"] = mask_logits
        state["masks"] = mask_logits > 0.5
        state["boxes"] = boxes
        state["scores"] = out_probs
        return state


__all__ = ["Sam3Processor"]
