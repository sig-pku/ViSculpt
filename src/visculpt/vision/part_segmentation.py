"""Kinematic part planning and deterministic Face Set lasso generation."""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast
from uuid import uuid4

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, UnidentifiedImageError
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from visculpt.bridge import JsonValue

from .sculpt_stroke import (
    ScreenshotViewportMapping,
    SculptStrokePlanningError,
)

type MaskArray = NDArray[np.uint8]
type Point = tuple[float, float]

_MASK_THRESHOLD = 128
_MAX_SCHEMA_SUBPARTS = 12
_MAX_SCHEMA_SYNONYMS = 5
_INSTANCE_QUALIFIERS = frozenset({"left", "right", "front", "rear"})
_LASSO_COLOR_PALETTE = (
    (46, 204, 113),
    (52, 152, 219),
    (241, 196, 15),
    (231, 76, 60),
    (155, 89, 182),
    (26, 188, 156),
    (230, 126, 34),
    (236, 112, 199),
)


class PartSegmentationError(RuntimeError):
    """Base class for user-facing part-segmentation failures."""


class PartSegmentationInputError(PartSegmentationError):
    """Raised when the screenshot or target description is invalid."""


class PartSegmentationVlmError(PartSegmentationError):
    """Raised when the VLM cannot produce a usable kinematic plan."""


class PartSegmentationSam3Error(PartSegmentationError):
    """Raised when one planned subpart cannot be segmented."""


class PartSegmentationNoMaskError(PartSegmentationSam3Error):
    """Raised only for a deterministic empty SAM3 segmentation."""


class PartSegmentationParentMaskError(PartSegmentationSam3Error):
    """Raised when the operation part has no usable parent mask."""


class PartSegmentationLassoError(PartSegmentationError):
    """Raised when a cleaned mask cannot produce a safe lasso."""


class PartSegmentationRpcError(PartSegmentationError):
    """Raised when Blender rejects a screenshot or lasso RPC request."""


class PartSegmentationArtifactError(PartSegmentationError):
    """Raised when diagnostic artifacts cannot be persisted."""


class KinematicSubpart(BaseModel):
    """One proximal-to-distal semantic segment suitable for SAM3."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=100)
    sam3_prompt: str = Field(min_length=1, max_length=80)
    fallback_prompts: list[str] = Field(
        min_length=0,
        max_length=_MAX_SCHEMA_SYNONYMS,
    )
    rationale: str = Field(min_length=1, max_length=500)

    @field_validator("label", "sam3_prompt")
    @classmethod
    def normalize_phrase(cls, value: str) -> str:
        """Keep semantic labels and SAM3 prompts short English phrases."""
        return _normalize_sam3_phrase(value)

    @field_validator("fallback_prompts")
    @classmethod
    def normalize_fallback_prompts(cls, values: list[str]) -> list[str]:
        """Normalize a bounded list of alternative SAM3 noun phrases."""
        return [_normalize_sam3_phrase(value) for value in values]

    @field_validator("rationale")
    @classmethod
    def normalize_rationale(cls, value: str) -> str:
        """Reject whitespace-only explanations."""
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("rationale must not be empty")
        return normalized

    @model_validator(mode="after")
    def validate_prompt_uniqueness(self) -> KinematicSubpart:
        """Prevent retries that repeat the primary prompt or each other."""
        prompts = [self.sam3_prompt, *self.fallback_prompts]
        normalized = [prompt.casefold() for prompt in prompts]
        if len(normalized) != len(set(normalized)):
            raise ValueError(
                "sam3_prompt and fallback_prompts must be unique"
            )
        return self


class KinematicPartPlan(BaseModel):
    """Schema-constrained VLM plan for one requested model part."""

    model_config = ConfigDict(extra="forbid")

    target_visible: bool
    parent_sam3_prompt: str | None
    subparts: list[KinematicSubpart] = Field(
        min_length=0,
        max_length=_MAX_SCHEMA_SUBPARTS,
    )
    rationale: str = Field(min_length=1, max_length=1000)

    @field_validator("rationale")
    @classmethod
    def normalize_rationale(cls, value: str) -> str:
        """Normalize the plan-level explanation."""
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("rationale must not be empty")
        return normalized

    @field_validator("parent_sam3_prompt")
    @classmethod
    def normalize_parent_prompt(cls, value: str | None) -> str | None:
        """Normalize the instance-specific parent prompt when present."""
        if value is None:
            return None
        return _normalize_sam3_phrase(value)

    @model_validator(mode="after")
    def validate_visibility_and_uniqueness(self) -> KinematicPartPlan:
        """Require a nonempty unique plan exactly when the target is visible."""
        if self.target_visible:
            if self.parent_sam3_prompt is None:
                raise ValueError(
                    "visible targets require parent_sam3_prompt"
                )
            if not self.subparts:
                raise ValueError(
                    "visible targets require at least one subpart"
                )
        elif self.parent_sam3_prompt is not None or self.subparts:
            raise ValueError(
                "invisible targets must not return a parent prompt or subparts"
            )
        labels = [item.label.casefold() for item in self.subparts]
        if len(labels) != len(set(labels)):
            raise ValueError("subpart labels must be unique")
        prompts = [item.sam3_prompt.casefold() for item in self.subparts]
        if len(prompts) != len(set(prompts)):
            raise ValueError("subpart SAM3 prompts must be unique")
        return self


@dataclass(frozen=True, slots=True)
class PartSegmentationConfig:
    """VLM, SAM3, lasso, and artifact settings for the composite Tool."""

    max_subparts: int = 8
    artifact_root: str = "output/tools/part-segmentation-with-sam3"
    llm_role: str = "translator"
    sam3_confidence_threshold: float = 0.5
    sam3_overlay_opacity: float = 0.45
    roi_padding_ratio: float = 0.15
    parent_mask_dilation_pixels: int = 4
    max_synonym_attempts: int = 2
    minimum_child_parent_containment: float = 0.8
    minimum_parent_coverage: float = 0.8
    maximum_pairwise_overlap_ratio: float = 0.2
    maximum_uncovered_component_ratio: float = 0.15
    max_instances_per_prompt: int = 8
    duplicate_candidate_iou: float = 0.95
    max_candidate_combinations: int = 50_000
    lasso_padding_pixels: int = 6
    lasso_padding_retry_multiplier: int = 2
    lasso_max_padding_increases: int = 5
    lasso_simplify_tolerance_pixels: float = 1.0
    max_lasso_points: int = 4096
    lasso_time_step_seconds: float = 0.01
    use_front_faces_only: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.max_subparts <= _MAX_SCHEMA_SUBPARTS:
            raise ValueError(
                f"max_subparts must be between 1 and {_MAX_SCHEMA_SUBPARTS}"
            )
        if not self.artifact_root.strip():
            raise ValueError("artifact_root must not be empty")
        if not self.llm_role.strip():
            raise ValueError("llm_role must not be empty")
        if not 0.05 <= self.sam3_confidence_threshold <= 0.95:
            raise ValueError(
                "sam3_confidence_threshold must be between 0.05 and 0.95"
            )
        if not 0.0 <= self.sam3_overlay_opacity <= 1.0:
            raise ValueError("sam3_overlay_opacity must be between 0 and 1")
        if not 0.0 <= self.roi_padding_ratio <= 1.0:
            raise ValueError("roi_padding_ratio must be between 0 and 1")
        if not 0 <= self.parent_mask_dilation_pixels <= 128:
            raise ValueError(
                "parent_mask_dilation_pixels must be between 0 and 128"
            )
        if not 0 <= self.max_synonym_attempts <= _MAX_SCHEMA_SYNONYMS:
            raise ValueError(
                "max_synonym_attempts must be between 0 and "
                f"{_MAX_SCHEMA_SYNONYMS}"
            )
        for name, value in (
            (
                "minimum_child_parent_containment",
                self.minimum_child_parent_containment,
            ),
            ("minimum_parent_coverage", self.minimum_parent_coverage),
            (
                "maximum_pairwise_overlap_ratio",
                self.maximum_pairwise_overlap_ratio,
            ),
            (
                "maximum_uncovered_component_ratio",
                self.maximum_uncovered_component_ratio,
            ),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if not 0.0 < self.duplicate_candidate_iou <= 1.0:
            raise ValueError(
                "duplicate_candidate_iou must be greater than 0 and at most 1"
            )
        if not 1 <= self.max_instances_per_prompt <= 128:
            raise ValueError(
                "max_instances_per_prompt must be between 1 and 128"
            )
        if not 1 <= self.max_candidate_combinations <= 10_000_000:
            raise ValueError(
                "max_candidate_combinations must be between 1 and 10000000"
            )
        if not 1 <= self.lasso_padding_pixels <= 192:
            raise ValueError("lasso_padding_pixels must be between 1 and 192")
        if not 1 <= self.lasso_padding_retry_multiplier <= 16:
            raise ValueError(
                "lasso_padding_retry_multiplier must be between 1 and 16"
            )
        if not 0 <= self.lasso_max_padding_increases <= 16:
            raise ValueError(
                "lasso_max_padding_increases must be between 0 and 16"
            )
        maximum_padding = self.lasso_padding_pixels * (
            self.lasso_padding_retry_multiplier
            ** self.lasso_max_padding_increases
        )
        if maximum_padding > 192:
            raise ValueError(
                "lasso padding retries must not exceed 192 pixels"
            )
        if not 0.0 <= self.lasso_simplify_tolerance_pixels <= 64.0:
            raise ValueError(
                "lasso_simplify_tolerance_pixels must be between 0 and 64"
            )
        if not 4 <= self.max_lasso_points <= 100_000:
            raise ValueError("max_lasso_points must be between 4 and 100000")
        if not 0.0 < self.lasso_time_step_seconds <= 1.0:
            raise ValueError(
                "lasso_time_step_seconds must be greater than 0 and at most 1"
            )


@dataclass(frozen=True, slots=True)
class FaceSetLassoGeometry:
    """One verified closed path in Blender region coordinates."""

    operator_path: tuple[dict[str, JsonValue], ...]
    screenshot_points: tuple[Point, ...]
    foreground_pixels: int
    component_count: int
    strategy: str
    padding_pixels: int
    missing_foreground_pixels: int

    def as_payload(self) -> dict[str, JsonValue]:
        """Return the complete deterministic geometry trace."""
        return {
            "coordinate_system": {
                "space": "VIEW_3D_WINDOW_REGION",
                "origin": "BOTTOM_LEFT",
                "units": "BLENDER_UI_PIXELS",
            },
            "closed": True,
            "point_count": len(self.operator_path),
            "unique_point_count": len(self.operator_path) - 1,
            "foreground_pixels": self.foreground_pixels,
            "component_count": self.component_count,
            "strategy": self.strategy,
            "padding_pixels": self.padding_pixels,
            "containment_check": {
                "missing_foreground_pixels": (
                    self.missing_foreground_pixels
                ),
                "complete": self.missing_foreground_pixels == 0,
            },
            "path": [dict(item) for item in self.operator_path],
        }


@dataclass(frozen=True, slots=True)
class PartSegmentationResult:
    """Completed Face Set side effect plus inspectable receipt."""

    payload: dict[str, JsonValue]

    def as_payload(self) -> dict[str, JsonValue]:
        """Return a JSON-compatible Tool result."""
        return dict(self.payload)


class PartSegmentationCompletion(Protocol):
    """Structural completion result supplied by the workflow LLM adapter."""

    value: BaseModel
    metadata: dict[str, JsonValue]


class PartSegmentationVlm(Protocol):
    """Minimal multimodal structured-completion surface."""

    def complete(
        self,
        *,
        role: str,
        system_prompt: str,
        user_prompt: str,
        image_paths: Sequence[str | Path],
        response_model: type[KinematicPartPlan],
    ) -> PartSegmentationCompletion:
        """Return one kinematic decomposition plan."""
        ...


class PartSegmentationSegmentTool(Protocol):
    """Direct invocation surface of the existing SAM3 Tool."""

    def invoke(self, input: dict[str, JsonValue]) -> object:
        """Segment one subpart and return a cleaned mask."""
        ...


class PartSegmentationComponentSelectorTool(Protocol):
    """Direct invocation surface of the semantic component selector."""

    def invoke(self, input: dict[str, JsonValue]) -> object:
        """Select one connected component from a cleaned parent mask."""
        ...


class PartSegmentationRpc(Protocol):
    """JSON-RPC surface used for screenshot metadata and Face Sets."""

    def send(self, payload: JsonValue) -> JsonValue:
        """Send one request and preserve the complete JSON-RPC envelope."""
        ...


class PartSegmentationPromptBuilder(Protocol):
    """English prompt builder kept in the centralized prompt module."""

    def __call__(
        self,
        *,
        part_description: str,
        max_subparts: int,
        max_synonym_attempts: int,
    ) -> str:
        """Build the kinematic decomposition user prompt."""
        ...


@dataclass(frozen=True, slots=True)
class _SegmentationAttempt:
    prompt: str
    status: str
    output_dir: Path
    response: dict[str, JsonValue] | None = None
    error: str | None = None

    def as_payload(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "prompt": self.prompt,
            "status": self.status,
            "output_dir": str(self.output_dir),
        }
        if self.response is not None:
            payload["response"] = self.response
        if self.error is not None:
            payload["error"] = self.error
        return payload


@dataclass(frozen=True, slots=True)
class _SubpartCandidate:
    """One independently cleaned SAM3 instance retained for joint selection."""

    candidate_id: str
    prompt: str
    prompt_index: int
    instance_index: int
    score: float
    box_xyxy: tuple[float, float, float, float] | None
    segment_response: dict[str, JsonValue]
    constraint: dict[str, JsonValue]
    cleaned_mask_path: Path
    overlay_path: Path
    mask: MaskArray

    def as_payload(self) -> dict[str, JsonValue]:
        return {
            "candidate_id": self.candidate_id,
            "prompt": self.prompt,
            "prompt_index": self.prompt_index,
            "instance_index": self.instance_index,
            "score": self.score,
            "box_xyxy": (
                list(self.box_xyxy) if self.box_xyxy is not None else None
            ),
            "cleaned_mask_path": str(self.cleaned_mask_path),
            "overlay_path": str(self.overlay_path),
            "constraint": self.constraint,
        }


@dataclass(slots=True)
class _CandidatePool:
    """Bounded prompt attempts and retained instances for one subpart."""

    order: int
    specification: KinematicSubpart
    prompts: tuple[str, ...]
    output_dir: Path
    attempts: list[_SegmentationAttempt]
    candidates: list[_SubpartCandidate]
    rejected_candidates: list[dict[str, JsonValue]]

    def as_payload(self) -> dict[str, JsonValue]:
        return {
            "order": self.order,
            "specification": self.specification.model_dump(mode="json"),
            "prompts": list(self.prompts),
            "attempts": [item.as_payload() for item in self.attempts],
            "candidates": [item.as_payload() for item in self.candidates],
            "rejected_candidates": list(self.rejected_candidates),
        }


@dataclass(frozen=True, slots=True)
class _ParentSegmentation:
    prompt: str
    instance_index: int
    score: float
    segment_response: dict[str, JsonValue]
    cleaned_mask_path: Path
    overlay_path: str | None
    mask: MaskArray
    foreground_pixels: int
    component_count: int
    component_selection: dict[str, JsonValue] | None
    roi_box: tuple[int, int, int, int]
    roi_image_path: Path
    roi_mask_path: Path

    def as_payload(self) -> dict[str, JsonValue]:
        left, top, right, bottom = self.roi_box
        return {
            "sam3_prompt": self.prompt,
            "selected_instance_index": self.instance_index,
            "selected_instance_score": self.score,
            "segment_response": self.segment_response,
            "cleaned_mask_path": str(self.cleaned_mask_path),
            "sam3_overlay_path": self.overlay_path,
            "foreground_pixels": self.foreground_pixels,
            "component_count": self.component_count,
            "component_selection": self.component_selection,
            "roi": {
                "coordinate_system": "IMAGE_TOP_LEFT_PIXELS",
                "left": left,
                "top": top,
                "right_exclusive": right,
                "bottom_exclusive": bottom,
                "width": right - left,
                "height": bottom - top,
                "image_path": str(self.roi_image_path),
                "parent_mask_path": str(self.roi_mask_path),
            },
        }


@dataclass(slots=True)
class _PreparedSubpart:
    order: int
    specification: KinematicSubpart
    selected_prompt: str
    attempts: tuple[_SegmentationAttempt, ...]
    segment_response: dict[str, JsonValue]
    constraint: dict[str, JsonValue]
    cleaned_mask_path: Path
    overlay_path: str | None
    lasso: FaceSetLassoGeometry
    rpc_request: dict[str, JsonValue]
    rpc_response: dict[str, JsonValue] | None = None


class PartSegmentationRunner:
    """Plan semantic subparts, segment them, and create Blender Face Sets."""

    def __init__(
        self,
        *,
        llm: PartSegmentationVlm,
        segment_tool: PartSegmentationSegmentTool,
        mask_component_selector_tool: PartSegmentationComponentSelectorTool,
        rpc_client: PartSegmentationRpc,
        system_prompt: str,
        prompt_builder: PartSegmentationPromptBuilder,
        config: PartSegmentationConfig | None = None,
        workdir: Path | None = None,
    ) -> None:
        self._llm = llm
        self._segment_tool = segment_tool
        self._mask_component_selector_tool = mask_component_selector_tool
        self._rpc_client = rpc_client
        self._system_prompt = system_prompt
        self._prompt_builder = prompt_builder
        self.config = config or PartSegmentationConfig()
        self._workdir = (workdir or Path.cwd()).resolve()

    def run(
        self,
        *,
        image_path: str | Path,
        part_description: str,
        artifact_root: str | None = None,
        parent_mask_path: str | Path | None = None,
    ) -> PartSegmentationResult:
        """Execute the complete, preparation-first Face Set workflow."""
        image_file = _resolve_image_path(image_path)
        description = _normalize_description(part_description)
        image = _load_rgb_image(image_file)
        run_dir = _create_run_dir(
            artifact_root or self.config.artifact_root,
            workdir=self._workdir,
            image_stem=image_file.stem,
        )
        trace_path = run_dir / "part-segmentation-trace.json"
        viewport_reference_path = run_dir / "viewport-reference.png"

        viewport_response, viewport_metadata = self._capture_viewport(
            viewport_reference_path
        )
        try:
            mapping = ScreenshotViewportMapping.from_metadata(
                image_width=image.width,
                image_height=image.height,
                metadata=viewport_metadata,
            )
        except SculptStrokePlanningError as error:
            raise PartSegmentationRpcError(
                "The input screenshot does not match the current Blender "
                f"VIEW_3D region: {error}"
            ) from error

        plan, llm_metadata = self._plan_subparts(
            image_path=image_file,
            part_description=description,
        )
        parent = (
            self._prepare_parent(
                image_path=image_file,
                image=image,
                plan=plan,
                run_dir=run_dir,
            )
            if parent_mask_path is None
            else self._prepare_manual_parent(
                image=image,
                plan=plan,
                run_dir=run_dir,
                parent_mask_path=parent_mask_path,
            )
        )
        prepared, candidate_selection = self._prepare_subparts(
            image_path=image_file,
            image=image,
            plan=plan,
            parent=parent,
            mapping=mapping,
            run_dir=run_dir,
        )
        validation = _validate_part_hierarchy(
            parent=parent,
            prepared=prepared,
        )
        visualization_path = run_dir / "part-face-set-lassos.png"
        _render_lasso_visualization(
            image,
            prepared=prepared,
            mapping=mapping,
            output_path=visualization_path,
        )

        trace = self._trace_payload(
            status="prepared",
            image_path=image_file,
            part_description=description,
            viewport_reference_path=viewport_reference_path,
            viewport_response=viewport_response,
            mapping=mapping,
            plan=plan,
            llm_metadata=llm_metadata,
            parent=parent,
            prepared=prepared,
            candidate_selection=candidate_selection,
            validation=validation,
            visualization_path=visualization_path,
            face_set_execution=None,
        )
        _write_json(trace_path, trace)

        prepared, candidate_selection, face_set_execution = (
            self._execute_face_set_partition(
                image=image,
                parent=parent,
                prepared=prepared,
                candidate_selection=candidate_selection,
                mapping=mapping,
                run_dir=run_dir,
            )
        )
        validation = _validate_part_hierarchy(
            parent=parent,
            prepared=prepared,
        )
        _render_lasso_visualization(
            image,
            prepared=prepared,
            mapping=mapping,
            output_path=visualization_path,
        )

        trace = self._trace_payload(
            status="completed",
            image_path=image_file,
            part_description=description,
            viewport_reference_path=viewport_reference_path,
            viewport_response=viewport_response,
            mapping=mapping,
            plan=plan,
            llm_metadata=llm_metadata,
            parent=parent,
            prepared=prepared,
            candidate_selection=candidate_selection,
            validation=validation,
            visualization_path=visualization_path,
            face_set_execution=face_set_execution,
        )
        _write_json(trace_path, trace)
        return PartSegmentationResult(
            payload={
                "status": "completed",
                "semantic_output": None,
                "subpart_count": len(prepared),
                "pose_ik_segments": len(prepared),
                "segmentation_mode": cast(
                    dict[str, JsonValue],
                    candidate_selection["fallback"],
                )["mode"],
                "fallback": candidate_selection["fallback"],
                "parent": parent.as_payload(),
                "validation": validation,
                "candidate_selection": candidate_selection,
                "face_set_execution": face_set_execution,
                "subparts": [
                    {
                        "order": item.order,
                        "label": item.specification.label,
                        "sam3_prompt": item.specification.sam3_prompt,
                        "fallback_prompts": list(
                            item.specification.fallback_prompts
                        ),
                        "selected_prompt": item.selected_prompt,
                        "attempts": [
                            attempt.as_payload()
                            for attempt in item.attempts
                        ],
                        "constraint": item.constraint,
                        "lasso_point_count": len(
                            item.lasso.operator_path
                        ),
                        "lasso_strategy": item.lasso.strategy,
                        "lasso_padding_pixels": (
                            item.lasso.padding_pixels
                        ),
                        "cleaned_mask_path": str(
                            item.cleaned_mask_path
                        ),
                        "sam3_overlay_path": item.overlay_path,
                        "rpc_response": item.rpc_response,
                    }
                    for item in prepared
                ],
                "viewport_mapping": mapping.as_payload(),
                "artifacts": {
                    "directory": str(run_dir),
                    "trace_path": str(trace_path),
                    "lasso_visualization_path": str(
                        visualization_path
                    ),
                    "viewport_reference_path": str(
                        viewport_reference_path
                    ),
                    "candidate_selection_path": cast(
                        str,
                        candidate_selection["selection_path"],
                    ),
                    "candidate_visualization_path": cast(
                        str,
                        candidate_selection[
                            "candidate_visualization_path"
                        ],
                    ),
                },
                "llm_metadata": llm_metadata,
            }
        )

    def _capture_viewport(
        self,
        output_path: Path,
    ) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
        """Capture current metadata without adding a third Tool input."""
        request: dict[str, JsonValue] = {
            "jsonrpc": "2.0",
            "id": f"part-segmentation-screenshot-{uuid4().hex}",
            "method": "get_screenshot",
            "params": {
                "output": "file",
                "filepath": str(output_path),
                "redraw": False,
            },
        }
        response = self._send_rpc(request, label="get_screenshot")
        result = _rpc_result(response, label="get_screenshot")
        if result.get("encoding") != "file":
            raise PartSegmentationRpcError(
                "get_screenshot did not return file encoding"
            )
        if not output_path.is_file():
            raise PartSegmentationRpcError(
                "get_screenshot did not create the viewport reference PNG"
            )
        return response, result

    def _plan_subparts(
        self,
        *,
        image_path: Path,
        part_description: str,
    ) -> tuple[KinematicPartPlan, dict[str, JsonValue]]:
        user_prompt = self._prompt_builder(
            part_description=part_description,
            max_subparts=self.config.max_subparts,
            max_synonym_attempts=self.config.max_synonym_attempts,
        )
        try:
            completion = self._llm.complete(
                role=self.config.llm_role,
                system_prompt=self._system_prompt,
                user_prompt=user_prompt,
                image_paths=[image_path],
                response_model=KinematicPartPlan,
            )
        except Exception as error:
            raise PartSegmentationVlmError(
                f"Part segmentation VLM request failed: {error}"
            ) from error
        if not isinstance(completion.value, KinematicPartPlan):
            raise PartSegmentationVlmError(
                "Part segmentation VLM returned an unexpected output"
            )
        plan = completion.value
        if not plan.target_visible:
            raise PartSegmentationVlmError(
                "The requested model part is not visibly identifiable in "
                "the input screenshot"
            )
        if len(plan.subparts) > self.config.max_subparts:
            raise PartSegmentationVlmError(
                "The VLM returned more subparts than the configured maximum"
            )
        assert plan.parent_sam3_prompt is not None
        requested_qualifiers = _identity_qualifiers(part_description)
        parent_qualifiers = _identity_qualifiers(
            plan.parent_sam3_prompt
        )
        missing_qualifiers = requested_qualifiers - parent_qualifiers
        if missing_qualifiers:
            raise PartSegmentationVlmError(
                "The parent SAM3 prompt dropped requested instance "
                "qualifier(s): " + ", ".join(sorted(missing_qualifiers))
            )
        return plan, dict(completion.metadata)

    def _prepare_parent(
        self,
        *,
        image_path: Path,
        image: Image.Image,
        plan: KinematicPartPlan,
        run_dir: Path,
    ) -> _ParentSegmentation:
        """Segment the requested instance before any generic child prompt."""
        prompt = plan.parent_sam3_prompt
        if prompt is None:
            raise PartSegmentationVlmError(
                "The visible target plan has no parent SAM3 prompt"
            )
        parent_dir = run_dir / "parent"
        _create_artifact_directory(parent_dir, label="parent segment")
        try:
            response = self._segment_prompt(
                image_path=image_path,
                prompt=prompt,
                output_dir=parent_dir,
            )
            parent_instance = _select_parent_instance(
                response,
                expected_size=image.size,
                prompt=prompt,
            )
        except PartSegmentationNoMaskError as error:
            raise PartSegmentationParentMaskError(
                f"SAM3 produced no usable parent mask for {prompt}: {error}"
            ) from error
        raw_mask_path = parent_instance["cleaned_mask_path"]
        assert isinstance(raw_mask_path, Path)
        raw_mask = parent_instance["mask"]
        assert isinstance(raw_mask, np.ndarray)
        component_count = _component_count(raw_mask)
        component_selection: dict[str, JsonValue] | None = None
        selected_overlay_path: Path | None = None
        if component_count > 1:
            selection_dir = parent_dir / "component-selection"
            _create_artifact_directory(
                selection_dir,
                label="parent component selection",
            )
            selection_response = self._mask_component_selector_tool.invoke(
                {
                    "image_path": str(image_path),
                    "cleaned_mask_path": str(raw_mask_path),
                    "part_description": prompt,
                    "overlay_opacity": self.config.sam3_overlay_opacity,
                    "output_dir": str(selection_dir),
                }
            )
            component_selection = _component_selection_result(
                selection_response,
                label=prompt,
            )
            selected_path = _component_selection_path(
                component_selection,
                key="selected_mask_path",
                label=prompt,
            )
            selected_overlay_path = _component_selection_path(
                component_selection,
                key="selected_overlay_path",
                label=prompt,
            )
            parent_mask = _binary_mask(
                selected_path,
                expected_size=image.size,
                label=f"selected parent component {prompt}",
            )
            if _component_count(parent_mask) != 1:
                raise PartSegmentationVlmError(
                    "Component selector did not return exactly one parent "
                    f"region for {prompt}"
                )
        else:
            parent_mask = raw_mask.copy()
        foreground_pixels = int(np.count_nonzero(parent_mask))
        if foreground_pixels == 0:
            raise PartSegmentationParentMaskError(
                f"SAM3 parent mask for {prompt} has no foreground"
            )
        cleaned_mask_path = parent_dir / "parent-cleaned-mask.png"
        _save_mask(parent_mask, cleaned_mask_path)
        roi_box = _expanded_foreground_box(
            parent_mask,
            padding_ratio=self.config.roi_padding_ratio,
        )
        roi_image_path = parent_dir / "parent-roi.png"
        _save_png(image.crop(roi_box), roi_image_path)
        left, top, right, bottom = roi_box
        roi_mask_path = parent_dir / "parent-roi-mask.png"
        _save_mask(parent_mask[top:bottom, left:right], roi_mask_path)
        return _ParentSegmentation(
            prompt=prompt,
            instance_index=cast(int, parent_instance["instance_index"]),
            score=cast(float, parent_instance["score"]),
            segment_response=response,
            cleaned_mask_path=cleaned_mask_path,
            overlay_path=(
                str(selected_overlay_path)
                if selected_overlay_path is not None
                else _optional_result_path(response, "overlay_path")
            ),
            mask=parent_mask,
            foreground_pixels=foreground_pixels,
            component_count=component_count,
            component_selection=component_selection,
            roi_box=roi_box,
            roi_image_path=roi_image_path,
            roi_mask_path=roi_mask_path,
        )

    def _prepare_manual_parent(
        self,
        *,
        image: Image.Image,
        plan: KinematicPartPlan,
        run_dir: Path,
        parent_mask_path: str | Path,
    ) -> _ParentSegmentation:
        """Use a user-confirmed parent mask instead of calling SAM3 again."""
        prompt = plan.parent_sam3_prompt
        if prompt is None:
            raise PartSegmentationVlmError(
                "The visible target plan has no parent SAM3 prompt"
            )
        source_path = Path(
            os.path.expandvars(os.path.expanduser(str(parent_mask_path)))
        ).resolve()
        parent_mask = _binary_mask(
            source_path,
            expected_size=image.size,
            label=f"manual parent {prompt}",
        )
        foreground_pixels = int(np.count_nonzero(parent_mask))
        if foreground_pixels == 0:
            raise PartSegmentationParentMaskError(
                f"Manual parent mask for {prompt} has no foreground"
            )
        parent_dir = run_dir / "parent"
        _create_artifact_directory(parent_dir, label="manual parent segment")
        cleaned_mask_path = parent_dir / "parent-cleaned-mask.png"
        overlay_path = parent_dir / "parent-manual-overlay.png"
        _save_mask(parent_mask, cleaned_mask_path)
        _render_mask_overlay(
            image,
            mask=parent_mask,
            opacity=self.config.sam3_overlay_opacity,
            output_path=overlay_path,
        )
        roi_box = _expanded_foreground_box(
            parent_mask,
            padding_ratio=self.config.roi_padding_ratio,
        )
        left, top, right, bottom = roi_box
        roi_image_path = parent_dir / "parent-roi.png"
        roi_mask_path = parent_dir / "parent-roi-mask.png"
        _save_png(image.crop(roi_box), roi_image_path)
        _save_mask(parent_mask[top:bottom, left:right], roi_mask_path)
        synthetic_response: dict[str, JsonValue] = {
            "result": {
                "source": "manual_mask_override",
                "prompt": prompt,
                "cleaned_mask_path": str(cleaned_mask_path),
                "cleaned_overlay_path": str(overlay_path),
            }
        }
        return _ParentSegmentation(
            prompt=prompt,
            instance_index=-1,
            score=1.0,
            segment_response=synthetic_response,
            cleaned_mask_path=cleaned_mask_path,
            overlay_path=str(overlay_path),
            mask=parent_mask,
            foreground_pixels=foreground_pixels,
            component_count=_component_count(parent_mask),
            component_selection={
                "source": "manual_mask_override",
                "source_mask_path": str(source_path),
            },
            roi_box=roi_box,
            roi_image_path=roi_image_path,
            roi_mask_path=roi_mask_path,
        )

    def _prepare_subparts(
        self,
        *,
        image_path: Path,
        image: Image.Image,
        plan: KinematicPartPlan,
        parent: _ParentSegmentation,
        mapping: ScreenshotViewportMapping,
        run_dir: Path,
    ) -> tuple[list[_PreparedSubpart], dict[str, JsonValue]]:
        """Build candidate pools and select one globally consistent assignment."""
        pools: list[_CandidatePool] = []
        for order, specification in enumerate(plan.subparts, start=1):
            segment_dir = run_dir / "segments" / (
                f"{order:02d}-{_slug(specification.label)}"
            )
            _create_artifact_directory(
                segment_dir,
                label="subpart segment",
            )
            pools.append(
                _CandidatePool(
                    order=order,
                    specification=specification,
                    prompts=_effective_child_prompts(
                        specification,
                        parent_prompt=parent.prompt,
                        max_synonym_attempts=(
                            self.config.max_synonym_attempts
                        ),
                    ),
                    output_dir=segment_dir,
                    attempts=[],
                    candidates=[],
                    rejected_candidates=[],
                )
            )

        for pool in pools:
            self._collect_candidate_attempt(
                pool=pool,
                prompt_index=0,
                image_path=image_path,
                image=image,
                parent=parent,
            )

        selection_rounds: list[dict[str, JsonValue]] = []
        selected: list[_SubpartCandidate] | None = None
        selection_metrics: dict[str, JsonValue] = {}
        while True:
            selected, selection_metrics = _select_candidate_assignment(
                pools=pools,
                parent=parent,
                config=self.config,
            )
            selection_rounds.append(
                {
                    "round": len(selection_rounds) + 1,
                    "candidate_counts": [
                        len(pool.candidates) for pool in pools
                    ],
                    "selection": selection_metrics,
                }
            )
            if selected is not None:
                break

            empty_pools = [pool for pool in pools if not pool.candidates]
            expansion_targets = (
                empty_pools
                if empty_pools
                else [
                    pool
                    for pool in pools
                    if len(pool.attempts) < len(pool.prompts)
                ]
            )
            expansion_targets = [
                pool
                for pool in expansion_targets
                if len(pool.attempts) < len(pool.prompts)
            ]
            if not expansion_targets:
                break
            for pool in expansion_targets:
                self._collect_candidate_attempt(
                    pool=pool,
                    prompt_index=len(pool.attempts),
                    image_path=image_path,
                    image=image,
                    parent=parent,
                )

        candidate_visualization_path = run_dir / "candidate-pools.png"
        _render_candidate_pool_visualization(
            image,
            pools=pools,
            selected=selected or (),
            output_path=candidate_visualization_path,
        )
        selection_path = run_dir / "candidate-selection.json"
        selection_payload: dict[str, JsonValue] = {
            "format": "sam3-instance-candidate-selection/v1",
            "status": "selected" if selected is not None else "failed",
            "algorithm": "bounded-global-instance-assignment/v1",
            "rounds": selection_rounds,
            "pools": [pool.as_payload() for pool in pools],
            "selected_candidate_ids": (
                [item.candidate_id for item in selected]
                if selected is not None
                else []
            ),
            "candidate_visualization_path": str(
                candidate_visualization_path
            ),
            "selection_path": str(selection_path),
        }
        if selected is None:
            empty_labels = [
                pool.specification.label
                for pool in pools
                if not pool.candidates
            ]
            detail = (
                "no retained candidate for " + ", ".join(empty_labels)
                if empty_labels
                else str(selection_metrics.get("failure_reason", ""))
            )
            fallback_reason = (
                "SAM3 global instance selection found no feasible subpart "
                "assignment after bounded synonym attempts: " + detail
            )
            fallback = self._prepare_parent_fallback(
                image=image,
                parent=parent,
                pools=pools,
                mapping=mapping,
                run_dir=run_dir,
                reason=fallback_reason,
            )
            selection_payload["status"] = "fallback_parent"
            selection_payload["fallback"] = {
                "applied": True,
                "mode": "WHOLE_PART_SINGLE_FACE_SET",
                "reason": fallback_reason,
                "failed_subparts": empty_labels,
                "pose_ik_segments": 1,
            }
            _write_json(selection_path, selection_payload)
            return [fallback], selection_payload

        try:
            selected, partition = _enforce_exact_parent_partition(
                selected,
                parent=parent,
                image=image,
                overlay_opacity=self.config.sam3_overlay_opacity,
            )
        except PartSegmentationSam3Error as error:
            fallback_reason = (
                "Selected SAM3 subparts could not form an exact parent-mask "
                f"partition: {error}"
            )
            fallback = self._prepare_parent_fallback(
                image=image,
                parent=parent,
                pools=pools,
                mapping=mapping,
                run_dir=run_dir,
                reason=fallback_reason,
            )
            selection_payload["status"] = "fallback_parent"
            selection_payload["fallback"] = {
                "applied": True,
                "mode": "WHOLE_PART_SINGLE_FACE_SET",
                "reason": fallback_reason,
                "failed_subparts": [
                    pool.specification.label for pool in pools
                ],
                "pose_ik_segments": 1,
            }
            _write_json(selection_path, selection_payload)
            return [fallback], selection_payload

        selection_payload["parent_partition"] = partition
        selection_payload["fallback"] = {
            "applied": False,
            "mode": "KINEMATIC_SUBPARTS",
        }
        selection_payload["selected"] = [
            item.as_payload() for item in selected
        ]
        _write_json(selection_path, selection_payload)

        prepared: list[_PreparedSubpart] = []
        for pool, candidate in zip(pools, selected, strict=True):
            specification = pool.specification
            order = pool.order
            try:
                lasso = generate_face_set_lasso(
                    cleaned_mask_path=candidate.cleaned_mask_path,
                    mapping=mapping,
                    padding_pixels=self.config.lasso_padding_pixels,
                    simplify_tolerance_pixels=(
                        self.config.lasso_simplify_tolerance_pixels
                    ),
                    max_lasso_points=self.config.max_lasso_points,
                    time_step_seconds=self.config.lasso_time_step_seconds,
                    path_name=(
                        f"face-set-{order:02d}-"
                        f"{_slug(specification.label)}"
                    ),
                )
            except PartSegmentationLassoError:
                raise
            except Exception as error:
                raise PartSegmentationLassoError(
                    f"Cannot generate lasso for "
                    f"{specification.label}: {error}"
                ) from error
            prepared.append(
                _PreparedSubpart(
                    order=order,
                    specification=specification,
                    selected_prompt=candidate.prompt,
                    attempts=_selected_attempts(
                        pool.attempts,
                        selected_prompt=candidate.prompt,
                    ),
                    segment_response=candidate.segment_response,
                    constraint=candidate.constraint,
                    cleaned_mask_path=candidate.cleaned_mask_path,
                    overlay_path=str(candidate.overlay_path),
                    lasso=lasso,
                    rpc_request=_lasso_rpc_request(
                        lasso=lasso,
                        mapping=mapping,
                        use_front_faces_only=(
                            self.config.use_front_faces_only
                        ),
                    ),
                )
            )
        return prepared, selection_payload

    def _prepare_parent_fallback(
        self,
        *,
        image: Image.Image,
        parent: _ParentSegmentation,
        pools: Sequence[_CandidatePool],
        mapping: ScreenshotViewportMapping,
        run_dir: Path,
        reason: str,
        padding_pixels: int | None = None,
        inherited_attempts: Sequence[_SegmentationAttempt] = (),
    ) -> _PreparedSubpart:
        """Use the complete parent mask when any child plan is unusable."""
        fallback_dir = run_dir / "whole-parent-fallback"
        _create_artifact_directory(
            fallback_dir,
            label="whole-parent Face Set fallback",
        )
        mask_path = fallback_dir / "whole-parent-mask.png"
        overlay_path = fallback_dir / "whole-parent-overlay.png"
        _save_mask(parent.mask, mask_path)
        _render_mask_overlay(
            image,
            mask=parent.mask,
            opacity=self.config.sam3_overlay_opacity,
            output_path=overlay_path,
        )
        specification = KinematicSubpart(
            label=parent.prompt,
            sam3_prompt=parent.prompt,
            fallback_prompts=[],
            rationale=(
                "Use the complete operation part as one Face Set because the "
                "planned kinematic subparts were not all usable."
            ),
        )
        lasso = generate_face_set_lasso(
            cleaned_mask_path=mask_path,
            mapping=mapping,
            padding_pixels=(
                self.config.lasso_padding_pixels
                if padding_pixels is None
                else padding_pixels
            ),
            simplify_tolerance_pixels=(
                self.config.lasso_simplify_tolerance_pixels
            ),
            max_lasso_points=self.config.max_lasso_points,
            time_step_seconds=self.config.lasso_time_step_seconds,
            path_name=f"face-set-01-{_slug(parent.prompt)}-fallback",
        )
        attempts = (
            tuple(
                attempt
                for pool in pools
                for attempt in pool.attempts
            )
            + tuple(inherited_attempts)
        )
        return _PreparedSubpart(
            order=1,
            specification=specification,
            selected_prompt=parent.prompt,
            attempts=attempts,
            segment_response=parent.segment_response,
            constraint={
                "algorithm": "whole-parent-face-set-fallback/v1",
                "fallback_applied": True,
                "reason": reason,
                "parent_mask_path": str(parent.cleaned_mask_path),
                "final_mask_path": str(mask_path),
                "parent_foreground_pixels": parent.foreground_pixels,
                "final_foreground_pixels": int(
                    np.count_nonzero(parent.mask)
                ),
                "subset_of_parent": True,
                "covers_parent": True,
                "pose_ik_segments": 1,
            },
            cleaned_mask_path=mask_path,
            overlay_path=str(overlay_path),
            lasso=lasso,
            rpc_request=_lasso_rpc_request(
                lasso=lasso,
                mapping=mapping,
                use_front_faces_only=self.config.use_front_faces_only,
            ),
        )

    def _collect_candidate_attempt(
        self,
        *,
        pool: _CandidatePool,
        prompt_index: int,
        image_path: Path,
        image: Image.Image,
        parent: _ParentSegmentation,
    ) -> None:
        """Add independently cleaned instances from one bounded prompt call."""
        prompt = pool.prompts[prompt_index]
        attempt_dir = pool.output_dir / (
            f"attempt-{prompt_index + 1:02d}-{_slug(prompt)}"
        )
        _create_artifact_directory(
            attempt_dir,
            label="SAM3 synonym attempt",
        )
        response: dict[str, JsonValue] | None = None
        try:
            response = self._segment_prompt(
                image_path=image_path,
                prompt=prompt,
                output_dir=attempt_dir,
            )
            candidates, rejected = _build_instance_candidates(
                image=image,
                response=response,
                parent=parent,
                prompt=prompt,
                prompt_index=prompt_index,
                subpart_order=pool.order,
                output_dir=attempt_dir,
                parent_dilation_pixels=(
                    self.config.parent_mask_dilation_pixels
                ),
                minimum_containment=(
                    self.config.minimum_child_parent_containment
                ),
                overlay_opacity=self.config.sam3_overlay_opacity,
                max_instances=self.config.max_instances_per_prompt,
            )
            pool.rejected_candidates.extend(rejected)
            added = _append_distinct_candidates(
                pool,
                candidates,
                duplicate_iou=self.config.duplicate_candidate_iou,
            )
            if not candidates:
                raise PartSegmentationSam3Error(
                    f"SAM3 prompt {prompt} produced no parent-contained "
                    "instance candidate"
                )
            status = "candidates_collected" if added else "duplicates_only"
            pool.attempts.append(
                _SegmentationAttempt(
                    prompt=prompt,
                    status=status,
                    output_dir=attempt_dir,
                    response=response,
                )
            )
        except PartSegmentationSam3Error as error:
            pool.attempts.append(
                _SegmentationAttempt(
                    prompt=prompt,
                    status="failed",
                    output_dir=attempt_dir,
                    response=response,
                    error=str(error),
                )
            )

    def _segment_prompt(
        self,
        *,
        image_path: Path,
        prompt: str,
        output_dir: Path,
    ) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "image_path": str(image_path),
            "prompt": prompt,
            "confidence_threshold": self.config.sam3_confidence_threshold,
            "overlay_opacity": self.config.sam3_overlay_opacity,
            "output_dir": str(output_dir),
        }
        try:
            response = self._segment_tool.invoke(payload)
        except Exception as error:
            raise PartSegmentationSam3Error(
                f"SAM3 failed for {prompt}: {error}"
            ) from error
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError as error:
                raise PartSegmentationSam3Error(
                    f"SAM3 failed for {prompt}: {response}"
                ) from error
        if not isinstance(response, dict):
            raise PartSegmentationSam3Error(
                f"SAM3 returned an invalid response for {prompt}"
            )
        sam3_error = response.get("sam3_error")
        if _is_empty_sam3_error(sam3_error):
            raise PartSegmentationNoMaskError(
                f"SAM3 produced no mask for {prompt}"
            )
        if sam3_error is not None:
            raise PartSegmentationSam3Error(
                f"SAM3 failed for {prompt}: "
                + json.dumps(
                    sam3_error,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        result = response.get("result")
        if not isinstance(result, dict):
            raise PartSegmentationSam3Error(
                f"SAM3 response for {prompt} has no result"
            )
        return cast(dict[str, JsonValue], response)

    def _send_rpc(
        self,
        request: dict[str, JsonValue],
        *,
        label: str,
    ) -> dict[str, JsonValue]:
        try:
            response = self._rpc_client.send(request)
        except Exception as error:
            raise PartSegmentationRpcError(
                f"Blender RPC failed during {label}: {error}"
            ) from error
        if not isinstance(response, dict):
            raise PartSegmentationRpcError(
                f"Blender RPC returned a non-object response during {label}"
            )
        if "error" in response:
            raise PartSegmentationRpcError(
                f"Blender RPC rejected {label}: "
                + json.dumps(
                    response["error"],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        _rpc_result(cast(dict[str, JsonValue], response), label=label)
        return cast(dict[str, JsonValue], response)

    def _execute_face_set_partition(
        self,
        *,
        image: Image.Image,
        parent: _ParentSegmentation,
        prepared: list[_PreparedSubpart],
        candidate_selection: dict[str, JsonValue],
        mapping: ScreenshotViewportMapping,
        run_dir: Path,
    ) -> tuple[
        list[_PreparedSubpart],
        dict[str, JsonValue],
        dict[str, JsonValue],
    ]:
        """Apply a verified padding ladder, then commit one safe fallback."""
        attempts: list[dict[str, JsonValue]] = []
        maximum_increases = self.config.lasso_max_padding_increases
        multiplier = self.config.lasso_padding_retry_multiplier
        base_padding = self.config.lasso_padding_pixels
        for increase_index in range(maximum_increases + 1):
            padding = base_padding * multiplier**increase_index
            self._regenerate_lassos(
                prepared,
                mapping=mapping,
                padding_pixels=padding,
            )
            request = _lasso_batch_rpc_request(
                prepared=prepared,
                mapping=mapping,
                padding_pixels=padding,
                use_front_faces_only=self.config.use_front_faces_only,
                rollback_on_residual=True,
            )
            response = self._send_rpc(
                request,
                label=f"verified Face Set batch at {padding}px padding",
            )
            result = _rpc_result(
                response,
                label=f"verified Face Set batch at {padding}px padding",
            )
            coverage = _face_set_coverage_result(result)
            attempt: dict[str, JsonValue] = {
                "attempt": increase_index + 1,
                "padding_pixels": padding,
                "request": request,
                "response": response,
                "coverage": coverage,
            }
            attempts.append(attempt)
            if coverage["complete"] is True:
                for item in prepared:
                    item.rpc_response = response
                candidate_selection["face_set_coverage"] = {
                    "status": "complete",
                    "padding_pixels": padding,
                    "attempt_count": len(attempts),
                    "coverage": coverage,
                }
                _rewrite_candidate_selection(candidate_selection)
                return prepared, candidate_selection, {
                    "format": "verified-face-set-execution/v1",
                    "status": "complete",
                    "attempts": attempts,
                    "final_padding_pixels": padding,
                    "fallback_commit": None,
                }
            rollback = result.get("rollback")
            if not isinstance(rollback, dict) or (
                rollback.get("performed") is not True
                or rollback.get("restored") is not True
            ):
                raise PartSegmentationRpcError(
                    "Blender reported residual original faces without "
                    "restoring the pre-attempt Face Set state"
                )

        final_padding = base_padding * multiplier**maximum_increases
        fallback_value = candidate_selection.get("fallback")
        already_parent_fallback = bool(
            isinstance(fallback_value, dict)
            and fallback_value.get("applied") is True
        )
        residual_counts = [
            cast(dict[str, JsonValue], attempt["coverage"])[
                "residual_original_face_count"
            ]
            for attempt in attempts
        ]
        retry_reason = (
            "Residual original faces remained after the default padding "
            f"and {maximum_increases} larger-padding attempts "
            f"({residual_counts})."
        )
        if not already_parent_fallback:
            inherited_attempts = tuple(
                attempt
                for item in prepared
                for attempt in item.attempts
            )
            prepared = [
                self._prepare_parent_fallback(
                    image=image,
                    parent=parent,
                    pools=(),
                    mapping=mapping,
                    run_dir=run_dir,
                    reason=retry_reason,
                    padding_pixels=final_padding,
                    inherited_attempts=inherited_attempts,
                )
            ]
            candidate_selection["status"] = "fallback_parent"
            candidate_selection["fallback"] = {
                "applied": True,
                "mode": "WHOLE_PART_SINGLE_FACE_SET",
                "reason": retry_reason,
                "failed_subparts": [],
                "pose_ik_segments": 1,
                "trigger": "RESIDUAL_ORIGINAL_FACES",
            }
        else:
            existing_fallback = cast(
                dict[str, JsonValue],
                fallback_value,
            )
            existing_reason = existing_fallback.get("reason")
            if isinstance(existing_reason, str):
                existing_fallback["reason"] = (
                    existing_reason + " " + retry_reason
                )

        self._regenerate_lassos(
            prepared,
            mapping=mapping,
            padding_pixels=final_padding,
        )
        fallback_request = _lasso_batch_rpc_request(
            prepared=prepared,
            mapping=mapping,
            padding_pixels=final_padding,
            use_front_faces_only=self.config.use_front_faces_only,
            rollback_on_residual=True,
        )
        fallback_response = self._send_rpc(
            fallback_request,
            label="whole-operation-part Face Set fallback",
        )
        fallback_result = _rpc_result(
            fallback_response,
            label="whole-operation-part Face Set fallback",
        )
        fallback_coverage = _face_set_coverage_result(fallback_result)
        if fallback_coverage["complete"] is not True:
            residual = fallback_coverage["residual_original_face_count"]
            raise PartSegmentationRpcError(
                "Whole-operation-part Face Set fallback still left "
                f"{residual} original faces; the transaction was rolled back"
            )
        for item in prepared:
            item.rpc_response = fallback_response
        candidate_selection["face_set_coverage"] = {
            "status": "fallback_committed",
            "padding_pixels": final_padding,
            "attempt_count": len(attempts),
            "coverage": fallback_coverage,
        }
        _rewrite_candidate_selection(candidate_selection)
        return prepared, candidate_selection, {
            "format": "verified-face-set-execution/v1",
            "status": "fallback_committed",
            "attempts": attempts,
            "final_padding_pixels": final_padding,
            "fallback_commit": {
                "request": fallback_request,
                "response": fallback_response,
                "coverage": fallback_coverage,
            },
        }

    def _regenerate_lassos(
        self,
        prepared: Sequence[_PreparedSubpart],
        *,
        mapping: ScreenshotViewportMapping,
        padding_pixels: int,
    ) -> None:
        """Regenerate all deterministic paths for one padding attempt."""
        for item in prepared:
            path_name = (
                str(item.lasso.operator_path[0].get("name", "")).strip()
                if item.lasso.operator_path
                else ""
            ) or (
                f"face-set-{item.order:02d}-"
                f"{_slug(item.specification.label)}"
            )
            item.lasso = generate_face_set_lasso(
                cleaned_mask_path=item.cleaned_mask_path,
                mapping=mapping,
                padding_pixels=padding_pixels,
                simplify_tolerance_pixels=(
                    self.config.lasso_simplify_tolerance_pixels
                ),
                max_lasso_points=self.config.max_lasso_points,
                time_step_seconds=self.config.lasso_time_step_seconds,
                path_name=path_name,
            )
            item.rpc_request = _lasso_rpc_request(
                lasso=item.lasso,
                mapping=mapping,
                use_front_faces_only=self.config.use_front_faces_only,
            )

    def _trace_payload(
        self,
        *,
        status: str,
        image_path: Path,
        part_description: str,
        viewport_reference_path: Path,
        viewport_response: dict[str, JsonValue],
        mapping: ScreenshotViewportMapping,
        plan: KinematicPartPlan,
        llm_metadata: dict[str, JsonValue],
        parent: _ParentSegmentation,
        prepared: Sequence[_PreparedSubpart],
        candidate_selection: dict[str, JsonValue],
        validation: dict[str, JsonValue],
        visualization_path: Path,
        face_set_execution: dict[str, JsonValue] | None,
    ) -> dict[str, JsonValue]:
        return {
            "format": "part-segmentation-with-sam3/v5",
            "status": status,
            "input": {
                "image_path": str(image_path),
                "part_description": part_description,
            },
            "config": {
                "max_subparts": self.config.max_subparts,
                "llm_role": self.config.llm_role,
                "sam3_confidence_threshold": (
                    self.config.sam3_confidence_threshold
                ),
                "sam3_overlay_opacity": self.config.sam3_overlay_opacity,
                "roi_padding_ratio": self.config.roi_padding_ratio,
                "parent_mask_dilation_pixels": (
                    self.config.parent_mask_dilation_pixels
                ),
                "max_synonym_attempts": self.config.max_synonym_attempts,
                "minimum_child_parent_containment": (
                    self.config.minimum_child_parent_containment
                ),
                "minimum_parent_coverage": (
                    self.config.minimum_parent_coverage
                ),
                "maximum_pairwise_overlap_ratio": (
                    self.config.maximum_pairwise_overlap_ratio
                ),
                "maximum_uncovered_component_ratio": (
                    self.config.maximum_uncovered_component_ratio
                ),
                "max_instances_per_prompt": (
                    self.config.max_instances_per_prompt
                ),
                "duplicate_candidate_iou": (
                    self.config.duplicate_candidate_iou
                ),
                "max_candidate_combinations": (
                    self.config.max_candidate_combinations
                ),
                "lasso_padding_pixels": self.config.lasso_padding_pixels,
                "lasso_padding_retry_multiplier": (
                    self.config.lasso_padding_retry_multiplier
                ),
                "lasso_max_padding_increases": (
                    self.config.lasso_max_padding_increases
                ),
                "lasso_simplify_tolerance_pixels": (
                    self.config.lasso_simplify_tolerance_pixels
                ),
                "max_lasso_points": self.config.max_lasso_points,
                "lasso_time_step_seconds": (
                    self.config.lasso_time_step_seconds
                ),
                "use_front_faces_only": self.config.use_front_faces_only,
            },
            "viewport": {
                "reference_path": str(viewport_reference_path),
                "capture_response": viewport_response,
                "mapping": mapping.as_payload(),
            },
            "vlm": {
                "plan": plan.model_dump(mode="json"),
                "metadata": llm_metadata,
            },
            "parent": parent.as_payload(),
            "candidate_selection": candidate_selection,
            "validation": validation,
            "face_set_execution": face_set_execution,
            "subparts": [
                {
                    "order": item.order,
                    "specification": item.specification.model_dump(
                        mode="json"
                    ),
                    "selected_prompt": item.selected_prompt,
                    "attempts": [
                        attempt.as_payload()
                        for attempt in item.attempts
                    ],
                    "segment_response": item.segment_response,
                    "constraint": item.constraint,
                    "cleaned_mask_path": str(item.cleaned_mask_path),
                    "sam3_overlay_path": item.overlay_path,
                    "lasso": item.lasso.as_payload(),
                    "rpc_request": item.rpc_request,
                    "rpc_response": item.rpc_response,
                }
                for item in prepared
            ],
            "artifacts": {
                "lasso_visualization_path": str(visualization_path),
            },
        }


def _result_instances(
    response: Mapping[str, JsonValue],
    *,
    label: str,
) -> list[dict[str, JsonValue]]:
    result = response.get("result")
    instances = result.get("instances") if isinstance(result, dict) else None
    if not isinstance(instances, list):
        raise PartSegmentationSam3Error(
            f"SAM3 response for {label} has no per-instance masks; restart "
            "the updated SAM3 inference service"
        )
    normalized: list[dict[str, JsonValue]] = []
    for expected_index, instance in enumerate(instances):
        if not isinstance(instance, dict):
            raise PartSegmentationSam3Error(
                f"SAM3 instance metadata for {label} is invalid"
            )
        if instance.get("instance_index") != expected_index:
            raise PartSegmentationSam3Error(
                f"SAM3 instance indices for {label} are not contiguous"
            )
        normalized.append(cast(dict[str, JsonValue], instance))
    return normalized


def _select_parent_instance(
    response: Mapping[str, JsonValue],
    *,
    expected_size: tuple[int, int],
    prompt: str,
) -> dict[str, object]:
    """Select one instance-specific parent without unioning SAM3 outputs."""
    candidates: list[dict[str, object]] = []
    for instance in _result_instances(response, label=f"parent {prompt}"):
        index = cast(int, instance["instance_index"])
        path_value = instance.get("cleaned_mask_path")
        if not isinstance(path_value, str):
            continue
        path = Path(path_value).expanduser().resolve()
        mask = _binary_mask(
            path,
            expected_size=expected_size,
            label=f"parent {prompt} instance {index}",
        )
        score_value = instance.get("score", 0.0)
        score = (
            float(score_value)
            if isinstance(score_value, (int, float))
            and not isinstance(score_value, bool)
            and math.isfinite(float(score_value))
            else 0.0
        )
        candidates.append(
            {
                "instance_index": index,
                "score": score,
                "cleaned_mask_path": path,
                "mask": mask,
                "foreground_pixels": int(np.count_nonzero(mask)),
            }
        )
    if not candidates:
        raise PartSegmentationNoMaskError(
            f"SAM3 produced no cleaned parent instance for {prompt}"
        )
    return max(
        candidates,
        key=lambda item: (
            cast(float, item["score"]),
            cast(int, item["foreground_pixels"]),
            -cast(int, item["instance_index"]),
        ),
    )


def _build_instance_candidates(
    *,
    image: Image.Image,
    response: dict[str, JsonValue],
    parent: _ParentSegmentation,
    prompt: str,
    prompt_index: int,
    subpart_order: int,
    output_dir: Path,
    parent_dilation_pixels: int,
    minimum_containment: float,
    overlay_opacity: float,
    max_instances: int,
) -> tuple[list[_SubpartCandidate], list[dict[str, JsonValue]]]:
    """Convert each SAM3 instance into a separately constrained candidate."""
    left, top, right, bottom = parent.roi_box
    dilated_parent = _dilate_mask(parent.mask, parent_dilation_pixels) > 0
    parent_logical = parent.mask > 0
    accepted: list[_SubpartCandidate] = []
    rejected: list[dict[str, JsonValue]] = []
    for instance in _result_instances(response, label=prompt):
        instance_index = cast(int, instance["instance_index"])
        path_value = instance.get("cleaned_mask_path")
        if not isinstance(path_value, str):
            rejected.append(
                {
                    "prompt": prompt,
                    "prompt_index": prompt_index,
                    "instance_index": instance_index,
                    "reason": "instance rejected during independent cleanup",
                }
            )
            continue
        raw_mask_path = Path(path_value).expanduser().resolve()
        raw_mask = _binary_mask(
            raw_mask_path,
            expected_size=image.size,
            label=f"child {prompt} instance {instance_index}",
        )
        search_mask = np.zeros_like(raw_mask)
        search_mask[top:bottom, left:right] = raw_mask[
            top:bottom,
            left:right,
        ]
        search_logical = search_mask > 0
        search_pixels = int(np.count_nonzero(search_logical))
        containment_pixels = int(
            np.count_nonzero(search_logical & dilated_parent)
        )
        containment = (
            containment_pixels / search_pixels if search_pixels else 0.0
        )
        rejection_base: dict[str, JsonValue] = {
            "prompt": prompt,
            "prompt_index": prompt_index,
            "instance_index": instance_index,
            "search_foreground_pixels": search_pixels,
            "parent_containment_ratio": round(containment, 8),
        }
        if search_pixels == 0:
            rejected.append(
                {**rejection_base, "reason": "outside parent ROI"}
            )
            continue
        if containment < minimum_containment:
            rejected.append(
                {
                    **rejection_base,
                    "reason": "below minimum parent containment",
                }
            )
            continue

        constrained = np.where(
            search_logical & dilated_parent,
            255,
            0,
        ).astype(np.uint8)
        foreground_pixels = int(np.count_nonzero(constrained))
        if foreground_pixels == 0:
            rejected.append(
                {**rejection_base, "reason": "empty after parent constraint"}
            )
            continue
        candidate_id = (
            f"s{subpart_order:02d}-p{prompt_index + 1:02d}-"
            f"i{instance_index:03d}"
        )
        candidate_mask_path = output_dir / (
            f"{candidate_id}-constrained-mask.png"
        )
        candidate_overlay_path = output_dir / (
            f"{candidate_id}-constrained-overlay.png"
        )
        _save_mask(constrained, candidate_mask_path)
        _render_mask_overlay(
            image,
            mask=constrained,
            opacity=overlay_opacity,
            output_path=candidate_overlay_path,
        )
        score = _instance_score(instance)
        box_xyxy = _instance_box(instance)
        actual_parent_overlap = int(
            np.count_nonzero((constrained > 0) & parent_logical)
        )
        accepted.append(
            _SubpartCandidate(
                candidate_id=candidate_id,
                prompt=prompt,
                prompt_index=prompt_index,
                instance_index=instance_index,
                score=score,
                box_xyxy=box_xyxy,
                segment_response=response,
                constraint={
                    "algorithm": "parent-constrained-sam3-instance/v2",
                    "prompt": prompt,
                    "sam3_input_space": "FULL_IMAGE",
                    "sam3_instance_index": instance_index,
                    "sam3_instance_score": score,
                    "sam3_instance_box_xyxy": (
                        list(box_xyxy) if box_xyxy is not None else None
                    ),
                    "raw_cleaned_mask_path": str(raw_mask_path),
                    "parent_mask_path": str(parent.cleaned_mask_path),
                    "search_roi": {
                        "coordinate_system": "IMAGE_TOP_LEFT_PIXELS",
                        "left": left,
                        "top": top,
                        "right_exclusive": right,
                        "bottom_exclusive": bottom,
                    },
                    "parent_dilation_pixels": parent_dilation_pixels,
                    "minimum_parent_containment": minimum_containment,
                    "search_foreground_pixels": search_pixels,
                    "selected_foreground_pixels": foreground_pixels,
                    "parent_containment_ratio_before_clip": round(
                        containment,
                        8,
                    ),
                    "actual_parent_overlap_pixels": actual_parent_overlap,
                    "actual_parent_containment_ratio": round(
                        actual_parent_overlap / foreground_pixels,
                        8,
                    ),
                    "component_count": _component_count(constrained),
                    "constrained_mask_path": str(candidate_mask_path),
                    "constrained_overlay_path": str(
                        candidate_overlay_path
                    ),
                },
                cleaned_mask_path=candidate_mask_path,
                overlay_path=candidate_overlay_path,
                mask=constrained,
            )
        )

    accepted.sort(
        key=lambda item: (
            -item.score,
            -int(np.count_nonzero(item.mask)),
            item.instance_index,
        )
    )
    for candidate in accepted[max_instances:]:
        rejected.append(
            {
                "candidate_id": candidate.candidate_id,
                "prompt": prompt,
                "instance_index": candidate.instance_index,
                "reason": "exceeds max_instances_per_prompt",
            }
        )
    return accepted[:max_instances], rejected


def _instance_score(instance: Mapping[str, JsonValue]) -> float:
    value = instance.get("score", 0.0)
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ):
        return float(value)
    return 0.0


def _instance_box(
    instance: Mapping[str, JsonValue],
) -> tuple[float, float, float, float] | None:
    value = instance.get("box_xyxy")
    if not isinstance(value, list) or len(value) != 4:
        return None
    if not all(
        isinstance(item, (int, float))
        and not isinstance(item, bool)
        and math.isfinite(float(item))
        for item in value
    ):
        return None
    return cast(
        tuple[float, float, float, float],
        tuple(float(item) for item in value),
    )


def _append_distinct_candidates(
    pool: _CandidatePool,
    candidates: Sequence[_SubpartCandidate],
    *,
    duplicate_iou: float,
) -> int:
    added = 0
    for candidate in candidates:
        duplicate = next(
            (
                existing
                for existing in pool.candidates
                if _mask_iou(existing.mask, candidate.mask) >= duplicate_iou
            ),
            None,
        )
        if duplicate is not None:
            pool.rejected_candidates.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "prompt": candidate.prompt,
                    "instance_index": candidate.instance_index,
                    "reason": "duplicate candidate",
                    "duplicate_of": duplicate.candidate_id,
                    "intersection_over_union": round(
                        _mask_iou(duplicate.mask, candidate.mask),
                        8,
                    ),
                }
            )
            continue
        pool.candidates.append(candidate)
        added += 1
    pool.candidates.sort(
        key=lambda item: (
            item.prompt_index,
            -item.score,
            item.instance_index,
            item.candidate_id,
        )
    )
    return added


def _mask_iou(left: MaskArray, right: MaskArray) -> float:
    left_logical = left > 0
    right_logical = right > 0
    union = int(np.count_nonzero(left_logical | right_logical))
    if union == 0:
        return 0.0
    intersection = int(np.count_nonzero(left_logical & right_logical))
    return intersection / union


def _select_candidate_assignment(
    *,
    pools: Sequence[_CandidatePool],
    parent: _ParentSegmentation,
    config: PartSegmentationConfig,
) -> tuple[list[_SubpartCandidate] | None, dict[str, JsonValue]]:
    """Exhaustively rank feasible assignments under a deterministic bound."""
    empty_labels = [
        pool.specification.label for pool in pools if not pool.candidates
    ]
    if empty_labels:
        return None, {
            "feasible": False,
            "failure_reason": "one or more candidate pools are empty",
            "empty_subparts": empty_labels,
            "search_steps": 0,
            "complete_assignments_evaluated": 0,
        }

    best: list[_SubpartCandidate] | None = None
    best_rank: tuple[object, ...] | None = None
    best_metrics: dict[str, JsonValue] = {}
    search_steps = 0
    complete_assignments = 0
    feasible_assignments = 0
    pairwise_pruned = 0
    coverage_rejected = 0
    uncovered_rejected = 0
    truncated = False
    parent_logical = parent.mask > 0
    parent_pixels = int(np.count_nonzero(parent_logical))
    pairwise_cache = _precompute_candidate_overlaps(pools)

    def search(chosen: list[_SubpartCandidate]) -> None:
        nonlocal best
        nonlocal best_rank
        nonlocal best_metrics
        nonlocal search_steps
        nonlocal complete_assignments
        nonlocal feasible_assignments
        nonlocal pairwise_pruned
        nonlocal coverage_rejected
        nonlocal uncovered_rejected
        nonlocal truncated
        if truncated:
            return
        if len(chosen) == len(pools):
            complete_assignments += 1
            metrics = _candidate_assignment_metrics(
                chosen,
                parent_logical=parent_logical,
                parent_pixels=parent_pixels,
                pairwise_cache=pairwise_cache,
            )
            coverage = cast(float, metrics["parent_coverage_ratio"])
            largest_uncovered = cast(
                float,
                metrics["largest_uncovered_component_ratio"],
            )
            if coverage < config.minimum_parent_coverage:
                coverage_rejected += 1
                return
            if (
                largest_uncovered
                > config.maximum_uncovered_component_ratio
            ):
                uncovered_rejected += 1
                return
            feasible_assignments += 1
            rank: tuple[object, ...] = (
                cast(float, metrics["maximum_pairwise_overlap_ratio"]),
                cast(float, metrics["total_pairwise_overlap_ratio"]),
                -coverage,
                -cast(float, metrics["total_sam3_score"]),
                tuple(item.candidate_id for item in chosen),
            )
            if best_rank is None or rank < best_rank:
                best = list(chosen)
                best_rank = rank
                best_metrics = metrics
            return

        pool = pools[len(chosen)]
        for candidate in pool.candidates:
            if search_steps >= config.max_candidate_combinations:
                truncated = True
                return
            search_steps += 1
            if any(
                pairwise_cache[_candidate_pair_key(candidate, item)]
                > config.maximum_pairwise_overlap_ratio
                for item in chosen
            ):
                pairwise_pruned += 1
                continue
            chosen.append(candidate)
            search(chosen)
            chosen.pop()
            if truncated:
                return

    search([])
    common: dict[str, JsonValue] = {
        "algorithm": "bounded-global-instance-assignment/v1",
        "candidate_counts": [len(pool.candidates) for pool in pools],
        "search_steps": search_steps,
        "search_step_limit": config.max_candidate_combinations,
        "complete_assignments_evaluated": complete_assignments,
        "feasible_assignments": feasible_assignments,
        "pairwise_pruned_partial_assignments": pairwise_pruned,
        "precomputed_pairwise_relations": len(pairwise_cache),
        "pairwise_relations": [
            {
                "left_candidate_id": key[0],
                "right_candidate_id": key[1],
                "overlap_ratio_of_smaller_mask": round(ratio, 8),
                "passes_hard_constraint": (
                    ratio <= config.maximum_pairwise_overlap_ratio
                ),
            }
            for key, ratio in sorted(pairwise_cache.items())
        ],
        "coverage_rejected_assignments": coverage_rejected,
        "uncovered_component_rejected_assignments": uncovered_rejected,
        "truncated": truncated,
        "thresholds": {
            "minimum_parent_coverage": config.minimum_parent_coverage,
            "maximum_pairwise_overlap_ratio": (
                config.maximum_pairwise_overlap_ratio
            ),
            "maximum_uncovered_component_ratio": (
                config.maximum_uncovered_component_ratio
            ),
        },
    }
    if truncated:
        return None, {
            **common,
            "feasible": False,
            "failure_reason": "candidate search exceeded deterministic bound",
        }
    if best is None:
        return None, {
            **common,
            "feasible": False,
            "failure_reason": "no assignment satisfies all hard constraints",
        }
    return best, {
        **common,
        "feasible": True,
        "selected_candidate_ids": [item.candidate_id for item in best],
        "selected_metrics": best_metrics,
    }


def _candidate_assignment_metrics(
    candidates: Sequence[_SubpartCandidate],
    *,
    parent_logical: NDArray[np.bool_],
    parent_pixels: int,
    pairwise_cache: Mapping[tuple[str, str], float],
) -> dict[str, JsonValue]:
    logical_masks = [item.mask > 0 for item in candidates]
    union = np.logical_or.reduce(logical_masks)
    coverage_pixels = int(np.count_nonzero(union & parent_logical))
    coverage = coverage_pixels / parent_pixels if parent_pixels else 0.0
    uncovered = np.where(parent_logical & ~union, 255, 0).astype(np.uint8)
    largest_uncovered_pixels = _largest_component_area(uncovered)
    largest_uncovered_ratio = (
        largest_uncovered_pixels / parent_pixels if parent_pixels else 0.0
    )
    pairwise: list[dict[str, JsonValue]] = []
    ratios: list[float] = []
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            ratio = pairwise_cache[_candidate_pair_key(left, right)]
            ratios.append(ratio)
            pairwise.append(
                {
                    "left_candidate_id": left.candidate_id,
                    "right_candidate_id": right.candidate_id,
                    "overlap_ratio_of_smaller_mask": round(ratio, 8),
                }
            )
    return {
        "parent_foreground_pixels": parent_pixels,
        "covered_parent_pixels": coverage_pixels,
        "parent_coverage_ratio": round(coverage, 8),
        "largest_uncovered_component_pixels": largest_uncovered_pixels,
        "largest_uncovered_component_ratio": round(
            largest_uncovered_ratio,
            8,
        ),
        "maximum_pairwise_overlap_ratio": round(max(ratios, default=0.0), 8),
        "total_pairwise_overlap_ratio": round(sum(ratios), 8),
        "total_sam3_score": round(sum(item.score for item in candidates), 8),
        "pairwise_overlaps": pairwise,
    }


def _pairwise_overlap_ratio(left: MaskArray, right: MaskArray) -> float:
    left_logical = left > 0
    right_logical = right > 0
    smaller = min(
        int(np.count_nonzero(left_logical)),
        int(np.count_nonzero(right_logical)),
    )
    if smaller == 0:
        return 0.0
    intersection = int(np.count_nonzero(left_logical & right_logical))
    return intersection / smaller


def _candidate_pair_key(
    left: _SubpartCandidate,
    right: _SubpartCandidate,
) -> tuple[str, str]:
    if left.candidate_id <= right.candidate_id:
        return left.candidate_id, right.candidate_id
    return right.candidate_id, left.candidate_id


def _precompute_candidate_overlaps(
    pools: Sequence[_CandidatePool],
) -> dict[tuple[str, str], float]:
    overlaps: dict[tuple[str, str], float] = {}
    for left_index, left_pool in enumerate(pools):
        for right_pool in pools[left_index + 1 :]:
            for left in left_pool.candidates:
                for right in right_pool.candidates:
                    overlaps[_candidate_pair_key(left, right)] = (
                        _pairwise_overlap_ratio(left.mask, right.mask)
                    )
    return overlaps


def _enforce_exact_parent_partition(
    candidates: Sequence[_SubpartCandidate],
    *,
    parent: _ParentSegmentation,
    image: Image.Image,
    overlay_opacity: float,
) -> tuple[list[_SubpartCandidate], dict[str, JsonValue]]:
    """Turn selected child masks into an exact partition of the parent."""
    if not candidates:
        raise PartSegmentationSam3Error(
            "Cannot partition a parent mask without child candidates"
        )
    parent_logical = parent.mask > 0
    parent_pixels = int(np.count_nonzero(parent_logical))
    if parent_pixels == 0:
        raise PartSegmentationSam3Error(
            "Cannot partition an empty parent mask"
        )
    raw_stack = np.stack([item.mask > 0 for item in candidates], axis=0)
    stack = raw_stack & parent_logical[np.newaxis, ...]
    empty_candidates = [
        candidate.candidate_id
        for candidate, mask in zip(candidates, stack, strict=True)
        if not np.any(mask)
    ]
    if empty_candidates:
        raise PartSegmentationSam3Error(
            "child candidates became empty after exact parent clipping: "
            + ", ".join(empty_candidates)
        )

    membership = np.sum(stack, axis=0)
    shared = membership > 1
    shared_pixels = int(np.count_nonzero(shared))
    uncovered = parent_logical & (membership == 0)
    uncovered_pixels = int(np.count_nonzero(uncovered))

    owners = np.full(parent_logical.shape, -1, dtype=np.int16)
    best_distance = np.full(
        parent_logical.shape,
        np.inf,
        dtype=np.float32,
    )
    for index, mask in enumerate(stack):
        exclusive = mask & (membership == 1)
        seed = exclusive if np.any(exclusive) else mask
        distance_source = np.where(seed, 0, 255).astype(np.uint8)
        distances = cv2.distanceTransform(
            distance_source,
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE,
        )
        eligible = parent_logical & ((membership == 0) | mask)
        better = eligible & (distances < best_distance)
        owners[better] = index
        best_distance[better] = distances[better]

    if np.any(parent_logical & (owners < 0)):
        raise PartSegmentationSam3Error(
            "nearest-seed partition left parent pixels without an owner"
        )
    resolved = [
        parent_logical & (owners == index)
        for index in range(len(candidates))
    ]
    if any(not np.any(mask) for mask in resolved):
        emptied = [
            candidate.candidate_id
            for candidate, mask in zip(candidates, resolved, strict=True)
            if not np.any(mask)
        ]
        raise PartSegmentationSam3Error(
            "exact parent partition emptied child candidates: "
            + ", ".join(emptied)
        )

    resolved_stack = np.stack(resolved, axis=0)
    resolved_membership = np.sum(resolved_stack, axis=0)
    missing_after = int(
        np.count_nonzero(parent_logical & (resolved_membership == 0))
    )
    shared_after = int(
        np.count_nonzero(parent_logical & (resolved_membership > 1))
    )
    outside_after = int(
        np.count_nonzero(
            np.logical_or.reduce(resolved_stack) & ~parent_logical
        )
    )
    if missing_after or shared_after or outside_after:
        raise PartSegmentationSam3Error(
            "exact parent partition invariant failed "
            f"(missing={missing_after}, shared={shared_after}, "
            f"outside={outside_after})"
        )

    output: list[_SubpartCandidate] = []
    partition_items: list[dict[str, JsonValue]] = []
    for index, (candidate, logical) in enumerate(
        zip(candidates, resolved, strict=True)
    ):
        mask = np.where(logical, 255, 0).astype(np.uint8)
        path = candidate.cleaned_mask_path.with_name(
            f"{candidate.candidate_id}-partition-mask.png"
        )
        overlay_path = candidate.overlay_path.with_name(
            f"{candidate.candidate_id}-partition-overlay.png"
        )
        _save_mask(mask, path)
        _render_mask_overlay(
            image,
            mask=mask,
            opacity=overlay_opacity,
            output_path=overlay_path,
        )
        raw_pixels = int(np.count_nonzero(raw_stack[index]))
        clipped_pixels = int(np.count_nonzero(stack[index]))
        final_pixels = int(np.count_nonzero(logical))
        record: dict[str, JsonValue] = {
            "candidate_id": candidate.candidate_id,
            "order": index + 1,
            "foreground_pixels_before": raw_pixels,
            "foreground_pixels_inside_parent_before": clipped_pixels,
            "outside_parent_pixels_removed": raw_pixels - clipped_pixels,
            "foreground_pixels_after": final_pixels,
            "uncovered_parent_pixels_assigned": int(
                np.count_nonzero(uncovered & (owners == index))
            ),
            "shared_parent_pixels_owned": int(
                np.count_nonzero(shared & (owners == index))
            ),
            "subset_of_parent": True,
        }
        constraint = dict(candidate.constraint)
        constraint["parent_partition"] = record
        constraint["constrained_mask_path"] = str(path)
        constraint["constrained_overlay_path"] = str(overlay_path)
        output.append(
            replace(
                candidate,
                constraint=constraint,
                cleaned_mask_path=path,
                overlay_path=overlay_path,
                mask=mask,
            )
        )
        partition_items.append(record)
    return output, {
        "algorithm": "exact-parent-nearest-seed-partition/v1",
        "status": "applied",
        "parent_foreground_pixels": parent_pixels,
        "outside_parent_pixels_removed": int(
            np.count_nonzero(raw_stack & ~parent_logical[np.newaxis, ...])
        ),
        "uncovered_parent_pixels_before": uncovered_pixels,
        "shared_pixels_before": shared_pixels,
        "uncovered_parent_pixels_after": missing_after,
        "shared_pixels_after": shared_after,
        "outside_parent_pixels_after": outside_after,
        "union_equals_parent": True,
        "all_subparts_within_parent": True,
        "tie_breaker": "lower proximal-to-distal subpart order",
        "subparts": partition_items,
    }


def _selected_attempts(
    attempts: Sequence[_SegmentationAttempt],
    *,
    selected_prompt: str,
) -> tuple[_SegmentationAttempt, ...]:
    return tuple(
        replace(
            attempt,
            status=(
                "selected"
                if attempt.prompt == selected_prompt
                else (
                    attempt.status
                    if attempt.status == "failed"
                    else "not_selected"
                )
            ),
        )
        for attempt in attempts
    )


def _validate_part_hierarchy(
    *,
    parent: _ParentSegmentation,
    prepared: Sequence[_PreparedSubpart],
) -> dict[str, JsonValue]:
    """Require child masks to form an exact partition of the parent mask."""
    parent_logical = parent.mask > 0
    parent_pixels = int(np.count_nonzero(parent_logical))
    if parent_pixels == 0:
        raise PartSegmentationSam3Error(
            "Cannot validate subparts against an empty parent mask"
        )
    child_masks: list[NDArray[np.bool_]] = []
    child_metrics: list[dict[str, JsonValue]] = []
    expected_size = (parent.mask.shape[1], parent.mask.shape[0])
    outside_parent_pixels = 0
    for item in prepared:
        child = _binary_mask(
            item.cleaned_mask_path,
            expected_size=expected_size,
            label=item.specification.label,
        ) > 0
        child_pixels = int(np.count_nonzero(child))
        if child_pixels == 0:
            raise PartSegmentationSam3Error(
                f"Constrained mask for {item.specification.label} is empty"
            )
        containment_pixels = int(np.count_nonzero(child & parent_logical))
        outside_pixels = int(np.count_nonzero(child & ~parent_logical))
        outside_parent_pixels += outside_pixels
        containment = containment_pixels / child_pixels
        if outside_pixels:
            raise PartSegmentationSam3Error(
                f"Final mask for {item.specification.label} contains "
                f"{outside_pixels} pixels outside the parent mask"
            )
        child_masks.append(child)
        child_metrics.append(
            {
                "label": item.specification.label,
                "selected_prompt": item.selected_prompt,
                "foreground_pixels": child_pixels,
                "parent_containment_ratio": round(containment, 8),
                "outside_parent_pixels": outside_pixels,
                "subset_of_parent": outside_pixels == 0,
            }
        )

    union = np.logical_or.reduce(child_masks)
    covered_parent_pixels = int(np.count_nonzero(union & parent_logical))
    parent_coverage = covered_parent_pixels / parent_pixels
    pairwise: list[dict[str, JsonValue]] = []
    maximum_overlap = 0.0
    for left_index, left_mask in enumerate(child_masks):
        for right_index in range(left_index + 1, len(child_masks)):
            right_mask = child_masks[right_index]
            intersection = int(np.count_nonzero(left_mask & right_mask))
            smaller = min(
                int(np.count_nonzero(left_mask)),
                int(np.count_nonzero(right_mask)),
            )
            ratio = intersection / smaller if smaller else 0.0
            maximum_overlap = max(maximum_overlap, ratio)
            pairwise.append(
                {
                    "left_label": prepared[left_index].specification.label,
                    "right_label": prepared[right_index].specification.label,
                    "intersection_pixels": intersection,
                    "overlap_ratio_of_smaller_mask": round(ratio, 8),
                }
            )

    uncovered = np.where(parent_logical & ~union, 255, 0).astype(np.uint8)
    uncovered_pixels = int(np.count_nonzero(uncovered))
    largest_uncovered_pixels = _largest_component_area(uncovered)
    largest_uncovered_ratio = largest_uncovered_pixels / parent_pixels
    union_outside_parent_pixels = int(
        np.count_nonzero(union & ~parent_logical)
    )
    failures: list[str] = []
    if covered_parent_pixels != parent_pixels:
        failures.append(
            f"parent union is missing {uncovered_pixels} foreground pixels"
        )
    if maximum_overlap > 0.0:
        failures.append(
            f"final subpart masks overlap by ratio {maximum_overlap:.4f}"
        )
    if outside_parent_pixels or union_outside_parent_pixels:
        failures.append(
            "one or more final subpart masks extend outside the parent mask"
        )
    if failures:
        raise PartSegmentationSam3Error(
            "Parent-constrained subpart validation failed: "
            + "; ".join(failures)
        )
    return {
        "algorithm": "exact-parent-partition-validation/v2",
        "passed": True,
        "parent_foreground_pixels": parent_pixels,
        "covered_parent_pixels": covered_parent_pixels,
        "parent_coverage_ratio": round(parent_coverage, 8),
        "uncovered_parent_pixels": uncovered_pixels,
        "union_equals_parent": (
            covered_parent_pixels == parent_pixels
            and union_outside_parent_pixels == 0
        ),
        "all_subparts_within_parent": outside_parent_pixels == 0,
        "disjoint_subparts": maximum_overlap == 0.0,
        "outside_parent_pixels": outside_parent_pixels,
        "union_outside_parent_pixels": union_outside_parent_pixels,
        "maximum_pairwise_overlap_ratio": round(maximum_overlap, 8),
        "largest_uncovered_component_pixels": largest_uncovered_pixels,
        "largest_uncovered_component_ratio": round(
            largest_uncovered_ratio,
            8,
        ),
        "subparts": child_metrics,
        "pairwise_overlaps": pairwise,
    }


def generate_face_set_lasso(
    *,
    cleaned_mask_path: str | Path,
    mapping: ScreenshotViewportMapping,
    padding_pixels: int,
    simplify_tolerance_pixels: float,
    max_lasso_points: int,
    time_step_seconds: float,
    path_name: str,
) -> FaceSetLassoGeometry:
    """Convert one cleaned mask into a verified closed lasso path."""
    if padding_pixels < 1:
        raise PartSegmentationLassoError(
            "lasso padding must be at least one pixel"
        )
    if max_lasso_points < 4:
        raise PartSegmentationLassoError(
            "max_lasso_points must allow three vertices plus closure"
        )
    if not math.isfinite(time_step_seconds) or time_step_seconds <= 0.0:
        raise PartSegmentationLassoError(
            "lasso time step must be a positive finite number"
        )
    mask_path = Path(cleaned_mask_path).expanduser().resolve()
    grayscale = _load_grayscale_mask(mask_path)
    if grayscale.shape != (mapping.image_height, mapping.image_width):
        raise PartSegmentationLassoError(
            "cleaned mask dimensions do not match the input screenshot"
        )
    try:
        logical = cv2.resize(
            grayscale,
            (mapping.region_width, mapping.region_height),
            interpolation=cv2.INTER_NEAREST,
        )
        logical = np.where(
            logical >= _MASK_THRESHOLD,
            255,
            0,
        ).astype(np.uint8)
        foreground_pixels = int(np.count_nonzero(logical))
        if foreground_pixels == 0:
            raise PartSegmentationLassoError(
                "cleaned SAM3 mask has no foreground"
            )
        component_count = _component_count(logical)
        padded = _dilate_mask(logical, padding_pixels)
        contours, _ = cv2.findContours(
            padded,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contours = [item for item in contours if len(item) >= 3]
        if not contours:
            raise PartSegmentationLassoError(
                "cleaned SAM3 mask has no usable external contour"
            )

        if component_count == 1 and len(contours) == 1:
            candidate = _simplify_contour(
                contours[0],
                tolerance=min(
                    simplify_tolerance_pixels,
                    max(0.0, padding_pixels * 0.5),
                ),
            )
            strategy = "dilated_external_contour"
        else:
            candidate = cv2.convexHull(np.concatenate(contours, axis=0))
            strategy = "dilated_multi_component_convex_hull"

        candidate = _limit_contour_points(
            candidate,
            max_unique_points=max_lasso_points - 1,
            padded_mask=padded,
        )
        missing = _missing_mask_pixels(logical, candidate)
        if missing:
            candidate = cv2.convexHull(np.concatenate(contours, axis=0))
            strategy += "_containment_hull_fallback"
            candidate = _limit_contour_points(
                candidate,
                max_unique_points=max_lasso_points - 1,
                padded_mask=padded,
            )
            missing = _missing_mask_pixels(logical, candidate)
        if missing:
            candidate = _mask_bounding_box_contour(padded)
            strategy += "_bounding_box_fallback"
            missing = _missing_mask_pixels(logical, candidate)
        if missing:
            raise PartSegmentationLassoError(
                "generated lasso does not completely contain the cleaned mask"
            )

        screenshot_points = _closed_contour_points(candidate)
    except cv2.error as error:
        raise PartSegmentationLassoError(
            f"OpenCV lasso planning failed: {error}"
        ) from error

    _validate_closed_points(
        screenshot_points,
        width=mapping.region_width,
        height=mapping.region_height,
        max_lasso_points=max_lasso_points,
    )
    operator_path = tuple(
        {
            "name": path_name[:128],
            "loc": [
                round(x, 6),
                round(mapping.region_height - 1.0 - y, 6),
            ],
            "time": round(index * time_step_seconds, 6),
        }
        for index, (x, y) in enumerate(screenshot_points)
    )
    return FaceSetLassoGeometry(
        operator_path=operator_path,
        screenshot_points=tuple(screenshot_points),
        foreground_pixels=foreground_pixels,
        component_count=component_count,
        strategy=strategy,
        padding_pixels=padding_pixels,
        missing_foreground_pixels=missing,
    )


def _simplify_contour(
    contour: NDArray[np.int32],
    *,
    tolerance: float,
) -> NDArray[np.int32]:
    if tolerance <= 0.0:
        return contour
    return cv2.approxPolyDP(contour, tolerance, True)


def _limit_contour_points(
    contour: NDArray[np.int32],
    *,
    max_unique_points: int,
    padded_mask: MaskArray,
) -> NDArray[np.int32]:
    if len(contour) <= max_unique_points:
        return contour
    perimeter = cv2.arcLength(contour, True)
    low = 0.0
    high = max(1.0, perimeter)
    best: NDArray[np.int32] | None = None
    for _ in range(32):
        midpoint = (low + high) * 0.5
        simplified = cv2.approxPolyDP(contour, midpoint, True)
        if 3 <= len(simplified) <= max_unique_points:
            best = simplified
            high = midpoint
        elif len(simplified) > max_unique_points:
            low = midpoint
        else:
            high = midpoint
    if best is not None and _missing_mask_pixels(padded_mask, best) == 0:
        return best
    return _mask_bounding_box_contour(padded_mask)


def _mask_bounding_box_contour(mask: MaskArray) -> NDArray[np.int32]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise PartSegmentationLassoError("cannot bound an empty mask")
    left = int(xs.min())
    right = int(xs.max())
    top = int(ys.min())
    bottom = int(ys.max())
    return np.asarray(
        [[[left, top]], [[right, top]], [[right, bottom]], [[left, bottom]]],
        dtype=np.int32,
    )


def _closed_contour_points(
    contour: NDArray[np.int32],
) -> list[Point]:
    points = [
        (float(item[0][0]), float(item[0][1])) for item in contour
    ]
    deduplicated: list[Point] = []
    for point in points:
        if not deduplicated or point != deduplicated[-1]:
            deduplicated.append(point)
    if deduplicated and deduplicated[0] == deduplicated[-1]:
        deduplicated.pop()
    if len(set(deduplicated)) < 3:
        raise PartSegmentationLassoError(
            "generated lasso has fewer than three distinct points"
        )
    deduplicated.append(deduplicated[0])
    return deduplicated


def _missing_mask_pixels(
    mask: MaskArray,
    contour: NDArray[np.int32],
) -> int:
    canvas = np.zeros_like(mask)
    cv2.fillPoly(canvas, [contour.astype(np.int32)], 255)
    return int(np.count_nonzero((mask > 0) & (canvas == 0)))


def _dilate_mask(mask: MaskArray, padding_pixels: int) -> MaskArray:
    kernel_size = padding_pixels * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    return cv2.dilate(mask, kernel, iterations=1)


def _component_count(mask: MaskArray) -> int:
    count, _ = cv2.connectedComponents(mask, connectivity=8)
    return max(0, int(count) - 1)


def _validate_closed_points(
    points: Sequence[Point],
    *,
    width: int,
    height: int,
    max_lasso_points: int,
) -> None:
    if len(points) > max_lasso_points:
        raise PartSegmentationLassoError(
            "generated lasso exceeds the configured point limit"
        )
    if len(points) < 4 or points[0] != points[-1]:
        raise PartSegmentationLassoError(
            "generated lasso is not a closed three-vertex polygon"
        )
    unique = set(points[:-1])
    if len(unique) < 3:
        raise PartSegmentationLassoError(
            "generated lasso has fewer than three distinct points"
        )
    xs = [point[0] for point in unique]
    ys = [point[1] for point in unique]
    if min(xs) < 0 or max(xs) >= width:
        raise PartSegmentationLassoError(
            "generated lasso x coordinate is outside the viewport"
        )
    if min(ys) < 0 or max(ys) >= height:
        raise PartSegmentationLassoError(
            "generated lasso y coordinate is outside the viewport"
        )
    if min(xs) == max(xs) or min(ys) == max(ys):
        raise PartSegmentationLassoError(
            "generated lasso must span nonzero width and height"
        )


def _lasso_rpc_request(
    *,
    lasso: FaceSetLassoGeometry,
    mapping: ScreenshotViewportMapping,
    use_front_faces_only: bool,
) -> dict[str, JsonValue]:
    params: dict[str, JsonValue] = {
        "path": [dict(point) for point in lasso.operator_path],
        "use_front_faces_only": use_front_faces_only,
    }
    if mapping.window_index is not None:
        params["window_index"] = mapping.window_index
    if mapping.area_index is not None:
        params["area_index"] = mapping.area_index
    return {
        "jsonrpc": "2.0",
        "id": f"part-segmentation-lasso-{uuid4().hex}",
        "method": "sculpt_face_set_lasso",
        "params": params,
    }


def _lasso_batch_rpc_request(
    *,
    prepared: Sequence[_PreparedSubpart],
    mapping: ScreenshotViewportMapping,
    padding_pixels: int,
    use_front_faces_only: bool,
    rollback_on_residual: bool,
) -> dict[str, JsonValue]:
    """Build one transactional, coverage-verified lasso request."""
    params: dict[str, JsonValue] = {
        "paths": [
            {
                "label": item.specification.label,
                "path": [
                    dict(point) for point in item.lasso.operator_path
                ],
            }
            for item in prepared
        ],
        "padding_pixels": padding_pixels,
        "rollback_on_residual": rollback_on_residual,
        "use_front_faces_only": use_front_faces_only,
    }
    if mapping.window_index is not None:
        params["window_index"] = mapping.window_index
    if mapping.area_index is not None:
        params["area_index"] = mapping.area_index
    return {
        "jsonrpc": "2.0",
        "id": f"part-segmentation-lasso-batch-{uuid4().hex}",
        "method": "sculpt_face_set_lasso_batch",
        "params": params,
    }


def _face_set_coverage_result(
    result: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    """Validate the RPC coverage contract before routing retries."""
    coverage = result.get("coverage")
    if not isinstance(coverage, dict):
        raise PartSegmentationRpcError(
            "Face Set batch response has no coverage object"
        )
    complete = coverage.get("complete")
    residual_count = coverage.get("residual_original_face_count")
    if not isinstance(complete, bool):
        raise PartSegmentationRpcError(
            "Face Set coverage complete must be a boolean"
        )
    if (
        isinstance(residual_count, bool)
        or not isinstance(residual_count, int)
        or residual_count < 0
    ):
        raise PartSegmentationRpcError(
            "Face Set coverage residual count must be nonnegative"
        )
    return cast(dict[str, JsonValue], coverage)


def _rewrite_candidate_selection(
    candidate_selection: Mapping[str, JsonValue],
) -> None:
    """Keep the standalone candidate receipt synchronized with retries."""
    selection_path = candidate_selection.get("selection_path")
    if isinstance(selection_path, str):
        _write_json(Path(selection_path), candidate_selection)


def _render_lasso_visualization(
    image: Image.Image,
    *,
    prepared: Sequence[_PreparedSubpart],
    mapping: ScreenshotViewportMapping,
    output_path: Path,
) -> None:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    scale_x = mapping.image_width / mapping.region_width
    scale_y = mapping.image_height / mapping.region_height
    for index, item in enumerate(prepared):
        color = _LASSO_COLOR_PALETTE[index % len(_LASSO_COLOR_PALETTE)]
        try:
            with Image.open(item.cleaned_mask_path) as mask_image:
                mask = mask_image.convert("L")
                if mask.size != base.size:
                    mask = mask.resize(base.size, Image.Resampling.NEAREST)
                tint = Image.new("RGBA", base.size, (*color, 70))
                overlay.alpha_composite(
                    Image.composite(
                        tint,
                        Image.new("RGBA", base.size, (0, 0, 0, 0)),
                        mask,
                    )
                )
        except (OSError, UnidentifiedImageError) as error:
            raise PartSegmentationArtifactError(
                f"Cannot render cleaned mask {item.cleaned_mask_path}"
            ) from error
        points = [
            (x * scale_x, y * scale_y)
            for x, y in item.lasso.screenshot_points
        ]
        draw.line(points, fill=(*color, 255), width=4, joint="curve")
        if points:
            label = f"{item.order}. {item.specification.sam3_prompt}"
            label_origin = (points[0][0] + 6, points[0][1] + 6)
            text_box = draw.textbbox(label_origin, label)
            draw.rectangle(text_box, fill=(10, 14, 20, 210))
            draw.text(label_origin, label, fill=(*color, 255))
    rendered = Image.alpha_composite(base, overlay).convert("RGB")
    _save_png(rendered, output_path)


def _render_candidate_pool_visualization(
    image: Image.Image,
    *,
    pools: Sequence[_CandidatePool],
    selected: Sequence[_SubpartCandidate],
    output_path: Path,
) -> None:
    """Render every retained instance and emphasize the global assignment."""
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    selected_ids = {item.candidate_id for item in selected}
    draw = ImageDraw.Draw(overlay)
    color_index = 0
    for pool in pools:
        for candidate in pool.candidates:
            color = _LASSO_COLOR_PALETTE[
                color_index % len(_LASSO_COLOR_PALETTE)
            ]
            color_index += 1
            alpha = 100 if candidate.candidate_id in selected_ids else 35
            mask_image = Image.fromarray(candidate.mask, mode="L")
            tint = Image.new("RGBA", base.size, (*color, alpha))
            overlay.alpha_composite(
                Image.composite(
                    tint,
                    Image.new("RGBA", base.size, (0, 0, 0, 0)),
                    mask_image,
                )
            )
            contours, _ = cv2.findContours(
                candidate.mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            width = 4 if candidate.candidate_id in selected_ids else 1
            for contour in contours:
                points = [
                    (int(point[0][0]), int(point[0][1]))
                    for point in contour
                ]
                if len(points) >= 2:
                    draw.line(
                        [*points, points[0]],
                        fill=(*color, 255),
                        width=width,
                        joint="curve",
                    )
            ys, xs = np.nonzero(candidate.mask)
            if len(xs):
                marker = "SELECTED" if candidate.candidate_id in selected_ids else "candidate"
                label = (
                    f"{candidate.candidate_id} {marker} "
                    f"score={candidate.score:.3f}"
                )
                origin = (int(xs.min()) + 3, int(ys.min()) + 3)
                text_box = draw.textbbox(origin, label)
                draw.rectangle(text_box, fill=(10, 14, 20, 210))
                draw.text(origin, label, fill=(*color, 255))
    _save_png(Image.alpha_composite(base, overlay).convert("RGB"), output_path)


def _rpc_result(
    response: Mapping[str, JsonValue],
    *,
    label: str,
) -> dict[str, JsonValue]:
    result = response.get("result")
    if not isinstance(result, dict):
        raise PartSegmentationRpcError(
            f"Blender RPC response for {label} has no object result"
        )
    return cast(dict[str, JsonValue], result)


def _result_path(
    response: Mapping[str, JsonValue],
    key: str,
    *,
    label: str,
) -> Path:
    result = response.get("result")
    value = result.get(key) if isinstance(result, dict) else None
    if not isinstance(value, str):
        raise PartSegmentationSam3Error(
            f"SAM3 response for {label} has no {key}"
        )
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise PartSegmentationSam3Error(
            f"SAM3 {key} for {label} is not a file: {path}"
        )
    return path


def _optional_result_path(
    response: Mapping[str, JsonValue],
    key: str,
) -> str | None:
    result = response.get("result")
    value = result.get(key) if isinstance(result, dict) else None
    return value if isinstance(value, str) else None


def _component_selection_result(
    response: object,
    *,
    label: str,
) -> dict[str, JsonValue]:
    """Parse one component-selector Tool response without hiding errors."""
    payload = response
    if isinstance(response, str):
        try:
            payload = json.loads(response)
        except json.JSONDecodeError as error:
            raise PartSegmentationVlmError(
                f"Component selector for {label} returned invalid JSON"
            ) from error
    if not isinstance(payload, dict):
        raise PartSegmentationVlmError(
            f"Component selector for {label} returned no object"
        )
    error_payload = payload.get("mask_component_selection_error")
    if isinstance(error_payload, dict):
        message = error_payload.get("message")
        detail = message if isinstance(message, str) else str(error_payload)
        raise PartSegmentationVlmError(
            f"Component selector for {label} failed: {detail}"
        )
    result = payload.get("result")
    if not isinstance(result, dict):
        raise PartSegmentationVlmError(
            f"Component selector for {label} has no object result"
        )
    return cast(dict[str, JsonValue], result)


def _component_selection_path(
    result: Mapping[str, JsonValue],
    *,
    key: str,
    label: str,
) -> Path:
    value = result.get(key)
    if not isinstance(value, str):
        raise PartSegmentationVlmError(
            f"Component selector for {label} has no {key}"
        )
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise PartSegmentationVlmError(
            f"Component selector {key} for {label} is not a file: {path}"
        )
    return path


def _binary_mask(
    path: Path,
    *,
    expected_size: tuple[int, int],
    label: str,
) -> MaskArray:
    """Load one SAM3 mask and verify its exact coordinate space."""
    try:
        with Image.open(path) as source:
            grayscale = np.asarray(source.convert("L"), dtype=np.uint8)
    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ) as error:
        raise PartSegmentationSam3Error(
            f"Cannot read SAM3 mask for {label}: {path}"
        ) from error
    width, height = expected_size
    if grayscale.shape != (height, width):
        raise PartSegmentationSam3Error(
            f"SAM3 mask for {label} has size "
            f"{grayscale.shape[1]}x{grayscale.shape[0]}, expected "
            f"{width}x{height}"
        )
    binary = np.where(
        grayscale >= _MASK_THRESHOLD,
        255,
        0,
    ).astype(np.uint8)
    if not np.any(binary):
        raise PartSegmentationSam3Error(
            f"SAM3 mask for {label} has no foreground"
        )
    return binary


def _component_count(mask: MaskArray) -> int:
    """Count 8-connected foreground regions in one parent mask."""
    try:
        count, _, _, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    except cv2.error as error:
        raise PartSegmentationSam3Error(
            f"Cannot analyze parent SAM3 mask: {error}"
        ) from error
    return max(0, int(count) - 1)


def _largest_component_area(mask: MaskArray) -> int:
    if not np.any(mask):
        return 0
    try:
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    except cv2.error as error:
        raise PartSegmentationSam3Error(
            f"Cannot analyze uncovered parent mask: {error}"
        ) from error
    return max(
        (int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, count)),
        default=0,
    )


def _expanded_foreground_box(
    mask: MaskArray,
    *,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise PartSegmentationSam3Error(
            "Cannot create an ROI from an empty parent mask"
        )
    left = int(xs.min())
    right = int(xs.max()) + 1
    top = int(ys.min())
    bottom = int(ys.max()) + 1
    padding_x = math.ceil((right - left) * padding_ratio)
    padding_y = math.ceil((bottom - top) * padding_ratio)
    height, width = mask.shape
    return (
        max(0, left - padding_x),
        max(0, top - padding_y),
        min(width, right + padding_x),
        min(height, bottom + padding_y),
    )


def _render_mask_overlay(
    image: Image.Image,
    *,
    mask: MaskArray,
    opacity: float,
    output_path: Path,
) -> None:
    base = image.convert("RGBA")
    alpha = int(round(255 * opacity))
    tint = Image.new("RGBA", base.size, (46, 204, 113, alpha))
    mask_image = Image.fromarray(mask, mode="L")
    transparent = Image.new("RGBA", base.size, (0, 0, 0, 0))
    overlay = Image.composite(tint, transparent, mask_image)
    _save_png(Image.alpha_composite(base, overlay).convert("RGB"), output_path)


def _create_artifact_directory(path: Path, *, label: str) -> None:
    try:
        path.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        raise PartSegmentationArtifactError(
            f"Cannot create {label} artifact directory: {path}"
        ) from error


def _resolve_image_path(value: str | Path) -> Path:
    path = Path(
        os.path.expandvars(os.path.expanduser(str(value)))
    ).resolve()
    if not path.is_file():
        raise PartSegmentationInputError(
            f"input screenshot is not a file: {path}"
        )
    return path


def _normalize_description(value: str) -> str:
    normalized = " ".join(value.strip().split())
    if not normalized:
        raise PartSegmentationInputError(
            "part description must not be empty"
        )
    return normalized


def _normalize_sam3_phrase(value: str) -> str:
    normalized = " ".join(value.strip().split())
    if not normalized:
        raise ValueError("SAM3 phrase must not be empty")
    if not normalized.isascii():
        raise ValueError("SAM3 phrase must be English ASCII text")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 .'-]*", normalized):
        raise ValueError("SAM3 phrase contains unsupported characters")
    return normalized


def _is_empty_sam3_error(value: JsonValue | None) -> bool:
    """Recognize only the tool's deterministic empty-foreground result."""
    if not isinstance(value, dict):
        return False
    message = value.get("message")
    return (
        value.get("type") == "mask_cleanup_error"
        and isinstance(message, str)
        and "no foreground" in message.casefold()
    )


def _identity_qualifiers(value: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[A-Za-z]+", value)
        if token.casefold() in _INSTANCE_QUALIFIERS
    }


def _effective_child_prompts(
    specification: KinematicSubpart,
    *,
    parent_prompt: str,
    max_synonym_attempts: int,
) -> list[str]:
    """Remove parent-resolved identity words before invoking SAM3."""
    qualifiers = _identity_qualifiers(parent_prompt)
    source_prompts = [
        specification.sam3_prompt,
        *specification.fallback_prompts[:max_synonym_attempts],
    ]
    effective: list[str] = []
    seen: set[str] = set()
    for source in source_prompts:
        words = source.split()
        filtered = [
            word
            for word in words
            if word.strip(".'-").casefold() not in qualifiers
        ]
        prompt = " ".join(filtered).strip()
        if not prompt:
            raise PartSegmentationVlmError(
                f"Subpart {specification.label} has no category words after "
                "removing parent instance qualifiers"
            )
        normalized = prompt.casefold()
        if normalized not in seen:
            effective.append(prompt)
            seen.add(normalized)
    return effective


def _load_rgb_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as source:
            return source.convert("RGB")
    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ) as error:
        raise PartSegmentationInputError(
            f"cannot read input screenshot: {path}"
        ) from error


def _load_grayscale_mask(path: Path) -> MaskArray:
    try:
        with Image.open(path) as source:
            return np.asarray(source.convert("L"), dtype=np.uint8)
    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ) as error:
        raise PartSegmentationLassoError(
            f"cannot read cleaned SAM3 mask: {path}"
        ) from error


def _create_run_dir(
    artifact_root: str,
    *,
    workdir: Path,
    image_stem: str,
) -> Path:
    root = Path(
        os.path.expandvars(os.path.expanduser(artifact_root))
    )
    if not root.is_absolute():
        root = workdir / root
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = root.resolve() / (
        f"{timestamp}-{_slug(image_stem)}-{uuid4().hex[:8]}"
    )
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        raise PartSegmentationArtifactError(
            f"Cannot create artifact directory: {run_dir}"
        ) from error
    return run_dir


def _slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return (normalized or "part")[:64]


def _write_json(path: Path, payload: Mapping[str, JsonValue]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError) as error:
        raise PartSegmentationArtifactError(
            f"Cannot write part-segmentation trace: {path}"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _save_png(image: Image.Image, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        image.save(temporary, format="PNG")
        os.replace(temporary, path)
    except OSError as error:
        raise PartSegmentationArtifactError(
            f"Cannot write lasso visualization: {path}"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _save_mask(mask: MaskArray, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        Image.fromarray(mask, mode="L").save(temporary, format="PNG")
        os.replace(temporary, path)
    except OSError as error:
        raise PartSegmentationArtifactError(
            f"Cannot write part-segmentation mask: {path}"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)
