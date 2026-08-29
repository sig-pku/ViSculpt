"""Structured LLM outputs used by the Sculpt Agent workflow."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

SUBTASK_DESCRIPTION_PATTERN = (
    r"^Use .+ brush at .+ to perform a (Smear|Drag|Draw) operation, "
    r"achieving .+\.$"
)
OPERATION_LOCATION_PATTERN = r"^[A-Za-z0-9]+([ -][A-Za-z0-9]+)*$"


class OperationMethod(StrEnum):
    """Operation families currently understood by the workflow."""

    SMEAR = "Smear"
    DRAG = "Drag"
    DRAW = "Draw"


class BrushScale(StrEnum):
    """Semantic brush footprint selected before segmentation."""

    LOCAL = "LOCAL"
    REGIONAL = "REGIONAL"
    BROAD = "BROAD"


class DrawScaleTier(StrEnum):
    """Semantic footprint tier for mask-fitted Draw content."""

    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    LARGE = "LARGE"


class EffectIntensity(StrEnum):
    """Minimum visible effect requested by the user intent."""

    SUBTLE = "SUBTLE"
    MEDIUM_VISIBLE = "MEDIUM_VISIBLE"
    STRONG = "STRONG"


class EffectAppropriateness(StrEnum):
    """VLM judgment of the visible edit relative to user intent."""

    TOO_WEAK = "TOO_WEAK"
    APPROPRIATE = "APPROPRIATE"
    EXCESSIVE_FOR_INSTRUCTION = "EXCESSIVE_FOR_INSTRUCTION"
    WRONG_EFFECT = "WRONG_EFFECT"
    WRONG_REGION = "WRONG_REGION"
    INCONCLUSIVE = "INCONCLUSIVE"


class EffectMagnitude(StrEnum):
    """Descriptive, non-normative visible edit magnitude."""

    SUBTLE = "SUBTLE"
    MODERATE = "MODERATE"
    LARGE = "LARGE"
    DRAMATIC = "DRAMATIC"


class SurfaceRetryScope(StrEnum):
    """Earliest invalid spatial stage for Smear and Draw retries."""

    RESELECT_VIEW = "RESELECT_VIEW"
    RESEGMENT = "RESEGMENT"
    REUSE_SEGMENTATION = "REUSE_SEGMENTATION"


class StandardView(StrEnum):
    """Six orthographic views available to View Selector."""

    FRONT = "FRONT"
    BACK = "BACK"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    TOP = "TOP"
    BOTTOM = "BOTTOM"


class DecomposedSubtask(BaseModel):
    """One natural-language geometry-editing subtask."""

    model_config = ConfigDict(extra="forbid")

    description: str = Field(
        min_length=20,
        max_length=500,
        pattern=SUBTASK_DESCRIPTION_PATTERN,
    )
    operation_method: OperationMethod

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: str) -> str:
        """Keep descriptions compact and consistently punctuated."""
        description = " ".join(value.strip().split())
        if not description:
            raise ValueError("description must not be empty")
        return description

    @field_validator("operation_method", mode="before")
    @classmethod
    def normalize_operation(cls, value: object) -> object:
        """Accept case-insensitive operation names."""
        if isinstance(value, str):
            normalized = value.strip().casefold()
            mapping = {
                item.value.casefold(): item.value
                for item in OperationMethod
            }
            return mapping.get(normalized, value)
        return value

    @model_validator(mode="after")
    def validate_description_format(self) -> Self:
        """Require the user-requested text template and operation name."""
        lowered = self.description.casefold()
        required = (
            lowered.startswith("use ")
            and " brush " in lowered
            and " at " in lowered
            and " operation" in lowered
            and "achiev" in lowered
            and self.operation_method.value.casefold() in lowered
        )
        if not required:
            raise ValueError(
                "description must follow the required Use/brush/location/"
                "operation/achieving template"
            )
        return self


class DecomposerOutput(BaseModel):
    """Structured output of the Decomposer node."""

    model_config = ConfigDict(extra="forbid")

    subtasks: list[DecomposedSubtask] = Field(
        min_length=1,
        max_length=20,
    )


class SculptIntent(BaseModel):
    """Translator-selected semantic Sculpt intent."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    operation_location: str = Field(
        min_length=1,
        max_length=64,
        pattern=OPERATION_LOCATION_PATTERN,
        description=(
            "The concrete execution location. For Smear this is a complete "
            "region; for Drag this is the precise mouse-down contact, such "
            "as Right Ear Tip rather than Right Ear."
        ),
    )
    part_to_be_changed: str = Field(
        min_length=1,
        max_length=64,
        pattern=OPERATION_LOCATION_PATTERN,
        description=(
            "The complete semantic model part that the user wants to "
            "change and that SAM3 should segment, such as Right Ear."
        ),
    )
    sculpt_brush: str = Field(min_length=1, max_length=128)
    brush_scale: BrushScale | None = None
    brush_strength: float | None = Field(default=None, ge=0.0, le=1.0)
    brush_direction: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "Exact runtime Direction identifier, or null when the selected "
            "local brush does not expose Direction. Pose Drag intents omit "
            "this field together with brush_scale and brush_strength. Draw "
            "intents must select ADD or SUBTRACT."
        ),
    )
    draw_pattern_description: str | None = Field(
        default=None,
        max_length=256,
        description=(
            "A concise English word or phrase for a generated Draw pattern. "
            "Exactly one of this field and draw_text is used by Draw."
        ),
    )
    draw_text: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "Exact printable English ASCII text for a text Draw operation. "
            "Exactly one of this field and draw_pattern_description is used "
            "by Draw."
        ),
    )
    draw_scale_tier: DrawScaleTier | None = Field(
        default=None,
        description=(
            "SMALL, MEDIUM, or LARGE footprint for Draw content. The "
            "workflow defaults omitted Draw values to MEDIUM; non-Draw "
            "intents omit this field."
        ),
    )
    effect_intensity: EffectIntensity

    @field_validator("operation_location", "part_to_be_changed")
    @classmethod
    def normalize_location(cls, value: str) -> str:
        """Keep both location roles as simple English noun phrases."""
        location = " ".join(value.strip().split())
        if not location.isascii():
            raise ValueError(
                "location semantics must be simple English words or phrases"
            )
        return location

    @field_validator("sculpt_brush")
    @classmethod
    def normalize_brush(cls, value: str) -> str:
        """Strip the selected Blender brush asset name."""
        brush = value.strip()
        if not brush or any(ord(character) < 32 for character in brush):
            raise ValueError("sculpt_brush is invalid")
        return brush

    @field_validator("brush_direction", mode="before")
    @classmethod
    def normalize_direction(cls, value: object) -> object:
        """Normalize a runtime-provided Blender Direction identifier."""
        if isinstance(value, str):
            direction = value.strip().upper()
            if not direction or any(
                ord(character) < 32 for character in direction
            ):
                raise ValueError("brush_direction is invalid")
            return direction
        return value

    @field_validator("draw_pattern_description")
    @classmethod
    def normalize_draw_pattern(cls, value: str | None) -> str | None:
        """Keep generated-pattern semantics compact and English-only."""
        if value is None:
            return None
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("draw_pattern_description must not be blank")
        if not normalized.isascii() or any(
            ord(character) < 32 for character in normalized
        ):
            raise ValueError(
                "draw_pattern_description must be printable English text"
            )
        return normalized

    @field_validator("draw_text")
    @classmethod
    def normalize_draw_text(cls, value: str | None) -> str | None:
        """Preserve exact glyph case while accepting printable English ASCII."""
        if value is None:
            return None
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("draw_text must not be blank")
        if any(
            ord(character) < 32 or ord(character) > 126
            for character in normalized
        ):
            raise ValueError("draw_text supports printable English ASCII only")
        return normalized

    @field_validator(
        "brush_scale",
        "draw_scale_tier",
        "effect_intensity",
        mode="before",
    )
    @classmethod
    def normalize_uppercase_enum(cls, value: object) -> object:
        """Accept case-insensitive semantic enum values."""
        if isinstance(value, str):
            return value.strip().upper()
        return value


class TranslatedSubtask(BaseModel):
    """Parameters associated with one Decomposer subtask index."""

    model_config = ConfigDict(extra="forbid")

    subtask_index: int = Field(ge=0)
    intent: SculptIntent


class TranslatorOutput(BaseModel):
    """Structured output of the Translator node."""

    model_config = ConfigDict(extra="forbid")

    translations: list[TranslatedSubtask] = Field(
        min_length=1,
        max_length=20,
    )


class ViewSelection(BaseModel):
    """View Selector decision for one subtask attempt."""

    model_config = ConfigDict(extra="forbid")

    view: StandardView
    reason: str = Field(min_length=1, max_length=800)

    @field_validator("view", mode="before")
    @classmethod
    def normalize_view(cls, value: object) -> object:
        """Accept case-insensitive standard views."""
        if isinstance(value, str):
            return value.strip().upper()
        return value


class DragDirectionPlan(BaseModel):
    """One VLM-planned straight Drag gesture in screenshot coordinates."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    anchor_target_valid: bool
    anchor_target_analysis: str = Field(min_length=1, max_length=1200)
    direction: list[float] = Field(
        min_length=2,
        max_length=2,
        description=(
            "Screen-space direction [x, y], where positive x points right "
            "and positive y points down."
        ),
    )
    distance_pixels: int = Field(ge=1, le=10_000)
    reason: str = Field(min_length=1, max_length=1200)

    @model_validator(mode="after")
    def validate_direction(self) -> Self:
        """Reject a vector that cannot define a line segment."""
        x, y = self.direction
        if abs(x) <= 1e-9 and abs(y) <= 1e-9:
            raise ValueError("direction must be nonzero")
        return self


class DragGradingAssessment(BaseModel):
    """Explicit VLM checks required before accepting a Drag edit."""

    model_config = ConfigDict(extra="forbid")

    target_identity_correct: bool
    motion_direction_correct: bool
    target_motion_visible: bool
    spatial_goal_reached: bool
    non_target_geometry_stable: bool
    analysis: str = Field(min_length=1, max_length=1600)


class GraderOutput(BaseModel):
    """Visual grading without parameter-repair responsibilities."""

    model_config = ConfigDict(extra="forbid")

    effect_appropriateness: EffectAppropriateness
    effect_magnitude: EffectMagnitude
    visual_evidence: list[str] = Field(min_length=1, max_length=8)
    instruction_compliance: int = Field(ge=0, le=5)
    visual_quality: int = Field(ge=0, le=5)
    geometric_plausibility: int = Field(ge=0, le=5)
    analysis: str = Field(min_length=1, max_length=3000)
    drag_assessment: DragGradingAssessment | None = None

    @field_validator("visual_evidence")
    @classmethod
    def normalize_evidence(cls, value: list[str]) -> list[str]:
        """Require concise, non-empty observations."""
        normalized = [" ".join(item.strip().split()) for item in value]
        if any(not item for item in normalized):
            raise ValueError("visual_evidence entries must not be empty")
        return normalized

    @property
    def total_score(self) -> int:
        """Return the deterministic aggregate score."""
        return (
            self.instruction_compliance
            + self.visual_quality
            + self.geometric_plausibility
        )

    @property
    def score_qualifies(self) -> bool:
        """Apply score-only criteria before deterministic acceptance."""
        return (
            self.total_score > 9
            and self.instruction_compliance >= 3
            and self.visual_quality >= 2
            and self.geometric_plausibility >= 2
        )


class RetryPlannerOutput(BaseModel):
    """LLM repair plan created only after a visual grading failure."""

    model_config = ConfigDict(extra="forbid")

    analysis: str = Field(min_length=1, max_length=3000)
    revised_subtask_description: str = Field(
        min_length=20,
        max_length=500,
        pattern=SUBTASK_DESCRIPTION_PATTERN,
    )
    revised_intent: SculptIntent
    recommended_view: StandardView
    surface_retry_scope: SurfaceRetryScope | None = None
    segmentation_prompt: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=OPERATION_LOCATION_PATTERN,
        description=(
            "Replacement SAM3 prompt used only when a Smear or Draw retry "
            "selects RESEGMENT."
        ),
    )
    regenerate_svg_pattern: bool = Field(
        default=False,
        description=(
            "Whether a pattern-based Draw retry must regenerate its SVG. "
            "This decision is independent of surface_retry_scope."
        ),
    )

    @field_validator("recommended_view", mode="before")
    @classmethod
    def normalize_view(cls, value: object) -> object:
        """Accept case-insensitive retry views."""
        if isinstance(value, str):
            return value.strip().upper()
        return value

    @field_validator("segmentation_prompt")
    @classmethod
    def normalize_segmentation_prompt(
        cls,
        value: str | None,
    ) -> str | None:
        """Keep replacement SAM3 prompts concise and unambiguous."""
        if value is None:
            return None
        prompt = " ".join(value.strip().split())
        if not prompt.isascii():
            raise ValueError("segmentation_prompt must be English ASCII")
        return prompt

    @model_validator(mode="after")
    def validate_surface_retry(self) -> Self:
        """Require a replacement prompt only for deterministic resegmentation."""
        if (
            self.surface_retry_scope is SurfaceRetryScope.RESEGMENT
            and self.segmentation_prompt is None
        ):
            raise ValueError(
                "RESEGMENT requires a replacement segmentation_prompt"
            )
        return self
