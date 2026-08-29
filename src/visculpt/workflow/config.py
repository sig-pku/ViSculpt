"""Centralized configuration for the Sculpt Agent workflow."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from visculpt.bridge import BlenderRpcConfig
from visculpt.vision import (
    MaskTrajectoryFitConfig,
    PartSegmentationConfig,
    QuadLocConfig,
    Sam3GradioConfig,
    TextTrajectoryConfig,
)

from .errors import WorkflowConfigError

DEFAULT_CONFIG_PATH = Path(__file__).with_name("default_config.toml")
LLM_ROLES = (
    "decomposer",
    "translator",
    "view_selector",
    "quadloc",
    "svg_pattern_generator",
    "grader",
    "retry_planner",
)
STANDARD_VIEWS = ("FRONT", "BACK", "LEFT", "RIGHT", "TOP", "BOTTOM")
OPERATION_METHODS = ("Smear", "Drag", "Draw")
IMPLEMENTED_OPERATION_METHODS = ("Smear", "Drag", "Draw")
OPENAI_COMPATIBLE_EFFORTS = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)
ANTHROPIC_COMPATIBLE_EFFORTS = (
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)


@dataclass(frozen=True, slots=True)
class LlmConfig:
    """Provider-neutral LLM configuration."""

    provider: str
    base_url: str
    endpoint_path: str
    api_key_env: str
    api_key_mode: str
    secrets_file: str
    timeout_seconds: float
    max_output_tokens: int
    effort: str | None
    anthropic_version: str
    schema_profile: str
    models: dict[str, str]
    max_retries: int
    retry_backoff_seconds: float

    def model_for(self, role: str) -> str:
        """Return the configured model for one workflow role."""
        try:
            return self.models[role]
        except KeyError as error:
            raise WorkflowConfigError(
                f"No LLM model is configured for role {role}"
            ) from error


@dataclass(frozen=True, slots=True)
class LlmProviderPresetConfig:
    """One non-secret provider preset exposed to the Web client."""

    preset_id: str
    label: str
    base_url: str
    api_key_env: str
    api_key_mode: str
    schema_profile: str
    openai_endpoint_path: str
    anthropic_endpoint_path: str | None


@dataclass(frozen=True, slots=True)
class LlmRuntimeConfig:
    """Runtime model-switching and connectivity-test configuration."""

    test_prompt_path: str
    test_image_path: str
    presets: tuple[LlmProviderPresetConfig, ...]

    def preset_for_base_url(
        self,
        base_url: str,
    ) -> LlmProviderPresetConfig | None:
        """Return the preset whose normalized Base URL matches exactly."""
        normalized = base_url.rstrip("/").casefold()
        return next(
            (
                preset
                for preset in self.presets
                if preset.base_url.casefold() == normalized
            ),
            None,
        )


@dataclass(frozen=True, slots=True)
class ServiceConfig:
    """Local Blender and SAM3 endpoint configuration."""

    blender_rpc_url: str
    blender_rpc_timeout_seconds: float
    blender_rpc_access_token_env: str
    blender_rpc_max_response_bytes: int
    sam3_url: str
    sam3_timeout_seconds: float

    def blender_rpc_config(
        self,
        *,
        environment: Mapping[str, str] | None = None,
    ) -> BlenderRpcConfig:
        """Translate the configured loopback URL to the Bridge config."""
        port = _loopback_port(
            self.blender_rpc_url,
            expected_path="/rpc",
            label="blender_rpc_url",
        )
        values = os.environ if environment is None else environment
        return BlenderRpcConfig(
            port=port,
            access_token=values.get(
                self.blender_rpc_access_token_env,
                "",
            ),
            timeout=self.blender_rpc_timeout_seconds,
            max_response_bytes=self.blender_rpc_max_response_bytes,
        )

    def sam3_config(self) -> Sam3GradioConfig:
        """Translate the configured loopback URL to the SAM3 config."""
        port = _loopback_port(
            self.sam3_url,
            expected_path="",
            label="sam3_url",
        )
        return Sam3GradioConfig(
            port=port,
            timeout=self.sam3_timeout_seconds,
        )


@dataclass(frozen=True, slots=True)
class MinimumEffectConfig:
    """One-sided target-region visibility thresholds."""

    enabled: bool
    capture_warmup_redraws: int
    baseline_capture_attempts: int
    comparison_blur_kernel_size: int
    baseline_noise_percentile: float
    pixel_delta_floor: float
    maximum_baseline_noise: float
    minimum_evaluation_pixels: int
    roi_padding_ratio: float
    no_effect_fraction: float
    subtle_minimum_mean_abs_diff: float
    subtle_minimum_changed_fraction: float
    medium_minimum_mean_abs_diff: float
    medium_minimum_changed_fraction: float
    strong_minimum_mean_abs_diff: float
    strong_minimum_changed_fraction: float


@dataclass(frozen=True, slots=True)
class ParameterResolutionConfig:
    """Mask-relative Sculpt settings and bounded retry policy."""

    local_size_ratio: float
    regional_size_ratio: float
    broad_size_ratio: float
    retry_dose_multiplier: float
    maximum_brush_size: int
    maximum_brush_strength: float
    maximum_pass_count: int


@dataclass(frozen=True, slots=True)
class SculptOperationDefaultsConfig:
    """Deterministic Sculpt settings for one operation family."""

    dyntopo_enabled: bool
    dyntopo_detail_size: float


@dataclass(frozen=True, slots=True)
class RoiFocusConfig:
    """Orthographic ROI focus policy shared by Smear and Draw."""

    enabled: bool
    margin_ratio: float
    maximum_zoom_factor: float


@dataclass(frozen=True, slots=True)
class DragWorkflowConfig:
    """Deterministic bounds and Pose defaults for straight Drag gestures."""

    llm_role: str
    minimum_distance_pixels: float
    maximum_distance_ratio: float
    stroke_spacing_pixels: float
    safe_anchor_minimum_margin_pixels: float
    safe_anchor_brush_radius_ratio: float
    safe_anchor_component_depth_ratio: float
    brush_anchor_extent_percentile: float
    pose_brush_name: str
    pose_brush_size: int
    pose_brush_strength: float
    pose_deformation_target: str
    pose_rotation_origins: str
    pose_origin_offset: float
    pose_smooth_iterations: int
    pose_connected_only: bool
    pose_max_element_distance: float


@dataclass(frozen=True, slots=True)
class DrawWorkflowConfig:
    """Configuration for LLM-generated Draw pattern artifacts."""

    llm_role: str
    artifact_root: str
    trajectory_artifact_root: str
    trajectory_point_spacing_pixels: float
    trajectory_flattening_spacing_pixels: float
    fit_artifact_root: str
    fit: MaskTrajectoryFitConfig
    text_trajectory_artifact_root: str
    text: TextTrajectoryConfig
    font_name: str
    stroke_method: str
    brush_spacing_percent: int
    use_space_attenuation: bool
    auto_smooth_factor: float
    finishing_smooth_enabled: bool
    finishing_smooth_brush_name: str
    finishing_smooth_size_ratio: float
    finishing_smooth_strength: float
    finishing_smooth_direction: str
    finishing_smooth_dyntopo_enabled: bool
    brush_size_ratio: float
    minimum_brush_size: int
    maximum_brush_size: int

    def __post_init__(self) -> None:
        """Validate related deterministic Draw bounds."""
        if (
            self.trajectory_flattening_spacing_pixels
            > self.trajectory_point_spacing_pixels
        ):
            raise WorkflowConfigError(
                "workflow.draw.trajectory_flattening_spacing_pixels must "
                "not exceed trajectory_point_spacing_pixels"
            )
        if self.minimum_brush_size > self.maximum_brush_size:
            raise WorkflowConfigError(
                "workflow.draw.minimum_brush_size must not exceed "
                "maximum_brush_size"
            )


@dataclass(frozen=True, slots=True)
class WorkflowRuntimeConfig:
    """Deterministic graph and artifact settings."""

    artifact_root: str
    decomposer_views: tuple[str, ...]
    translator_views: tuple[str, ...]
    standard_views: tuple[str, ...]
    frame: str
    max_subtasks: int
    max_subtask_attempts: int
    require_execution_approval: bool
    blender_session_lease_seconds: float
    snapshot_compress: bool
    load_ui_on_restore: bool
    sam3_confidence_threshold: float
    sam3_overlay_opacity: float
    roi_focus: RoiFocusConfig
    quadloc: QuadLocConfig
    part_segmentation: PartSegmentationConfig
    drag: DragWorkflowConfig
    draw: DrawWorkflowConfig
    minimum_effect: MinimumEffectConfig
    parameter_resolution: ParameterResolutionConfig
    operation_defaults: dict[str, SculptOperationDefaultsConfig]
    recursion_limit: int

    @property
    def effective_recursion_limit(self) -> int:
        """Return a safe graph-step budget for every configured retry path."""
        # Fixed nodes cover initialization, planning, and finalization. Reserve
        # preparation plus the complete select, execute, grade, repair, and
        # restore path for every subtask attempt.
        derived = 32 + self.max_subtasks * (
            8 + self.max_subtask_attempts * 32
        )
        return max(self.recursion_limit, derived)

    def defaults_for_operation(
        self,
        operation_method: str,
    ) -> SculptOperationDefaultsConfig:
        """Return fixed settings for one executable operation method."""
        try:
            return self.operation_defaults[operation_method]
        except KeyError as error:
            raise WorkflowConfigError(
                "No deterministic defaults are configured for operation "
                f"{operation_method}"
            ) from error


@dataclass(frozen=True, slots=True)
class SculptWorkflowConfig:
    """Complete non-secret workflow configuration."""

    llm: LlmConfig
    llm_runtime: LlmRuntimeConfig
    services: ServiceConfig
    workflow: WorkflowRuntimeConfig
    source_path: Path

    def artifact_root(self, workdir: Path | None = None) -> Path:
        """Resolve the artifact root against the runtime working directory."""
        root = Path(
            os.path.expandvars(
                os.path.expanduser(self.workflow.artifact_root)
            )
        )
        if not root.is_absolute():
            root = (workdir or Path.cwd()) / root
        return root.resolve()

    def api_key(
        self,
        *,
        environment: Mapping[str, str] | None = None,
        workdir: Path | None = None,
    ) -> str | None:
        """Read an optional secret without storing it in graph state."""
        if self.llm.api_key_mode == "none":
            return None
        values = os.environ if environment is None else environment
        direct = values.get(self.llm.api_key_env, "").strip()
        if direct:
            return direct
        secrets_path = Path(
            os.path.expandvars(os.path.expanduser(self.llm.secrets_file))
        )
        if not secrets_path.is_absolute():
            secrets_path = (workdir or Path.cwd()) / secrets_path
        if not secrets_path.exists():
            if self.llm.api_key_mode == "required":
                raise WorkflowConfigError(
                    f"Missing secret {self.llm.api_key_env} in the "
                    "environment or configured secrets file"
                )
            return None
        secrets = _read_dotenv(secrets_path)
        secret = secrets.get(self.llm.api_key_env, "").strip()
        if not secret and self.llm.api_key_mode == "required":
            raise WorkflowConfigError(
                f"Missing secret {self.llm.api_key_env} in the environment "
                f"or configured secrets file"
            )
        return secret or None


def load_workflow_config(
    path: str | Path | None = None,
) -> SculptWorkflowConfig:
    """Load and validate the single TOML workflow config file."""
    source = DEFAULT_CONFIG_PATH if path is None else Path(path)
    source = Path(
        os.path.expandvars(os.path.expanduser(str(source)))
    ).resolve()
    try:
        with source.open("rb") as stream:
            payload = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise WorkflowConfigError(
            f"Cannot load workflow config {source}: {error}"
        ) from error

    llm_section = _section(payload, "llm")
    models_section = _section(llm_section, "models")
    provider = _string(llm_section, "provider").lower()
    if provider not in {"openai_compatible", "anthropic"}:
        raise WorkflowConfigError(
            "llm.provider must be openai_compatible or anthropic"
        )
    models = {
        role: _string(models_section, role) for role in LLM_ROLES
    }
    schema_profile = _string(llm_section, "schema_profile").lower()
    if schema_profile not in {"full", "gemini_compatible"}:
        raise WorkflowConfigError(
            "llm.schema_profile must be full or gemini_compatible"
        )
    api_key_mode = _string(llm_section, "api_key_mode").lower()
    if api_key_mode not in {"required", "if_present", "none"}:
        raise WorkflowConfigError(
            "llm.api_key_mode must be required, if_present, or none"
        )
    effort = _string(
        llm_section,
        "effort",
        allow_empty=True,
    ).lower()
    allowed_efforts = (
        ANTHROPIC_COMPATIBLE_EFFORTS
        if provider == "anthropic"
        else OPENAI_COMPATIBLE_EFFORTS
    )
    if effort and effort not in allowed_efforts:
        choices = ", ".join(allowed_efforts)
        raise WorkflowConfigError(
            f"llm.effort must be empty or one of {choices} for {provider}"
        )
    llm = LlmConfig(
        provider=provider,
        base_url=_http_url(_string(llm_section, "base_url")),
        endpoint_path=_endpoint_path(
            _string(llm_section, "endpoint_path")
        ),
        api_key_env=_string(llm_section, "api_key_env"),
        api_key_mode=api_key_mode,
        secrets_file=_string(llm_section, "secrets_file"),
        timeout_seconds=_positive_number(
            llm_section,
            "timeout_seconds",
        ),
        max_output_tokens=_bounded_integer(
            llm_section,
            "max_output_tokens",
            minimum=1,
            maximum=65_536,
        ),
        effort=effort or None,
        anthropic_version=_string(llm_section, "anthropic_version"),
        schema_profile=schema_profile,
        models=models,
        max_retries=_bounded_integer(
            llm_section,
            "max_retries",
            minimum=0,
            maximum=5,
        ),
        retry_backoff_seconds=_bounded_number(
            llm_section,
            "retry_backoff_seconds",
            minimum=0.0,
            maximum=60.0,
        ),
    )
    llm_runtime = _llm_runtime_config(
        _section(llm_section, "runtime")
    )

    service_section = _section(payload, "services")
    services = ServiceConfig(
        blender_rpc_url=_string(service_section, "blender_rpc_url"),
        blender_rpc_timeout_seconds=_positive_number(
            service_section,
            "blender_rpc_timeout_seconds",
        ),
        blender_rpc_access_token_env=_string(
            service_section,
            "blender_rpc_access_token_env",
        ),
        blender_rpc_max_response_bytes=_bounded_integer(
            service_section,
            "blender_rpc_max_response_bytes",
            minimum=1024,
            maximum=1024 * 1024 * 1024,
        ),
        sam3_url=_string(service_section, "sam3_url"),
        sam3_timeout_seconds=_positive_number(
            service_section,
            "sam3_timeout_seconds",
        ),
    )
    services.blender_rpc_config()
    services.sam3_config()

    workflow_section = _section(payload, "workflow")
    minimum_effect_section = _section(
        workflow_section,
        "minimum_effect",
    )
    parameter_resolution_section = _section(
        workflow_section,
        "parameter_resolution",
    )
    quadloc_section = _section(workflow_section, "quadloc")
    part_segmentation_section = _section(
        workflow_section,
        "part_segmentation",
    )
    drag_section = _section(workflow_section, "drag")
    draw_section = _section(workflow_section, "draw")
    roi_focus_section = _section(workflow_section, "roi_focus")
    draw_fit_section = _section(draw_section, "fit")
    draw_text_section = _section(draw_section, "text")
    operation_defaults = _operation_defaults(
        _section(workflow_section, "operation_defaults")
    )
    standard_views = _view_list(workflow_section, "standard_views")
    if set(standard_views) != set(STANDARD_VIEWS) or len(
        standard_views
    ) != len(STANDARD_VIEWS):
        raise WorkflowConfigError(
            "workflow.standard_views must contain each of FRONT, BACK, "
            "LEFT, RIGHT, TOP, and BOTTOM exactly once"
        )
    frame = _string(workflow_section, "frame").upper()
    if frame not in {"KEEP", "SELECTED", "ALL"}:
        raise WorkflowConfigError(
            "workflow.frame must be KEEP, SELECTED, or ALL"
        )
    workflow = WorkflowRuntimeConfig(
        artifact_root=_string(workflow_section, "artifact_root"),
        decomposer_views=_view_list(
            workflow_section,
            "decomposer_views",
        ),
        translator_views=_view_list(
            workflow_section,
            "translator_views",
        ),
        standard_views=standard_views,
        frame=frame,
        max_subtasks=_bounded_integer(
            workflow_section,
            "max_subtasks",
            minimum=1,
            maximum=20,
        ),
        max_subtask_attempts=_bounded_integer(
            workflow_section,
            "max_subtask_attempts",
            minimum=1,
            maximum=3,
        ),
        require_execution_approval=_boolean(
            workflow_section,
            "require_execution_approval",
        ),
        blender_session_lease_seconds=_positive_number(
            workflow_section,
            "blender_session_lease_seconds",
        ),
        snapshot_compress=_boolean(
            workflow_section,
            "snapshot_compress",
        ),
        load_ui_on_restore=_boolean(
            workflow_section,
            "load_ui_on_restore",
        ),
        sam3_confidence_threshold=_bounded_number(
            workflow_section,
            "sam3_confidence_threshold",
            minimum=0.05,
            maximum=0.95,
        ),
        sam3_overlay_opacity=_bounded_number(
            workflow_section,
            "sam3_overlay_opacity",
            minimum=0.0,
            maximum=1.0,
        ),
        roi_focus=RoiFocusConfig(
            enabled=_boolean(roi_focus_section, "enabled"),
            margin_ratio=_bounded_number(
                roi_focus_section,
                "margin_ratio",
                minimum=0.0,
                maximum=2.0,
            ),
            maximum_zoom_factor=_bounded_number(
                roi_focus_section,
                "maximum_zoom_factor",
                minimum=1.0,
                maximum=100.0,
            ),
        ),
        quadloc=QuadLocConfig(
            max_depth=_bounded_integer(
                quadloc_section,
                "max_depth",
                minimum=1,
                maximum=10,
            ),
            max_backtracks=_bounded_integer(
                quadloc_section,
                "max_backtracks",
                minimum=0,
                maximum=20,
            ),
            region_expansion_ratio=_bounded_number(
                quadloc_section,
                "region_expansion_ratio",
                minimum=0.0,
                maximum=1.0,
            ),
            overlay_opacity=_bounded_number(
                quadloc_section,
                "overlay_opacity",
                minimum=0.01,
                maximum=0.99,
            ),
            artifact_root=_string(quadloc_section, "artifact_root"),
            model_segmentation_prompt=_string(
                quadloc_section,
                "model_segmentation_prompt",
            ),
            sam3_confidence_threshold=_bounded_number(
                quadloc_section,
                "sam3_confidence_threshold",
                minimum=0.05,
                maximum=0.95,
            ),
            sam3_overlay_opacity=_bounded_number(
                quadloc_section,
                "sam3_overlay_opacity",
                minimum=0.0,
                maximum=1.0,
            ),
            llm_role=_quadloc_llm_role(quadloc_section),
            nearest_point_chunk_size=_bounded_integer(
                quadloc_section,
                "nearest_point_chunk_size",
                minimum=1,
                maximum=10_000_000,
            ),
        ),
        part_segmentation=PartSegmentationConfig(
            max_subparts=_bounded_integer(
                part_segmentation_section,
                "max_subparts",
                minimum=1,
                maximum=12,
            ),
            artifact_root=_string(
                part_segmentation_section,
                "artifact_root",
            ),
            llm_role=_workflow_llm_role(
                part_segmentation_section,
                label="workflow.part_segmentation.llm_role",
            ),
            sam3_confidence_threshold=_bounded_number(
                part_segmentation_section,
                "sam3_confidence_threshold",
                minimum=0.05,
                maximum=0.95,
            ),
            sam3_overlay_opacity=_bounded_number(
                part_segmentation_section,
                "sam3_overlay_opacity",
                minimum=0.0,
                maximum=1.0,
            ),
            roi_padding_ratio=_bounded_number(
                part_segmentation_section,
                "roi_padding_ratio",
                minimum=0.0,
                maximum=1.0,
            ),
            parent_mask_dilation_pixels=_bounded_integer(
                part_segmentation_section,
                "parent_mask_dilation_pixels",
                minimum=0,
                maximum=128,
            ),
            max_synonym_attempts=_bounded_integer(
                part_segmentation_section,
                "max_synonym_attempts",
                minimum=0,
                maximum=5,
            ),
            minimum_child_parent_containment=_bounded_number(
                part_segmentation_section,
                "minimum_child_parent_containment",
                minimum=0.0,
                maximum=1.0,
            ),
            minimum_parent_coverage=_bounded_number(
                part_segmentation_section,
                "minimum_parent_coverage",
                minimum=0.0,
                maximum=1.0,
            ),
            maximum_pairwise_overlap_ratio=_bounded_number(
                part_segmentation_section,
                "maximum_pairwise_overlap_ratio",
                minimum=0.0,
                maximum=1.0,
            ),
            maximum_uncovered_component_ratio=_bounded_number(
                part_segmentation_section,
                "maximum_uncovered_component_ratio",
                minimum=0.0,
                maximum=1.0,
            ),
            max_instances_per_prompt=_bounded_integer(
                part_segmentation_section,
                "max_instances_per_prompt",
                minimum=1,
                maximum=128,
            ),
            duplicate_candidate_iou=_bounded_number(
                part_segmentation_section,
                "duplicate_candidate_iou",
                minimum=0.000001,
                maximum=1.0,
            ),
            max_candidate_combinations=_bounded_integer(
                part_segmentation_section,
                "max_candidate_combinations",
                minimum=1,
                maximum=10_000_000,
            ),
            lasso_padding_pixels=_bounded_integer(
                part_segmentation_section,
                "lasso_padding_pixels",
                minimum=1,
                maximum=192,
            ),
            lasso_padding_retry_multiplier=_bounded_integer(
                part_segmentation_section,
                "lasso_padding_retry_multiplier",
                minimum=1,
                maximum=16,
            ),
            lasso_max_padding_increases=_bounded_integer(
                part_segmentation_section,
                "lasso_max_padding_increases",
                minimum=0,
                maximum=16,
            ),
            lasso_simplify_tolerance_pixels=_bounded_number(
                part_segmentation_section,
                "lasso_simplify_tolerance_pixels",
                minimum=0.0,
                maximum=64.0,
            ),
            max_lasso_points=_bounded_integer(
                part_segmentation_section,
                "max_lasso_points",
                minimum=4,
                maximum=100_000,
            ),
            lasso_time_step_seconds=_bounded_number(
                part_segmentation_section,
                "lasso_time_step_seconds",
                minimum=0.000001,
                maximum=1.0,
            ),
            use_front_faces_only=_boolean(
                part_segmentation_section,
                "use_front_faces_only",
            ),
        ),
        drag=DragWorkflowConfig(
            llm_role=_workflow_llm_role(
                drag_section,
                label="workflow.drag.llm_role",
            ),
            minimum_distance_pixels=_bounded_number(
                drag_section,
                "minimum_distance_pixels",
                minimum=1.0,
                maximum=10_000.0,
            ),
            maximum_distance_ratio=_bounded_number(
                drag_section,
                "maximum_distance_ratio",
                minimum=0.01,
                maximum=1.0,
            ),
            stroke_spacing_pixels=_bounded_number(
                drag_section,
                "stroke_spacing_pixels",
                minimum=0.5,
                maximum=100.0,
            ),
            safe_anchor_minimum_margin_pixels=_bounded_number(
                drag_section,
                "safe_anchor_minimum_margin_pixels",
                minimum=0.0,
                maximum=1_000.0,
            ),
            safe_anchor_brush_radius_ratio=_bounded_number(
                drag_section,
                "safe_anchor_brush_radius_ratio",
                minimum=0.0,
                maximum=1.0,
            ),
            safe_anchor_component_depth_ratio=_bounded_number(
                drag_section,
                "safe_anchor_component_depth_ratio",
                minimum=0.01,
                maximum=1.0,
            ),
            brush_anchor_extent_percentile=_bounded_number(
                drag_section,
                "brush_anchor_extent_percentile",
                minimum=50.0,
                maximum=100.0,
            ),
            pose_brush_name=_string(drag_section, "pose_brush_name"),
            pose_brush_size=_bounded_integer(
                drag_section,
                "pose_brush_size",
                minimum=1,
                maximum=10_000,
            ),
            pose_brush_strength=_bounded_number(
                drag_section,
                "pose_brush_strength",
                minimum=0.0,
                maximum=1.0,
            ),
            pose_deformation_target=_enum_string(
                drag_section,
                "pose_deformation_target",
                allowed={"GEOMETRY", "CLOTH_SIM"},
            ),
            pose_rotation_origins=_enum_string(
                drag_section,
                "pose_rotation_origins",
                allowed={"FACE_SETS"},
            ),
            pose_origin_offset=_bounded_number(
                drag_section,
                "pose_origin_offset",
                minimum=0.0,
                maximum=2.0,
            ),
            pose_smooth_iterations=_bounded_integer(
                drag_section,
                "pose_smooth_iterations",
                minimum=0,
                maximum=100,
            ),
            pose_connected_only=_boolean(
                drag_section,
                "pose_connected_only",
            ),
            pose_max_element_distance=_bounded_number(
                drag_section,
                "pose_max_element_distance",
                minimum=0.0,
                maximum=10.0,
            ),
        ),
        draw=DrawWorkflowConfig(
            llm_role=_workflow_llm_role(
                draw_section,
                label="workflow.draw.llm_role",
            ),
            artifact_root=_string(draw_section, "artifact_root"),
            trajectory_artifact_root=_string(
                draw_section,
                "trajectory_artifact_root",
            ),
            trajectory_point_spacing_pixels=_bounded_number(
                draw_section,
                "trajectory_point_spacing_pixels",
                minimum=0.25,
                maximum=64.0,
            ),
            trajectory_flattening_spacing_pixels=_bounded_number(
                draw_section,
                "trajectory_flattening_spacing_pixels",
                minimum=0.1,
                maximum=16.0,
            ),
            fit_artifact_root=_string(
                draw_fit_section,
                "artifact_root",
            ),
            fit=MaskTrajectoryFitConfig(
                containment_margin_pixels=_bounded_number(
                    draw_fit_section,
                    "containment_margin_pixels",
                    minimum=0.0,
                    maximum=32.0,
                ),
                small_boundary_clearance_ratio=_bounded_number(
                    draw_fit_section,
                    "small_boundary_clearance_ratio",
                    minimum=0.000001,
                    maximum=0.999999,
                ),
                medium_boundary_clearance_ratio=_bounded_number(
                    draw_fit_section,
                    "medium_boundary_clearance_ratio",
                    minimum=0.000001,
                    maximum=0.999999,
                ),
                large_boundary_clearance_ratio=_bounded_number(
                    draw_fit_section,
                    "large_boundary_clearance_ratio",
                    minimum=0.000001,
                    maximum=0.999999,
                ),
                optimization_max_dimension=_bounded_integer(
                    draw_fit_section,
                    "optimization_max_dimension",
                    minimum=64,
                    maximum=2_048,
                ),
                rotation_coarse_step_degrees=_bounded_number(
                    draw_fit_section,
                    "rotation_coarse_step_degrees",
                    minimum=0.1,
                    maximum=180.0,
                ),
                rotation_refine_step_degrees=_bounded_number(
                    draw_fit_section,
                    "rotation_refine_step_degrees",
                    minimum=0.1,
                    maximum=180.0,
                ),
                rotation_fine_step_degrees=_bounded_number(
                    draw_fit_section,
                    "rotation_fine_step_degrees",
                    minimum=0.1,
                    maximum=180.0,
                ),
                scale_search_iterations=_bounded_integer(
                    draw_fit_section,
                    "scale_search_iterations",
                    minimum=4,
                    maximum=20,
                ),
            ),
            text_trajectory_artifact_root=_string(
                draw_text_section,
                "artifact_root",
            ),
            text=TextTrajectoryConfig(
                maximum_characters=_bounded_integer(
                    draw_text_section,
                    "maximum_characters",
                    minimum=1,
                    maximum=256,
                ),
                maximum_layout_lanes=_bounded_integer(
                    draw_text_section,
                    "maximum_layout_lanes",
                    minimum=1,
                    maximum=32,
                ),
                base_font_size=_bounded_number(
                    draw_text_section,
                    "base_font_size",
                    minimum=16.0,
                    maximum=512.0,
                ),
                skeleton_raster_scale=_bounded_number(
                    draw_text_section,
                    "skeleton_raster_scale",
                    minimum=1.0,
                    maximum=8.0,
                ),
                centerline_sample_spacing=_bounded_number(
                    draw_text_section,
                    "centerline_sample_spacing",
                    minimum=0.25,
                    maximum=16.0,
                ),
                centerline_simplify_tolerance=_bounded_number(
                    draw_text_section,
                    "centerline_simplify_tolerance",
                    minimum=0.0,
                    maximum=4.0,
                ),
                skeleton_spur_prune_length=_bounded_number(
                    draw_text_section,
                    "skeleton_spur_prune_length",
                    minimum=0.0,
                    maximum=16.0,
                ),
                line_gap_ratio=_bounded_number(
                    draw_text_section,
                    "line_gap_ratio",
                    minimum=0.0,
                    maximum=1.0,
                ),
                canonical_canvas_margin=_bounded_number(
                    draw_text_section,
                    "canonical_canvas_margin",
                    minimum=0.0,
                    maximum=127.999,
                ),
            ),
            font_name=_string(draw_section, "font_name"),
            stroke_method=_enum_string(
                draw_section,
                "stroke_method",
                allowed={
                    "DOTS",
                    "DRAG_DOT",
                    "SPACE",
                    "AIRBRUSH",
                    "ANCHORED",
                    "LINE",
                    "CURVE",
                },
            ),
            brush_spacing_percent=_bounded_integer(
                draw_section,
                "brush_spacing_percent",
                minimum=1,
                maximum=1_000,
            ),
            use_space_attenuation=_boolean(
                draw_section,
                "use_space_attenuation",
            ),
            auto_smooth_factor=_bounded_number(
                draw_section,
                "auto_smooth_factor",
                minimum=0.0,
                maximum=1.0,
            ),
            finishing_smooth_enabled=_boolean(
                draw_section,
                "finishing_smooth_enabled",
            ),
            finishing_smooth_brush_name=_string(
                draw_section,
                "finishing_smooth_brush_name",
            ),
            finishing_smooth_size_ratio=_bounded_number(
                draw_section,
                "finishing_smooth_size_ratio",
                minimum=0.1,
                maximum=2.0,
            ),
            finishing_smooth_strength=_bounded_number(
                draw_section,
                "finishing_smooth_strength",
                minimum=0.0,
                maximum=1.0,
            ),
            finishing_smooth_direction=_string(
                draw_section,
                "finishing_smooth_direction",
            ).upper(),
            finishing_smooth_dyntopo_enabled=_boolean(
                draw_section,
                "finishing_smooth_dyntopo_enabled",
            ),
            brush_size_ratio=_bounded_number(
                draw_section,
                "brush_size_ratio",
                minimum=0.001,
                maximum=1.0,
            ),
            minimum_brush_size=_bounded_integer(
                draw_section,
                "minimum_brush_size",
                minimum=1,
                maximum=10_000,
            ),
            maximum_brush_size=_bounded_integer(
                draw_section,
                "maximum_brush_size",
                minimum=1,
                maximum=10_000,
            ),
        ),
        minimum_effect=MinimumEffectConfig(
            enabled=_boolean(minimum_effect_section, "enabled"),
            capture_warmup_redraws=_bounded_integer(
                minimum_effect_section,
                "capture_warmup_redraws",
                minimum=1,
                maximum=5,
            ),
            baseline_capture_attempts=_bounded_integer(
                minimum_effect_section,
                "baseline_capture_attempts",
                minimum=1,
                maximum=5,
            ),
            comparison_blur_kernel_size=_bounded_integer(
                minimum_effect_section,
                "comparison_blur_kernel_size",
                minimum=1,
                maximum=9,
            ),
            baseline_noise_percentile=_bounded_number(
                minimum_effect_section,
                "baseline_noise_percentile",
                minimum=50.0,
                maximum=100.0,
            ),
            pixel_delta_floor=_bounded_number(
                minimum_effect_section,
                "pixel_delta_floor",
                minimum=0.0,
                maximum=255.0,
            ),
            maximum_baseline_noise=_bounded_number(
                minimum_effect_section,
                "maximum_baseline_noise",
                minimum=0.0,
                maximum=255.0,
            ),
            minimum_evaluation_pixels=_bounded_integer(
                minimum_effect_section,
                "minimum_evaluation_pixels",
                minimum=1,
                maximum=100_000_000,
            ),
            roi_padding_ratio=_bounded_number(
                minimum_effect_section,
                "roi_padding_ratio",
                minimum=0.0,
                maximum=1.0,
            ),
            no_effect_fraction=_bounded_number(
                minimum_effect_section,
                "no_effect_fraction",
                minimum=0.0,
                maximum=1.0,
            ),
            subtle_minimum_mean_abs_diff=_bounded_number(
                minimum_effect_section,
                "subtle_minimum_mean_abs_diff",
                minimum=0.0,
                maximum=255.0,
            ),
            subtle_minimum_changed_fraction=_bounded_number(
                minimum_effect_section,
                "subtle_minimum_changed_fraction",
                minimum=0.0,
                maximum=1.0,
            ),
            medium_minimum_mean_abs_diff=_bounded_number(
                minimum_effect_section,
                "medium_minimum_mean_abs_diff",
                minimum=0.0,
                maximum=255.0,
            ),
            medium_minimum_changed_fraction=_bounded_number(
                minimum_effect_section,
                "medium_minimum_changed_fraction",
                minimum=0.0,
                maximum=1.0,
            ),
            strong_minimum_mean_abs_diff=_bounded_number(
                minimum_effect_section,
                "strong_minimum_mean_abs_diff",
                minimum=0.0,
                maximum=255.0,
            ),
            strong_minimum_changed_fraction=_bounded_number(
                minimum_effect_section,
                "strong_minimum_changed_fraction",
                minimum=0.0,
                maximum=1.0,
            ),
        ),
        parameter_resolution=ParameterResolutionConfig(
            local_size_ratio=_bounded_number(
                parameter_resolution_section,
                "local_size_ratio",
                minimum=0.01,
                maximum=2.0,
            ),
            regional_size_ratio=_bounded_number(
                parameter_resolution_section,
                "regional_size_ratio",
                minimum=0.01,
                maximum=2.0,
            ),
            broad_size_ratio=_bounded_number(
                parameter_resolution_section,
                "broad_size_ratio",
                minimum=0.01,
                maximum=2.0,
            ),
            retry_dose_multiplier=_bounded_number(
                parameter_resolution_section,
                "retry_dose_multiplier",
                minimum=1.0,
                maximum=4.0,
            ),
            maximum_brush_size=_bounded_integer(
                parameter_resolution_section,
                "maximum_brush_size",
                minimum=1,
                maximum=10_000,
            ),
            maximum_brush_strength=_bounded_number(
                parameter_resolution_section,
                "maximum_brush_strength",
                minimum=0.0,
                maximum=1.0,
            ),
            maximum_pass_count=_bounded_integer(
                parameter_resolution_section,
                "maximum_pass_count",
                minimum=1,
                maximum=20,
            ),
        ),
        operation_defaults=operation_defaults,
        recursion_limit=_bounded_integer(
            workflow_section,
            "recursion_limit",
            minimum=10,
            maximum=1000,
        ),
    )
    _validate_runtime_relationships(workflow)
    return SculptWorkflowConfig(
        llm=llm,
        llm_runtime=llm_runtime,
        services=services,
        workflow=workflow,
        source_path=source,
    )


def _llm_runtime_config(
    payload: Mapping[str, object],
) -> LlmRuntimeConfig:
    """Parse provider presets and fixed multimodal test assets."""
    raw_presets = payload.get("presets")
    if not isinstance(raw_presets, list) or not raw_presets:
        raise WorkflowConfigError(
            "llm.runtime.presets must be a non-empty array of tables"
        )
    presets: list[LlmProviderPresetConfig] = []
    preset_ids: set[str] = set()
    base_urls: set[str] = set()
    for index, raw_preset in enumerate(raw_presets):
        if not isinstance(raw_preset, dict):
            raise WorkflowConfigError(
                f"llm.runtime.presets[{index}] must be a config table"
            )
        preset_id = _string(raw_preset, "id")
        if preset_id in preset_ids:
            raise WorkflowConfigError(
                f"Duplicate llm.runtime preset id {preset_id}"
            )
        base_url = _http_url(_string(raw_preset, "base_url"))
        base_key = base_url.casefold()
        if base_key in base_urls:
            raise WorkflowConfigError(
                f"Duplicate llm.runtime preset Base URL {base_url}"
            )
        api_key_mode = _string(raw_preset, "api_key_mode").lower()
        if api_key_mode not in {"required", "if_present", "none"}:
            raise WorkflowConfigError(
                "llm.runtime preset api_key_mode must be required, "
                "if_present, or none"
            )
        schema_profile = _string(
            raw_preset,
            "schema_profile",
        ).lower()
        if schema_profile not in {"full", "gemini_compatible"}:
            raise WorkflowConfigError(
                "llm.runtime preset schema_profile must be full or "
                "gemini_compatible"
            )
        raw_anthropic_path = raw_preset.get("anthropic_endpoint_path")
        if raw_anthropic_path is not None and not isinstance(
            raw_anthropic_path,
            str,
        ):
            raise WorkflowConfigError(
                "anthropic_endpoint_path must be a string when provided"
            )
        anthropic_path = (
            None
            if raw_anthropic_path is None
            else _endpoint_path(raw_anthropic_path.strip())
        )
        presets.append(
            LlmProviderPresetConfig(
                preset_id=preset_id,
                label=_string(raw_preset, "label"),
                base_url=base_url,
                api_key_env=_string(raw_preset, "api_key_env"),
                api_key_mode=api_key_mode,
                schema_profile=schema_profile,
                openai_endpoint_path=_endpoint_path(
                    _string(raw_preset, "openai_endpoint_path")
                ),
                anthropic_endpoint_path=anthropic_path,
            )
        )
        preset_ids.add(preset_id)
        base_urls.add(base_key)
    return LlmRuntimeConfig(
        test_prompt_path=_string(payload, "test_prompt_path"),
        test_image_path=_string(payload, "test_image_path"),
        presets=tuple(presets),
    )


def _quadloc_llm_role(payload: Mapping[str, object]) -> str:
    """Validate the dedicated multimodal model role used by QuadLoc."""
    return _workflow_llm_role(
        payload,
        label="workflow.quadloc.llm_role",
    )


def _workflow_llm_role(
    payload: Mapping[str, object],
    *,
    label: str,
) -> str:
    """Validate one configured workflow model role."""
    role = _string(payload, "llm_role")
    if role not in LLM_ROLES:
        raise WorkflowConfigError(
            f"{label} must be one of " + ", ".join(LLM_ROLES)
        )
    return role


def _validate_runtime_relationships(
    workflow: WorkflowRuntimeConfig,
) -> None:
    """Validate ordering constraints across independently parsed fields."""
    effect = workflow.minimum_effect
    if effect.pixel_delta_floor > effect.maximum_baseline_noise:
        raise WorkflowConfigError(
            "minimum_effect.pixel_delta_floor cannot exceed "
            "maximum_baseline_noise"
        )
    if effect.comparison_blur_kernel_size % 2 == 0:
        raise WorkflowConfigError(
            "minimum_effect.comparison_blur_kernel_size must be odd"
        )
    mean_thresholds = (
        effect.subtle_minimum_mean_abs_diff,
        effect.medium_minimum_mean_abs_diff,
        effect.strong_minimum_mean_abs_diff,
    )
    fraction_thresholds = (
        effect.subtle_minimum_changed_fraction,
        effect.medium_minimum_changed_fraction,
        effect.strong_minimum_changed_fraction,
    )
    if list(mean_thresholds) != sorted(mean_thresholds):
        raise WorkflowConfigError(
            "minimum-effect mean thresholds must increase from SUBTLE "
            "through STRONG"
        )
    if list(fraction_thresholds) != sorted(fraction_thresholds):
        raise WorkflowConfigError(
            "minimum-effect fraction thresholds must increase from "
            "SUBTLE through STRONG"
        )
    if effect.no_effect_fraction > effect.subtle_minimum_changed_fraction:
        raise WorkflowConfigError(
            "minimum_effect.no_effect_fraction cannot exceed the SUBTLE "
            "minimum changed fraction"
        )
    resolution = workflow.parameter_resolution
    size_ratios = (
        resolution.local_size_ratio,
        resolution.regional_size_ratio,
        resolution.broad_size_ratio,
    )
    if not size_ratios[0] < size_ratios[1] < size_ratios[2]:
        raise WorkflowConfigError(
            "parameter-resolution size ratios must satisfy "
            "LOCAL < REGIONAL < BROAD"
        )


def _operation_defaults(
    payload: Mapping[str, object],
) -> dict[str, SculptOperationDefaultsConfig]:
    """Parse deterministic per-operation Sculpt settings."""
    parsed: dict[str, SculptOperationDefaultsConfig] = {}
    canonical_names = {name.casefold(): name for name in OPERATION_METHODS}
    for raw_name, raw_settings in payload.items():
        canonical_name = canonical_names.get(raw_name.casefold())
        if canonical_name is None:
            raise WorkflowConfigError(
                f"Unknown workflow.operation_defaults operation {raw_name}"
            )
        if canonical_name in parsed:
            raise WorkflowConfigError(
                "workflow.operation_defaults contains duplicate operation "
                f"{canonical_name}"
            )
        if not isinstance(raw_settings, dict):
            raise WorkflowConfigError(
                "workflow.operation_defaults."
                f"{raw_name} must be a config section"
            )
        parsed[canonical_name] = SculptOperationDefaultsConfig(
            dyntopo_enabled=_boolean(raw_settings, "dyntopo_enabled"),
            dyntopo_detail_size=_bounded_number(
                raw_settings,
                "dyntopo_detail_size",
                minimum=0.5,
                maximum=40.0,
            ),
        )
    missing = [
        name for name in IMPLEMENTED_OPERATION_METHODS if name not in parsed
    ]
    if missing:
        raise WorkflowConfigError(
            "workflow.operation_defaults is missing implemented operations: "
            + ", ".join(missing)
        )
    return parsed


def _read_dotenv(path: Path) -> dict[str, str]:
    """Read simple dotenv assignments without logging secret values."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise WorkflowConfigError(
            f"Cannot read configured secrets file {path}"
        ) from error
    values: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:].lstrip()
        key, raw_value = stripped.split("=", maxsplit=1)
        key = key.strip()
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {
            "'",
            '"',
        }:
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def _section(
    payload: Mapping[str, object],
    name: str,
) -> Mapping[str, object]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise WorkflowConfigError(f"Missing [{name}] config section")
    return value


def _string(
    payload: Mapping[str, object],
    name: str,
    *,
    allow_empty: bool = False,
) -> str:
    value = payload.get(name)
    if not isinstance(value, str):
        raise WorkflowConfigError(f"{name} must be a string")
    stripped = value.strip()
    if not stripped and not allow_empty:
        raise WorkflowConfigError(f"{name} must not be empty")
    return stripped


def _enum_string(
    payload: Mapping[str, object],
    name: str,
    *,
    allowed: set[str],
) -> str:
    """Read one uppercase identifier from a closed configuration set."""
    value = _string(payload, name).upper()
    if value not in allowed:
        raise WorkflowConfigError(
            f"{name} must be one of " + ", ".join(sorted(allowed))
        )
    return value


def _positive_number(
    payload: Mapping[str, object],
    name: str,
) -> float:
    return _bounded_number(
        payload,
        name,
        minimum=0.001,
        maximum=86_400.0,
    )


def _bounded_number(
    payload: Mapping[str, object],
    name: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkflowConfigError(f"{name} must be a number")
    number = float(value)
    if not minimum <= number <= maximum:
        raise WorkflowConfigError(
            f"{name} must be between {minimum:g} and {maximum:g}"
        )
    return number


def _bounded_integer(
    payload: Mapping[str, object],
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkflowConfigError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise WorkflowConfigError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return value


def _boolean(payload: Mapping[str, object], name: str) -> bool:
    value = payload.get(name)
    if not isinstance(value, bool):
        raise WorkflowConfigError(f"{name} must be a boolean")
    return value


def _view_list(
    payload: Mapping[str, object],
    name: str,
) -> tuple[str, ...]:
    value = payload.get(name)
    if not isinstance(value, list) or not value:
        raise WorkflowConfigError(f"{name} must be a non-empty array")
    views: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise WorkflowConfigError(f"{name} must contain strings")
        view = item.strip().upper()
        if view not in STANDARD_VIEWS:
            raise WorkflowConfigError(
                f"{name} contains unsupported view {view}"
            )
        if view in views:
            raise WorkflowConfigError(f"{name} contains duplicate {view}")
        views.append(view)
    return tuple(views)


def _http_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise WorkflowConfigError("llm.base_url must be an HTTP(S) URL")
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise WorkflowConfigError(
            "llm.base_url must not contain credentials, query, or fragment"
        )
    return value.rstrip("/")


def _endpoint_path(value: str) -> str:
    parsed = urlsplit(value)
    if (
        not value.startswith("/")
        or value.startswith("//")
        or parsed.scheme
        or parsed.netloc
        or parsed.query
        or parsed.fragment
    ):
        raise WorkflowConfigError(
            "llm.endpoint_path must be an absolute URL path without "
            "credentials, query, or fragment"
        )
    if any(part in {".", ".."} for part in parsed.path.split("/")):
        raise WorkflowConfigError(
            "llm.endpoint_path must not contain dot path segments"
        )
    return parsed.path


def _loopback_port(
    value: str,
    *,
    expected_path: str,
    label: str,
) -> int:
    parsed = urlsplit(value)
    if parsed.scheme != "http" or parsed.hostname not in {
        "127.0.0.1",
        "localhost",
    }:
        raise WorkflowConfigError(
            f"{label} must use HTTP loopback host 127.0.0.1 or localhost"
        )
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise WorkflowConfigError(
            f"{label} must not contain credentials, query, or fragment"
        )
    path = parsed.path.rstrip("/")
    if path != expected_path:
        expected = expected_path or "/"
        raise WorkflowConfigError(f"{label} path must be {expected}")
    try:
        port = parsed.port
    except ValueError as error:
        raise WorkflowConfigError(f"{label} has an invalid port") from error
    if port is None:
        raise WorkflowConfigError(f"{label} must include an explicit port")
    return port
