"""Runtime-only workflow dependencies kept outside LangGraph state."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from langchain_core.tools import BaseTool

from visculpt.bridge import BlenderRpcClient, JsonValue
from visculpt.tools import (
    create_change_view_tool,
    create_enter_sculpt_mode_tool,
    create_execute_sculpt_stroke_tool,
    create_fit_svg_trajectories_to_mask_tool,
    create_focus_viewport_roi_tool,
    create_generate_svg_pattern_tool,
    create_get_screenshot_tool,
    create_get_sculpt_brush_capabilities_tool,
    create_load_blender_state_tool,
    create_part_segmentation_with_sam3_tool,
    create_plan_sculpt_strokes_tool,
    create_quadloc_tool,
    create_restore_sculpt_viewport_ui_tool,
    create_restore_viewport_state_tool,
    create_save_blender_state_tool,
    create_segment_with_sam3_tool,
    create_select_mask_component_tool,
    create_set_sculpt_settings_tool,
    create_svg_to_mouse_trajectories_tool,
    create_text_to_mouse_trajectories_tool,
)
from visculpt.vision.sam3 import (
    Sam3GradioClient,
    Sam3SegmentationResult,
)

from .config import SculptWorkflowConfig
from .llm import HttpStructuredLlm, StructuredMultimodalLlm
from .services import check_blender_rpc_ready, check_sam3_ready
from .token_usage import (
    TokenUsageStore,
    default_token_usage_database_path,
)

LlmBoundToolFactory = Callable[[StructuredMultimodalLlm], BaseTool]


class Sam3Segmenter(Protocol):
    """SAM3 segmentation-only surface used by view prefiltering."""

    def segment(
        self,
        *,
        image_path: str | Path,
        prompt: str,
        confidence_threshold: float = 0.5,
        overlay_opacity: float = 0.45,
        output_dir: str | Path | None = None,
    ) -> Sam3SegmentationResult:
        """Return a persisted semantic segmentation result."""
        ...


@dataclass(frozen=True, slots=True)
class WorkflowTools:
    """Tool instances used by deterministic workflow nodes."""

    enter_sculpt_mode: BaseTool
    restore_sculpt_viewport_ui: BaseTool
    change_view: BaseTool
    focus_viewport_roi: BaseTool
    restore_viewport_state: BaseTool
    get_screenshot: BaseTool
    get_sculpt_brush_capabilities: BaseTool
    save_blender_state: BaseTool
    load_blender_state: BaseTool
    set_sculpt_settings: BaseTool
    segment_with_sam3: BaseTool
    select_mask_component: BaseTool
    part_segmentation_with_sam3: BaseTool
    quadloc: BaseTool
    plan_sculpt_strokes: BaseTool
    execute_sculpt_stroke: BaseTool
    generate_svg_pattern: BaseTool
    svg_to_mouse_trajectories: BaseTool
    fit_svg_trajectories_to_mask: BaseTool
    text_to_mouse_trajectories: BaseTool


@dataclass(frozen=True, slots=True)
class WorkflowDependencies:
    """Non-serializable clients and probes injected into graph nodes."""

    llm: StructuredMultimodalLlm
    tools: WorkflowTools
    sam3_segmenter: Sam3Segmenter
    blender_ready: Callable[[], dict[str, JsonValue]]
    sam3_ready: Callable[[], dict[str, JsonValue]]
    part_segmentation_tool_factory: LlmBoundToolFactory | None = None
    mask_component_selector_tool_factory: LlmBoundToolFactory | None = None
    quadloc_tool_factory: LlmBoundToolFactory | None = None
    generate_svg_pattern_tool_factory: LlmBoundToolFactory | None = None
    blender_client: BlenderRpcClient | None = None
    sam3_client: Sam3GradioClient | None = None
    token_usage_store: TokenUsageStore | None = None


def create_default_workflow_dependencies(
    config: SculptWorkflowConfig,
    *,
    workdir: Path | None = None,
) -> WorkflowDependencies:
    """Construct live local tools and the configured remote LLM adapter."""
    blender_client = BlenderRpcClient(config.services.blender_rpc_config())
    sam3_config = config.services.sam3_config()
    sam3_segmenter = Sam3GradioClient(sam3_config)
    llm = HttpStructuredLlm(
        config.llm,
        api_key=config.api_key(workdir=workdir),
    )
    segment_tool = create_segment_with_sam3_tool(client=sam3_segmenter)

    def build_mask_component_selector_tool(
        run_llm: StructuredMultimodalLlm,
    ) -> BaseTool:
        return create_select_mask_component_tool(
            llm=run_llm,
            llm_role="translator",
            workdir=workdir,
        )

    def build_part_segmentation_tool(
        run_llm: StructuredMultimodalLlm,
    ) -> BaseTool:
        return create_part_segmentation_with_sam3_tool(
            llm=run_llm,
            segment_tool=segment_tool,
            mask_component_selector_tool=(
                build_mask_component_selector_tool(run_llm)
            ),
            client=blender_client,
            config=config.workflow.part_segmentation,
            workdir=workdir,
        )

    def build_quadloc_tool(
        run_llm: StructuredMultimodalLlm,
    ) -> BaseTool:
        return create_quadloc_tool(
            llm=run_llm,
            segment_tool=segment_tool,
            config=config.workflow.quadloc,
            workdir=workdir,
        )

    def build_generate_svg_pattern_tool(
        run_llm: StructuredMultimodalLlm,
    ) -> BaseTool:
        return create_generate_svg_pattern_tool(
            llm=run_llm,
            llm_role=config.workflow.draw.llm_role,
        )

    tools = WorkflowTools(
        enter_sculpt_mode=create_enter_sculpt_mode_tool(
            client=blender_client
        ),
        restore_sculpt_viewport_ui=(
            create_restore_sculpt_viewport_ui_tool(
                client=blender_client
            )
        ),
        change_view=create_change_view_tool(client=blender_client),
        focus_viewport_roi=create_focus_viewport_roi_tool(
            client=blender_client
        ),
        restore_viewport_state=create_restore_viewport_state_tool(
            client=blender_client
        ),
        get_screenshot=create_get_screenshot_tool(client=blender_client),
        get_sculpt_brush_capabilities=(
            create_get_sculpt_brush_capabilities_tool(
                client=blender_client
            )
        ),
        save_blender_state=create_save_blender_state_tool(
            client=blender_client
        ),
        load_blender_state=create_load_blender_state_tool(
            client=blender_client
        ),
        set_sculpt_settings=create_set_sculpt_settings_tool(
            client=blender_client
        ),
        segment_with_sam3=segment_tool,
        select_mask_component=build_mask_component_selector_tool(llm),
        part_segmentation_with_sam3=build_part_segmentation_tool(llm),
        quadloc=build_quadloc_tool(llm),
        plan_sculpt_strokes=create_plan_sculpt_strokes_tool(),
        execute_sculpt_stroke=create_execute_sculpt_stroke_tool(
            client=blender_client
        ),
        generate_svg_pattern=build_generate_svg_pattern_tool(llm),
        svg_to_mouse_trajectories=(
            create_svg_to_mouse_trajectories_tool(
                point_spacing_pixels=(
                    config.workflow.draw.trajectory_point_spacing_pixels
                ),
                flattening_spacing_pixels=(
                    config.workflow.draw
                    .trajectory_flattening_spacing_pixels
                ),
            )
        ),
        fit_svg_trajectories_to_mask=(
            create_fit_svg_trajectories_to_mask_tool(
                config=config.workflow.draw.fit,
            )
        ),
        text_to_mouse_trajectories=(
            create_text_to_mouse_trajectories_tool(
                config=config.workflow.draw.text,
                fit_config=config.workflow.draw.fit,
            )
        ),
    )
    return WorkflowDependencies(
        llm=llm,
        tools=tools,
        sam3_segmenter=sam3_segmenter,
        blender_ready=lambda: check_blender_rpc_ready(blender_client),
        sam3_ready=lambda: check_sam3_ready(sam3_segmenter.config),
        part_segmentation_tool_factory=build_part_segmentation_tool,
        mask_component_selector_tool_factory=(
            build_mask_component_selector_tool
        ),
        quadloc_tool_factory=build_quadloc_tool,
        generate_svg_pattern_tool_factory=(
            build_generate_svg_pattern_tool
        ),
        blender_client=blender_client,
        sam3_client=sam3_segmenter,
        token_usage_store=TokenUsageStore(
            default_token_usage_database_path(workdir or Path.cwd())
        ),
    )
