"""JSON-serializable LangGraph state for the Sculpt Agent workflow."""

from __future__ import annotations

import re
from typing import Annotated, NotRequired, TypedDict
from uuid import uuid4

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages

from visculpt.bridge import JsonValue

from .events import WORKFLOW_EVENT_SCHEMA_VERSION

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class SculptWorkflowInput(TypedDict):
    """Minimal public input accepted by the Agent Server graph."""

    user_instruction: str
    run_id: NotRequired[str]


class SculptWorkflowState(TypedDict):
    """Shared state exposed to every workflow node and future Web clients."""

    run_id: str
    thread_id: str | None
    user_instruction: str
    messages: Annotated[list[AnyMessage], add_messages]
    workflow_dir: str
    state_artifact_path: str | None
    workflow_status: str
    llm_config: dict[str, JsonValue]
    llm_config_overridden: bool
    token_usage: dict[str, JsonValue]
    event_schema_version: str
    event_sequence: int
    service_status: dict[str, JsonValue]
    blender_session_lease: dict[str, JsonValue] | None
    initial_sculpt_response: JsonValue
    sculpt_viewport_ui_snapshot: dict[str, JsonValue] | None
    sculpt_viewport_ui_restore_response: JsonValue
    sculpt_brush_capabilities: dict[str, JsonValue] | None
    artifacts: list[dict[str, JsonValue]]
    decomposer_screenshots: dict[str, JsonValue]
    subtasks: list[dict[str, JsonValue]]
    translator_screenshots: dict[str, JsonValue]
    translations: list[dict[str, JsonValue]]
    current_subtask_index: int
    current_attempt: int
    checkpoint_path: str | None
    view_screenshots: dict[str, JsonValue]
    view_segmentation_results: dict[str, JsonValue]
    valid_views: list[str]
    rejected_drag_anchor_views: list[str]
    selected_view: str | None
    view_selection_reason: str | None
    view_selection_intervention: dict[str, JsonValue] | None
    manual_mask_request: dict[str, JsonValue] | None
    manual_mask_intervention: dict[str, JsonValue] | None
    manual_mask_result: dict[str, JsonValue] | None
    manual_mask_overrides: dict[str, JsonValue]
    execution_preparation: dict[str, JsonValue] | None
    segmentation_result: dict[str, JsonValue] | None
    quadloc_result: dict[str, JsonValue] | None
    drag_target_binding: dict[str, JsonValue] | None
    locked_drag_plan: dict[str, JsonValue] | None
    drag_direction_plan: dict[str, JsonValue] | None
    part_segmentation_result: dict[str, JsonValue] | None
    draw_pattern_result: dict[str, JsonValue] | None
    draw_trajectory_result: dict[str, JsonValue] | None
    draw_fitted_trajectory_result: dict[str, JsonValue] | None
    resolved_sculpt_plan: dict[str, JsonValue] | None
    stroke_plan_result: dict[str, JsonValue] | None
    viewport_focus: dict[str, JsonValue] | None
    full_before_screenshot: dict[str, JsonValue] | None
    baseline_screenshot_a: dict[str, JsonValue] | None
    baseline_screenshot_b: dict[str, JsonValue] | None
    minimum_effect_baseline: dict[str, JsonValue] | None
    after_screenshot: dict[str, JsonValue] | None
    full_after_screenshot: dict[str, JsonValue] | None
    settings_result: JsonValue
    execution_results: list[JsonValue]
    execution_error: str | None
    minimum_effect_result: dict[str, JsonValue] | None
    grader_result: dict[str, JsonValue] | None
    acceptance_decision: dict[str, JsonValue] | None
    retry_feedback: dict[str, JsonValue] | None
    retry_directive: dict[str, JsonValue] | None
    retry_scope: str | None
    restore_action: str | None
    execution_decision: str | None
    current_approval_id: str | None
    applied_operation_ids: list[str]
    attempt_history: list[dict[str, JsonValue]]
    subtask_results: list[dict[str, JsonValue]]
    llm_calls: list[dict[str, JsonValue]]
    events: list[dict[str, JsonValue]]


def create_initial_workflow_state(
    user_instruction: str,
    *,
    run_id: str | None = None,
) -> SculptWorkflowState:
    """Create a complete initial state without any hidden runtime objects."""
    instruction = user_instruction.strip()
    if not instruction:
        raise ValueError("user_instruction must not be empty")
    identifier = uuid4().hex if run_id is None else run_id
    if RUN_ID_PATTERN.fullmatch(identifier) is None:
        raise ValueError(
            "run_id must contain only ASCII letters, digits, dots, "
            "underscores, and hyphens"
        )
    workflow_dir = f"run-{identifier}"
    return SculptWorkflowState(
        run_id=identifier,
        thread_id=None,
        user_instruction=instruction,
        messages=[HumanMessage(content=instruction)],
        workflow_dir=workflow_dir,
        state_artifact_path=None,
        workflow_status="created",
        llm_config={},
        llm_config_overridden=False,
        token_usage={},
        event_schema_version=WORKFLOW_EVENT_SCHEMA_VERSION,
        event_sequence=0,
        service_status={},
        blender_session_lease=None,
        initial_sculpt_response=None,
        sculpt_viewport_ui_snapshot=None,
        sculpt_viewport_ui_restore_response=None,
        sculpt_brush_capabilities=None,
        artifacts=[],
        decomposer_screenshots={},
        subtasks=[],
        translator_screenshots={},
        translations=[],
        current_subtask_index=0,
        current_attempt=0,
        checkpoint_path=None,
        view_screenshots={},
        view_segmentation_results={},
        valid_views=[],
        rejected_drag_anchor_views=[],
        selected_view=None,
        view_selection_reason=None,
        view_selection_intervention=None,
        manual_mask_request=None,
        manual_mask_intervention=None,
        manual_mask_result=None,
        manual_mask_overrides={},
        execution_preparation=None,
        segmentation_result=None,
        quadloc_result=None,
        drag_target_binding=None,
        locked_drag_plan=None,
        drag_direction_plan=None,
        part_segmentation_result=None,
        draw_pattern_result=None,
        draw_trajectory_result=None,
        draw_fitted_trajectory_result=None,
        resolved_sculpt_plan=None,
        stroke_plan_result=None,
        viewport_focus=None,
        full_before_screenshot=None,
        baseline_screenshot_a=None,
        baseline_screenshot_b=None,
        minimum_effect_baseline=None,
        after_screenshot=None,
        full_after_screenshot=None,
        settings_result=None,
        execution_results=[],
        execution_error=None,
        minimum_effect_result=None,
        grader_result=None,
        acceptance_decision=None,
        retry_feedback=None,
        retry_directive=None,
        retry_scope=None,
        restore_action=None,
        execution_decision=None,
        current_approval_id=None,
        applied_operation_ids=[],
        attempt_history=[],
        subtask_results=[],
        llm_calls=[],
        events=[],
    )
