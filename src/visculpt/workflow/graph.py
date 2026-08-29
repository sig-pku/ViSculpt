"""LangGraph implementation of the initial Blender Sculpt Agent workflow."""

from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from langchain_core.messages import AIMessage, message_to_dict
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.errors import GraphInterrupt
from langgraph.config import get_config
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from visculpt.bridge import JsonValue
from visculpt.vision import (
    DragAnchorCorrectionError,
    DragBrushSizeResolutionError,
    DragStrokePlanningError,
    DragTargetBindingError,
    DrawStrokePlanningError,
    MaskComponentSelectionError,
    SculptStrokePolylineVisualizationResult,
    bind_drag_target_component,
    cleaned_mask_roi,
    correct_drag_anchor,
    foreground_component_count,
    plan_drag_stroke,
    plan_draw_strokes,
    render_sculpt_stroke_polyline_visualizations,
    resolve_drag_brush_size,
    warp_cleaned_mask_to_focused_view,
)

from .artifacts import (
    WorkflowArtifact,
    WorkflowArtifactKind,
    create_artifact,
    merge_artifacts,
)
from .config import SculptWorkflowConfig, load_workflow_config
from .dependencies import (
    WorkflowDependencies,
    create_default_workflow_dependencies,
)
from .errors import WorkflowExecutionError, WorkflowLlmError
from .events import (
    WorkflowEventEmitter,
    WorkflowEventStatus,
    stream_token_usage_update,
)
from .idempotency import sculpt_operation_id
from .llm import HttpStructuredLlm, StructuredMultimodalLlm
from .models import (
    DecomposedSubtask,
    DecomposerOutput,
    DragDirectionPlan,
    DrawScaleTier,
    EffectAppropriateness,
    GraderOutput,
    OperationMethod,
    RetryPlannerOutput,
    SculptIntent,
    SurfaceRetryScope,
    TranslatedSubtask,
    TranslatorOutput,
    ViewSelection,
)
from .minimum_effect import (
    MinimumEffectBaseline,
    MinimumEffectResult,
    MinimumEffectVerdict,
    evaluate_minimum_effect,
    prepare_minimum_effect_baseline,
    retry_directive_for_minimum_effect,
    unmeasured_minimum_effect_result,
)
from .manual_mask import (
    ManualMaskError,
    ManualMaskPaintResponse,
    ManualMaskRasterResult,
    ManualMaskReviewResponse,
    rasterize_manual_mask,
)
from .parameter_resolution import resolve_sculpt_plan
from .prompts import (
    DECOMPOSER_SYSTEM_PROMPT,
    DRAG_DIRECTION_SYSTEM_PROMPT,
    GRADER_SYSTEM_PROMPT,
    RETRY_PLANNER_SYSTEM_PROMPT,
    TRANSLATOR_SYSTEM_PROMPT,
    VIEW_SELECTOR_SYSTEM_PROMPT,
    decomposer_user_prompt,
    drag_direction_user_prompt,
    grader_user_prompt,
    retry_planner_user_prompt,
    translator_user_prompt,
    view_selector_user_prompt,
)
from .runtime_llm import (
    RuntimeLlmSettings,
    create_runtime_llm,
    runtime_llm_settings_from_runnable,
)
from .state import (
    SculptWorkflowInput,
    SculptWorkflowState,
    create_initial_workflow_state,
)
from .token_usage import TokenUsageContext, TokenUsageRecorder
from .session import session_lease_for_artifact_root
from .sculpt_capabilities import (
    BlenderSculptCapabilities,
    SculptCapabilityError,
    parse_get_state_sculpt_capabilities,
)
from .view_prefilter import assess_view_segmentation

type StateUpdate = dict[str, Any]

_DRAG_RETRY_DOSE = "drag_dose"
_DRAG_RETRY_GESTURE = "drag_gesture"
_DRAG_RETRY_LOCALIZE = "drag_localize"
_DRAG_RETRY_VIEW = "reselect"
_DRAG_RETRY_INTENT = "intent"
_SURFACE_RETRY_RESEGMENT = "surface_resegment"
_SURFACE_RETRY_REUSE_SEGMENTATION = "surface_reuse_segmentation"
_MANUAL_MASK_SMEAR = "smear_target"
_MANUAL_MASK_DRAW = "draw_target"
_MANUAL_MASK_CHANGED_PART = "changed_part"
_MANUAL_MASK_MAX_BRUSH_SIZE = 300
_QUADLOC_MODEL_MASK = "quadloc_model"

_DRAG_PREPARATION_CHANGED_PART = "changed_part"
_DRAG_PREPARATION_QUADLOC = "quadloc"
_DRAG_PREPARATION_FINAL = "final"
_DRAG_PREPARATION_COMPLETED = "completed"


class _DragPreparationRetry(WorkflowExecutionError):
    """Carry the earliest invalid Drag stage into deterministic routing."""

    def __init__(self, message: str, *, retry_scope: str) -> None:
        super().__init__(message)
        self.retry_scope = retry_scope


class _Sam3NoSegmentation(WorkflowExecutionError):
    """Identify deterministic empty segmentation at one semantic call site."""

    def __init__(
        self,
        message: str,
        *,
        call_site: str,
        image_path: str,
        prompt: str,
    ) -> None:
        super().__init__(message)
        self.call_site = call_site
        self.image_path = image_path
        self.prompt = prompt


class SculptAgentWorkflow:
    """Compiled deterministic graph around multimodal planning nodes."""

    def __init__(
        self,
        config: SculptWorkflowConfig,
        dependencies: WorkflowDependencies,
        *,
        workdir: Path | None = None,
    ) -> None:
        self.config = config
        self.dependencies = dependencies
        self.workdir = workdir
        self.graph = self._build_graph()

    @classmethod
    def from_config(
        cls,
        config_path: str | Path | None = None,
        *,
        workdir: Path | None = None,
    ) -> SculptAgentWorkflow:
        """Load config, secrets, live clients, and compile the graph."""
        config = load_workflow_config(config_path)
        dependencies = create_default_workflow_dependencies(
            config,
            workdir=workdir,
        )
        return cls(config, dependencies, workdir=workdir)

    def invoke(
        self,
        user_instruction: str,
        *,
        run_id: str | None = None,
        workdir: Path | None = None,
    ) -> SculptWorkflowState:
        """Run the workflow to completion and return final shared state."""
        graph_input: SculptWorkflowInput = {
            "user_instruction": user_instruction,
        }
        if run_id is not None:
            graph_input["run_id"] = run_id
        result = cast(
            SculptWorkflowState,
            self.graph.invoke(
                graph_input,
                config=self._local_run_config(workdir),
            ),
        )
        state_path = result.get("state_artifact_path")
        if isinstance(state_path, str) and not Path(state_path).is_absolute():
            result = cast(SculptWorkflowState, dict(result))
            result["state_artifact_path"] = str(
                (
                    self.config.artifact_root(workdir)
                    / result["workflow_dir"]
                    / state_path
                ).resolve()
            )
        return result

    def stream(
        self,
        user_instruction: str,
        *,
        run_id: str | None = None,
        workdir: Path | None = None,
        stream_mode: str = "values",
    ) -> Iterator[object]:
        """Stream graph updates for a future Web client."""
        graph_input: SculptWorkflowInput = {
            "user_instruction": user_instruction,
        }
        if run_id is not None:
            graph_input["run_id"] = run_id
        return cast(
            Iterator[object],
            self.graph.stream(
                graph_input,
                config=self._local_run_config(workdir),
                stream_mode=stream_mode,
            ),
        )

    def _build_graph(self) -> CompiledStateGraph:
        builder = StateGraph(
            SculptWorkflowState,
            input_schema=SculptWorkflowInput,
        )
        builder.add_node("initialize", self._initialize)
        builder.add_node(
            "initial_check",
            self._guarded(self._initial_check),
        )
        builder.add_node("decomposer", self._guarded(self._decomposer))
        builder.add_node("translator", self._guarded(self._translator))
        builder.add_node(
            "prepare_subtask",
            self._guarded(self._prepare_subtask),
        )
        builder.add_node(
            "save_checkpoint",
            self._guarded(self._save_checkpoint),
        )
        builder.add_node(
            "view_selector",
            self._guarded(self._view_selector),
        )
        builder.add_node(
            "manual_view_selection",
            self._guarded_with_config(self._manual_view_selection),
        )
        builder.add_node(
            "prepare_manual_mask",
            self._guarded(self._prepare_manual_mask),
        )
        builder.add_node(
            "manual_mask_input",
            self._guarded_with_config(self._manual_mask_input),
        )
        builder.add_node(
            "manual_mask_review",
            self._guarded_with_config(self._manual_mask_review),
        )
        builder.add_node(
            "prepare_execution_dispatch",
            self._guarded(self._prepare_execution_dispatch),
        )
        builder.add_node(
            "prepare_drag_context",
            self._guarded(self._prepare_drag_context),
        )
        builder.add_node(
            "prepare_drag_changed_part",
            self._guarded(self._prepare_drag_changed_part),
        )
        builder.add_node(
            "prepare_drag_quadloc",
            self._guarded(self._prepare_drag_quadloc),
        )
        builder.add_node(
            "prepare_execution",
            self._guarded(self._prepare_execution),
        )
        builder.add_node(
            "approve_execution",
            self._guarded_with_config(self._approve_execution),
        )
        builder.add_node("executor", self._guarded(self._executor))
        builder.add_node(
            "minimum_effect",
            self._guarded(self._minimum_effect),
        )
        builder.add_node("grader", self._guarded(self._grader))
        builder.add_node(
            "acceptance_gate",
            self._guarded(self._acceptance_gate),
        )
        builder.add_node(
            "retry_planner",
            self._guarded(self._retry_planner),
        )
        builder.add_node(
            "restore_checkpoint",
            self._guarded(self._restore_checkpoint),
        )
        builder.add_node(
            "advance_subtask",
            self._guarded(self._advance_subtask),
        )
        builder.add_node("finalize", self._guarded(self._finalize))

        builder.add_edge(START, "initialize")
        builder.add_edge("initialize", "initial_check")
        builder.add_edge("initial_check", "decomposer")
        builder.add_edge("decomposer", "translator")
        builder.add_edge("translator", "prepare_subtask")
        builder.add_conditional_edges(
            "prepare_subtask",
            self._route_prepare,
            {
                "save": "save_checkpoint",
                "finish": "finalize",
            },
        )
        builder.add_edge("save_checkpoint", "view_selector")
        builder.add_conditional_edges(
            "view_selector",
            self._route_view_selector,
            {
                "execute": "prepare_execution_dispatch",
                "manual": "manual_view_selection",
            },
        )
        builder.add_conditional_edges(
            "manual_view_selection",
            self._route_manual_view_selection,
            {
                "execute": "prepare_execution_dispatch",
                "advance": "advance_subtask",
            },
        )
        builder.add_conditional_edges(
            "prepare_execution_dispatch",
            self._route_execution_dispatch,
            {
                "drag_context": "prepare_drag_context",
                "drag_changed_part": "prepare_drag_changed_part",
                "drag_quadloc": "prepare_drag_quadloc",
                "execute": "prepare_execution",
            },
        )
        builder.add_edge("prepare_drag_context", "prepare_drag_changed_part")
        builder.add_conditional_edges(
            "prepare_drag_changed_part",
            self._route_drag_changed_part,
            {
                "quadloc": "prepare_drag_quadloc",
                "manual_mask": "prepare_manual_mask",
                "restore": "restore_checkpoint",
            },
        )
        builder.add_conditional_edges(
            "prepare_drag_quadloc",
            self._route_drag_quadloc,
            {
                "execute": "prepare_execution",
                "restore": "restore_checkpoint",
            },
        )
        builder.add_conditional_edges(
            "prepare_execution",
            self._route_prepared_execution,
            {
                "approve": "approve_execution",
                "restore": "restore_checkpoint",
                "manual_mask": "prepare_manual_mask",
            },
        )
        builder.add_conditional_edges(
            "prepare_manual_mask",
            self._route_prepare_manual_mask,
            {
                "paint": "manual_mask_input",
                "advance": "advance_subtask",
            },
        )
        builder.add_conditional_edges(
            "manual_mask_input",
            self._route_manual_mask_input,
            {
                "review": "manual_mask_review",
                "advance": "advance_subtask",
            },
        )
        builder.add_conditional_edges(
            "manual_mask_review",
            self._route_manual_mask_review,
            {
                "execute": "prepare_execution_dispatch",
                "redraw": "manual_mask_input",
                "advance": "advance_subtask",
            },
        )
        builder.add_conditional_edges(
            "approve_execution",
            self._route_approval,
            {
                "execute": "executor",
                "restore": "restore_checkpoint",
                "advance": "advance_subtask",
            },
        )
        builder.add_conditional_edges(
            "executor",
            self._route_executor,
            {
                "effect": "minimum_effect",
                "advance": "advance_subtask",
            },
        )
        builder.add_conditional_edges(
            "minimum_effect",
            self._route_minimum_effect,
            {
                "grade": "grader",
                "restore": "restore_checkpoint",
            },
        )
        builder.add_edge("grader", "acceptance_gate")
        builder.add_conditional_edges(
            "acceptance_gate",
            self._route_acceptance,
            {
                "advance": "advance_subtask",
                "retry": "retry_planner",
                "restore": "restore_checkpoint",
            },
        )
        builder.add_edge("retry_planner", "restore_checkpoint")
        builder.add_conditional_edges(
            "restore_checkpoint",
            self._route_restore,
            {
                "same_view": "prepare_execution_dispatch",
                "reselect": "view_selector",
                "advance": "advance_subtask",
            },
        )
        builder.add_edge("advance_subtask", "prepare_subtask")
        builder.add_edge("finalize", END)
        return builder.compile(name="sculpt_agent")

    def _local_run_config(self, workdir: Path | None) -> RunnableConfig:
        """Keep direct Python/CLI runs non-interactive and testable."""
        return {
            "recursion_limit": (
                self.config.workflow.effective_recursion_limit
            ),
            "configurable": {
                "artifact_root": str(self.config.artifact_root(workdir)),
                "auto_approve": True,
                "auto_skip_no_valid_view": True,
                "auto_skip_manual_mask": True,
            },
        }

    def _initialize(
        self,
        graph_input: SculptWorkflowInput,
        config: RunnableConfig,
    ) -> StateUpdate:
        """Expand the minimal public input into complete durable State."""
        state = create_initial_workflow_state(
            graph_input["user_instruction"],
            run_id=graph_input.get("run_id"),
        )
        llm_settings, overridden = runtime_llm_settings_from_runnable(
            self.config,
            config.get("configurable", {}),
        )
        configurable = config.get("configurable", {})
        thread_value = configurable.get("thread_id")
        state["thread_id"] = (
            thread_value if isinstance(thread_value, str) else None
        )
        state["llm_config"] = llm_settings.model_dump(mode="json")
        state["llm_config_overridden"] = overridden
        usage_store = self.dependencies.token_usage_store
        if usage_store is not None:
            try:
                state["token_usage"] = usage_store.workflow_summary(
                    state["run_id"],
                    thread_id=state["thread_id"],
                    title=state["user_instruction"],
                )
            except Exception:
                # Usage accounting is observational and cannot block sculpting.
                pass
        emitter = WorkflowEventEmitter(state, node="initialize")
        emitter.workflow(
            status=WorkflowEventStatus.CREATED,
            workflow_status="created",
            message="Workflow state initialized",
        )
        state.update(cast(SculptWorkflowState, emitter.state_update()))
        return cast(StateUpdate, state)

    def _llm_for_state(
        self,
        state: SculptWorkflowState,
        *,
        call_site: str | None = None,
    ) -> StructuredMultimodalLlm:
        """Return the immutable per-run adapter or injected default fake."""
        base: StructuredMultimodalLlm
        if not state.get("llm_config_overridden", False):
            base = self.dependencies.llm
        else:
            settings = RuntimeLlmSettings.model_validate(state["llm_config"])
            base = create_runtime_llm(
                self.config,
                settings,
                workdir=self.workdir,
            )
        usage_store = self.dependencies.token_usage_store
        if usage_store is None or not isinstance(base, HttpStructuredLlm):
            return base
        site = call_site or "workflow_llm"
        recorder = TokenUsageRecorder(
            usage_store,
            TokenUsageContext(
                scope="workflow",
                run_id=state["run_id"],
                thread_id=state.get("thread_id"),
                workflow_title=state["user_instruction"],
                call_site=site,
            ),
            on_update=lambda update: stream_token_usage_update(
                state,
                update,
                node=site,
            ),
        )
        return base.with_call_observer(recorder)

    def _llm_bound_tool(
        self,
        state: SculptWorkflowState,
        *,
        fallback: BaseTool,
        factory: Callable[[StructuredMultimodalLlm], BaseTool] | None,
        call_site: str = "composite_tool",
    ) -> BaseTool:
        """Bind composite Tool model calls to the immutable Run snapshot."""
        if factory is None:
            return fallback
        return factory(self._llm_for_state(state, call_site=call_site))

    def _guarded(
        self,
        function: Callable[[SculptWorkflowState], StateUpdate],
    ) -> Callable[[SculptWorkflowState], StateUpdate]:
        """Refresh the Blender lease and release it after node failures."""

        def invoke(state: SculptWorkflowState) -> StateUpdate:
            try:
                if state.get("blender_session_lease") is not None:
                    self._lease(state).heartbeat(state["run_id"])
                update = function(state)
                return self._with_token_usage(state, update)
            except GraphInterrupt:
                raise
            except Exception as error:
                self._cleanup_failed_run(state, error)
                raise

        return invoke

    def _guarded_with_config(
        self,
        function: Callable[
            [SculptWorkflowState, RunnableConfig],
            StateUpdate,
        ],
    ) -> Callable[[SculptWorkflowState, RunnableConfig], StateUpdate]:
        """Guard a node that also consumes per-run configuration."""

        def invoke(
            state: SculptWorkflowState,
            config: RunnableConfig,
        ) -> StateUpdate:
            try:
                self._lease(state).heartbeat(state["run_id"])
                update = function(state, config)
                return self._with_token_usage(state, update)
            except GraphInterrupt:
                raise
            except Exception as error:
                self._cleanup_failed_run(state, error)
                raise

        return invoke

    def _with_token_usage(
        self,
        state: SculptWorkflowState,
        update: StateUpdate,
    ) -> StateUpdate:
        """Checkpoint the latest aggregate without persisting every call."""
        store = self.dependencies.token_usage_store
        if store is None:
            return update
        try:
            summary = store.workflow_summary(
                state["run_id"],
                thread_id=state.get("thread_id"),
                title=state["user_instruction"],
            )
        except Exception:
            # Usage is observational; storage failures must not alter geometry.
            return update
        return {**update, "token_usage": summary}

    def _cleanup_failed_run(
        self,
        state: SculptWorkflowState,
        original_error: Exception,
    ) -> None:
        """Roll back the active subtask, restore UI, and release its lease."""
        lease = self._lease(state)
        lease_status = lease.status()
        active = lease_status.get("lease")
        if not isinstance(active, dict) or active.get("run_id") != state[
            "run_id"
        ]:
            # A run cannot modify Blender after its lease is released or transferred.
            return
        cleanup_errors: list[str] = []
        checkpoint_value = state.get("checkpoint_path")
        if isinstance(checkpoint_value, str):
            try:
                self._load_blender_checkpoint(
                    state,
                    checkpoint_value,
                    label="load_blender_state(failure_rollback)",
                )
            except Exception as error:
                cleanup_errors.append(
                    f"Blender checkpoint rollback failed: {error}"
                )
        snapshot = state.get("sculpt_viewport_ui_snapshot")
        if isinstance(snapshot, dict):
            try:
                self._restore_sculpt_viewport_ui_snapshot(snapshot)
            except Exception as error:
                cleanup_errors.append(
                    f"Sculpt viewport UI restoration failed: {error}"
                )
        released = lease.release(state["run_id"])
        lease_status = lease.status()
        active = lease_status.get("lease")
        if (
            not released
            and isinstance(active, dict)
            and active.get("run_id") == state["run_id"]
        ):
            cleanup_errors.append("Blender session lease was not released")
        if cleanup_errors:
            raise WorkflowExecutionError(
                "Workflow failed and cleanup was incomplete: "
                + "; ".join(cleanup_errors)
            ) from original_error

    def _load_blender_checkpoint(
        self,
        state: SculptWorkflowState,
        checkpoint_value: str,
        *,
        label: str,
    ) -> JsonValue:
        """Load and verify one run-owned Blender transaction checkpoint."""
        checkpoint = self._run_path(state, checkpoint_value)
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise WorkflowExecutionError(
                f"Blender checkpoint is missing or empty: {checkpoint}"
            )
        response = _invoke_tool(
            self.dependencies.tools.load_blender_state,
            {
                "filepath": str(checkpoint),
                "load_ui": self.config.workflow.load_ui_on_restore,
            },
            label=label,
        )
        result = _rpc_result(response, label)
        loaded = result.get("filepath")
        if result.get("loaded") is not True:
            raise WorkflowExecutionError(
                f"{label} did not confirm that Blender loaded the checkpoint"
            )
        if not isinstance(loaded, str) or Path(loaded).resolve() != checkpoint:
            raise WorkflowExecutionError(
                f"{label} loaded an unexpected file: {loaded}"
            )
        return response

    def _restore_sculpt_viewport_ui_snapshot(
        self,
        snapshot: dict[str, JsonValue],
    ) -> JsonValue:
        """Restore one UI snapshot through the dedicated bounded Tool."""
        response = _invoke_tool(
            self.dependencies.tools.restore_sculpt_viewport_ui,
            {"snapshot": snapshot},
            label="restore_sculpt_viewport_ui",
        )
        result = _rpc_result(response, "restore_sculpt_viewport_ui")
        if result.get("restored") is not True:
            raise WorkflowExecutionError(
                "restore_sculpt_viewport_ui did not confirm restoration"
            )
        return response

    def _artifact_root(self) -> Path:
        config = get_config()
        configurable = config.get("configurable", {})
        root_value = configurable.get("artifact_root")
        return (
            Path(str(root_value)).resolve()
            if root_value
            else self.config.artifact_root()
        )

    def _workflow_dir(self, state: SculptWorkflowState) -> Path:
        root = self._artifact_root()
        stored = Path(state["workflow_dir"])
        resolved = (stored if stored.is_absolute() else root / stored).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise WorkflowExecutionError(
                "workflow_dir escapes the configured artifact root"
            ) from error
        return resolved

    def _run_path(
        self,
        state: SculptWorkflowState,
        value: str | Path,
    ) -> Path:
        workflow_dir = self._workflow_dir(state)
        stored = Path(value)
        resolved = (
            stored if stored.is_absolute() else workflow_dir / stored
        ).resolve()
        try:
            resolved.relative_to(workflow_dir)
        except ValueError as error:
            raise WorkflowExecutionError(
                "workflow path escapes the current run directory"
            ) from error
        return resolved

    def _lease(self, state: SculptWorkflowState):
        return session_lease_for_artifact_root(
            self._artifact_root(),
            lease_seconds=(
                self.config.workflow.blender_session_lease_seconds
            ),
        )

    def _initial_check(self, state: SculptWorkflowState) -> StateUpdate:
        workflow_dir = self._workflow_dir(state)
        workflow_dir.mkdir(parents=True, exist_ok=True)
        lease = self._lease(state).acquire(state["run_id"])
        emitter = WorkflowEventEmitter(state, node="initial_check")
        emitter.node(
            status=WorkflowEventStatus.RUNNING,
            phase="started",
            message="Checking local Blender and SAM3 services",
        )
        blender = self.dependencies.blender_ready()
        emitter.service(
            service="blender_rpc",
            ready=True,
            message="Blender RPC Server is ready",
        )
        sam3 = self.dependencies.sam3_ready()
        emitter.service(
            service="sam3",
            ready=True,
            message="SAM3 inference service is ready",
        )
        enter_response = _invoke_tool(
            self.dependencies.tools.enter_sculpt_mode,
            {"hide_viewport_ui": True},
            label="enter_sculpt_mode",
        )
        enter_result = _rpc_result(enter_response, "enter_sculpt_mode")
        snapshot: dict[str, JsonValue] | None = None
        try:
            viewport_ui = _required_mapping(
                enter_result,
                "viewport_ui",
                label="enter_sculpt_mode",
            )
            snapshot = cast(
                dict[str, JsonValue],
                dict(
                    _required_mapping(
                        viewport_ui,
                        "snapshot",
                        label="enter_sculpt_mode viewport_ui",
                    )
                ),
            )
            lease = self._lease(state).remember_viewport_ui_snapshot(
                state["run_id"],
                snapshot,
            )
            if (
                enter_result.get("workspace") != "Sculpting"
                or enter_result.get("mode") != "SCULPT"
            ):
                raise WorkflowExecutionError(
                    "enter_sculpt_mode did not enter Sculpting/SCULPT"
                )
            capability_response = _invoke_tool(
                self.dependencies.tools.get_sculpt_brush_capabilities,
                {},
                label="get_sculpt_brush_capabilities",
            )
            try:
                capabilities = parse_get_state_sculpt_capabilities(
                    capability_response
                )
            except SculptCapabilityError as error:
                raise WorkflowExecutionError(
                    "Cannot read local Sculpt brush capabilities: "
                    f"{error}"
                ) from error
        except Exception as error:
            if snapshot is not None:
                try:
                    self._restore_sculpt_viewport_ui_snapshot(snapshot)
                except Exception as cleanup_error:
                    raise WorkflowExecutionError(
                        "Initial check failed and the Sculpt viewport UI "
                        f"could not be restored: {cleanup_error}"
                    ) from error
            raise
        capability_payload = capabilities.model_dump(mode="json")
        emitter.node(
            status=WorkflowEventStatus.SUCCESS,
            phase="completed",
            message=(
                "Services are ready, Sculpt Mode is active, viewport UI "
                "distractions are hidden, and "
                f"{capabilities.brush_count} local brushes were indexed"
            ),
        )
        return {
            "workflow_status": "ready",
            "service_status": {"blender_rpc": blender, "sam3": sam3},
            "blender_session_lease": lease,
            "initial_sculpt_response": enter_response,
            "sculpt_viewport_ui_snapshot": snapshot,
            "sculpt_brush_capabilities": capability_payload,
            **emitter.state_update(),
        }

    def _decomposer(self, state: SculptWorkflowState) -> StateUpdate:
        capabilities = _capabilities_from_state(state)
        capability_payload = capabilities.model_dump(mode="json")
        emitter = WorkflowEventEmitter(state, node="decomposer")
        emitter.node(
            status=WorkflowEventStatus.RUNNING,
            phase="started",
            message="Capturing views for instruction decomposition",
        )
        screenshots, paths = self._capture_view_set(
            state,
            self._workflow_dir(state) / "decomposer",
            self.config.workflow.decomposer_views,
            emitter=emitter,
            label_prefix="Decomposer",
        )
        completion = self._llm_for_state(
            state,
            call_site="decomposer",
        ).complete(
            role="decomposer",
            system_prompt=DECOMPOSER_SYSTEM_PROMPT,
            user_prompt=decomposer_user_prompt(
                user_instruction=state["user_instruction"],
                screenshot_paths=paths,
                max_subtasks=self.config.workflow.max_subtasks,
                sculpt_capabilities=capability_payload,
            ),
            image_paths=list(paths.values()),
            response_model=DecomposerOutput,
        )
        if not isinstance(completion.value, DecomposerOutput):
            raise WorkflowLlmError(
                "Decomposer adapter returned the wrong response model"
            )
        if len(completion.value.subtasks) > (
            self.config.workflow.max_subtasks
        ):
            raise WorkflowLlmError(
                "Decomposer returned more subtasks than configured"
            )
        subtasks: list[JsonValue] = []
        for item in completion.value.subtasks:
            try:
                description = (
                    capabilities.canonicalize_subtask_description(
                        item.description
                    )
                )
            except SculptCapabilityError as error:
                raise WorkflowLlmError(
                    f"Decomposer selected an unavailable Sculpt brush: "
                    f"{error}"
                ) from error
            canonical = DecomposedSubtask(
                description=description,
                operation_method=item.operation_method,
            )
            subtasks.append(canonical.model_dump(mode="json"))
        artifacts = _screenshot_artifacts(screenshots)
        emitter.artifacts(
            artifact_ids=[item.artifact_id for item in artifacts],
            message="Decomposer screenshots are available",
        )
        emitter.node(
            status=WorkflowEventStatus.SUCCESS,
            phase="completed",
            message=f"Created {len(subtasks)} ordered subtasks",
        )
        return {
            "workflow_status": "planning",
            "decomposer_screenshots": screenshots,
            "subtasks": subtasks,
            "artifacts": merge_artifacts(state["artifacts"], artifacts),
            "llm_calls": [*state["llm_calls"], completion.metadata],
            **emitter.state_update(),
        }

    def _translator(self, state: SculptWorkflowState) -> StateUpdate:
        capabilities = _capabilities_from_state(state)
        capability_payload = capabilities.model_dump(mode="json")
        emitter = WorkflowEventEmitter(state, node="translator")
        emitter.node(
            status=WorkflowEventStatus.RUNNING,
            phase="started",
            message="Capturing views for parameter translation",
        )
        screenshots, paths = self._capture_view_set(
            state,
            self._workflow_dir(state) / "translator",
            self.config.workflow.translator_views,
            emitter=emitter,
            label_prefix="Translator",
        )
        completion = self._llm_for_state(
            state,
            call_site="translator",
        ).complete(
            role="translator",
            system_prompt=TRANSLATOR_SYSTEM_PROMPT,
            user_prompt=translator_user_prompt(
                subtasks=state["subtasks"],
                screenshot_paths=paths,
                sculpt_capabilities=capability_payload,
            ),
            image_paths=list(paths.values()),
            response_model=TranslatorOutput,
        )
        if not isinstance(completion.value, TranslatorOutput):
            raise WorkflowLlmError(
                "Translator adapter returned the wrong response model"
            )
        ordered = sorted(
            completion.value.translations,
            key=lambda item: item.subtask_index,
        )
        indices = [item.subtask_index for item in ordered]
        expected = list(range(len(state["subtasks"])))
        if indices != expected:
            raise WorkflowLlmError(
                "Translator must return each subtask_index exactly once"
            )
        validated_translations: list[TranslatedSubtask] = []
        for item in ordered:
            subtask = state["subtasks"][item.subtask_index]
            intent = _validated_sculpt_intent(
                item.intent,
                subtask=subtask,
                capabilities=capabilities,
                role="Translator",
            )
            validated_translations.append(
                TranslatedSubtask(
                    subtask_index=item.subtask_index,
                    intent=intent,
                )
            )
        translations = [
            item.model_dump(mode="json", exclude_none=True)
            for item in validated_translations
        ]
        artifacts = _screenshot_artifacts(screenshots)
        emitter.artifacts(
            artifact_ids=[item.artifact_id for item in artifacts],
            message="Translator screenshots are available",
        )
        emitter.node(
            status=WorkflowEventStatus.SUCCESS,
            phase="completed",
            message=(
                f"Translated {len(translations)} subtasks into settings"
            ),
        )
        return {
            "translator_screenshots": screenshots,
            "translations": cast(list[JsonValue], translations),
            "artifacts": merge_artifacts(state["artifacts"], artifacts),
            "llm_calls": [*state["llm_calls"], completion.metadata],
            **emitter.state_update(),
        }

    def _prepare_subtask(
        self,
        state: SculptWorkflowState,
    ) -> StateUpdate:
        emitter = WorkflowEventEmitter(state, node="prepare_subtask")
        index = state["current_subtask_index"]
        if index >= len(state["subtasks"]):
            emitter.workflow(
                status=WorkflowEventStatus.RUNNING,
                workflow_status="finishing",
                message="All subtasks have been processed",
            )
            return {
                "workflow_status": "finishing",
                **emitter.state_update(),
            }
        operation_method = str(
            state["subtasks"][index].get("operation_method", "")
        )
        emitter.subtask(
            status=WorkflowEventStatus.RUNNING,
            action="prepared",
            operation_method=operation_method,
            message=f"Preparing subtask {index + 1}",
        )
        return {
            "workflow_status": "executing",
            "current_attempt": 1,
            "checkpoint_path": None,
            "view_screenshots": {},
            "view_segmentation_results": {},
            "valid_views": [],
            "rejected_drag_anchor_views": [],
            "selected_view": None,
            "view_selection_reason": None,
            "view_selection_intervention": None,
            "manual_mask_request": None,
            "manual_mask_intervention": None,
            "manual_mask_result": None,
            "manual_mask_overrides": {},
            "execution_preparation": None,
            "segmentation_result": None,
            "quadloc_result": None,
            "drag_target_binding": None,
            "locked_drag_plan": None,
            "drag_direction_plan": None,
            "part_segmentation_result": None,
            "draw_pattern_result": None,
            "draw_trajectory_result": None,
            "draw_fitted_trajectory_result": None,
            "resolved_sculpt_plan": None,
            "stroke_plan_result": None,
            "viewport_focus": None,
            "full_before_screenshot": None,
            "baseline_screenshot_a": None,
            "baseline_screenshot_b": None,
            "minimum_effect_baseline": None,
            "after_screenshot": None,
            "full_after_screenshot": None,
            "settings_result": None,
            "execution_results": [],
            "execution_error": None,
            "minimum_effect_result": None,
            "grader_result": None,
            "acceptance_decision": None,
            "retry_feedback": None,
            "retry_directive": None,
            "retry_scope": None,
            "restore_action": None,
            "execution_decision": None,
            "current_approval_id": None,
            **emitter.state_update(),
        }

    def _save_checkpoint(self, state: SculptWorkflowState) -> StateUpdate:
        emitter = WorkflowEventEmitter(state, node="save_checkpoint")
        emitter.node(
            status=WorkflowEventStatus.RUNNING,
            phase="started",
            message="Saving the pre-subtask Blender state",
        )
        index = state["current_subtask_index"]
        workflow_dir = self._workflow_dir(state)
        snapshot_dir = workflow_dir / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = (
            snapshot_dir / f"subtask-{index + 1:03d}-before.blend"
        ).resolve()
        if not snapshot_path.exists():
            response = _invoke_tool(
                self.dependencies.tools.save_blender_state,
                {
                    "filepath": str(snapshot_path),
                    "overwrite": False,
                    "compress": self.config.workflow.snapshot_compress,
                },
                label="save_blender_state",
            )
            _rpc_result(response, "save_blender_state")
        if not snapshot_path.is_file():
            raise WorkflowExecutionError(
                f"Blender snapshot was not created at {snapshot_path}"
            )
        if snapshot_path.stat().st_size == 0:
            raise WorkflowExecutionError(
                f"Blender snapshot is empty at {snapshot_path}"
            )
        self._lease(state).remember_checkpoint(
            state["run_id"],
            str(snapshot_path),
        )
        artifact = create_artifact(
            run_id=state["run_id"],
            workflow_dir=workflow_dir,
            path=snapshot_path,
            kind=WorkflowArtifactKind.BLENDER_STATE,
            label=f"Subtask {index + 1} rollback state",
        )
        emitter.artifacts(
            artifact_ids=[artifact.artifact_id],
            message="Blender rollback state is available",
        )
        emitter.node(
            status=WorkflowEventStatus.SUCCESS,
            phase="completed",
            message=f"Saved pre-subtask state to {snapshot_path.name}",
        )
        return {
            "checkpoint_path": artifact.relative_path,
            "artifacts": merge_artifacts(
                state["artifacts"],
                [artifact],
            ),
            **emitter.state_update(),
        }

    def _view_selector(self, state: SculptWorkflowState) -> StateUpdate:
        emitter = WorkflowEventEmitter(state, node="view_selector")
        emitter.node(
            status=WorkflowEventStatus.RUNNING,
            phase="started",
            message=(
                "Capturing and validating all six standard views with SAM3"
            ),
        )
        attempt_dir = self._attempt_dir(state)
        screenshots, paths = self._capture_view_set(
            state,
            attempt_dir / "view-selector",
            self.config.workflow.standard_views,
            emitter=emitter,
            label_prefix=(
                f"Subtask {state['current_subtask_index'] + 1} "
                f"attempt {state['current_attempt']}"
            ),
        )
        (
            evaluations,
            valid_paths,
            overlay_paths,
            summaries,
            probe_artifacts,
        ) = self._prefilter_views_with_sam3(
            state,
            screenshots=screenshots,
            screenshot_paths=paths,
            emitter=emitter,
        )
        rejected_views = set(state["rejected_drag_anchor_views"])
        for view in rejected_views:
            valid_paths.pop(view, None)
            overlay_paths.pop(view, None)
            summaries.pop(view, None)
            evaluation = evaluations.get(view)
            if isinstance(evaluation, dict):
                evaluation["workflow_rejection"] = {
                    "code": "DRAG_SAFE_ANCHOR_UNAVAILABLE",
                    "reason": (
                        "A previous attempt could not find a safe target-mask "
                        "mouse-down point in this view"
                    ),
                }
        screenshot_artifacts = _screenshot_artifacts(screenshots)
        artifacts = [*screenshot_artifacts, *probe_artifacts]
        emitter.artifacts(
            artifact_ids=[item.artifact_id for item in artifacts],
            message="View screenshots and SAM3 probe results are available",
        )

        if not valid_paths:
            index = state["current_subtask_index"]
            if rejected_views:
                reason = (
                    "No standard view remains usable after SAM3 validation "
                    "and Safe Drag Anchor rejection for "
                    f"{self._current_intent(state)['part_to_be_changed']}"
                )
                completion_message = (
                    "Manual view selection is required because no SAM3-valid "
                    "view remains with a safe Drag anchor"
                )
            else:
                reason = (
                    "No standard view produced a usable SAM3 mask for "
                    f"{self._current_intent(state)['part_to_be_changed']}"
                )
                completion_message = (
                    "Manual view selection is required because SAM3 rejected "
                    "all six standard views"
                )
            intervention_id = (
                f"{state['run_id']}:subtask-{index}:"
                f"attempt-{state['current_attempt']}:manual-view"
            )
            intervention: dict[str, JsonValue] = {
                "schema_version": "2.0",
                "intervention_id": intervention_id,
                "status": "pending",
                "decision": None,
                "selected_view": None,
                "reason": reason,
            }
            emitter.intervention(
                intervention_id=intervention_id,
                decision="pending",
                selected_view=None,
                message=(
                    f"Waiting for a manual operation view for subtask "
                    f"{index + 1}"
                ),
            )
            emitter.node(
                status=WorkflowEventStatus.WAITING,
                phase="completed",
                message=completion_message,
            )
            return {
                "view_screenshots": screenshots,
                "view_segmentation_results": evaluations,
                "valid_views": [],
                "selected_view": None,
                "view_selection_reason": reason,
                "view_selection_intervention": intervention,
                "artifacts": merge_artifacts(
                    state["artifacts"],
                    artifacts,
                ),
                **emitter.state_update(),
            }

        candidate_image_paths = _view_candidate_image_paths(
            valid_paths,
            overlay_paths,
        )
        completion = self._llm_for_state(
            state,
            call_site="view_selector",
        ).complete(
            role="view_selector",
            system_prompt=VIEW_SELECTOR_SYSTEM_PROMPT,
            user_prompt=view_selector_user_prompt(
                subtask=self._current_subtask(state),
                intent=self._current_intent(state),
                screenshot_paths=valid_paths,
                overlay_paths=overlay_paths,
                segmentation_summaries=summaries,
                retry_feedback=state["retry_feedback"],
            ),
            image_paths=candidate_image_paths,
            response_model=ViewSelection,
        )
        if not isinstance(completion.value, ViewSelection):
            raise WorkflowLlmError(
                "View Selector adapter returned the wrong response model"
            )
        selection = completion.value
        selected_view = selection.view.value
        if selected_view not in valid_paths:
            raise WorkflowLlmError(
                "View Selector selected a SAM3-invalid view "
                f"{selected_view}; allowed views are "
                f"{', '.join(valid_paths)}"
            )
        emitter.node(
            status=WorkflowEventStatus.SUCCESS,
            phase="completed",
            message=(
                f"Selected {selected_view} from "
                f"{len(valid_paths)} SAM3-valid views"
            ),
        )
        return {
            "view_screenshots": screenshots,
            "view_segmentation_results": evaluations,
            "valid_views": list(valid_paths),
            "selected_view": selected_view,
            "view_selection_reason": selection.reason,
            "view_selection_intervention": None,
            "artifacts": merge_artifacts(state["artifacts"], artifacts),
            "llm_calls": [*state["llm_calls"], completion.metadata],
            **emitter.state_update(),
        }

    def _manual_view_selection(
        self,
        state: SculptWorkflowState,
        config: RunnableConfig,
    ) -> StateUpdate:
        """Let the user override SAM3 view rejection or skip the subtask."""
        intervention = state["view_selection_intervention"]
        if not isinstance(intervention, dict):
            raise WorkflowExecutionError(
                "Manual view selection started without a pending request"
            )
        intervention_id = intervention.get("intervention_id")
        if not isinstance(intervention_id, str):
            raise WorkflowExecutionError(
                "Manual view selection request is missing intervention_id"
            )
        views: list[dict[str, JsonValue]] = []
        for view in self.config.workflow.standard_views:
            screenshot = state["view_screenshots"].get(view)
            if not isinstance(screenshot, dict):
                raise WorkflowExecutionError(
                    f"Manual view selection is missing the {view} screenshot"
                )
            artifact = screenshot.get("artifact")
            if not isinstance(artifact, dict):
                raise WorkflowExecutionError(
                    f"Manual view selection is missing the {view} artifact"
                )
            evaluation = state["view_segmentation_results"].get(view)
            assessment = (
                evaluation.get("assessment")
                if isinstance(evaluation, dict)
                else None
            )
            views.append(
                {
                    "view": view,
                    "screenshot_artifact": dict(artifact),
                    "assessment": (
                        dict(assessment)
                        if isinstance(assessment, dict)
                        else None
                    ),
                }
            )

        configurable = config.get("configurable", {})
        if configurable.get("auto_skip_no_valid_view") is True:
            response: object = {"decision": "skip"}
        else:
            response = interrupt(
                {
                    "schema_version": "2.0",
                    "type": "sculpt.view_selection.required",
                    "intervention_id": intervention_id,
                    "run_id": state["run_id"],
                    "subtask_index": state["current_subtask_index"],
                    "attempt": state["current_attempt"],
                    "question": (
                        "Choose an operation view despite SAM3 validation, "
                        "or skip this subtask."
                    ),
                    "subtask": self._current_subtask(state),
                    "views": views,
                    "allowed_decisions": ["select", "skip"],
                }
            )
        if not isinstance(response, Mapping):
            raise WorkflowExecutionError(
                "Manual view selection response must be an object"
            )
        decision = response.get("decision")
        if decision not in {"select", "skip"}:
            raise WorkflowExecutionError(
                "Manual view selection decision must be select or skip"
            )

        emitter = WorkflowEventEmitter(
            state,
            node="manual_view_selection",
        )
        if decision == "select":
            selected = response.get("view")
            allowed_views = set(self.config.workflow.standard_views)
            if not isinstance(selected, str) or selected not in allowed_views:
                raise WorkflowExecutionError(
                    "Manual view selection must choose one standard view"
                )
            if selected not in state["view_screenshots"]:
                raise WorkflowExecutionError(
                    f"Manual view selection has no screenshot for {selected}"
                )
            reason = (
                f"User manually selected {selected} after all standard "
                "views failed automatic validation"
            )
            resolved_intervention = {
                **intervention,
                "status": "selected",
                "decision": "select",
                "selected_view": selected,
            }
            emitter.intervention(
                intervention_id=intervention_id,
                decision="selected",
                selected_view=selected,
                message=f"User selected the {selected} operation view",
            )
            emitter.node(
                status=WorkflowEventStatus.SUCCESS,
                phase="completed",
                message=f"Continuing with the manually selected {selected} view",
            )
            return {
                "selected_view": selected,
                "view_selection_reason": reason,
                "view_selection_intervention": resolved_intervention,
                **emitter.state_update(),
            }

        method = OperationMethod(
            str(self._current_subtask(state)["operation_method"])
        )
        reason = (
            "User skipped the subtask after no standard view produced a "
            "usable automatic segmentation"
        )
        results = list(state["subtask_results"])
        results.append(
            {
                "subtask_index": state["current_subtask_index"],
                "status": "skipped_no_valid_view",
                "operation_method": method.value,
                "attempts": state["current_attempt"],
                "reason": reason,
                "view_segmentation_results": state[
                    "view_segmentation_results"
                ],
            }
        )
        resolved_intervention = {
            **intervention,
            "status": "skipped",
            "decision": "skip",
            "selected_view": None,
        }
        emitter.intervention(
            intervention_id=intervention_id,
            decision="skipped",
            selected_view=None,
            message="User skipped the subtask without selecting a view",
        )
        emitter.subtask(
            status=WorkflowEventStatus.SKIPPED,
            action="executed",
            operation_method=method.value,
            message=(
                f"Skipped subtask {state['current_subtask_index'] + 1}: "
                f"{reason}"
            ),
        )
        emitter.node(
            status=WorkflowEventStatus.SKIPPED,
            phase="completed",
            message="Manual view selection ended without a selected view",
        )
        return {
            "selected_view": None,
            "view_selection_reason": reason,
            "view_selection_intervention": resolved_intervention,
            "subtask_results": results,
            **emitter.state_update(),
        }

    def _manual_mask_override_path(
        self,
        state: SculptWorkflowState,
        *,
        call_site: str,
        selected_view: str | None = None,
    ) -> Path | None:
        value = state["manual_mask_overrides"].get(call_site)
        if not isinstance(value, dict):
            return None
        if (
            selected_view is not None
            and value.get("selected_view") != selected_view
        ):
            # A manual mask belongs to one screenshot coordinate system and view.
            return None
        mask_value = value.get("cleaned_mask_path")
        if not isinstance(mask_value, str):
            raise WorkflowExecutionError(
                f"Manual mask override {call_site} is missing cleaned_mask_path"
            )
        path = self._run_path(state, mask_value)
        if not path.is_file():
            raise WorkflowExecutionError(
                f"Manual mask override is not a file: {path}"
            )
        return path

    def _manual_segmentation_override(
        self,
        state: SculptWorkflowState,
        *,
        call_site: str,
        image_path: str,
        prompt: str,
    ) -> dict[str, JsonValue] | None:
        mask_path = self._manual_mask_override_path(
            state,
            call_site=call_site,
            selected_view=state["selected_view"],
        )
        if mask_path is None:
            return None
        value = state["manual_mask_overrides"].get(call_site)
        assert isinstance(value, dict)
        overlay_value = value.get("cleaned_overlay_path")
        overlay_path = (
            self._run_path(state, overlay_value)
            if isinstance(overlay_value, str)
            else None
        )
        if overlay_path is not None and not overlay_path.is_file():
            raise WorkflowExecutionError(
                f"Manual mask overlay is not a file: {overlay_path}"
            )
        result: dict[str, JsonValue] = {
            "source": "manual_mask_override",
            "image_path": image_path,
            "prompt": prompt,
            "mask_path": str(mask_path),
            "cleaned_mask_path": str(mask_path),
            "manual_mask_override": dict(value),
        }
        if overlay_path is not None:
            result["overlay_path"] = str(overlay_path)
            result["cleaned_overlay_path"] = str(overlay_path)
        return {
            "input": {
                "image_path": image_path,
                "prompt": prompt,
            },
            "result": result,
        }

    def _prepare_manual_mask(
        self,
        state: SculptWorkflowState,
    ) -> StateUpdate:
        """Restore geometry and obtain a whole-model clipping mask once."""
        request = state["manual_mask_request"]
        if not isinstance(request, dict):
            raise WorkflowExecutionError(
                "Manual mask preparation started without a request"
            )
        request_id = request.get("request_id")
        call_site = request.get("call_site")
        image_value = request.get("image_path")
        if not all(
            isinstance(value, str)
            for value in (request_id, call_site, image_value)
        ):
            raise WorkflowExecutionError("Manual mask request is incomplete")
        emitter = WorkflowEventEmitter(state, node="prepare_manual_mask")
        emitter.node(
            status=WorkflowEventStatus.RUNNING,
            phase="started",
            message="Preparing whole-model constraints for manual masking",
        )
        checkpoint = state["checkpoint_path"]
        if not isinstance(checkpoint, str):
            raise WorkflowExecutionError(
                "Manual mask recovery requires a Blender checkpoint"
            )
        self._load_blender_checkpoint(
            state,
            checkpoint,
            label="load_blender_state(manual_mask)",
        )
        image_path = self._run_path(state, cast(str, image_value))
        prompt = self.config.workflow.quadloc.model_segmentation_prompt
        output_dir = (
            self._attempt_dir(state)
            / "manual-mask"
            / cast(str, call_site)
            / "whole-model"
        ).resolve()
        try:
            model_segmentation = _invoke_sam3_tool(
                self.dependencies.tools.segment_with_sam3,
                {
                    "image_path": str(image_path),
                    "prompt": prompt,
                    "confidence_threshold": (
                        self.config.workflow.sam3_confidence_threshold
                    ),
                    "overlay_opacity": (
                        self.config.workflow.sam3_overlay_opacity
                    ),
                    "output_dir": str(output_dir),
                },
                label="segment_whole_model_with_sam3",
                call_site="manual_mask_whole_model",
                image_path=str(image_path),
                prompt=prompt,
            )
        except _Sam3NoSegmentation as error:
            return self._skip_for_manual_mask(
                state,
                emitter=emitter,
                status="skipped_no_whole_model_mask",
                reason=(
                    "SAM3 could not segment the complete model required for "
                    f"manual-mask clipping: {error}"
                ),
                intervention={
                    **request,
                    "status": "skipped",
                    "stage": "paint",
                },
            )

        workflow_dir = self._workflow_dir(state)
        model_payload = _result_object(
            model_segmentation,
            "segment_whole_model_with_sam3",
        )
        model_mask_value = model_payload.get("cleaned_mask_path")
        if not isinstance(model_mask_value, str):
            raise WorkflowExecutionError(
                "Whole-model segmentation is missing cleaned_mask_path"
            )
        source_artifact = create_artifact(
            run_id=state["run_id"],
            workflow_dir=workflow_dir,
            path=image_path,
            kind=WorkflowArtifactKind.SCREENSHOT,
            label="Manual mask source screenshot",
            metadata={
                "stage": "manual_mask",
                "call_site": cast(str, call_site),
            },
        )
        model_artifacts = _segment_artifacts(
            state,
            workflow_dir,
            model_segmentation,
            label_prefix="Whole-model",
            metadata={
                "stage": "manual_mask_constraint",
                "call_site": cast(str, call_site),
            },
        )
        mask_artifact = next(
            (
                item
                for item in model_artifacts
                if item.kind is WorkflowArtifactKind.MASK
                and "Noise-cleaned" in item.label
            ),
            None,
        )
        if mask_artifact is None:
            raise WorkflowExecutionError(
                "Cannot expose the whole-model mask artifact"
            )
        artifacts = [source_artifact, *model_artifacts]
        intervention: dict[str, JsonValue] = {
            **request,
            "intervention_id": cast(str, request_id),
            "status": "awaiting_paint",
            "stage": "paint",
            "source_artifact": source_artifact.model_dump(mode="json"),
            "model_mask_artifact": mask_artifact.model_dump(mode="json"),
            "model_segmentation": _relativize_run_paths(
                model_segmentation,
                workflow_dir,
            ),
        }
        emitter.artifacts(
            artifact_ids=[item.artifact_id for item in artifacts],
            message="Manual-mask source and whole-model constraint are ready",
        )
        emitter.intervention(
            intervention_id=cast(str, request_id),
            intervention_type="manual_mask",
            stage="paint",
            decision="pending",
            message="Waiting for the user to paint the missing target mask",
        )
        emitter.node(
            status=WorkflowEventStatus.WAITING,
            phase="completed",
            message="Manual mask paint input is ready",
        )
        return {
            "manual_mask_intervention": intervention,
            "manual_mask_result": None,
            "execution_error": None,
            "artifacts": merge_artifacts(state["artifacts"], artifacts),
            **emitter.state_update(),
        }

    def _manual_mask_input(
        self,
        state: SculptWorkflowState,
        config: RunnableConfig,
    ) -> StateUpdate:
        """Collect vector paint strokes and rasterize their model intersection."""
        intervention = state["manual_mask_intervention"]
        request = state["manual_mask_request"]
        if not isinstance(intervention, dict) or not isinstance(request, dict):
            raise WorkflowExecutionError(
                "Manual mask input started without an intervention"
            )
        intervention_id = intervention.get("intervention_id")
        if not isinstance(intervention_id, str):
            raise WorkflowExecutionError(
                "Manual mask intervention is missing intervention_id"
            )
        configurable = config.get("configurable", {})
        if configurable.get("auto_skip_manual_mask") is True:
            response: object = {"decision": "skip"}
        else:
            response = interrupt(
                {
                    "schema_version": "2.0",
                    "type": "sculpt.manual_mask.required",
                    "stage": "paint",
                    "intervention_id": intervention_id,
                    "run_id": state["run_id"],
                    "subtask_index": state["current_subtask_index"],
                    "attempt": state["current_attempt"],
                    "call_site": request["call_site"],
                    "part_description": request["part_description"],
                    "source_artifact": intervention["source_artifact"],
                    "image_width": request["image_width"],
                    "image_height": request["image_height"],
                    "revision": request["revision"],
                    "brush": {
                        "minimum_size": 1,
                        "maximum_size": _MANUAL_MASK_MAX_BRUSH_SIZE,
                        "default_size": min(
                            _MANUAL_MASK_MAX_BRUSH_SIZE,
                            max(
                                8,
                                round(
                                    min(
                                        int(request["image_width"]),
                                        int(request["image_height"]),
                                    )
                                    * 0.04
                                ),
                            ),
                        ),
                    },
                    "allowed_decisions": ["finish", "skip"],
                }
            )
        try:
            submission = ManualMaskPaintResponse.model_validate(response)
        except ValueError as error:
            raise WorkflowExecutionError(
                f"Invalid manual mask paint response: {error}"
            ) from error
        emitter = WorkflowEventEmitter(state, node="manual_mask_input")
        if submission.decision == "skip":
            return self._skip_for_manual_mask(
                state,
                emitter=emitter,
                status="skipped_manual_mask",
                reason="User skipped the subtask instead of painting a mask",
                intervention={
                    **intervention,
                    "status": "skipped",
                    "stage": "paint",
                },
            )

        model_segmentation = intervention.get("model_segmentation")
        if not isinstance(model_segmentation, dict):
            raise WorkflowExecutionError(
                "Manual mask intervention is missing model segmentation"
            )
        model_payload = _result_object(
            model_segmentation,
            "manual_mask_whole_model",
        )
        model_mask_value = model_payload.get("cleaned_mask_path")
        image_value = request.get("image_path")
        call_site = request.get("call_site")
        revision = request.get("revision")
        if (
            not isinstance(model_mask_value, str)
            or not isinstance(image_value, str)
            or not isinstance(call_site, str)
            or not isinstance(revision, int)
        ):
            raise WorkflowExecutionError(
                "Manual mask rasterization inputs are incomplete"
            )
        try:
            raster = rasterize_manual_mask(
                image_path=self._run_path(state, image_value),
                model_mask_path=self._run_path(state, model_mask_value),
                submission=submission,
                output_dir=(
                    self._attempt_dir(state)
                    / "manual-mask"
                    / call_site
                    / f"revision-{revision:02d}"
                ),
                overlay_opacity=self.config.workflow.sam3_overlay_opacity,
            )
        except ManualMaskError as error:
            raise WorkflowExecutionError(
                f"Cannot rasterize manual mask: {error}"
            ) from error
        workflow_dir = self._workflow_dir(state)
        raster_payload = cast(
            dict[str, JsonValue],
            _relativize_run_paths(
                raster.model_dump(mode="json"),
                workflow_dir,
            ),
        )
        artifacts = _manual_mask_artifacts(
            state,
            workflow_dir,
            raster,
            call_site=call_site,
            revision=revision,
        )
        by_kind = {item.label: item for item in artifacts}
        result: dict[str, JsonValue] = {
            **raster_payload,
            "revision": revision,
            "call_site": call_site,
            "painted_mask_artifact": by_kind[
                "User-painted mask"
            ].model_dump(mode="json"),
            "cleaned_mask_artifact": by_kind[
                "Model-clipped manual mask"
            ].model_dump(mode="json"),
            "cleaned_overlay_artifact": by_kind[
                "Model-clipped manual mask overlay"
            ].model_dump(mode="json"),
        }
        next_intervention = {
            **intervention,
            "status": "awaiting_review",
            "stage": "review",
            "revision": revision,
        }
        emitter.artifacts(
            artifact_ids=[item.artifact_id for item in artifacts],
            message="Manual mask intersection preview is ready",
        )
        emitter.intervention(
            intervention_id=intervention_id,
            intervention_type="manual_mask",
            stage="review",
            decision="painted",
            message="User-painted mask was clipped to the whole model",
        )
        emitter.node(
            status=WorkflowEventStatus.WAITING,
            phase="completed",
            message="Waiting for manual mask confirmation",
        )
        return {
            "manual_mask_intervention": next_intervention,
            "manual_mask_result": result,
            "artifacts": merge_artifacts(state["artifacts"], artifacts),
            **emitter.state_update(),
        }

    def _manual_mask_review(
        self,
        state: SculptWorkflowState,
        config: RunnableConfig,
    ) -> StateUpdate:
        """Confirm, redraw, or skip one model-clipped manual mask."""
        intervention = state["manual_mask_intervention"]
        request = state["manual_mask_request"]
        result = state["manual_mask_result"]
        if not all(isinstance(item, dict) for item in (intervention, request, result)):
            raise WorkflowExecutionError(
                "Manual mask review started without a rendered result"
            )
        intervention = cast(dict[str, JsonValue], intervention)
        request = cast(dict[str, JsonValue], request)
        result = cast(dict[str, JsonValue], result)
        intervention_id = intervention.get("intervention_id")
        foreground = result.get("intersection_foreground_pixels")
        if not isinstance(intervention_id, str) or not isinstance(foreground, int):
            raise WorkflowExecutionError("Manual mask review state is incomplete")
        configurable = config.get("configurable", {})
        if configurable.get("auto_skip_manual_mask") is True:
            response: object = {"decision": "skip"}
        else:
            response = interrupt(
                {
                    "schema_version": "2.0",
                    "type": "sculpt.manual_mask.required",
                    "stage": "review",
                    "intervention_id": intervention_id,
                    "run_id": state["run_id"],
                    "subtask_index": state["current_subtask_index"],
                    "attempt": state["current_attempt"],
                    "call_site": request["call_site"],
                    "part_description": request["part_description"],
                    "source_artifact": intervention["source_artifact"],
                    "mask_artifact": result["cleaned_mask_artifact"],
                    "overlay_artifact": result[
                        "cleaned_overlay_artifact"
                    ],
                    "intersection_foreground_pixels": foreground,
                    "confirm_allowed": foreground > 0,
                    "revision": request["revision"],
                    "allowed_decisions": ["confirm", "redraw", "skip"],
                }
            )
        try:
            review = ManualMaskReviewResponse.model_validate(response)
        except ValueError as error:
            raise WorkflowExecutionError(
                f"Invalid manual mask review response: {error}"
            ) from error
        emitter = WorkflowEventEmitter(state, node="manual_mask_review")
        if review.decision == "skip":
            return self._skip_for_manual_mask(
                state,
                emitter=emitter,
                status="skipped_manual_mask",
                reason="User rejected the manual mask and skipped the subtask",
                intervention={
                    **intervention,
                    "status": "skipped",
                    "stage": "review",
                },
            )
        if review.decision == "redraw":
            revision_value = request.get("revision")
            if not isinstance(revision_value, int):
                raise WorkflowExecutionError(
                    "Manual mask request has an invalid revision"
                )
            revised_request = {
                **request,
                "revision": revision_value + 1,
                "status": "awaiting_paint",
            }
            revised_intervention = {
                **intervention,
                "revision": revision_value + 1,
                "status": "awaiting_paint",
                "stage": "paint",
            }
            emitter.intervention(
                intervention_id=intervention_id,
                intervention_type="manual_mask",
                stage="paint",
                decision="redraw",
                message="User requested a new manual mask drawing",
            )
            emitter.node(
                status=WorkflowEventStatus.RETRYING,
                phase="completed",
                message="Returning to manual mask painting",
            )
            return {
                "manual_mask_request": revised_request,
                "manual_mask_intervention": revised_intervention,
                "manual_mask_result": None,
                **emitter.state_update(),
            }
        if foreground <= 0:
            raise WorkflowExecutionError(
                "An empty manual mask intersection cannot be confirmed"
            )
        call_site = request.get("call_site")
        if not isinstance(call_site, str):
            raise WorkflowExecutionError("Manual mask request has no call_site")
        overrides = dict(state["manual_mask_overrides"])
        overrides[call_site] = {
            "source": "user_painted_model_intersection",
            "call_site": call_site,
            "part_description": request["part_description"],
            "selected_view": state["selected_view"],
            "source_image_path": request["image_path"],
            "image_width": result["image_width"],
            "image_height": result["image_height"],
            "cleaned_mask_path": result["cleaned_mask_path"],
            "cleaned_overlay_path": result["cleaned_overlay_path"],
            "painted_mask_path": result["painted_mask_path"],
            "revision": result["revision"],
            "intersection_foreground_pixels": foreground,
        }
        confirmed = {
            **intervention,
            "status": "confirmed",
            "stage": "review",
        }
        emitter.intervention(
            intervention_id=intervention_id,
            intervention_type="manual_mask",
            stage="review",
            decision="confirmed",
            message="Manual mask was confirmed for Sculpt preparation",
        )
        emitter.node(
            status=WorkflowEventStatus.SUCCESS,
            phase="completed",
            message="Continuing with the confirmed manual mask",
        )
        return {
            "manual_mask_request": None,
            "manual_mask_intervention": confirmed,
            "manual_mask_overrides": overrides,
            "execution_error": None,
            **emitter.state_update(),
        }

    def _skip_for_manual_mask(
        self,
        state: SculptWorkflowState,
        *,
        emitter: WorkflowEventEmitter,
        status: str,
        reason: str,
        intervention: dict[str, JsonValue],
    ) -> StateUpdate:
        """Record one explicit no-mask skip after geometry is restored."""
        method = OperationMethod(
            str(self._current_subtask(state)["operation_method"])
        )
        results = list(state["subtask_results"])
        results.append(
            {
                "subtask_index": state["current_subtask_index"],
                "status": status,
                "operation_method": method.value,
                "attempts": state["current_attempt"],
                "last_selected_view": state["selected_view"],
                "reason": reason,
            }
        )
        intervention_id = intervention.get("intervention_id") or intervention.get(
            "request_id"
        )
        if isinstance(intervention_id, str):
            emitter.intervention(
                intervention_id=intervention_id,
                intervention_type="manual_mask",
                stage=cast(Literal["paint", "review"], intervention["stage"]),
                decision="skipped",
                message=reason,
            )
        emitter.subtask(
            status=WorkflowEventStatus.SKIPPED,
            action="executed",
            operation_method=method.value,
            message=f"Skipped subtask {state['current_subtask_index'] + 1}: {reason}",
        )
        emitter.node(
            status=WorkflowEventStatus.SKIPPED,
            phase="completed",
            message=reason,
        )
        return {
            "manual_mask_request": None,
            "manual_mask_intervention": intervention,
            "manual_mask_result": None,
            "execution_error": reason,
            "subtask_results": results,
            **emitter.state_update(),
        }

    def _prefilter_views_with_sam3(
        self,
        state: SculptWorkflowState,
        *,
        screenshots: Mapping[str, JsonValue],
        screenshot_paths: Mapping[str, str],
        emitter: WorkflowEventEmitter,
    ) -> tuple[
        dict[str, JsonValue],
        dict[str, str],
        dict[str, str],
        dict[str, Mapping[str, JsonValue]],
        list[WorkflowArtifact],
    ]:
        """Probe all views and keep only non-empty SAM3 segmentations."""
        workflow_dir = self._workflow_dir(state)
        prompt = str(self._current_intent(state)["part_to_be_changed"])
        evaluations: dict[str, JsonValue] = {}
        valid_paths: dict[str, str] = {}
        overlay_paths: dict[str, str] = {}
        summaries: dict[str, Mapping[str, JsonValue]] = {}
        artifacts: list[WorkflowArtifact] = []
        total = len(screenshot_paths)

        for index, (view, screenshot_path) in enumerate(
            screenshot_paths.items(),
            start=1,
        ):
            output_dir = (
                self._attempt_dir(state)
                / "view-selector"
                / "sam3-probes"
                / view.lower()
            )
            try:
                result = self.dependencies.sam3_segmenter.segment(
                    image_path=screenshot_path,
                    prompt=prompt,
                    confidence_threshold=(
                        self.config.workflow.sam3_confidence_threshold
                    ),
                    overlay_opacity=(
                        self.config.workflow.sam3_overlay_opacity
                    ),
                    output_dir=output_dir,
                )
                assessment = assess_view_segmentation(
                    mask_path=result.mask_path,
                    metadata=result.metadata,
                )
            except Exception as error:
                raise WorkflowExecutionError(
                    f"SAM3 view prefilter failed for {view}: {error}"
                ) from error

            payload = result.as_payload()
            assessment_payload = assessment.as_payload()
            relative_payload = _relativize_run_paths(
                cast(JsonValue, payload),
                workflow_dir,
            )
            screenshot = screenshots.get(view)
            screenshot_state_path = (
                screenshot.get("path")
                if isinstance(screenshot, dict)
                else None
            )
            evaluations[view] = {
                "view": view,
                "segmentation_prompt": prompt,
                "screenshot_path": screenshot_state_path,
                "assessment": assessment_payload,
                "segmentation": relative_payload,
            }
            summaries[view] = assessment_payload
            artifacts.extend(
                _segment_artifacts(
                    state,
                    workflow_dir,
                    payload,
                    label_prefix=f"{view} view probe",
                    metadata={
                        "stage": "view_prefilter",
                        "view": view,
                        "valid": assessment.valid,
                    },
                )
            )
            if assessment.valid:
                valid_paths[view] = screenshot_path
                overlay_paths[view] = result.overlay_path
                message = (
                    f"SAM3 accepted {view} view "
                    f"({assessment.instance_count} instances)"
                )
            else:
                message = (
                    f"SAM3 rejected {view} view: "
                    f"{assessment.invalid_reason}"
                )
            emitter.progress(
                label="SAM3 view prefilter",
                current=index,
                total=total,
                unit="view",
                message=message,
            )

        return (
            evaluations,
            valid_paths,
            overlay_paths,
            summaries,
            artifacts,
        )

    def _prepare_execution_dispatch(
        self,
        state: SculptWorkflowState,
    ) -> StateUpdate:
        """Route execution preparation without repeating completed stages."""
        del state
        return {}

    def _prepare_drag_context(
        self,
        state: SculptWorkflowState,
    ) -> StateUpdate:
        """Capture and persist the immutable screen-space Drag context."""
        selected_view = state["selected_view"]
        if selected_view is None:
            raise WorkflowExecutionError(
                "Drag context preparation requires a selected view"
            )
        subtask = self._current_subtask(state)
        if OperationMethod(
            str(subtask["operation_method"])
        ) is not OperationMethod.DRAG:
            raise WorkflowExecutionError(
                "Drag context preparation received a non-Drag subtask"
            )
        intent = self._current_intent(state)
        emitter = WorkflowEventEmitter(state, node="prepare_drag_context")
        emitter.node(
            status=WorkflowEventStatus.RUNNING,
            phase="started",
            message=f"Capturing the {selected_view} Drag context",
        )
        segment_input = self._change_and_capture(
            state,
            selected_view,
            self._attempt_dir(state) / "segment-input.png",
            label=(
                f"Subtask {state['current_subtask_index'] + 1} "
                f"attempt {state['current_attempt']} full before"
            ),
        )
        pose_drag = _is_pose_drag(
            subtask=subtask,
            intent=intent,
            pose_brush_name=self.config.workflow.drag.pose_brush_name,
        )
        context: dict[str, JsonValue] = {
            "schema_version": "execution-preparation/v1",
            "subtask_index": state["current_subtask_index"],
            "attempt": state["current_attempt"],
            "selected_view": selected_view,
            "operation_method": OperationMethod.DRAG.value,
            "stage": _DRAG_PREPARATION_CHANGED_PART,
            "completed_stages": ["capture"],
            "segment_input": segment_input,
            "pose_drag": pose_drag,
            "changed_part_description": str(intent["part_to_be_changed"]),
            "operation_location": str(intent["operation_location"]),
        }
        artifacts = _screenshot_artifacts({selected_view: segment_input})
        emitter.artifacts(
            artifact_ids=[item.artifact_id for item in artifacts],
            message="The reusable Drag screenshot is available",
        )
        emitter.node(
            status=WorkflowEventStatus.SUCCESS,
            phase="completed",
            message="Drag screenshot context was persisted",
        )
        return {
            "execution_preparation": context,
            "manual_mask_request": None,
            "execution_error": None,
            "restore_action": None,
            "artifacts": merge_artifacts(state["artifacts"], artifacts),
            **emitter.state_update(),
        }

    def _prepare_drag_changed_part(
        self,
        state: SculptWorkflowState,
    ) -> StateUpdate:
        """Resolve one changed-part mask and optional Pose Face Sets."""
        context = self._require_drag_preparation_context(
            state,
            expected_stage=_DRAG_PREPARATION_CHANGED_PART,
        )
        selected_view = cast(str, context["selected_view"])
        segment_input = _json_object_field(
            context,
            "segment_input",
            label="Drag preparation context",
        )
        image_path = str(
            self._run_path(state, str(segment_input["path"]))
        )
        part_description = str(context["changed_part_description"])
        pose_drag = bool(context["pose_drag"])
        attempt_dir = self._attempt_dir(state)
        workflow_dir = self._workflow_dir(state)
        emitter = WorkflowEventEmitter(
            state,
            node="prepare_drag_changed_part",
        )
        emitter.node(
            status=WorkflowEventStatus.RUNNING,
            phase="started",
            message=f"Resolving the changed part: {part_description}",
        )
        artifacts: list[WorkflowArtifact] = []
        part_segmentation_result: dict[str, JsonValue] | None = None
        segmentation_result: dict[str, JsonValue] | None = None
        try:
            # Manual drawing restores the pre-execution snapshot, so Face Sets
            # must return to the original operation view.
            view_response = _invoke_tool(
                self.dependencies.tools.change_view,
                {"view": selected_view, "frame": self.config.workflow.frame},
                label=f"change_view({selected_view})",
            )
            _rpc_result(view_response, f"change_view({selected_view})")
            if pose_drag:
                part_tool = self._llm_bound_tool(
                    state,
                    fallback=(
                        self.dependencies.tools.part_segmentation_with_sam3
                    ),
                    factory=self.dependencies.part_segmentation_tool_factory,
                    call_site="part_segmentation",
                )
                tool_input: dict[str, JsonValue] = {
                    "image_path": image_path,
                    "part_description": part_description,
                    "output_dir": str(
                        (attempt_dir / "part-segmentation").resolve()
                    ),
                }
                override_path = self._manual_mask_override_path(
                    state,
                    call_site=_MANUAL_MASK_CHANGED_PART,
                    selected_view=selected_view,
                )
                if override_path is not None:
                    tool_input["parent_mask_path"] = str(override_path)
                part_segmentation_result = _invoke_part_segmentation_tool(
                    part_tool,
                    tool_input,
                    image_path=image_path,
                    prompt=part_description,
                )
                part_payload = _result_object(
                    part_segmentation_result,
                    "part_segmentation_with_sam3",
                )
                face_set_count = _positive_result_integer(
                    part_payload,
                    "subpart_count",
                )
                kinematic_overlay_path = _part_segmentation_overlay_path(
                    part_payload
                )
                segmentation_result = (
                    _changed_part_segmentation_from_part_result(
                        part_segmentation_result,
                        image_path=image_path,
                        part_description=part_description,
                    )
                )
                context.update(
                    {
                        "face_set_count": face_set_count,
                        "kinematic_overlay_path": kinematic_overlay_path,
                        "changed_part_source": (
                            "part_segmentation_parent"
                        ),
                    }
                )
                artifacts.extend(
                    _part_segmentation_artifacts(
                        state,
                        workflow_dir,
                        part_segmentation_result,
                    )
                )
            else:
                segmentation_result = self._manual_segmentation_override(
                    state,
                    call_site=_MANUAL_MASK_CHANGED_PART,
                    image_path=image_path,
                    prompt=part_description,
                ) or _invoke_sam3_tool(
                    self.dependencies.tools.segment_with_sam3,
                    {
                        "image_path": image_path,
                        "prompt": part_description,
                        "confidence_threshold": (
                            self.config.workflow.sam3_confidence_threshold
                        ),
                        "overlay_opacity": (
                            self.config.workflow.sam3_overlay_opacity
                        ),
                        "output_dir": str(
                            (attempt_dir / "segment").resolve()
                        ),
                    },
                    label="segment_with_sam3",
                    call_site=_MANUAL_MASK_CHANGED_PART,
                    image_path=image_path,
                    prompt=part_description,
                )
                (
                    segmentation_result,
                    component_artifacts,
                ) = self._select_segment_component_if_needed(
                    state,
                    image_path=Path(image_path),
                    segmentation_result=segmentation_result,
                    part_description=part_description,
                    output_dir=(
                        attempt_dir / "segment-component-selection"
                    ),
                    workflow_dir=workflow_dir,
                )
                artifacts.extend(component_artifacts)
                context["changed_part_source"] = "segment_with_sam3"

            segment_payload = _result_object(
                segmentation_result,
                "changed_part segmentation",
            )
            cleaned_mask = segment_payload.get("cleaned_mask_path")
            if not isinstance(cleaned_mask, str):
                raise WorkflowExecutionError(
                    "Changed-part segmentation is missing cleaned_mask_path"
                )
            context["changed_part_mask_path"] = cleaned_mask
            context["stage"] = _DRAG_PREPARATION_QUADLOC
            context["completed_stages"] = [
                "capture",
                "changed_part",
                *(["face_sets"] if pose_drag else []),
            ]
            artifacts.extend(
                _segment_artifacts(
                    state,
                    workflow_dir,
                    segmentation_result,
                    label_prefix="Changed-part",
                    metadata={
                        "stage": "drag_changed_part",
                        "view": selected_view,
                    },
                )
            )
        except _Sam3NoSegmentation as error:
            request = self._manual_mask_request_from_error(
                state,
                error=error,
                segment_input=segment_input,
            )
            emitter.node(
                status=WorkflowEventStatus.WAITING,
                phase="completed",
                message="Waiting for a manually painted changed-part mask",
            )
            return {
                "execution_preparation": context,
                "manual_mask_request": request,
                "manual_mask_result": None,
                "execution_error": str(error),
                "restore_action": None,
                "retry_scope": None,
                **emitter.state_update(),
            }
        except (WorkflowExecutionError, WorkflowLlmError, ValueError) as error:
            return self._drag_preparation_failure_update(
                state,
                emitter=emitter,
                context=context,
                message=str(error),
                code="drag_changed_part_failed",
                retry_scope="same_view",
            )

        emitter.artifacts(
            artifact_ids=[item.artifact_id for item in artifacts],
            message="Changed-part mask and Face Set evidence are available",
        )
        emitter.node(
            status=WorkflowEventStatus.SUCCESS,
            phase="completed",
            message=(
                "Changed-part mask and Pose Face Sets were persisted"
                if pose_drag
                else "Changed-part mask was persisted"
            ),
        )
        return {
            "execution_preparation": cast(
                dict[str, JsonValue],
                _relativize_run_paths(context, workflow_dir),
            ),
            "segmentation_result": _relative_optional_object(
                segmentation_result,
                workflow_dir,
            ),
            "part_segmentation_result": _relative_optional_object(
                part_segmentation_result,
                workflow_dir,
            ),
            "manual_mask_request": None,
            "manual_mask_result": None,
            "execution_error": None,
            "restore_action": None,
            "retry_scope": None,
            "artifacts": merge_artifacts(state["artifacts"], artifacts),
            **emitter.state_update(),
        }

    def _prepare_drag_quadloc(
        self,
        state: SculptWorkflowState,
    ) -> StateUpdate:
        """Localize the Drag contact point and persist the QuadLoc result."""
        context = self._require_drag_preparation_context(
            state,
            expected_stage=_DRAG_PREPARATION_QUADLOC,
        )
        segment_input = _json_object_field(
            context,
            "segment_input",
            label="Drag preparation context",
        )
        image_path = str(
            self._run_path(state, str(segment_input["path"]))
        )
        operation_location = str(context["operation_location"])
        emitter = WorkflowEventEmitter(state, node="prepare_drag_quadloc")
        emitter.node(
            status=WorkflowEventStatus.RUNNING,
            phase="started",
            message=f"Localizing the Drag start point at {operation_location}",
        )
        try:
            quadloc_tool = self._llm_bound_tool(
                state,
                fallback=self.dependencies.tools.quadloc,
                factory=self.dependencies.quadloc_tool_factory,
                call_site="quadloc",
            )
            quadloc_result = _invoke_quadloc_tool(
                quadloc_tool,
                {
                    "image_path": image_path,
                    "location_description": operation_location,
                    "output_dir": str(
                        (self._attempt_dir(state) / "quadloc").resolve()
                    ),
                },
                image_path=image_path,
            )
            payload = _result_object(quadloc_result, "quadloc")
            coordinate = payload.get("coordinate")
            if not isinstance(coordinate, dict):
                raise WorkflowExecutionError(
                    "QuadLoc result is missing coordinate"
                )
            _integer_coordinate(coordinate, "x")
            _integer_coordinate(coordinate, "y")
        except _Sam3NoSegmentation as error:
            return self._drag_preparation_failure_update(
                state,
                emitter=emitter,
                context=context,
                message=str(error),
                code="sam3_no_quadloc_model_mask",
                retry_scope=None,
                skip_status="skipped_no_quadloc_model_mask",
                force_skip=True,
            )
        except (WorkflowExecutionError, WorkflowLlmError, ValueError) as error:
            return self._drag_preparation_failure_update(
                state,
                emitter=emitter,
                context=context,
                message=str(error),
                code="drag_quadloc_failed",
                retry_scope="same_view",
            )

        workflow_dir = self._workflow_dir(state)
        context["stage"] = _DRAG_PREPARATION_FINAL
        completed = context.get("completed_stages")
        context["completed_stages"] = [
            *(
                list(completed)
                if isinstance(completed, list)
                else ["capture", "changed_part"]
            ),
            "quadloc",
        ]
        emitter.node(
            status=WorkflowEventStatus.SUCCESS,
            phase="completed",
            message="QuadLoc result was persisted for final Drag planning",
        )
        return {
            "execution_preparation": cast(
                dict[str, JsonValue],
                _relativize_run_paths(context, workflow_dir),
            ),
            "quadloc_result": _relative_optional_object(
                quadloc_result,
                workflow_dir,
            ),
            "execution_error": None,
            "restore_action": None,
            "retry_scope": None,
            **emitter.state_update(),
        }

    def _require_drag_preparation_context(
        self,
        state: SculptWorkflowState,
        *,
        expected_stage: str | None = None,
    ) -> dict[str, JsonValue]:
        """Validate and copy the preparation record for this exact attempt."""
        value = state["execution_preparation"]
        if not isinstance(value, dict):
            raise WorkflowExecutionError(
                "Drag execution preparation context is missing"
            )
        context = _json_object_copy(value)
        expected = {
            "subtask_index": state["current_subtask_index"],
            "attempt": state["current_attempt"],
            "selected_view": state["selected_view"],
            "operation_method": OperationMethod.DRAG.value,
        }
        for key, expected_value in expected.items():
            if context.get(key) != expected_value:
                raise WorkflowExecutionError(
                    f"Drag preparation context has stale {key}"
                )
        if expected_stage is not None and context.get("stage") != expected_stage:
            raise WorkflowExecutionError(
                "Drag preparation context is at stage "
                f"{context.get('stage')!r}, expected {expected_stage!r}"
            )
        _json_object_field(
            context,
            "segment_input",
            label="Drag preparation context",
        )
        return context

    def _manual_mask_request_from_error(
        self,
        state: SculptWorkflowState,
        *,
        error: _Sam3NoSegmentation,
        segment_input: Mapping[str, JsonValue],
    ) -> dict[str, JsonValue]:
        """Persist one deterministic missing-mask request in screenshot space."""
        workflow_dir = self._workflow_dir(state)
        relative_input = _relativize_run_paths(
            {"image_path": error.image_path},
            workflow_dir,
        )
        image_value = (
            relative_input.get("image_path")
            if isinstance(relative_input, dict)
            else None
        )
        if not isinstance(image_value, str):
            raise WorkflowExecutionError(
                "Cannot persist manual mask source image"
            ) from error
        image_width, image_height = _screenshot_dimensions(segment_input)
        return {
            "schema_version": "2.0",
            "request_id": (
                f"{state['run_id']}:subtask-"
                f"{state['current_subtask_index']}:attempt-"
                f"{state['current_attempt']}:manual-mask:{error.call_site}"
            ),
            "call_site": error.call_site,
            "image_path": image_value,
            "part_description": error.prompt,
            "image_width": image_width,
            "image_height": image_height,
            "reason": str(error),
            "revision": 1,
            "status": "model_mask_required",
        }

    def _drag_preparation_failure_update(
        self,
        state: SculptWorkflowState,
        *,
        emitter: WorkflowEventEmitter,
        context: Mapping[str, JsonValue],
        message: str,
        code: str,
        retry_scope: str | None,
        skip_status: str = "skipped_after_max_attempts",
        force_skip: bool = False,
    ) -> StateUpdate:
        """Convert one staged Drag failure into the existing bounded routing."""
        can_retry = (
            not force_skip
            and state["current_attempt"]
            < self.config.workflow.max_subtask_attempts
        )
        restore_action = "retry" if can_retry else "skip"
        emitter.error(
            code=code,
            retryable=can_retry,
            message=message,
        )
        emitter.node(
            status=WorkflowEventStatus.ERROR,
            phase="completed",
            message=message,
        )
        results = list(state["subtask_results"])
        if restore_action == "skip":
            results.append(
                {
                    "subtask_index": state["current_subtask_index"],
                    "status": skip_status,
                    "operation_method": OperationMethod.DRAG.value,
                    "attempts": state["current_attempt"],
                    "last_selected_view": state["selected_view"],
                    "reason": message,
                }
            )
        return {
            "execution_preparation": cast(
                dict[str, JsonValue],
                _relativize_run_paths(
                    dict(context),
                    self._workflow_dir(state),
                ),
            ),
            "manual_mask_request": None,
            "execution_error": message,
            "restore_action": restore_action,
            "retry_scope": retry_scope if can_retry else None,
            "subtask_results": results,
            **emitter.state_update(),
        }

    def _prepare_execution(
        self,
        state: SculptWorkflowState,
    ) -> StateUpdate:
        """Resolve segmentation, settings, strokes, and two clean baselines."""
        selected_view = state["selected_view"]
        if selected_view is None:
            raise WorkflowExecutionError(
                "Execution preparation started without a selected view"
            )
        subtask = self._current_subtask(state)
        method = OperationMethod(str(subtask["operation_method"]))
        emitter = WorkflowEventEmitter(state, node="prepare_execution")
        emitter.node(
            status=WorkflowEventStatus.RUNNING,
            phase="started",
            message=f"Resolving the {selected_view} Sculpt execution plan",
        )
        attempt_dir = self._attempt_dir(state)
        workflow_dir = self._workflow_dir(state)
        intent = self._current_intent(state)
        artifacts: list[WorkflowArtifact] = []
        segmentation_result: dict[str, JsonValue] | None = None
        quadloc_result: dict[str, JsonValue] | None = None
        drag_target_binding: dict[str, JsonValue] | None = None
        locked_drag_plan = state["locked_drag_plan"]
        drag_direction_plan: dict[str, JsonValue] | None = None
        part_segmentation_result: dict[str, JsonValue] | None = None
        draw_pattern_result: dict[str, JsonValue] | None = None
        draw_trajectory_result: dict[str, JsonValue] | None = None
        draw_fitted_trajectory_result: dict[str, JsonValue] | None = None
        resolved_plan_payload: dict[str, JsonValue] | None = None
        stroke_plan_result: dict[str, JsonValue] | None = None
        viewport_focus: dict[str, JsonValue] | None = None
        full_before: dict[str, JsonValue] | None = None
        settings_result: JsonValue = None
        baseline_a: dict[str, JsonValue] | None = None
        baseline_b: dict[str, JsonValue] | None = None
        minimum_baseline: dict[str, JsonValue] | None = None
        segment_input: dict[str, JsonValue] | None = None
        drag_context: dict[str, JsonValue] | None = None
        baseline_cleaned_mask: str | None = None
        baseline_stroke_plan: Mapping[str, JsonValue] | None = None
        baseline_stability_exhausted = False
        safe_anchor_requires_reselect = False
        preparation_retry_scope: str | None = None
        rejected_drag_anchor_views = list(
            state["rejected_drag_anchor_views"]
        )
        preparation_error: str | None = None
        manual_mask_request: dict[str, JsonValue] | None = None
        deterministic_skip_status: str | None = None
        retry_feedback = state["retry_feedback"]
        llm_calls = list(state["llm_calls"])

        if (
            method is OperationMethod.DRAG
            and state["retry_scope"]
            in {_DRAG_RETRY_DOSE, _DRAG_RETRY_GESTURE}
            and isinstance(locked_drag_plan, dict)
        ):
            return self._prepare_locked_drag_retry(
                state,
                emitter=emitter,
                selected_view=selected_view,
                subtask=subtask,
                intent=intent,
                attempt_dir=attempt_dir,
                workflow_dir=workflow_dir,
                locked_plan=locked_drag_plan,
            )

        try:
            if method is OperationMethod.DRAG:
                drag_context = self._require_drag_preparation_context(
                    state,
                    expected_stage=_DRAG_PREPARATION_FINAL,
                )
                segment_input = _json_object_field(
                    drag_context,
                    "segment_input",
                    label="Drag preparation context",
                )
                if not isinstance(state["segmentation_result"], dict):
                    raise WorkflowExecutionError(
                        "Drag preparation did not persist changed_part"
                    )
                if not isinstance(state["quadloc_result"], dict):
                    raise WorkflowExecutionError(
                        "Drag preparation did not persist QuadLoc"
                    )
                segmentation_result = _json_object_copy(
                    state["segmentation_result"]
                )
                quadloc_result = _json_object_copy(state["quadloc_result"])
                if isinstance(state["part_segmentation_result"], dict):
                    part_segmentation_result = _json_object_copy(
                        state["part_segmentation_result"]
                    )
            else:
                segment_input = self._change_and_capture(
                    state,
                    selected_view,
                    attempt_dir / "segment-input.png",
                    label=(
                        f"Subtask {state['current_subtask_index'] + 1} "
                        f"attempt {state['current_attempt']} full before"
                    ),
                )
                artifacts.extend(
                    _screenshot_artifacts({selected_view: segment_input})
                )
            if method in {OperationMethod.SMEAR, OperationMethod.DRAW}:
                full_before = segment_input
            if method is OperationMethod.SMEAR:
                image_path = str(
                    self._run_path(state, str(segment_input["path"]))
                )
                prompt = _surface_segmentation_prompt(state, intent)
                segmentation_result = _reused_surface_segmentation(
                    state,
                    image_path=image_path,
                    prompt=prompt,
                )
                if segmentation_result is None:
                    segmentation_result = self._manual_segmentation_override(
                        state,
                        call_site=_MANUAL_MASK_SMEAR,
                        image_path=image_path,
                        prompt=prompt,
                    ) or _invoke_sam3_tool(
                        self.dependencies.tools.segment_with_sam3,
                        {
                            "image_path": image_path,
                            "prompt": prompt,
                            "confidence_threshold": (
                                self.config.workflow
                                .sam3_confidence_threshold
                            ),
                            "overlay_opacity": (
                                self.config.workflow.sam3_overlay_opacity
                            ),
                            "output_dir": str(
                                (attempt_dir / "segment").resolve()
                            ),
                        },
                        label="segment_with_sam3",
                        call_site=_MANUAL_MASK_SMEAR,
                        image_path=image_path,
                        prompt=prompt,
                    )
                    (
                        segmentation_result,
                        component_artifacts,
                    ) = self._select_segment_component_if_needed(
                        state,
                        image_path=self._run_path(
                            state,
                            str(segment_input["path"]),
                        ),
                        segmentation_result=segmentation_result,
                        part_description=prompt,
                        output_dir=(
                            attempt_dir / "segment-component-selection"
                        ),
                        workflow_dir=workflow_dir,
                    )
                    artifacts.extend(component_artifacts)
                artifacts.extend(
                    _segment_artifacts(
                        state,
                        workflow_dir,
                        segmentation_result,
                        label_prefix=(
                            "Reused"
                            if state["retry_scope"]
                            in {
                                "same_view",
                                _SURFACE_RETRY_REUSE_SEGMENTATION,
                            }
                            else None
                        ),
                    )
                )
                segment_payload = _result_object(
                    segmentation_result,
                    "segment_with_sam3",
                )
                if self.config.workflow.roi_focus.enabled:
                    (
                        segment_input,
                        segmentation_result,
                        viewport_focus,
                        focus_artifacts,
                    ) = self._focus_segmented_execution(
                        state,
                        segment_input=segment_input,
                        segmentation_result=segmentation_result,
                        attempt_dir=attempt_dir,
                        workflow_dir=workflow_dir,
                        selected_view=selected_view,
                    )
                    artifacts.extend(focus_artifacts)
                    segment_payload = _result_object(
                        segmentation_result,
                        "segment_with_sam3",
                    )
                cleaned_mask = segment_payload.get("cleaned_mask_path")
                if not isinstance(cleaned_mask, str):
                    raise WorkflowExecutionError(
                        "segment_with_sam3 is missing cleaned_mask_path"
                    )
                resolved_plan = resolve_sculpt_plan(
                    intent=SculptIntent.model_validate(intent),
                    cleaned_mask_path=cleaned_mask,
                    screenshot_metadata=cast(
                        Mapping[str, object],
                        segment_input["metadata"],
                    ),
                    config=self.config.workflow.parameter_resolution,
                    operation_defaults=(
                        self.config.workflow.defaults_for_operation(
                            method.value
                        )
                    ),
                    retry_directive=state["retry_directive"],
                )
                resolved_plan_payload = resolved_plan.model_dump(mode="json")
                settings = resolved_plan.settings.model_dump(mode="json")
                stroke_plan_result = _invoke_tool(
                    self.dependencies.tools.plan_sculpt_strokes,
                    {
                        "cleaned_mask_path": cleaned_mask,
                        "sculpt_brush": settings["sculpt_brush"],
                        "brush_size": settings["brush_size"],
                        "brush_strength": settings["brush_strength"],
                        "brush_direction": settings["brush_direction"],
                        "screenshot_metadata": segment_input["metadata"],
                        "output_dir": str(
                            (attempt_dir / "stroke-plan").resolve()
                        ),
                    },
                    label="plan_sculpt_strokes",
                )
                artifacts.extend(
                    _stroke_plan_artifacts(
                        state,
                        workflow_dir,
                        stroke_plan_result,
                    )
                )
                plan_payload = _result_object(
                    stroke_plan_result,
                    "plan_sculpt_strokes",
                )
                stroke_plan = plan_payload.get("stroke_plan")
                overlay_path = segment_payload.get("overlay_path")
                if not isinstance(stroke_plan, dict) or not isinstance(
                    overlay_path,
                    str,
                ):
                    raise WorkflowExecutionError(
                        "Prepared Smear result cannot render trajectories"
                    )
                trajectory_visualization = (
                    render_sculpt_stroke_polyline_visualizations(
                        stroke_plan=cast(Mapping[str, object], stroke_plan),
                        mask_path=cleaned_mask,
                        overlay_path=overlay_path,
                        output_dir=(
                            attempt_dir / "stroke-plan" / "visualizations"
                        ),
                    )
                )
                artifacts.extend(
                    _stroke_trajectory_artifacts(
                        state,
                        workflow_dir,
                        trajectory_visualization,
                    )
                )
                settings_result = _invoke_tool(
                    self.dependencies.tools.set_sculpt_settings,
                    settings,
                    label="set_sculpt_settings",
                )

            elif method is OperationMethod.DRAG:
                image_path = str(
                    self._run_path(state, str(segment_input["path"]))
                )
                pose_drag = _is_pose_drag(
                    subtask=subtask,
                    intent=intent,
                    pose_brush_name=(
                        self.config.workflow.drag.pose_brush_name
                    ),
                )
                defaults = self.config.workflow.defaults_for_operation(
                    method.value
                )
                part_description = str(intent["part_to_be_changed"])
                face_set_count: int | None = None
                kinematic_overlay_path: str | None = None
                prepared_drag_checkpoint_path: str | None = None
                if pose_drag:
                    if part_segmentation_result is None:
                        raise WorkflowExecutionError(
                            "Pose Drag preparation is missing Face Sets"
                        )
                    face_set_count = _positive_result_integer(
                        drag_context,
                        "face_set_count",
                    )
                    overlay_value = drag_context.get(
                        "kinematic_overlay_path"
                    )
                    if not isinstance(overlay_value, str):
                        raise WorkflowExecutionError(
                            "Pose Drag preparation is missing its Face Set "
                            "overlay"
                        )
                    kinematic_overlay_path = str(
                        self._run_path(state, overlay_value)
                    )
                if quadloc_result is None or segmentation_result is None:
                    raise WorkflowExecutionError(
                        "Staged Drag preparation is incomplete"
                    )
                quadloc_payload = _result_object(
                    quadloc_result,
                    "quadloc",
                )
                coordinate = quadloc_payload.get("coordinate")
                if not isinstance(coordinate, dict):
                    raise WorkflowExecutionError(
                        "QuadLoc result is missing coordinate"
                    )
                start_x = _integer_coordinate(coordinate, "x")
                start_y = _integer_coordinate(coordinate, "y")
                width, height = _screenshot_dimensions(segment_input)
                segment_payload = _result_object(
                    segmentation_result,
                    "changed_part segmentation",
                )
                cleaned_mask_value = segment_payload.get(
                    "cleaned_mask_path"
                )
                if not isinstance(cleaned_mask_value, str):
                    raise WorkflowExecutionError(
                        "Changed-part segmentation is missing cleaned_mask_path"
                    )
                cleaned_mask = str(
                    self._run_path(state, cleaned_mask_value)
                )

                if pose_drag:
                    if face_set_count is None:
                        raise WorkflowExecutionError(
                            "Pose Drag did not resolve its kinematic chain"
                        )
                    settings: dict[str, JsonValue] = {
                        "sculpt_brush": intent["sculpt_brush"],
                        "brush_size": (
                            self.config.workflow.drag.pose_brush_size
                        ),
                        "brush_strength": (
                            self.config.workflow.drag.pose_brush_strength
                        ),
                        "brush_direction": None,
                        # Dyntopo distorts the Face Set IK scope for Pose strokes.
                        "dyntopo_enabled": False,
                        "dyntopo_detail_size": (
                            defaults.dyntopo_detail_size
                        ),
                        "use_unified_size": False,
                        "use_unified_strength": False,
                        "use_size_pressure": False,
                        "use_strength_pressure": False,
                        "deformation_target": (
                            self.config.workflow.drag
                            .pose_deformation_target
                        ),
                        "rotation_origins": (
                            self.config.workflow.drag
                            .pose_rotation_origins
                        ),
                        "pose_origin_offset": (
                            self.config.workflow.drag.pose_origin_offset
                        ),
                        "smooth_iterations": (
                            self.config.workflow.drag
                            .pose_smooth_iterations
                        ),
                        "pose_ik_segments": face_set_count,
                        "connected_only": (
                            self.config.workflow.drag.pose_connected_only
                        ),
                        "max_element_distance": (
                            self.config.workflow.drag
                            .pose_max_element_distance
                        ),
                    }
                    resolved_plan_payload = {
                        "settings": settings,
                        "stroke_policy": {
                            "pass_count": 1,
                            "target_coverage": 1.0,
                            "dose_multiplier": 1.0,
                        },
                        "resolution_context": {
                            "mode": "POSE_FACE_SETS",
                            "part_description": part_description,
                            "face_set_count": face_set_count,
                            "reason_codes": [
                                "POSE_FIXED_SIZE_STRENGTH",
                                "POSE_IK_FROM_FACE_SETS",
                                "POSE_DYNTOPO_DISABLED",
                                "POSE_DISCONNECTED_PARTS_ENABLED",
                            ],
                        },
                    }
                else:
                    resolved_plan = resolve_sculpt_plan(
                        intent=SculptIntent.model_validate(intent),
                        cleaned_mask_path=cleaned_mask,
                        screenshot_metadata=cast(
                            Mapping[str, object],
                            segment_input["metadata"],
                        ),
                        config=self.config.workflow.parameter_resolution,
                        operation_defaults=defaults,
                        retry_directive=None,
                    )
                    resolved_plan_payload = resolved_plan.model_dump(
                        mode="json"
                    )
                    settings = resolved_plan.settings.model_dump(mode="json")

                try:
                    contact_anchor = correct_drag_anchor(
                        cleaned_mask_path=cleaned_mask,
                        start_x=start_x,
                        start_y=start_y,
                        # The first phase establishes reliable contact without
                        # changing component identity through the final footprint.
                        brush_size=1,
                        minimum_margin_pixels=(
                            self.config.workflow.drag
                            .safe_anchor_minimum_margin_pixels
                        ),
                        brush_radius_ratio=(
                            self.config.workflow.drag
                            .safe_anchor_brush_radius_ratio
                        ),
                        component_depth_ratio=(
                            self.config.workflow.drag
                            .safe_anchor_component_depth_ratio
                        ),
                        expected_width=width,
                        expected_height=height,
                    )
                except DragAnchorCorrectionError as error:
                    safe_anchor_requires_reselect = True
                    if selected_view not in rejected_drag_anchor_views:
                        rejected_drag_anchor_views.append(selected_view)
                    preparation_error = (
                        f"Safe Drag Anchor rejected {selected_view}: {error}"
                    )
                    retry_feedback = _safe_anchor_retry_feedback(
                        retry_feedback,
                        rejected_view=selected_view,
                        message=preparation_error,
                        rejected_views=rejected_drag_anchor_views,
                        standard_views=(
                            self.config.workflow.standard_views
                        ),
                    )
                    raise WorkflowExecutionError(
                        preparation_error
                    ) from error

                contact_anchor_payload = contact_anchor.as_payload()
                contact_anchor_payload["phase"] = "CONTACT"
                contact_anchor_payload["part_to_be_changed"] = str(
                    intent["part_to_be_changed"]
                )
                start_x = contact_anchor.corrected_x
                start_y = contact_anchor.corrected_y

                try:
                    target_binding = bind_drag_target_component(
                        cleaned_mask_path=cleaned_mask,
                        image_path=image_path,
                        anchor_x=start_x,
                        anchor_y=start_y,
                        output_dir=attempt_dir / "drag-target",
                        expected_width=width,
                        expected_height=height,
                    )
                except DragTargetBindingError as error:
                    raise _DragPreparationRetry(
                        f"Drag target component binding failed: {error}",
                        retry_scope=_DRAG_RETRY_LOCALIZE,
                    ) from error

                if not pose_drag:
                    try:
                        drag_size = resolve_drag_brush_size(
                            component_mask_path=(
                                target_binding.component_mask_path
                            ),
                            anchor_x=start_x,
                            anchor_y=start_y,
                            preliminary_brush_size=int(
                                settings["brush_size"]
                            ),
                            support_radius_ratio=float(
                                resolved_plan.resolution_context
                                .brush_size_ratio
                            ),
                            extent_percentile=(
                                self.config.workflow.drag
                                .brush_anchor_extent_percentile
                            ),
                            maximum_brush_size=(
                                self.config.workflow.parameter_resolution
                                .maximum_brush_size
                            ),
                            screenshot_metadata=cast(
                                Mapping[str, object],
                                segment_input["metadata"],
                            ),
                        )
                    except DragBrushSizeResolutionError as error:
                        raise _DragPreparationRetry(
                            "Drag Brush Size resolution failed: "
                            f"{error}",
                            retry_scope=_DRAG_RETRY_LOCALIZE,
                        ) from error
                    settings["brush_size"] = drag_size.brush_size
                    if resolved_plan_payload is None:
                        raise WorkflowExecutionError(
                            "Drag execution plan was not resolved"
                        )
                    resolved_plan_payload["settings"] = settings
                    resolution_context = resolved_plan_payload.get(
                        "resolution_context"
                    )
                    if not isinstance(resolution_context, dict):
                        raise WorkflowExecutionError(
                            "Drag resolution context is invalid"
                        )
                    drag_size_payload = drag_size.as_payload()
                    resolution_context["drag_brush_size"] = (
                        drag_size_payload
                    )
                    reason_codes = resolution_context.get("reason_codes")
                    if isinstance(reason_codes, list):
                        resolution_context["reason_codes"] = [
                            *reason_codes,
                            *drag_size_payload["reason_codes"],
                        ]

                try:
                    safe_anchor = correct_drag_anchor(
                        cleaned_mask_path=(
                            target_binding.component_mask_path
                        ),
                        start_x=start_x,
                        start_y=start_y,
                        brush_size=int(settings["brush_size"]),
                        minimum_margin_pixels=(
                            self.config.workflow.drag
                            .safe_anchor_minimum_margin_pixels
                        ),
                        brush_radius_ratio=(
                            self.config.workflow.drag
                            .safe_anchor_brush_radius_ratio
                        ),
                        component_depth_ratio=(
                            self.config.workflow.drag
                            .safe_anchor_component_depth_ratio
                        ),
                        expected_width=width,
                        expected_height=height,
                    )
                except DragAnchorCorrectionError as error:
                    safe_anchor_requires_reselect = True
                    if selected_view not in rejected_drag_anchor_views:
                        rejected_drag_anchor_views.append(selected_view)
                    preparation_error = (
                        "Influence Safe Drag Anchor rejected "
                        f"{selected_view}: {error}"
                    )
                    retry_feedback = _safe_anchor_retry_feedback(
                        retry_feedback,
                        rejected_view=selected_view,
                        message=preparation_error,
                        rejected_views=rejected_drag_anchor_views,
                        standard_views=(
                            self.config.workflow.standard_views
                        ),
                    )
                    raise WorkflowExecutionError(
                        preparation_error
                    ) from error

                start_x = safe_anchor.corrected_x
                start_y = safe_anchor.corrected_y
                try:
                    target_binding = bind_drag_target_component(
                        cleaned_mask_path=(
                            target_binding.component_mask_path
                        ),
                        image_path=image_path,
                        anchor_x=start_x,
                        anchor_y=start_y,
                        output_dir=attempt_dir / "drag-target",
                        expected_width=width,
                        expected_height=height,
                    )
                except DragTargetBindingError as error:
                    raise _DragPreparationRetry(
                        "Final Drag target component binding failed: "
                        f"{error}",
                        retry_scope=_DRAG_RETRY_LOCALIZE,
                    ) from error

                safe_anchor_payload = safe_anchor.as_payload()
                safe_anchor_payload["phase"] = "INFLUENCE"
                safe_anchor_payload["part_to_be_changed"] = str(
                    intent["part_to_be_changed"]
                )
                coordinate = {"x": start_x, "y": start_y}
                quadloc_payload["coordinate"] = coordinate
                quadloc_payload["contact_anchor"] = (
                    contact_anchor_payload
                )
                quadloc_payload["safe_anchor"] = safe_anchor_payload
                quadloc_result = {
                    **quadloc_result,
                    "result": quadloc_payload,
                }
                drag_target_binding = target_binding.as_payload()
                drag_target_binding["part_to_be_changed"] = str(
                    intent["part_to_be_changed"]
                )
                drag_target_binding["operation_location"] = str(
                    intent["operation_location"]
                )
                artifacts.extend(
                    _drag_target_artifacts(
                        state,
                        workflow_dir,
                        binding=drag_target_binding,
                    )
                )

                completion = self._llm_for_state(
                    state,
                    call_site="drag_direction",
                ).complete(
                    role=self.config.workflow.drag.llm_role,
                    system_prompt=DRAG_DIRECTION_SYSTEM_PROMPT,
                    user_prompt=drag_direction_user_prompt(
                        subtask=subtask,
                        intent=intent,
                        selected_view=selected_view,
                        start_coordinate=coordinate,
                        image_width=width,
                        image_height=height,
                        retry_feedback=retry_feedback,
                        retry_directive=state["retry_directive"],
                        kinematic_overlay_attached=(
                            kinematic_overlay_path is not None
                        ),
                    ),
                    image_paths=[
                        image_path,
                        target_binding.anchor_overlay_path,
                        *(
                            [kinematic_overlay_path]
                            if kinematic_overlay_path is not None
                            else []
                        ),
                    ],
                    response_model=DragDirectionPlan,
                )
                if not isinstance(completion.value, DragDirectionPlan):
                    raise WorkflowLlmError(
                        "Drag Planner returned the wrong response model"
                    )
                llm_calls.append(completion.metadata)
                direction = completion.value
                drag_direction_plan = direction.model_dump(mode="json")
                if not direction.anchor_target_valid:
                    raise _DragPreparationRetry(
                        "Drag Planner rejected the localized target: "
                        + direction.anchor_target_analysis,
                        retry_scope=_DRAG_RETRY_LOCALIZE,
                    )

                if pose_drag:
                    prepared_checkpoint = (
                        attempt_dir / "locked-pose-context.blend"
                    ).resolve()
                    prepared_response = _invoke_tool(
                        self.dependencies.tools.save_blender_state,
                        {"filepath": str(prepared_checkpoint)},
                        label="save_blender_state(locked_pose_context)",
                    )
                    _rpc_result(
                        prepared_response,
                        "save_blender_state(locked_pose_context)",
                    )
                    prepared_drag_checkpoint_path = str(
                        prepared_checkpoint
                    )

                settings_result = _invoke_tool(
                    self.dependencies.tools.set_sculpt_settings,
                    settings,
                    label="set_sculpt_settings",
                )
                distance_multiplier = _drag_distance_multiplier(
                    state["retry_directive"]
                )
                try:
                    drag_plan = plan_drag_stroke(
                        image_path=image_path,
                        start_x=start_x,
                        start_y=start_y,
                        direction_x=direction.direction[0],
                        direction_y=direction.direction[1],
                        distance_pixels=direction.distance_pixels,
                        distance_multiplier=distance_multiplier,
                        brush_size=int(settings["brush_size"]),
                        brush_strength=float(settings["brush_strength"]),
                        brush_direction=cast(
                            str | None,
                            settings.get("brush_direction"),
                        ),
                        screenshot_metadata=cast(
                            dict[str, object],
                            segment_input["metadata"],
                        ),
                        output_dir=attempt_dir / "drag-plan",
                        minimum_distance_pixels=(
                            self.config.workflow.drag
                            .minimum_distance_pixels
                        ),
                        maximum_distance_ratio=(
                            self.config.workflow.drag
                            .maximum_distance_ratio
                        ),
                        stroke_spacing_pixels=(
                            self.config.workflow.drag
                            .stroke_spacing_pixels
                        ),
                    )
                except DragStrokePlanningError as error:
                    raise WorkflowExecutionError(str(error)) from error
                stroke_plan_result = {
                    "input": {
                        "image_path": image_path,
                        "start_coordinate": coordinate,
                        "direction": list(direction.direction),
                        "distance_pixels": direction.distance_pixels,
                    },
                    "result": drag_plan.as_payload(),
                }
                if resolved_plan_payload is None:
                    raise WorkflowExecutionError(
                        "Drag execution plan was not resolved"
                    )
                resolved_plan_payload["drag"] = {
                    "quadloc": quadloc_payload,
                    "target_binding": drag_target_binding,
                    "direction_plan": drag_direction_plan,
                    "gesture": drag_plan.stroke_plan["gesture"],
                }
                locked_drag_plan = {
                    "schema_version": "locked-drag-plan/v1",
                    "subtask_index": state["current_subtask_index"],
                    "selected_view": selected_view,
                    "screenshot_dimensions": {
                        "width": width,
                        "height": height,
                    },
                    "segmentation_result": segmentation_result,
                    "quadloc_result": quadloc_result,
                    "target_binding": drag_target_binding,
                    "safe_anchor": coordinate,
                    "direction_plan": drag_direction_plan,
                    "base_distance_pixels": direction.distance_pixels,
                    "base_settings": settings,
                    "base_resolved_sculpt_plan": resolved_plan_payload,
                    "pose_drag": pose_drag,
                    "part_description": part_description,
                    "face_set_count": face_set_count,
                    "part_segmentation_result": part_segmentation_result,
                    "kinematic_overlay_path": kinematic_overlay_path,
                    "prepared_checkpoint_path": (
                        prepared_drag_checkpoint_path
                    ),
                }
                baseline_cleaned_mask = (
                    target_binding.component_mask_path
                )
                artifacts.extend(
                    _stroke_plan_artifacts(
                        state,
                        workflow_dir,
                        stroke_plan_result,
                    )
                )
                artifacts.extend(
                    _drag_trajectory_artifacts(
                        state,
                        workflow_dir,
                        visualization_path=drag_plan.visualization_path,
                    )
                )

            elif method is OperationMethod.DRAW:
                image_path = str(
                    self._run_path(state, str(segment_input["path"]))
                )
                draw_prompt = _surface_segmentation_prompt(state, intent)
                segmentation_result = _reused_surface_segmentation(
                    state,
                    image_path=image_path,
                    prompt=draw_prompt,
                )
                if segmentation_result is None:
                    segmentation_result = self._manual_segmentation_override(
                        state,
                        call_site=_MANUAL_MASK_DRAW,
                        image_path=image_path,
                        prompt=draw_prompt,
                    ) or _invoke_sam3_tool(
                        self.dependencies.tools.segment_with_sam3,
                        {
                            "image_path": image_path,
                            "prompt": draw_prompt,
                            "confidence_threshold": (
                                self.config.workflow
                                .sam3_confidence_threshold
                            ),
                            "overlay_opacity": (
                                self.config.workflow.sam3_overlay_opacity
                            ),
                            "output_dir": str(
                                (attempt_dir / "segment").resolve()
                            ),
                        },
                        label="segment_with_sam3",
                        call_site=_MANUAL_MASK_DRAW,
                        image_path=image_path,
                        prompt=draw_prompt,
                    )
                    (
                        segmentation_result,
                        component_artifacts,
                    ) = self._select_segment_component_if_needed(
                        state,
                        image_path=Path(image_path),
                        segmentation_result=segmentation_result,
                        part_description=draw_prompt,
                        output_dir=(
                            attempt_dir / "segment-component-selection"
                        ),
                        workflow_dir=workflow_dir,
                    )
                    artifacts.extend(component_artifacts)
                artifacts.extend(
                    _segment_artifacts(
                        state,
                        workflow_dir,
                        segmentation_result,
                        label_prefix=(
                            "Reused"
                            if state["retry_scope"]
                            in {
                                "same_view",
                                _SURFACE_RETRY_REUSE_SEGMENTATION,
                            }
                            else None
                        ),
                    )
                )
                segment_payload = _result_object(
                    segmentation_result,
                    "segment_with_sam3",
                )
                if self.config.workflow.roi_focus.enabled:
                    (
                        segment_input,
                        segmentation_result,
                        viewport_focus,
                        focus_artifacts,
                    ) = self._focus_segmented_execution(
                        state,
                        segment_input=segment_input,
                        segmentation_result=segmentation_result,
                        attempt_dir=attempt_dir,
                        workflow_dir=workflow_dir,
                        selected_view=selected_view,
                    )
                    artifacts.extend(focus_artifacts)
                    segment_payload = _result_object(
                        segmentation_result,
                        "segment_with_sam3",
                    )
                    image_path = str(
                        self._run_path(
                            state,
                            str(segment_input["path"]),
                        )
                    )
                cleaned_mask = segment_payload.get("cleaned_mask_path")
                if not isinstance(cleaned_mask, str):
                    raise WorkflowExecutionError(
                        "Draw segmentation is missing cleaned_mask_path"
                    )

                draw_dir = (attempt_dir / "draw").resolve()
                draw_dir.mkdir(parents=True, exist_ok=True)
                pattern_description = intent.get(
                    "draw_pattern_description"
                )
                draw_text = intent.get("draw_text")
                draw_scale_tier = str(intent["draw_scale_tier"])
                if isinstance(pattern_description, str):
                    reused_draw_source = _reused_draw_pattern_source(state)
                    if reused_draw_source is None:
                        generator = self._llm_bound_tool(
                            state,
                            fallback=(
                                self.dependencies.tools.generate_svg_pattern
                            ),
                            factory=(
                                self.dependencies
                                .generate_svg_pattern_tool_factory
                            ),
                            call_site="svg_pattern_generator",
                        )
                        draw_pattern_result = _invoke_tool(
                            generator,
                            {"pattern_description": pattern_description},
                            label="generate_svg_pattern",
                        )
                        pattern_payload = _result_object(
                            draw_pattern_result,
                            "generate_svg_pattern",
                        )
                        svg = pattern_payload.get("svg")
                        if not isinstance(svg, str):
                            raise WorkflowExecutionError(
                                "generate_svg_pattern is missing SVG markup"
                            )
                        svg_path = draw_dir / "generated-pattern.svg"
                        _write_workflow_text(svg_path, svg)
                        pattern_payload["svg_path"] = str(svg_path)
                        draw_pattern_result = {
                            **draw_pattern_result,
                            "result": pattern_payload,
                        }
                        llm_metadata = pattern_payload.get("llm")
                        if isinstance(llm_metadata, dict):
                            llm_calls.append(
                                cast(dict[str, JsonValue], llm_metadata)
                            )

                        draw_trajectory_result = _invoke_tool(
                            self.dependencies.tools.svg_to_mouse_trajectories,
                            {"svg": svg},
                            label="svg_to_mouse_trajectories",
                        )
                    else:
                        (
                            draw_pattern_result,
                            draw_trajectory_result,
                        ) = reused_draw_source
                    source_plan = _result_object(
                        draw_trajectory_result,
                        "svg_to_mouse_trajectories",
                    )
                    if reused_draw_source is None:
                        source_plan_path = (
                            draw_dir / "source-trajectories.json"
                        )
                        _write_workflow_json(source_plan_path, source_plan)
                        source_plan["trajectory_plan_path"] = str(
                            source_plan_path
                        )
                        draw_trajectory_result = {
                            **draw_trajectory_result,
                            "result": source_plan,
                        }
                    draw_fitted_trajectory_result = _invoke_tool(
                        self.dependencies.tools.fit_svg_trajectories_to_mask,
                        {
                            "mask_path": cleaned_mask,
                            "trajectory_plan": source_plan,
                            "scale_tier": draw_scale_tier,
                        },
                        label="fit_svg_trajectories_to_mask",
                    )
                    fitted_plan = _result_object(
                        draw_fitted_trajectory_result,
                        "fit_svg_trajectories_to_mask",
                    )
                    content_kind = "PATTERN"
                elif isinstance(draw_text, str):
                    draw_trajectory_result = _invoke_tool(
                        self.dependencies.tools.text_to_mouse_trajectories,
                        {
                            "text": draw_text,
                            "font_name": (
                                self.config.workflow.draw.font_name
                            ),
                            "mask_path": cleaned_mask,
                            "scale_tier": draw_scale_tier,
                        },
                        label="text_to_mouse_trajectories",
                    )
                    fitted_plan = _result_object(
                        draw_trajectory_result,
                        "text_to_mouse_trajectories",
                    )
                    draw_fitted_trajectory_result = _json_object_copy(
                        draw_trajectory_result
                    )
                    content_kind = "TEXT"
                else:
                    raise WorkflowExecutionError(
                        "Draw intent has neither pattern nor text content"
                    )

                fitted_plan_path = draw_dir / "fitted-trajectories.json"
                _write_workflow_json(fitted_plan_path, fitted_plan)
                fitted_plan["trajectory_plan_path"] = str(
                    fitted_plan_path
                )
                if draw_fitted_trajectory_result is None:
                    raise WorkflowExecutionError(
                        "Draw trajectory fitting returned no result"
                    )
                draw_fitted_trajectory_result = {
                    **draw_fitted_trajectory_result,
                    "result": fitted_plan,
                }

                defaults = self.config.workflow.defaults_for_operation(
                    method.value
                )
                capabilities = _capabilities_from_state(state)
                draw_brush = capabilities.canonical_brush_name("Draw")
                translated_draw_direction = intent.get("brush_direction")
                if not isinstance(translated_draw_direction, str):
                    raise WorkflowExecutionError(
                        "Draw intent is missing Translator-selected "
                        "brush_direction"
                    )
                draw_direction = capabilities.normalize_direction(
                    draw_brush,
                    translated_draw_direction,
                )
                if draw_direction not in {"ADD", "SUBTRACT"}:
                    raise WorkflowExecutionError(
                        "Draw brush Direction must be ADD or SUBTRACT"
                    )
                size_multiplier, pass_count, dose_multiplier = (
                    _draw_retry_policy(
                        state["retry_directive"],
                        maximum_pass_count=(
                            self.config.workflow.parameter_resolution
                            .maximum_pass_count
                        ),
                    )
                )
                draw_plan = plan_draw_strokes(
                    image_path=image_path,
                    trajectory_plan=fitted_plan,
                    screenshot_metadata=cast(
                        Mapping[str, object],
                        segment_input["metadata"],
                    ),
                    output_dir=draw_dir / "stroke-plan",
                    brush_size_ratio=(
                        self.config.workflow.draw.brush_size_ratio
                    ),
                    minimum_brush_size=(
                        self.config.workflow.draw.minimum_brush_size
                    ),
                    maximum_brush_size=(
                        self.config.workflow.draw.maximum_brush_size
                    ),
                    size_multiplier=size_multiplier,
                    brush_strength=1.0,
                    brush_direction=draw_direction,
                )
                stroke_plan_result = {
                    "input": {
                        "image_path": image_path,
                        "content_kind": content_kind,
                        "cleaned_mask_path": cleaned_mask,
                    },
                    "result": draw_plan.as_payload(),
                }
                draw_settings = _required_mapping(
                    draw_plan.stroke_plan,
                    "sculpt_settings",
                    label="Draw stroke plan",
                )
                settings = {
                    "sculpt_brush": draw_brush,
                    "brush_size": int(draw_settings["brush_size"]),
                    "brush_strength": 1.0,
                    "brush_direction": draw_direction,
                    "dyntopo_enabled": defaults.dyntopo_enabled,
                    "dyntopo_detail_size": defaults.dyntopo_detail_size,
                    "use_unified_size": False,
                    "use_unified_strength": False,
                    "use_size_pressure": False,
                    "use_strength_pressure": False,
                    "stroke_method": (
                        self.config.workflow.draw.stroke_method
                    ),
                    "brush_spacing_percent": (
                        self.config.workflow.draw.brush_spacing_percent
                    ),
                    "use_space_attenuation": (
                        self.config.workflow.draw.use_space_attenuation
                    ),
                    "auto_smooth_factor": (
                        self.config.workflow.draw.auto_smooth_factor
                    ),
                }
                finishing_smooth: dict[str, JsonValue] = {
                    "enabled": (
                        self.config.workflow.draw.finishing_smooth_enabled
                    ),
                    "algorithm": "same_trajectory_low_strength_smooth/v1",
                    "trajectory_pass_count": 1,
                }
                if self.config.workflow.draw.finishing_smooth_enabled:
                    smooth_brush = capabilities.canonical_brush_name(
                        self.config.workflow.draw
                        .finishing_smooth_brush_name
                    )
                    smooth_direction = capabilities.normalize_direction(
                        smooth_brush,
                        self.config.workflow.draw
                        .finishing_smooth_direction,
                    )
                    smooth_size = max(
                        1,
                        min(
                            10_000,
                            round(
                                int(draw_settings["brush_size"])
                                * self.config.workflow.draw
                                .finishing_smooth_size_ratio
                            ),
                        ),
                    )
                    finishing_smooth["settings"] = {
                        "sculpt_brush": smooth_brush,
                        "brush_size": smooth_size,
                        "brush_strength": (
                            self.config.workflow.draw
                            .finishing_smooth_strength
                        ),
                        "brush_direction": smooth_direction,
                        "dyntopo_enabled": (
                            self.config.workflow.draw
                            .finishing_smooth_dyntopo_enabled
                        ),
                        "dyntopo_detail_size": (
                            defaults.dyntopo_detail_size
                        ),
                        "use_unified_size": False,
                        "use_unified_strength": False,
                        "use_size_pressure": False,
                        "use_strength_pressure": False,
                        "stroke_method": (
                            self.config.workflow.draw.stroke_method
                        ),
                        "brush_spacing_percent": (
                            self.config.workflow.draw
                            .brush_spacing_percent
                        ),
                        "use_space_attenuation": (
                            self.config.workflow.draw
                            .use_space_attenuation
                        ),
                    }
                resolved_plan_payload = {
                    "settings": settings,
                    "stroke_policy": {
                        "pass_count": pass_count,
                        "target_coverage": 1.0,
                        "dose_multiplier": dose_multiplier,
                    },
                    "resolution_context": {
                        "mode": "MASK_FITTED_DRAW_TRAJECTORIES",
                        "content_kind": content_kind,
                        "draw_scale_tier": draw_scale_tier,
                        "brush_size_policy": draw_plan.stroke_plan[
                            "size_resolution"
                        ],
                        "trajectory_summary": fitted_plan.get("summary"),
                        "reason_codes": [
                            "DRAW_BRUSH_FIXED",
                            "DRAW_STRENGTH_FIXED_ONE",
                            "DRAW_SIZE_FROM_FITTED_TRAJECTORY",
                            "DRAW_DYNTOPO_FROM_OPERATION_DEFAULTS",
                            *(
                                ["DRAW_FINISHING_SMOOTH_PASS"]
                                if self.config.workflow.draw
                                .finishing_smooth_enabled
                                else []
                            ),
                        ],
                    },
                    "finishing_smooth": finishing_smooth,
                    "draw": {
                        "pattern_description": pattern_description,
                        "text": draw_text,
                        "font_name": (
                            self.config.workflow.draw.font_name
                            if content_kind == "TEXT"
                            else None
                        ),
                        "scale_tier": draw_scale_tier,
                        "adaptive_sizing": fitted_plan.get("sizing"),
                        "trajectory_format": fitted_plan.get("format"),
                    },
                }
                settings_result = _invoke_tool(
                    self.dependencies.tools.set_sculpt_settings,
                    settings,
                    label="set_sculpt_settings",
                )
                artifacts.extend(
                    _stroke_plan_artifacts(
                        state,
                        workflow_dir,
                        stroke_plan_result,
                    )
                )
                artifacts.extend(
                    _draw_artifacts(
                        state,
                        workflow_dir,
                        pattern_result=draw_pattern_result,
                        trajectory_result=draw_trajectory_result,
                        fitted_result=draw_fitted_trajectory_result,
                        visualization_path=draw_plan.visualization_path,
                    )
                )
                baseline_cleaned_mask = cleaned_mask

            minimum_effect_enabled = (
                method in {
                    OperationMethod.SMEAR,
                    OperationMethod.DRAG,
                    OperationMethod.DRAW,
                }
                and self.config.workflow.minimum_effect.enabled
            )
            if minimum_effect_enabled:
                if segmentation_result is None or stroke_plan_result is None:
                    raise WorkflowExecutionError(
                        f"{method.value} preparation is missing segmentation "
                        "or strokes"
                    )
                segment_payload = _result_object(
                    segmentation_result,
                    "segment_with_sam3",
                )
                plan_payload = _result_object(
                    stroke_plan_result,
                    "plan_sculpt_strokes",
                )
                stroke_plan = plan_payload.get("stroke_plan")
                cleaned_mask = segment_payload.get("cleaned_mask_path")
                if not isinstance(stroke_plan, dict) or not isinstance(
                    cleaned_mask,
                    str,
                ):
                    raise WorkflowExecutionError(
                        f"Prepared {method.value} result is incomplete"
                    )
                if baseline_cleaned_mask is None:
                    baseline_cleaned_mask = cleaned_mask
                baseline_stroke_plan = cast(
                    Mapping[str, JsonValue],
                    stroke_plan,
                )

            capture_limit = (
                self.config.workflow.minimum_effect.baseline_capture_attempts
                if minimum_effect_enabled
                else 1
            )
            prepared: MinimumEffectBaseline | None = None
            for capture_index in range(1, capture_limit + 1):
                self._warm_current_view()
                baseline_a = self._capture_current_view(
                    state,
                    attempt_dir / "baseline-a.png",
                    label=(
                        f"Subtask {state['current_subtask_index'] + 1} "
                        f"attempt {state['current_attempt']} baseline A"
                    ),
                    redraw=False,
                )
                baseline_b = self._capture_current_view(
                    state,
                    attempt_dir / "baseline-b.png",
                    label=(
                        f"Subtask {state['current_subtask_index'] + 1} "
                        f"attempt {state['current_attempt']} baseline B"
                    ),
                    redraw=False,
                )
                if not minimum_effect_enabled:
                    break
                if (
                    baseline_cleaned_mask is None
                    or baseline_stroke_plan is None
                ):
                    raise WorkflowExecutionError(
                        "Minimum-effect baseline inputs are missing"
                    )
                prepared = prepare_minimum_effect_baseline(
                    baseline_a_path=self._run_path(
                        state,
                        str(baseline_a["path"]),
                    ),
                    baseline_b_path=self._run_path(
                        state,
                        str(baseline_b["path"]),
                    ),
                    cleaned_mask_path=baseline_cleaned_mask,
                    stroke_plan=baseline_stroke_plan,
                    output_dir=attempt_dir / "minimum-effect",
                    config=self.config.workflow.minimum_effect,
                    evaluation_strategy=(
                        "semantic_mask"
                        if method is OperationMethod.DRAG
                        else "stroke_intersection"
                    ),
                )
                prepared = prepared.model_copy(
                    update={"capture_attempt_count": capture_index}
                )
                if prepared.ready:
                    if capture_index > 1:
                        prepared = prepared.model_copy(
                            update={
                                "reason_codes": ["BASELINE_RECAPTURED"]
                            }
                        )
                    break
                retryable_noise = prepared.reason_codes == [
                    "BASELINE_NOISE_TOO_HIGH"
                ]
                if not retryable_noise:
                    break

            baseline_stability_exhausted = bool(
                prepared is not None
                and not prepared.ready
                and prepared.reason_codes == ["BASELINE_NOISE_TOO_HIGH"]
                and prepared.capture_attempt_count >= capture_limit
            )

            if baseline_a is None or baseline_b is None:
                raise WorkflowExecutionError(
                    "Stable baseline capture did not produce screenshots"
                )
            artifacts.extend(
                _screenshot_artifacts(
                    {"baseline_a": baseline_a, "baseline_b": baseline_b}
                )
            )
            if prepared is not None:
                minimum_baseline = prepared.model_dump(mode="json")
                artifacts.extend(
                    _minimum_effect_artifacts(
                        state,
                        workflow_dir,
                        minimum_baseline,
                    )
                )
                if not prepared.ready:
                    preparation_error = (
                        "Minimum-effect baseline is inconclusive: "
                        + ", ".join(prepared.reason_codes)
                    )
        except _Sam3NoSegmentation as error:
            preparation_error = str(error)
            if error.call_site == _QUADLOC_MODEL_MASK:
                deterministic_skip_status = "skipped_no_quadloc_model_mask"
            else:
                if segment_input is None:
                    raise WorkflowExecutionError(
                        "Manual mask request is missing its source screenshot"
                    ) from error
                manual_mask_request = self._manual_mask_request_from_error(
                    state,
                    error=error,
                    segment_input=segment_input,
                )
        except (WorkflowExecutionError, WorkflowLlmError, ValueError) as error:
            preparation_error = str(error)
            if isinstance(error, _DragPreparationRetry):
                preparation_retry_scope = error.retry_scope
                retry_feedback = {
                    "source": "drag_preparation",
                    "failed_attempt": state["current_attempt"],
                    "previous_view": selected_view,
                    "retry_scope": error.retry_scope,
                    "analysis": preparation_error,
                }

        if viewport_focus is not None:
            try:
                self._apply_viewport_snapshot(
                    viewport_focus,
                    snapshot_name="before",
                )
            except WorkflowExecutionError as error:
                preparation_error = _combine_errors(
                    preparation_error,
                    f"Cannot restore the full viewport after ROI focus: {error}",
                )

        if drag_context is not None and preparation_error is None:
            drag_context["stage"] = _DRAG_PREPARATION_COMPLETED
            completed_stages = drag_context.get("completed_stages")
            drag_context["completed_stages"] = [
                *(
                    list(completed_stages)
                    if isinstance(completed_stages, list)
                    else ["capture", "changed_part", "quadloc"]
                ),
                "final_plan",
            ]

        approval_id = (
            f"{state['run_id']}:subtask-{state['current_subtask_index']}:"
            f"attempt-{state['current_attempt']}:execute"
        )
        emitter.artifacts(
            artifact_ids=[item.artifact_id for item in artifacts],
            message="Resolved execution and baseline artifacts are available",
        )
        restore_action: str | None = None
        retry_scope: str | None = None
        if manual_mask_request is not None:
            restore_action = None
            retry_scope = None
        elif preparation_error:
            if deterministic_skip_status is not None:
                restore_action = "skip"
                retry_scope = None
            elif baseline_stability_exhausted:
                restore_action = "skip"
                retry_scope = None
            else:
                restore_action = (
                    "retry"
                    if state["current_attempt"]
                    < self.config.workflow.max_subtask_attempts
                    else "skip"
                )
                retry_scope = preparation_retry_scope or (
                    "reselect"
                    if safe_anchor_requires_reselect
                    else "same_view"
                )
            emitter.error(
                code=(
                    "sam3_no_quadloc_model_mask"
                    if deterministic_skip_status is not None
                    else "baseline_stability_failed"
                    if baseline_stability_exhausted
                    else "drag_target_identity_invalid"
                    if preparation_retry_scope
                    == _DRAG_RETRY_LOCALIZE
                    else "drag_safe_anchor_unavailable"
                    if safe_anchor_requires_reselect
                    else "execution_preparation_failed"
                ),
                retryable=restore_action == "retry",
                message=preparation_error,
            )
        elif self.config.workflow.require_execution_approval:
            emitter.approval(
                approval_id=approval_id,
                decision="pending",
                message="Sculpt execution is waiting for approval",
            )
        emitter.node(
            status=(
                WorkflowEventStatus.WAITING
                if manual_mask_request is not None
                else WorkflowEventStatus.ERROR
                if preparation_error
                else
                WorkflowEventStatus.WAITING
                if self.config.workflow.require_execution_approval
                else WorkflowEventStatus.SUCCESS
            ),
            phase="completed",
            message=(
                "Waiting for a manually painted segmentation mask"
                if manual_mask_request is not None
                else preparation_error
                or "Segmentation, resolved parameters, strokes, and baselines are ready"
            ),
        )
        results = list(state["subtask_results"])
        if preparation_error and restore_action == "skip":
            skipped_result: dict[str, JsonValue] = {
                "subtask_index": state["current_subtask_index"],
                "status": (
                    deterministic_skip_status
                    if deterministic_skip_status is not None
                    else "skipped_unstable_baseline"
                    if baseline_stability_exhausted
                    else "skipped_after_max_attempts"
                ),
                "operation_method": method.value,
                "attempts": (
                    0
                    if baseline_stability_exhausted
                    else state["current_attempt"]
                ),
                "last_selected_view": selected_view,
                "reason": preparation_error,
            }
            if minimum_baseline is not None:
                skipped_result["baseline_capture_attempts"] = (
                    minimum_baseline.get("capture_attempt_count", 1)
                )
                skipped_result["last_minimum_effect_baseline"] = (
                    _relativize_run_paths(minimum_baseline, workflow_dir)
                )
            results.append(skipped_result)
        return {
            "manual_mask_request": manual_mask_request,
            "manual_mask_result": None,
            "execution_preparation": _relative_optional_object(
                drag_context,
                workflow_dir,
            ),
            "segmentation_result": (
                cast(
                    dict[str, JsonValue],
                    _relativize_run_paths(segmentation_result, workflow_dir),
                )
                if segmentation_result is not None
                else None
            ),
            "quadloc_result": (
                cast(
                    dict[str, JsonValue],
                    _relativize_run_paths(quadloc_result, workflow_dir),
                )
                if quadloc_result is not None
                else None
            ),
            "drag_target_binding": (
                cast(
                    dict[str, JsonValue],
                    _relativize_run_paths(
                        drag_target_binding,
                        workflow_dir,
                    ),
                )
                if drag_target_binding is not None
                else None
            ),
            "locked_drag_plan": (
                cast(
                    dict[str, JsonValue],
                    _relativize_run_paths(
                        locked_drag_plan,
                        workflow_dir,
                    ),
                )
                if locked_drag_plan is not None
                else state["locked_drag_plan"]
            ),
            "drag_direction_plan": drag_direction_plan,
            "part_segmentation_result": (
                cast(
                    dict[str, JsonValue],
                    _relativize_run_paths(
                        part_segmentation_result,
                        workflow_dir,
                    ),
                )
                if part_segmentation_result is not None
                else None
            ),
            "draw_pattern_result": _relative_optional_object(
                draw_pattern_result,
                workflow_dir,
            ),
            "draw_trajectory_result": _relative_optional_object(
                draw_trajectory_result,
                workflow_dir,
            ),
            "draw_fitted_trajectory_result": _relative_optional_object(
                draw_fitted_trajectory_result,
                workflow_dir,
            ),
            "resolved_sculpt_plan": _relative_optional_object(
                resolved_plan_payload,
                workflow_dir,
            ),
            "stroke_plan_result": (
                cast(
                    dict[str, JsonValue],
                    _relativize_run_paths(stroke_plan_result, workflow_dir),
                )
                if stroke_plan_result is not None
                else None
            ),
            "viewport_focus": _relative_optional_object(
                viewport_focus,
                workflow_dir,
            ),
            "full_before_screenshot": full_before,
            "baseline_screenshot_a": baseline_a,
            "baseline_screenshot_b": baseline_b,
            "minimum_effect_baseline": (
                cast(
                    dict[str, JsonValue],
                    _relativize_run_paths(minimum_baseline, workflow_dir),
                )
                if minimum_baseline is not None
                else None
            ),
            "settings_result": settings_result,
            "after_screenshot": None,
            "full_after_screenshot": None,
            "execution_results": [],
            "execution_error": preparation_error,
            "minimum_effect_result": None,
            "grader_result": None,
            "acceptance_decision": None,
            "current_approval_id": approval_id,
            "execution_decision": None,
            "restore_action": restore_action,
            "retry_scope": retry_scope,
            "retry_feedback": retry_feedback,
            "rejected_drag_anchor_views": rejected_drag_anchor_views,
            "subtask_results": results,
            "llm_calls": llm_calls,
            "artifacts": merge_artifacts(state["artifacts"], artifacts),
            **emitter.state_update(),
        }

    def _prepare_locked_drag_retry(
        self,
        state: SculptWorkflowState,
        *,
        emitter: WorkflowEventEmitter,
        selected_view: str,
        subtask: Mapping[str, JsonValue],
        intent: Mapping[str, JsonValue],
        attempt_dir: Path,
        workflow_dir: Path,
        locked_plan: Mapping[str, JsonValue],
    ) -> StateUpdate:
        """Rebuild only the invalid Drag stages after restoring a checkpoint."""
        retry_scope = state["retry_scope"]
        if retry_scope not in {_DRAG_RETRY_DOSE, _DRAG_RETRY_GESTURE}:
            raise WorkflowExecutionError(
                f"Unsupported locked Drag retry scope: {retry_scope}"
            )
        artifacts: list[WorkflowArtifact] = []
        llm_calls = list(state["llm_calls"])
        segmentation_result: dict[str, JsonValue] | None = None
        quadloc_result: dict[str, JsonValue] | None = None
        drag_target_binding: dict[str, JsonValue] | None = None
        direction_payload: dict[str, JsonValue] | None = None
        part_segmentation_result: dict[str, JsonValue] | None = None
        resolved_payload: dict[str, JsonValue] | None = None
        stroke_plan_result: dict[str, JsonValue] | None = None
        settings_result: JsonValue = None
        baseline_a: dict[str, JsonValue] | None = None
        baseline_b: dict[str, JsonValue] | None = None
        minimum_baseline: dict[str, JsonValue] | None = None
        preparation_error: str | None = None
        preparation_retry_scope: str | None = None
        baseline_stability_exhausted = False
        next_locked_plan = _json_object_copy(locked_plan)

        try:
            if locked_plan.get("subtask_index") != state[
                "current_subtask_index"
            ]:
                raise _DragPreparationRetry(
                    "Locked Drag plan belongs to another subtask",
                    retry_scope=_DRAG_RETRY_LOCALIZE,
                )
            if locked_plan.get("selected_view") != selected_view:
                raise _DragPreparationRetry(
                    "Locked Drag plan view no longer matches selected_view",
                    retry_scope=_DRAG_RETRY_VIEW,
                )
            segment_input = self._change_and_capture(
                state,
                selected_view,
                attempt_dir / "segment-input.png",
                label=(
                    f"Subtask {state['current_subtask_index'] + 1} "
                    f"attempt {state['current_attempt']} locked Drag input"
                ),
            )
            artifacts.extend(
                _screenshot_artifacts({selected_view: segment_input})
            )
            image_path = str(
                self._run_path(state, str(segment_input["path"]))
            )
            width, height = _screenshot_dimensions(segment_input)
            expected_dimensions = _required_mapping(
                locked_plan,
                "screenshot_dimensions",
                label="Locked Drag plan",
            )
            if (
                _integer_coordinate(expected_dimensions, "width") != width
                or _integer_coordinate(expected_dimensions, "height")
                != height
            ):
                raise _DragPreparationRetry(
                    "Viewport dimensions changed since Drag localization",
                    retry_scope=_DRAG_RETRY_LOCALIZE,
                )

            segmentation_result = _json_object_field(
                locked_plan,
                "segmentation_result",
                label="Locked Drag plan",
            )
            quadloc_result = _json_object_field(
                locked_plan,
                "quadloc_result",
                label="Locked Drag plan",
            )
            locked_target = _json_object_field(
                locked_plan,
                "target_binding",
                label="Locked Drag plan",
            )
            locked_mask_value = locked_target.get("component_mask_path")
            if not isinstance(locked_mask_value, str):
                raise _DragPreparationRetry(
                    "Locked Drag plan is missing its component mask",
                    retry_scope=_DRAG_RETRY_LOCALIZE,
                )
            locked_mask_path = str(
                self._run_path(state, locked_mask_value)
            )
            base_settings = _json_object_field(
                locked_plan,
                "base_settings",
                label="Locked Drag plan",
            )
            pose_drag = bool(locked_plan.get("pose_drag"))
            settings = _drag_retry_settings(
                base_settings=base_settings,
                directive=state["retry_directive"],
                pose_drag=pose_drag,
                maximum_brush_size=(
                    self.config.workflow.parameter_resolution
                    .maximum_brush_size
                ),
                maximum_brush_strength=(
                    self.config.workflow.parameter_resolution
                    .maximum_brush_strength
                ),
            )

            kinematic_overlay_path: str | None = None
            if pose_drag:
                stored_part_segmentation = locked_plan.get(
                    "part_segmentation_result"
                )
                if not isinstance(stored_part_segmentation, dict):
                    raise WorkflowExecutionError(
                        "Locked Pose Drag plan is missing Face Set evidence"
                    )
                part_segmentation_result = _json_object_copy(
                    cast(
                        Mapping[str, JsonValue],
                        stored_part_segmentation,
                    )
                )
                face_set_count = _positive_result_integer(
                    locked_plan,
                    "face_set_count",
                )
                settings["pose_ik_segments"] = face_set_count
                stored_overlay = locked_plan.get(
                    "kinematic_overlay_path"
                )
                if not isinstance(stored_overlay, str):
                    raise WorkflowExecutionError(
                        "Locked Pose Drag plan is missing its Face Set overlay"
                    )
                kinematic_overlay_path = str(
                    self._run_path(state, stored_overlay)
                )

            safe_anchor_value = _required_mapping(
                locked_plan,
                "safe_anchor",
                label="Locked Drag plan",
            )
            locked_x = _integer_coordinate(safe_anchor_value, "x")
            locked_y = _integer_coordinate(safe_anchor_value, "y")
            try:
                safe_anchor = correct_drag_anchor(
                    cleaned_mask_path=locked_mask_path,
                    start_x=locked_x,
                    start_y=locked_y,
                    brush_size=int(settings["brush_size"]),
                    minimum_margin_pixels=(
                        self.config.workflow.drag
                        .safe_anchor_minimum_margin_pixels
                    ),
                    brush_radius_ratio=(
                        self.config.workflow.drag
                        .safe_anchor_brush_radius_ratio
                    ),
                    component_depth_ratio=(
                        self.config.workflow.drag
                        .safe_anchor_component_depth_ratio
                    ),
                    expected_width=width,
                    expected_height=height,
                )
            except DragAnchorCorrectionError as error:
                raise _DragPreparationRetry(
                    "Locked Drag target cannot provide a safe retry anchor: "
                    f"{error}",
                    retry_scope=_DRAG_RETRY_LOCALIZE,
                ) from error
            coordinate: dict[str, JsonValue] = {
                "x": safe_anchor.corrected_x,
                "y": safe_anchor.corrected_y,
            }
            safe_anchor_payload = safe_anchor.as_payload()
            safe_anchor_payload["part_to_be_changed"] = str(
                intent["part_to_be_changed"]
            )
            quadloc_payload = _result_object(quadloc_result, "quadloc")
            quadloc_payload["coordinate"] = coordinate
            quadloc_payload["safe_anchor"] = safe_anchor_payload
            quadloc_result["result"] = quadloc_payload

            try:
                target_binding = bind_drag_target_component(
                    cleaned_mask_path=locked_mask_path,
                    image_path=image_path,
                    anchor_x=safe_anchor.corrected_x,
                    anchor_y=safe_anchor.corrected_y,
                    output_dir=attempt_dir / "drag-target",
                    expected_width=width,
                    expected_height=height,
                )
            except DragTargetBindingError as error:
                raise _DragPreparationRetry(
                    f"Locked Drag target binding failed: {error}",
                    retry_scope=_DRAG_RETRY_LOCALIZE,
                ) from error
            drag_target_binding = target_binding.as_payload()
            drag_target_binding["part_to_be_changed"] = str(
                intent["part_to_be_changed"]
            )
            drag_target_binding["operation_location"] = str(
                intent["operation_location"]
            )
            artifacts.extend(
                _drag_target_artifacts(
                    state,
                    workflow_dir,
                    binding=drag_target_binding,
                )
            )

            if retry_scope == _DRAG_RETRY_GESTURE:
                completion = self._llm_for_state(
                    state,
                    call_site="drag_direction_retry",
                ).complete(
                    role=self.config.workflow.drag.llm_role,
                    system_prompt=DRAG_DIRECTION_SYSTEM_PROMPT,
                    user_prompt=drag_direction_user_prompt(
                        subtask=subtask,
                        intent=intent,
                        selected_view=selected_view,
                        start_coordinate=coordinate,
                        image_width=width,
                        image_height=height,
                        retry_feedback=state["retry_feedback"],
                        retry_directive=None,
                        kinematic_overlay_attached=(
                            kinematic_overlay_path is not None
                        ),
                    ),
                    image_paths=[
                        image_path,
                        target_binding.anchor_overlay_path,
                        *(
                            [kinematic_overlay_path]
                            if kinematic_overlay_path is not None
                            else []
                        ),
                    ],
                    response_model=DragDirectionPlan,
                )
                if not isinstance(completion.value, DragDirectionPlan):
                    raise WorkflowLlmError(
                        "Drag Planner returned the wrong response model"
                    )
                llm_calls.append(completion.metadata)
                direction = completion.value
                if not direction.anchor_target_valid:
                    raise _DragPreparationRetry(
                        "Drag Planner rejected the locked target: "
                        + direction.anchor_target_analysis,
                        retry_scope=_DRAG_RETRY_LOCALIZE,
                    )
                distance_multiplier = 1.0
                next_locked_plan["direction_plan"] = (
                    direction.model_dump(mode="json")
                )
                next_locked_plan["base_distance_pixels"] = (
                    direction.distance_pixels
                )
            else:
                direction = DragDirectionPlan.model_validate(
                    _json_object_field(
                        locked_plan,
                        "direction_plan",
                        label="Locked Drag plan",
                    )
                )
                distance_multiplier = _drag_distance_multiplier(
                    state["retry_directive"]
                )
            direction_payload = direction.model_dump(mode="json")

            settings_result = _invoke_tool(
                self.dependencies.tools.set_sculpt_settings,
                settings,
                label="set_sculpt_settings",
            )
            base_distance = _positive_number_field(
                next_locked_plan,
                "base_distance_pixels",
                label="Locked Drag plan",
            )
            drag_plan = plan_drag_stroke(
                image_path=image_path,
                start_x=safe_anchor.corrected_x,
                start_y=safe_anchor.corrected_y,
                direction_x=direction.direction[0],
                direction_y=direction.direction[1],
                distance_pixels=base_distance,
                distance_multiplier=distance_multiplier,
                brush_size=int(settings["brush_size"]),
                brush_strength=float(settings["brush_strength"]),
                brush_direction=cast(
                    str | None,
                    settings.get("brush_direction"),
                ),
                screenshot_metadata=cast(
                    dict[str, object],
                    segment_input["metadata"],
                ),
                output_dir=attempt_dir / "drag-plan",
                minimum_distance_pixels=(
                    self.config.workflow.drag.minimum_distance_pixels
                ),
                maximum_distance_ratio=(
                    self.config.workflow.drag.maximum_distance_ratio
                ),
                stroke_spacing_pixels=(
                    self.config.workflow.drag.stroke_spacing_pixels
                ),
            )
            stroke_plan_result = {
                "input": {
                    "image_path": image_path,
                    "start_coordinate": coordinate,
                    "direction": list(direction.direction),
                    "distance_pixels": base_distance,
                },
                "result": drag_plan.as_payload(),
            }
            base_resolved = _json_object_field(
                locked_plan,
                "base_resolved_sculpt_plan",
                label="Locked Drag plan",
            )
            resolved_payload = _json_object_copy(base_resolved)
            resolved_payload["settings"] = settings
            resolved_payload["drag"] = {
                "quadloc": quadloc_payload,
                "target_binding": drag_target_binding,
                "direction_plan": direction_payload,
                "gesture": drag_plan.stroke_plan["gesture"],
            }
            resolved_payload["retry"] = {
                "scope": retry_scope,
                "locked_localization_reused": True,
                "sam3_reused": True,
                "quadloc_reused": True,
                "direction_reused": (
                    retry_scope == _DRAG_RETRY_DOSE
                ),
                "directive": dict(state["retry_directive"] or {}),
            }
            artifacts.extend(
                _stroke_plan_artifacts(
                    state,
                    workflow_dir,
                    stroke_plan_result,
                )
            )
            artifacts.extend(
                _drag_trajectory_artifacts(
                    state,
                    workflow_dir,
                    visualization_path=drag_plan.visualization_path,
                )
            )

            minimum_effect_enabled = (
                self.config.workflow.minimum_effect.enabled
            )
            capture_limit = (
                self.config.workflow.minimum_effect
                .baseline_capture_attempts
                if minimum_effect_enabled
                else 1
            )
            prepared: MinimumEffectBaseline | None = None
            for capture_index in range(1, capture_limit + 1):
                self._warm_current_view()
                baseline_a = self._capture_current_view(
                    state,
                    attempt_dir / "baseline-a.png",
                    label=(
                        f"Subtask {state['current_subtask_index'] + 1} "
                        f"attempt {state['current_attempt']} baseline A"
                    ),
                    redraw=False,
                )
                baseline_b = self._capture_current_view(
                    state,
                    attempt_dir / "baseline-b.png",
                    label=(
                        f"Subtask {state['current_subtask_index'] + 1} "
                        f"attempt {state['current_attempt']} baseline B"
                    ),
                    redraw=False,
                )
                if not minimum_effect_enabled:
                    break
                prepared = prepare_minimum_effect_baseline(
                    baseline_a_path=self._run_path(
                        state,
                        str(baseline_a["path"]),
                    ),
                    baseline_b_path=self._run_path(
                        state,
                        str(baseline_b["path"]),
                    ),
                    cleaned_mask_path=(
                        target_binding.component_mask_path
                    ),
                    stroke_plan=cast(
                        Mapping[str, JsonValue],
                        drag_plan.stroke_plan,
                    ),
                    output_dir=attempt_dir / "minimum-effect",
                    config=self.config.workflow.minimum_effect,
                    evaluation_strategy="semantic_mask",
                )
                prepared = prepared.model_copy(
                    update={"capture_attempt_count": capture_index}
                )
                if prepared.ready:
                    if capture_index > 1:
                        prepared = prepared.model_copy(
                            update={
                                "reason_codes": ["BASELINE_RECAPTURED"]
                            }
                        )
                    break
                if prepared.reason_codes != ["BASELINE_NOISE_TOO_HIGH"]:
                    break
            if minimum_effect_enabled and prepared is None:
                raise WorkflowExecutionError(
                    "Locked Drag retry did not prepare a minimum-effect baseline"
                )
            if prepared is not None:
                minimum_baseline = prepared.model_dump(mode="json")
                baseline_stability_exhausted = bool(
                    not prepared.ready
                    and prepared.reason_codes == [
                        "BASELINE_NOISE_TOO_HIGH"
                    ]
                    and prepared.capture_attempt_count >= capture_limit
                )
                if not prepared.ready:
                    preparation_error = (
                        "Minimum-effect baseline is inconclusive: "
                        + ", ".join(prepared.reason_codes)
                    )
        except (
            DragStrokePlanningError,
            WorkflowExecutionError,
            WorkflowLlmError,
            ValueError,
        ) as error:
            preparation_error = str(error)
            if isinstance(error, _DragPreparationRetry):
                preparation_retry_scope = error.retry_scope

        if baseline_a is not None and baseline_b is not None:
            artifacts.extend(
                _screenshot_artifacts(
                    {"baseline_a": baseline_a, "baseline_b": baseline_b}
                )
            )
        if minimum_baseline is not None:
            artifacts.extend(
                _minimum_effect_artifacts(
                    state,
                    workflow_dir,
                    minimum_baseline,
                )
            )
        approval_id = (
            f"{state['run_id']}:subtask-{state['current_subtask_index']}:"
            f"attempt-{state['current_attempt']}:execute"
        )
        can_retry = state["current_attempt"] < (
            self.config.workflow.max_subtask_attempts
        )
        restore_action: str | None = None
        next_scope: str | None = retry_scope
        if preparation_error:
            restore_action = (
                "skip"
                if baseline_stability_exhausted or not can_retry
                else "retry"
            )
            next_scope = preparation_retry_scope or _DRAG_RETRY_LOCALIZE
            emitter.error(
                code=(
                    "baseline_stability_failed"
                    if baseline_stability_exhausted
                    else "locked_drag_retry_failed"
                ),
                retryable=restore_action == "retry",
                message=preparation_error,
            )
        elif self.config.workflow.require_execution_approval:
            emitter.approval(
                approval_id=approval_id,
                decision="pending",
                message="Sculpt execution is waiting for approval",
            )
        emitter.artifacts(
            artifact_ids=[item.artifact_id for item in artifacts],
            message="Locked Drag retry evidence is available",
        )
        emitter.node(
            status=(
                WorkflowEventStatus.ERROR
                if preparation_error
                else WorkflowEventStatus.WAITING
                if self.config.workflow.require_execution_approval
                else WorkflowEventStatus.SUCCESS
            ),
            phase="completed",
            message=(
                preparation_error
                or "Locked Drag localization was reused and the retry is ready"
            ),
        )
        results = list(state["subtask_results"])
        if preparation_error and restore_action == "skip":
            results.append(
                {
                    "subtask_index": state["current_subtask_index"],
                    "status": (
                        "skipped_unstable_baseline"
                        if baseline_stability_exhausted
                        else "skipped_after_max_attempts"
                    ),
                    "operation_method": OperationMethod.DRAG.value,
                    "attempts": state["current_attempt"],
                    "last_selected_view": selected_view,
                    "reason": preparation_error,
                }
            )
        return {
            "segmentation_result": segmentation_result,
            "quadloc_result": quadloc_result,
            "drag_target_binding": _relative_optional_object(
                drag_target_binding,
                workflow_dir,
            ),
            "locked_drag_plan": _relative_optional_object(
                next_locked_plan,
                workflow_dir,
            ),
            "drag_direction_plan": direction_payload,
            "part_segmentation_result": _relative_optional_object(
                part_segmentation_result,
                workflow_dir,
            ),
            "resolved_sculpt_plan": _relative_optional_object(
                resolved_payload,
                workflow_dir,
            ),
            "stroke_plan_result": _relative_optional_object(
                stroke_plan_result,
                workflow_dir,
            ),
            "baseline_screenshot_a": baseline_a,
            "baseline_screenshot_b": baseline_b,
            "minimum_effect_baseline": _relative_optional_object(
                minimum_baseline,
                workflow_dir,
            ),
            "settings_result": settings_result,
            "after_screenshot": None,
            "execution_results": [],
            "execution_error": preparation_error,
            "minimum_effect_result": None,
            "grader_result": None,
            "acceptance_decision": None,
            "current_approval_id": approval_id,
            "execution_decision": None,
            "restore_action": restore_action,
            "retry_scope": next_scope,
            "retry_feedback": (
                state["retry_feedback"]
                if preparation_error is None
                else {
                    "source": "locked_drag_retry",
                    "failed_attempt": state["current_attempt"],
                    "previous_view": selected_view,
                    "retry_scope": next_scope,
                    "analysis": preparation_error,
                }
            ),
            "subtask_results": results,
            "llm_calls": llm_calls,
            "artifacts": merge_artifacts(state["artifacts"], artifacts),
            **emitter.state_update(),
        }

    def _approve_execution(
        self,
        state: SculptWorkflowState,
        config: RunnableConfig,
    ) -> StateUpdate:
        """Pause Agent Server runs before the destructive Sculpt operator."""
        approval_id = state["current_approval_id"]
        if approval_id is None:
            raise WorkflowExecutionError("Missing execution approval ID")
        method = OperationMethod(
            str(self._current_subtask(state)["operation_method"])
        )
        configurable = config.get("configurable", {})
        auto_approve = configurable.get("auto_approve") is True
        if (
            method not in {
                OperationMethod.SMEAR,
                OperationMethod.DRAG,
                OperationMethod.DRAW,
            }
            or not self.config.workflow.require_execution_approval
            or auto_approve
        ):
            decision = "approve"
        else:
            before = _required_screenshot(
                state["baseline_screenshot_a"],
                "baseline A",
            )
            response = interrupt(
                {
                    "schema_version": "2.0",
                    "type": "sculpt.execution.approval",
                    "approval_id": approval_id,
                    "run_id": state["run_id"],
                    "subtask_index": state["current_subtask_index"],
                    "attempt": state["current_attempt"],
                    "question": "Approve this Blender Sculpt operation?",
                    "selected_view": state["selected_view"],
                    "subtask": self._current_subtask(state),
                    "intent": self._current_intent(state),
                    "resolved_sculpt_plan": state["resolved_sculpt_plan"],
                    "before_artifact": before.get("artifact"),
                    "allowed_decisions": ["approve", "reject"],
                }
            )
            if not isinstance(response, Mapping):
                raise WorkflowExecutionError(
                    "Execution approval response must be an object"
                )
            decision_value = response.get("decision")
            if decision_value not in {"approve", "reject"}:
                raise WorkflowExecutionError(
                    "Execution approval decision must be approve or reject"
                )
            decision = str(decision_value)

        emitter = WorkflowEventEmitter(state, node="approve_execution")
        emitter.approval(
            approval_id=approval_id,
            decision=cast(Any, decision),
            message=(
                "Sculpt execution approved"
                if decision == "approve"
                else "Sculpt execution rejected"
            ),
        )
        updates: StateUpdate = {
            "execution_decision": decision,
            **emitter.state_update(),
        }
        if decision == "reject":
            method = OperationMethod(
                str(self._current_subtask(state)["operation_method"])
            )
            results = list(state["subtask_results"])
            results.append(
                {
                    "subtask_index": state["current_subtask_index"],
                    "status": "rejected_by_user",
                    "operation_method": method.value,
                    "attempts": state["current_attempt"],
                    "selected_view": state["selected_view"],
                }
            )
            updates.update(
                {
                    "after_screenshot": state["baseline_screenshot_a"],
                    "execution_error": "Sculpt execution rejected by user",
                    "subtask_results": results,
                    "restore_action": "skip",
                }
            )
        return updates

    def _executor(self, state: SculptWorkflowState) -> StateUpdate:
        subtask = self._current_subtask(state)
        method = OperationMethod(str(subtask["operation_method"]))
        emitter = WorkflowEventEmitter(state, node="executor")
        emitter.node(
            status=WorkflowEventStatus.RUNNING,
            phase="started",
            message=f"Executing {method.value} subtask",
        )
        selected_view = state["selected_view"]
        if selected_view is None:
            raise WorkflowExecutionError(
                "Executor started without a selected view"
            )
        attempt_dir = self._attempt_dir(state)
        before = _required_screenshot(
            state["baseline_screenshot_a"],
            "baseline A",
        )

        if method not in {
            OperationMethod.SMEAR,
            OperationMethod.DRAG,
            OperationMethod.DRAW,
        }:
            results = list(state["subtask_results"])
            results.append(
                {
                    "subtask_index": state["current_subtask_index"],
                    "status": "skipped_not_implemented",
                    "operation_method": method.value,
                    "attempts": 0,
                    "selected_view": selected_view,
                    "before_screenshot": before["path"],
                }
            )
            emitter.subtask(
                status=WorkflowEventStatus.SKIPPED,
                action="executed",
                operation_method=method.value,
                message=(
                    f"Skipped {method.value}; only Smear, Drag, and Draw are "
                    "implemented"
                ),
            )
            emitter.node(
                status=WorkflowEventStatus.SKIPPED,
                phase="completed",
                message=f"Skipped unimplemented {method.value} operation",
            )
            return {
                "after_screenshot": before,
                "execution_error": (
                    f"{method.value} execution is not implemented yet"
                ),
                "subtask_results": results,
                **emitter.state_update(),
            }

        if state["execution_decision"] != "approve":
            raise WorkflowExecutionError(
                "Executor cannot run without an approved decision"
            )
        execution_results: list[JsonValue] = []
        execution_error: str | None = None
        discovered_artifacts: list[WorkflowArtifact] = []
        workflow_dir = self._workflow_dir(state)
        applied_operation_ids = list(state["applied_operation_ids"])
        focus = (
            state["viewport_focus"]
            if method in {OperationMethod.SMEAR, OperationMethod.DRAW}
            and isinstance(state["viewport_focus"], dict)
            else None
        )
        focus_applied = False
        try:
            if focus is not None:
                self._apply_viewport_snapshot(
                    focus,
                    snapshot_name="focused",
                )
                focus_applied = True
            plan_result = state["stroke_plan_result"]
            resolved = state["resolved_sculpt_plan"]
            if not isinstance(plan_result, dict) or not isinstance(
                resolved,
                dict,
            ):
                raise WorkflowExecutionError(
                    "Executor is missing the resolved stroke plan"
                )
            operator_calls = _operator_calls(plan_result)
            policy = resolved.get("stroke_policy")
            pass_count = (
                int(policy.get("pass_count", 1))
                if isinstance(policy, dict)
                else 1
            )
            scheduled_calls = [
                operator_call
                for _ in range(pass_count)
                for operator_call in operator_calls
            ]
            finishing = resolved.get("finishing_smooth")
            finishing_settings: dict[str, JsonValue] | None = None
            if (
                method is OperationMethod.DRAW
                and isinstance(finishing, dict)
                and finishing.get("enabled") is True
            ):
                raw_finishing_settings = finishing.get("settings")
                if not isinstance(raw_finishing_settings, dict):
                    raise WorkflowExecutionError(
                        "Draw finishing smooth pass has no settings"
                    )
                finishing_settings = _json_object_copy(
                    cast(
                        Mapping[str, JsonValue],
                        raw_finishing_settings,
                    )
                )
            total_calls = len(scheduled_calls) + (
                len(operator_calls) if finishing_settings is not None else 0
            )
            call_index = 0

            def execute_call(
                tool_input: dict[str, JsonValue],
                *,
                phase: str,
            ) -> None:
                nonlocal call_index
                call_index += 1
                operation_id = sculpt_operation_id(
                    run_id=state["run_id"],
                    subtask_index=state["current_subtask_index"],
                    attempt=state["current_attempt"],
                    call_index=call_index,
                    tool_input=tool_input,
                )
                response = _invoke_tool(
                    self.dependencies.tools.execute_sculpt_stroke,
                    {**tool_input, "operation_id": operation_id},
                    label=(
                        "execute_sculpt_stroke "
                        f"{call_index}/{total_calls} ({phase})"
                    ),
                )
                result = _rpc_result(response, "execute_sculpt_stroke")
                returned_id = result.get("operation_id")
                if returned_id != operation_id:
                    raise WorkflowExecutionError(
                        "execute_sculpt_stroke did not confirm operation_id"
                    )
                idempotency = result.get("idempotency")
                status = (
                    idempotency.get("status")
                    if isinstance(idempotency, dict)
                    else None
                )
                if status not in {"applied", "already_applied"}:
                    raise WorkflowExecutionError(
                        "execute_sculpt_stroke did not confirm a valid "
                        "idempotency status"
                    )
                if operation_id not in applied_operation_ids:
                    applied_operation_ids.append(operation_id)
                execution_results.append(
                    {
                        "operation_id": operation_id,
                        "status": status,
                        "phase": phase,
                        "response": response,
                    }
                )
                emitter.progress(
                    label="Sculpt strokes",
                    current=call_index,
                    total=total_calls,
                    unit="stroke",
                    message=(
                        f"Processed Sculpt stroke {call_index} of "
                        f"{total_calls} ({phase})"
                    ),
                )

            for operator_call in scheduled_calls:
                execute_call(
                    _operator_tool_input(operator_call),
                    phase="primary",
                )

            if finishing_settings is not None:
                primary_settings = resolved.get("settings")
                if not isinstance(primary_settings, dict):
                    raise WorkflowExecutionError(
                        "Draw plan has no primary settings to restore"
                    )
                finishing_size = finishing_settings.get("brush_size")
                if (
                    isinstance(finishing_size, bool)
                    or not isinstance(finishing_size, int)
                ):
                    raise WorkflowExecutionError(
                        "Draw finishing smooth size is invalid"
                    )
                finishing_error: WorkflowExecutionError | None = None
                try:
                    _invoke_tool(
                        self.dependencies.tools.set_sculpt_settings,
                        finishing_settings,
                        label="set Draw finishing Smooth settings",
                    )
                    for operator_call in operator_calls:
                        execute_call(
                            _draw_finishing_tool_input(
                                operator_call,
                                brush_size=finishing_size,
                            ),
                            phase="finishing_smooth",
                        )
                except WorkflowExecutionError as error:
                    finishing_error = error
                try:
                    _invoke_tool(
                        self.dependencies.tools.set_sculpt_settings,
                        _json_object_copy(
                            cast(
                                Mapping[str, JsonValue],
                                primary_settings,
                            )
                        ),
                        label="restore Draw settings after Smooth finish",
                    )
                except WorkflowExecutionError as restore_error:
                    if finishing_error is not None:
                        raise WorkflowExecutionError(
                            f"{finishing_error}; additionally failed to "
                            f"restore Draw settings: {restore_error}"
                        ) from restore_error
                    raise
                if finishing_error is not None:
                    raise finishing_error
        except WorkflowExecutionError as error:
            execution_error = str(error)

        after = before
        try:
            self._warm_current_view()
            after = self._capture_current_view(
                state,
                attempt_dir / "after.png",
                label=(
                    f"Subtask {state['current_subtask_index'] + 1} "
                    f"attempt {state['current_attempt']} focused after"
                ),
                redraw=False,
            )
            discovered_artifacts.extend(
                _screenshot_artifacts({selected_view: after})
            )
        except WorkflowExecutionError as error:
            execution_error = _combine_errors(
                execution_error,
                f"Cannot capture the post-stroke viewport: {error}",
            )
        finally:
            if focus_applied and focus is not None:
                try:
                    self._apply_viewport_snapshot(
                        focus,
                        snapshot_name="before",
                    )
                    focus_applied = False
                except WorkflowExecutionError as error:
                    execution_error = _combine_errors(
                        execution_error,
                        "Cannot restore the full viewport after Sculpt "
                        f"execution: {error}",
                    )

        full_after = after
        if focus is not None and not focus_applied:
            try:
                full_after = self._capture_current_view(
                    state,
                    attempt_dir / "full-after.png",
                    label=(
                        f"Subtask {state['current_subtask_index'] + 1} "
                        f"attempt {state['current_attempt']} full after"
                    ),
                )
                discovered_artifacts.extend(
                    _screenshot_artifacts({"full_after": full_after})
                )
            except WorkflowExecutionError as error:
                execution_error = _combine_errors(
                    execution_error,
                    f"Cannot capture the restored full viewport: {error}",
                )
        emitter.artifacts(
            artifact_ids=[
                artifact.artifact_id for artifact in discovered_artifacts
            ],
            message="Execution artifacts are available",
        )
        if execution_error:
            emitter.error(
                code="sculpt_execution_failed",
                retryable=True,
                message=execution_error,
            )
        emitter.subtask(
            status=(
                WorkflowEventStatus.ERROR
                if execution_error
                else WorkflowEventStatus.SUCCESS
            ),
            action="executed",
            operation_method=method.value,
            message=(
                execution_error
                or f"Executed {len(execution_results)} Sculpt gestures"
            ),
        )
        emitter.node(
            status=(
                WorkflowEventStatus.ERROR
                if execution_error
                else WorkflowEventStatus.SUCCESS
            ),
            phase="completed",
            message=execution_error or "Sculpt execution completed",
        )
        return {
            "after_screenshot": after,
            "full_after_screenshot": full_after,
            "execution_results": execution_results,
            "execution_error": execution_error,
            "applied_operation_ids": applied_operation_ids,
            "artifacts": merge_artifacts(
                state["artifacts"],
                discovered_artifacts,
            ),
            **emitter.state_update(),
        }

    def _minimum_effect(
        self,
        state: SculptWorkflowState,
    ) -> StateUpdate:
        """Reject only absent or insufficiently visible target changes."""
        emitter = WorkflowEventEmitter(state, node="minimum_effect")
        emitter.node(
            status=WorkflowEventStatus.RUNNING,
            phase="started",
            message="Checking the minimum visible target-region effect",
        )
        intent = self._current_intent(state)
        baseline_payload = state["minimum_effect_baseline"]
        after = _required_screenshot(state["after_screenshot"], "after")
        config = self.config.workflow.minimum_effect

        prepared: MinimumEffectBaseline | None = None
        if isinstance(baseline_payload, dict):
            absolute_baseline = dict(baseline_payload)
            for key in (
                "baseline_a_path",
                "baseline_b_path",
                "evaluation_mask_path",
                "before_roi_path",
            ):
                value = absolute_baseline.get(key)
                if isinstance(value, str):
                    absolute_baseline[key] = str(
                        self._run_path(state, value)
                    )
            prepared = MinimumEffectBaseline.model_validate(
                absolute_baseline
            )

        if not config.enabled:
            result = unmeasured_minimum_effect_result(
                verdict=MinimumEffectVerdict.VISIBLE,
                effect_intensity=str(intent["effect_intensity"]),
                config=config,
                baseline=prepared,
                reason_codes=["MINIMUM_EFFECT_GATE_DISABLED"],
            )
        elif prepared is None:
            result = unmeasured_minimum_effect_result(
                verdict=MinimumEffectVerdict.INCONCLUSIVE,
                effect_intensity=str(intent["effect_intensity"]),
                config=config,
                reason_codes=["MINIMUM_EFFECT_BASELINE_MISSING"],
            )
        elif state["execution_error"] is not None:
            result = unmeasured_minimum_effect_result(
                verdict=MinimumEffectVerdict.INCONCLUSIVE,
                effect_intensity=str(intent["effect_intensity"]),
                config=config,
                baseline=prepared,
                reason_codes=["SCULPT_EXECUTION_ERROR"],
            )
        else:
            try:
                result = evaluate_minimum_effect(
                    baseline=prepared,
                    after_path=self._run_path(state, str(after["path"])),
                    effect_intensity=str(intent["effect_intensity"]),
                    output_dir=self._attempt_dir(state) / "minimum-effect",
                    config=config,
                )
            except (OSError, ValueError):
                result = unmeasured_minimum_effect_result(
                    verdict=MinimumEffectVerdict.INCONCLUSIVE,
                    effect_intensity=str(intent["effect_intensity"]),
                    config=config,
                    baseline=prepared,
                    reason_codes=["MINIMUM_EFFECT_EVALUATION_ERROR"],
                )

        workflow_dir = self._workflow_dir(state)
        result_payload = cast(
            dict[str, JsonValue],
            _relativize_run_paths(
                result.model_dump(mode="json"),
                workflow_dir,
            ),
        )
        artifacts = _minimum_effect_artifacts(
            state,
            workflow_dir,
            result.model_dump(mode="json"),
        )
        visible = result.verdict is MinimumEffectVerdict.VISIBLE
        can_retry = state["current_attempt"] < (
            self.config.workflow.max_subtask_attempts
        )
        restore_action = None if visible else (
            "retry" if can_retry else "skip"
        )
        operation = OperationMethod(
            str(self._current_subtask(state)["operation_method"])
        )
        retry_directive = state["retry_directive"]
        if result.verdict in {
            MinimumEffectVerdict.NO_EFFECT,
            MinimumEffectVerdict.TOO_SUBTLE,
        } and can_retry:
            retry_multiplier = (
                self.config.workflow.parameter_resolution
                .retry_dose_multiplier
            )
            if operation is OperationMethod.DRAG:
                retry_directive = _next_drag_dose_directive(
                    result=result,
                    previous=state["retry_directive"],
                    retry_multiplier=retry_multiplier,
                    pose_drag=_is_pose_drag(
                        subtask=self._current_subtask(state),
                        intent=self._current_intent(state),
                        pose_brush_name=(
                            self.config.workflow.drag.pose_brush_name
                        ),
                    ),
                )
            else:
                retry_directive = retry_directive_for_minimum_effect(
                    result,
                    previous=state["retry_directive"],
                    retry_multiplier=retry_multiplier,
                )

        event_status = (
            WorkflowEventStatus.SUCCESS
            if visible
            else WorkflowEventStatus.RETRYING
            if can_retry
            else WorkflowEventStatus.SKIPPED
        )
        emitter.effect_visibility(
            verdict=result.verdict.value,
            target_mean_abs_diff=(
                result.metrics.target_mean_abs_diff
            ),
            target_changed_fraction=(
                result.metrics.target_changed_fraction
            ),
            status=event_status,
            message=(
                "Minimum visible effect reached"
                if visible
                else (
                    f"Minimum-effect verdict: {result.verdict.value}; "
                    f"action={restore_action}"
                )
            ),
        )
        emitter.artifacts(
            artifact_ids=[item.artifact_id for item in artifacts],
            message="Minimum-effect evidence is available",
        )

        updates: StateUpdate = {
            "minimum_effect_result": result_payload,
            "restore_action": restore_action,
            "retry_scope": None if visible else (
                _DRAG_RETRY_DOSE
                if operation is OperationMethod.DRAG
                else "same_view"
            ),
            "retry_directive": retry_directive,
            "acceptance_decision": (
                None
                if visible
                else {
                    "accepted": False,
                    "action": "retry" if can_retry else "restore",
                    "reason_codes": list(result.reason_codes),
                }
            ),
            "artifacts": merge_artifacts(state["artifacts"], artifacts),
        }
        if not visible:
            history = [
                *state["attempt_history"],
                _attempt_record(state, minimum_effect=result_payload),
            ]
            results = list(state["subtask_results"])
            if not can_retry:
                results.append(
                    {
                        "subtask_index": state["current_subtask_index"],
                        "status": "skipped_after_max_attempts",
                        "operation_method": operation.value,
                        "attempts": state["current_attempt"],
                        "last_selected_view": state["selected_view"],
                        "last_minimum_effect": result_payload,
                    }
                )
            emitter.subtask(
                status=event_status,
                action="graded",
                operation_method=operation.value,
                message=(
                    f"Effect gate returned {result.verdict.value}; "
                    f"action={restore_action}"
                ),
            )
            updates.update(
                {
                    "attempt_history": history,
                    "subtask_results": results,
                    "retry_feedback": {
                        "source": "minimum_effect",
                        "failed_attempt": state["current_attempt"],
                        "verdict": result.verdict.value,
                        "reason_codes": list(result.reason_codes),
                    },
                }
            )
        emitter.node(
            status=event_status,
            phase="completed",
            message=(
                "Visible result forwarded to the visual Grader"
                if visible
                else "Result will be restored before the next action"
            ),
        )
        updates.update(emitter.state_update())
        return updates

    def _grader(self, state: SculptWorkflowState) -> StateUpdate:
        """Use the VLM only after deterministic visibility is established."""
        emitter = WorkflowEventEmitter(state, node="grader")
        emitter.node(
            status=WorkflowEventStatus.RUNNING,
            phase="started",
            message="Judging visible effect quality and appropriateness",
        )
        selected_view = state["selected_view"]
        minimum_effect = state["minimum_effect_result"]
        if selected_view is None or not isinstance(minimum_effect, dict):
            raise WorkflowExecutionError(
                "Grader is missing selected view or minimum-effect result"
            )
        image_paths, image_labels = self._attempt_evidence_images(state)
        completion = self._llm_for_state(
            state,
            call_site="grader",
        ).complete(
            role="grader",
            system_prompt=GRADER_SYSTEM_PROMPT,
            user_prompt=grader_user_prompt(
                user_instruction=state["user_instruction"],
                subtask=self._current_subtask(state),
                intent=self._current_intent(state),
                selected_view=selected_view,
                minimum_effect=minimum_effect,
                image_labels=image_labels,
            ),
            image_paths=image_paths,
            response_model=GraderOutput,
        )
        if not isinstance(completion.value, GraderOutput):
            raise WorkflowLlmError(
                "Grader adapter returned the wrong response model"
            )
        grader = completion.value
        payload = grader.model_dump(mode="json")
        payload["total_score"] = grader.total_score
        payload["score_qualifies"] = grader.score_qualifies
        emitter.node(
            status=WorkflowEventStatus.SUCCESS,
            phase="completed",
            message=(
                "Visual Grader completed with appropriateness "
                f"{grader.effect_appropriateness.value}"
            ),
        )
        return {
            "grader_result": payload,
            "llm_calls": [*state["llm_calls"], completion.metadata],
            **emitter.state_update(),
        }

    def _acceptance_gate(
        self,
        state: SculptWorkflowState,
    ) -> StateUpdate:
        """Combine deterministic visibility with the structured VLM verdict."""
        emitter = WorkflowEventEmitter(state, node="acceptance_gate")
        emitter.node(
            status=WorkflowEventStatus.RUNNING,
            phase="started",
            message="Applying deterministic acceptance rules",
        )
        minimum_payload = state["minimum_effect_result"]
        grader_payload = state["grader_result"]
        if not isinstance(minimum_payload, dict) or not isinstance(
            grader_payload,
            dict,
        ):
            raise WorkflowExecutionError(
                "Acceptance Gate is missing effect or Grader output"
            )
        minimum = MinimumEffectResult.model_validate(minimum_payload)
        grader = GraderOutput.model_validate(
            {
                key: grader_payload[key]
                for key in GraderOutput.model_fields
                if key in grader_payload
            }
        )
        reasons: list[str] = []
        if minimum.verdict is not MinimumEffectVerdict.VISIBLE:
            reasons.append("MINIMUM_EFFECT_NOT_VISIBLE")
        if grader.effect_appropriateness is not (
            EffectAppropriateness.APPROPRIATE
        ):
            reasons.append(
                f"EFFECT_{grader.effect_appropriateness.value}"
            )
        if not grader.score_qualifies:
            reasons.append("GRADER_SCORE_BELOW_ACCEPTANCE_THRESHOLD")
        operation = OperationMethod(
            str(self._current_subtask(state)["operation_method"])
        )
        if operation is OperationMethod.DRAG:
            drag_assessment = grader.drag_assessment
            if drag_assessment is None:
                reasons.append("DRAG_ASSESSMENT_MISSING")
            else:
                if not drag_assessment.target_identity_correct:
                    reasons.append("DRAG_TARGET_IDENTITY_INCORRECT")
                if not drag_assessment.motion_direction_correct:
                    reasons.append("DRAG_MOTION_DIRECTION_INCORRECT")
                if not drag_assessment.target_motion_visible:
                    reasons.append("DRAG_TARGET_MOTION_NOT_VISIBLE")
                if not drag_assessment.spatial_goal_reached:
                    reasons.append("DRAG_SPATIAL_GOAL_NOT_REACHED")
                if not drag_assessment.non_target_geometry_stable:
                    reasons.append("DRAG_NON_TARGET_GEOMETRY_UNSTABLE")
        if state["execution_error"] is not None:
            reasons.append("SCULPT_EXECUTION_ERROR")
        accepted = not reasons
        can_retry = state["current_attempt"] < (
            self.config.workflow.max_subtask_attempts
        )
        selected_retry_scope: str | None = None
        if not accepted and can_retry:
            selected_retry_scope = (
                _classify_drag_retry_scope(grader)
                if operation is OperationMethod.DRAG
                else "reselect"
            )
        action = (
            "advance"
            if accepted
            else "retry"
            if can_retry
            else "restore"
        )
        decision: dict[str, JsonValue] = {
            "accepted": accepted,
            "action": action,
            "reason_codes": reasons,
            "retry_scope": selected_retry_scope,
        }
        history = [
            *state["attempt_history"],
            _attempt_record(
                state,
                minimum_effect=minimum_payload,
                grader=grader_payload,
                acceptance=decision,
            ),
        ]
        results = list(state["subtask_results"])
        if accepted:
            results.append(
                {
                    "subtask_index": state["current_subtask_index"],
                    "status": "passed",
                    "operation_method": operation.value,
                    "attempts": state["current_attempt"],
                    "selected_view": state["selected_view"],
                    "resolved_sculpt_plan": state[
                        "resolved_sculpt_plan"
                    ],
                    "minimum_effect": minimum_payload,
                    "grader": grader_payload,
                }
            )
        elif not can_retry:
            results.append(
                {
                    "subtask_index": state["current_subtask_index"],
                    "status": "skipped_after_max_attempts",
                    "operation_method": operation.value,
                    "attempts": state["current_attempt"],
                    "last_selected_view": state["selected_view"],
                    "last_minimum_effect": minimum_payload,
                    "last_grader": grader_payload,
                    "reason_codes": reasons,
                }
            )
        status = (
            WorkflowEventStatus.PASSED
            if accepted
            else WorkflowEventStatus.RETRYING
            if can_retry
            else WorkflowEventStatus.SKIPPED
        )
        emitter.subtask(
            status=status,
            action="graded",
            operation_method=operation.value,
            message=(
                f"Attempt accepted with score {grader.total_score}"
                if accepted
                else (
                    f"Attempt rejected ({', '.join(reasons)}); "
                    f"action={action}"
                )
            ),
        )
        emitter.node(
            status=(
                WorkflowEventStatus.SUCCESS if accepted else status
            ),
            phase="completed",
            message=(
                "Acceptance Gate passed"
                if accepted
                else "Acceptance Gate requested repair"
                if can_retry
                else "Maximum attempts reached; change will be restored"
            ),
        )
        return {
            "acceptance_decision": decision,
            "attempt_history": history,
            "subtask_results": results,
            "restore_action": None if accepted else (
                "retry" if can_retry else "skip"
            ),
            "retry_scope": selected_retry_scope,
            **emitter.state_update(),
        }

    def _retry_planner(
        self,
        state: SculptWorkflowState,
    ) -> StateUpdate:
        """Create a semantic repair plan after a visible-result failure."""
        emitter = WorkflowEventEmitter(state, node="retry_planner")
        emitter.node(
            status=WorkflowEventStatus.RUNNING,
            phase="started",
            message="Planning a visual-result repair",
        )
        selected_view = state["selected_view"]
        resolved_plan = state["resolved_sculpt_plan"]
        minimum_effect = state["minimum_effect_result"]
        grader = state["grader_result"]
        if (
            selected_view is None
            or not isinstance(resolved_plan, dict)
            or not isinstance(minimum_effect, dict)
            or not isinstance(grader, dict)
        ):
            raise WorkflowExecutionError(
                "Retry Planner is missing current-attempt evidence"
            )
        operation = OperationMethod(
            str(self._current_subtask(state)["operation_method"])
        )
        if operation is OperationMethod.DRAG:
            grader_model = GraderOutput.model_validate(
                {
                    key: grader[key]
                    for key in GraderOutput.model_fields
                    if key in grader
                }
            )
            scope = _classify_drag_retry_scope(grader_model)
            if scope != _DRAG_RETRY_INTENT:
                directive = state["retry_directive"]
                if scope == _DRAG_RETRY_DOSE:
                    directive = _force_drag_dose_directive(
                        previous=directive,
                        retry_multiplier=(
                            self.config.workflow.parameter_resolution
                            .retry_dose_multiplier
                        ),
                        pose_drag=_is_pose_drag(
                            subtask=self._current_subtask(state),
                            intent=self._current_intent(state),
                            pose_brush_name=(
                                self.config.workflow.drag.pose_brush_name
                            ),
                        ),
                        cause=str(
                            grader_model.effect_appropriateness.value
                        ),
                    )
                else:
                    directive = None
                feedback: dict[str, JsonValue] = {
                    "source": "failure_scoped_drag_retry",
                    "failed_attempt": state["current_attempt"],
                    "previous_view": selected_view,
                    "retry_scope": scope,
                    "minimum_effect": minimum_effect,
                    "grader": grader,
                }
                emitter.node(
                    status=WorkflowEventStatus.RETRYING,
                    phase="completed",
                    message=(
                        "Drag retry will restart from scope " + scope
                    ),
                )
                return {
                    "retry_feedback": feedback,
                    "retry_directive": directive,
                    "retry_scope": scope,
                    "restore_action": "retry",
                    **emitter.state_update(),
                }
        image_paths, image_labels = self._attempt_evidence_images(state)
        segment_context = _segment_context(
            state["segmentation_result"],
            state["stroke_plan_result"],
        )
        capabilities = _capabilities_from_state(state)
        completion = self._llm_for_state(
            state,
            call_site="retry_planner",
        ).complete(
            role="retry_planner",
            system_prompt=RETRY_PLANNER_SYSTEM_PROMPT,
            user_prompt=retry_planner_user_prompt(
                user_instruction=state["user_instruction"],
                subtask=self._current_subtask(state),
                intent=self._current_intent(state),
                resolved_plan=resolved_plan,
                selected_view=selected_view,
                minimum_effect=minimum_effect,
                grader=grader,
                segment_context=segment_context,
                image_labels=image_labels,
                sculpt_capabilities=capabilities.model_dump(mode="json"),
            ),
            image_paths=image_paths,
            response_model=RetryPlannerOutput,
        )
        if not isinstance(completion.value, RetryPlannerOutput):
            raise WorkflowLlmError(
                "Retry Planner adapter returned the wrong response model"
            )
        repair = completion.value
        index = state["current_subtask_index"]
        try:
            revised_description = (
                capabilities.canonicalize_subtask_description(
                    repair.revised_subtask_description
                )
            )
        except SculptCapabilityError as error:
            raise WorkflowLlmError(
                f"Retry Planner selected an unavailable Sculpt brush: {error}"
            ) from error
        revised_subtask = DecomposedSubtask(
            description=revised_description,
            operation_method=operation,
        )
        revised_intent = _validated_sculpt_intent(
            repair.revised_intent,
            subtask=revised_subtask.model_dump(mode="json"),
            capabilities=capabilities,
            role="Retry Planner",
        )
        current_subtask = self._current_subtask(state)
        current_intent = self._current_intent(state)
        next_retry_scope = "reselect"
        recommended_view = repair.recommended_view.value
        segmentation_prompt: str | None = None
        regenerate_svg_pattern = False
        if _is_pose_drag(
            subtask=current_subtask,
            intent=current_intent,
            pose_brush_name=self.config.workflow.drag.pose_brush_name,
        ):
            revised_subtask = DecomposedSubtask.model_validate(
                current_subtask
            )
            preserved_payload = repair.revised_intent.model_dump(
                mode="json",
                exclude_none=True,
            )
            preserved_payload["operation_location"] = current_intent[
                "operation_location"
            ]
            preserved_payload["part_to_be_changed"] = current_intent[
                "part_to_be_changed"
            ]
            preserved_payload["sculpt_brush"] = current_intent[
                "sculpt_brush"
            ]
            for field in (
                "brush_scale",
                "brush_strength",
                "brush_direction",
            ):
                preserved_payload.pop(field, None)
            revised_intent = _validated_sculpt_intent(
                SculptIntent.model_validate(preserved_payload),
                subtask=current_subtask,
                capabilities=capabilities,
                role="Retry Planner",
            )
        elif operation in {OperationMethod.SMEAR, OperationMethod.DRAW}:
            surface_scope = repair.surface_retry_scope
            if surface_scope is None:
                raise WorkflowLlmError(
                    f"Retry Planner omitted surface_retry_scope for "
                    f"{operation.value}"
                )
            next_retry_scope = {
                SurfaceRetryScope.RESELECT_VIEW: "reselect",
                SurfaceRetryScope.RESEGMENT: _SURFACE_RETRY_RESEGMENT,
                SurfaceRetryScope.REUSE_SEGMENTATION: (
                    _SURFACE_RETRY_REUSE_SEGMENTATION
                ),
            }[surface_scope]
            if surface_scope is not SurfaceRetryScope.RESELECT_VIEW:
                recommended_view = selected_view
            segmentation_prompt = (
                repair.segmentation_prompt
                if surface_scope is SurfaceRetryScope.RESEGMENT
                else None
            )
            if (
                surface_scope is SurfaceRetryScope.RESEGMENT
                and segmentation_prompt is None
            ):
                raise WorkflowLlmError(
                    "Retry Planner selected RESEGMENT without a new "
                    "segmentation_prompt"
                )
            if surface_scope is SurfaceRetryScope.RESEGMENT:
                previous_prompt = (
                    segment_context.get("prompt")
                    if isinstance(segment_context, dict)
                    and isinstance(segment_context.get("prompt"), str)
                    else current_intent.get("part_to_be_changed")
                )
                if (
                    isinstance(previous_prompt, str)
                    and segmentation_prompt is not None
                    and segmentation_prompt.casefold()
                    == previous_prompt.casefold()
                ):
                    raise WorkflowLlmError(
                        "Retry Planner repeated the deterministic SAM3 prompt"
                    )

            revised_payload = revised_intent.model_dump(mode="json")
            if surface_scope is SurfaceRetryScope.REUSE_SEGMENTATION:
                revised_payload["operation_location"] = current_intent[
                    "operation_location"
                ]
                revised_payload["part_to_be_changed"] = current_intent[
                    "part_to_be_changed"
                ]
                if operation is OperationMethod.SMEAR and all(
                    revised_payload.get(field) == current_intent.get(field)
                    for field in (
                        "sculpt_brush",
                        "brush_scale",
                        "brush_strength",
                        "brush_direction",
                        "effect_intensity",
                    )
                ):
                    raise WorkflowLlmError(
                        "Retry Planner reused Smear segmentation without "
                        "changing any brush parameter"
                    )

            if operation is OperationMethod.DRAW:
                current_pattern = current_intent.get(
                    "draw_pattern_description"
                )
                current_text = current_intent.get("draw_text")
                if isinstance(current_pattern, str):
                    regenerate_svg_pattern = bool(
                        repair.regenerate_svg_pattern
                    )
                    revised_payload["draw_text"] = None
                    if not regenerate_svg_pattern:
                        revised_payload["draw_pattern_description"] = (
                            current_pattern
                        )
                    else:
                        revised_pattern = revised_payload.get(
                            "draw_pattern_description"
                        )
                        if (
                            not isinstance(revised_pattern, str)
                            or " ".join(revised_pattern.split()).casefold()
                            == " ".join(current_pattern.split()).casefold()
                        ):
                            raise WorkflowLlmError(
                                "Retry Planner requested SVG regeneration "
                                "without changing draw_pattern_description"
                            )
                elif isinstance(current_text, str):
                    revised_payload["draw_pattern_description"] = None
                    revised_payload["draw_text"] = current_text
                    regenerate_svg_pattern = False
                else:
                    raise WorkflowExecutionError(
                        "Current Draw intent has no stable content kind"
                    )
            else:
                regenerate_svg_pattern = False

            revised_intent = _validated_sculpt_intent(
                SculptIntent.model_validate(revised_payload),
                subtask=revised_subtask.model_dump(mode="json"),
                capabilities=capabilities,
                role="Retry Planner",
            )
        subtasks = [dict(item) for item in state["subtasks"]]
        subtasks[index] = revised_subtask.model_dump(mode="json")
        translations = [dict(item) for item in state["translations"]]
        for position, translation in enumerate(translations):
            if translation.get("subtask_index") == index:
                translations[position] = {
                    "subtask_index": index,
                    "intent": revised_intent.model_dump(
                        mode="json",
                        exclude_none=True,
                    ),
                }
                break
        else:
            raise WorkflowExecutionError(
                f"Retry Planner cannot find translation for subtask {index}"
            )
        retry_feedback: dict[str, JsonValue] = {
            "source": "retry_planner",
            "failed_attempt": state["current_attempt"],
            "previous_view": selected_view,
            "recommended_view": recommended_view,
            "retry_scope": next_retry_scope,
            "surface_retry_scope": (
                repair.surface_retry_scope.value
                if operation in {OperationMethod.SMEAR, OperationMethod.DRAW}
                and repair.surface_retry_scope is not None
                else None
            ),
            "segmentation_prompt": segmentation_prompt,
            "regenerate_svg_pattern": regenerate_svg_pattern,
            "analysis": repair.analysis,
            "minimum_effect": minimum_effect,
            "grader": grader,
        }
        attempt_history = list(state["attempt_history"])
        if (
            operation in {OperationMethod.SMEAR, OperationMethod.DRAW}
            and attempt_history
            and attempt_history[-1].get("attempt")
            == state["current_attempt"]
        ):
            latest_attempt = dict(attempt_history[-1])
            latest_attempt["retry_plan"] = retry_feedback
            attempt_history[-1] = latest_attempt
        emitter.node(
            status=WorkflowEventStatus.RETRYING,
            phase="completed",
            message=(
                f"Repair plan ready; retry scope {next_retry_scope}"
            ),
        )
        return {
            "subtasks": subtasks,
            "translations": translations,
            "retry_feedback": retry_feedback,
            "attempt_history": attempt_history,
            "retry_directive": None,
            "retry_scope": next_retry_scope,
            "restore_action": "retry",
            "llm_calls": [*state["llm_calls"], completion.metadata],
            **emitter.state_update(),
        }

    def _restore_checkpoint(
        self,
        state: SculptWorkflowState,
    ) -> StateUpdate:
        emitter = WorkflowEventEmitter(state, node="restore_checkpoint")
        emitter.node(
            status=WorkflowEventStatus.RUNNING,
            phase="started",
            message="Restoring the pre-subtask Blender state",
        )
        checkpoint_value = state["checkpoint_path"]
        if checkpoint_value is None:
            raise WorkflowExecutionError(
                "Cannot restore a subtask without checkpoint_path"
            )
        action = state["restore_action"]
        if action not in {"retry", "skip"}:
            raise WorkflowExecutionError(
                f"Unknown restore action {action!r}"
            )
        operation = OperationMethod(
            str(self._current_subtask(state)["operation_method"])
        )
        surface_operation = operation in {
            OperationMethod.SMEAR,
            OperationMethod.DRAW,
        }
        reuse_surface_segmentation = (
            action == "retry"
            and surface_operation
            and state["retry_scope"]
            in {"same_view", _SURFACE_RETRY_REUSE_SEGMENTATION}
            and isinstance(state["segmentation_result"], dict)
        )
        retry_feedback = state["retry_feedback"]
        regenerate_svg_pattern = bool(
            isinstance(retry_feedback, dict)
            and retry_feedback.get("failed_attempt")
            == state["current_attempt"]
            and retry_feedback.get("regenerate_svg_pattern") is True
        )
        preserve_draw_source = (
            action == "retry"
            and operation is OperationMethod.DRAW
            and not regenerate_svg_pattern
            and isinstance(state["draw_pattern_result"], dict)
            and isinstance(state["draw_trajectory_result"], dict)
        )
        manual_mask_overrides = dict(state["manual_mask_overrides"])
        if (
            action == "retry"
            and surface_operation
            and not reuse_surface_segmentation
        ):
            manual_mask_overrides.pop(
                _MANUAL_MASK_SMEAR
                if operation is OperationMethod.SMEAR
                else _MANUAL_MASK_DRAW,
                None,
            )
        restore_checkpoint_value = checkpoint_value
        locked_plan = state["locked_drag_plan"]
        if (
            action == "retry"
            and state["retry_scope"]
            in {_DRAG_RETRY_DOSE, _DRAG_RETRY_GESTURE}
            and isinstance(locked_plan, dict)
            and isinstance(
                locked_plan.get("prepared_checkpoint_path"),
                str,
            )
        ):
            restore_checkpoint_value = cast(
                str,
                locked_plan["prepared_checkpoint_path"],
            )
        self._load_blender_checkpoint(
            state,
            restore_checkpoint_value,
            label="load_blender_state",
        )
        next_attempt = (
            state["current_attempt"] + 1
            if action == "retry"
            else state["current_attempt"]
        )
        emitter.node(
            status=WorkflowEventStatus.SUCCESS,
            phase="completed",
            message=f"Restored pre-subtask state; action={action}",
        )
        preserve_view = action == "retry" and state["retry_scope"] in {
            "same_view",
            _DRAG_RETRY_DOSE,
            _DRAG_RETRY_GESTURE,
            _DRAG_RETRY_LOCALIZE,
            _SURFACE_RETRY_RESEGMENT,
            _SURFACE_RETRY_REUSE_SEGMENTATION,
        }
        preserve_locked_drag = (
            action == "retry"
            and state["retry_scope"]
            in {_DRAG_RETRY_DOSE, _DRAG_RETRY_GESTURE}
        )
        operation_prefix = (
            f"{state['run_id']}:subtask-"
            f"{state['current_subtask_index']}:"
        )
        return {
            "current_attempt": next_attempt,
            "view_screenshots": (
                state["view_screenshots"] if preserve_view else {}
            ),
            "view_segmentation_results": (
                state["view_segmentation_results"] if preserve_view else {}
            ),
            "valid_views": (
                list(state["valid_views"]) if preserve_view else []
            ),
            "selected_view": (
                state["selected_view"] if preserve_view else None
            ),
            "view_selection_reason": (
                state["view_selection_reason"] if preserve_view else None
            ),
            "view_selection_intervention": (
                state["view_selection_intervention"]
                if preserve_view
                else None
            ),
            "manual_mask_request": None,
            "manual_mask_intervention": (
                state["manual_mask_intervention"]
                if manual_mask_overrides
                else None
            ),
            "manual_mask_result": None,
            "manual_mask_overrides": manual_mask_overrides,
            "execution_preparation": None,
            "segmentation_result": (
                state["segmentation_result"]
                if reuse_surface_segmentation
                else None
            ),
            "quadloc_result": None,
            "drag_target_binding": None,
            "locked_drag_plan": (
                state["locked_drag_plan"]
                if preserve_locked_drag
                else None
            ),
            "drag_direction_plan": None,
            "part_segmentation_result": None,
            "draw_pattern_result": (
                state["draw_pattern_result"]
                if preserve_draw_source
                else None
            ),
            "draw_trajectory_result": (
                state["draw_trajectory_result"]
                if preserve_draw_source
                else None
            ),
            "draw_fitted_trajectory_result": None,
            "resolved_sculpt_plan": None,
            "stroke_plan_result": None,
            "viewport_focus": None,
            "full_before_screenshot": None,
            "baseline_screenshot_a": None,
            "baseline_screenshot_b": None,
            "minimum_effect_baseline": None,
            "after_screenshot": None,
            "full_after_screenshot": None,
            "settings_result": None,
            "execution_results": [],
            "execution_error": None,
            "applied_operation_ids": [
                operation_id
                for operation_id in state["applied_operation_ids"]
                if not operation_id.startswith(operation_prefix)
            ],
            "minimum_effect_result": None,
            "grader_result": None,
            "acceptance_decision": None,
            "execution_decision": None,
            "current_approval_id": None,
            **emitter.state_update(),
        }

    def _advance_subtask(
        self,
        state: SculptWorkflowState,
    ) -> StateUpdate:
        emitter = WorkflowEventEmitter(state, node="advance_subtask")
        if state["checkpoint_path"] is not None:
            self._lease(state).clear_checkpoint(state["run_id"])
        next_index = state["current_subtask_index"] + 1
        emitter.subtask(
            status=WorkflowEventStatus.SUCCESS,
            action="advanced",
            operation_method=None,
            message=f"Advanced to subtask index {next_index}",
        )
        return {
            "current_subtask_index": next_index,
            "current_attempt": 0,
            "checkpoint_path": None,
            "view_screenshots": {},
            "view_segmentation_results": {},
            "valid_views": [],
            "rejected_drag_anchor_views": [],
            "selected_view": None,
            "view_selection_reason": None,
            "view_selection_intervention": None,
            "manual_mask_request": None,
            "manual_mask_intervention": None,
            "manual_mask_result": None,
            "manual_mask_overrides": {},
            "execution_preparation": None,
            "segmentation_result": None,
            "quadloc_result": None,
            "drag_target_binding": None,
            "locked_drag_plan": None,
            "drag_direction_plan": None,
            "part_segmentation_result": None,
            "draw_pattern_result": None,
            "draw_trajectory_result": None,
            "draw_fitted_trajectory_result": None,
            "resolved_sculpt_plan": None,
            "stroke_plan_result": None,
            "viewport_focus": None,
            "full_before_screenshot": None,
            "baseline_screenshot_a": None,
            "baseline_screenshot_b": None,
            "minimum_effect_baseline": None,
            "after_screenshot": None,
            "full_after_screenshot": None,
            "settings_result": None,
            "execution_results": [],
            "execution_error": None,
            "minimum_effect_result": None,
            "grader_result": None,
            "acceptance_decision": None,
            "retry_feedback": None,
            "retry_directive": None,
            "retry_scope": None,
            "restore_action": None,
            "execution_decision": None,
            "current_approval_id": None,
            **emitter.state_update(),
        }

    def _finalize(self, state: SculptWorkflowState) -> StateUpdate:
        workflow_status = _final_workflow_status(state)
        event_status = (
            WorkflowEventStatus.COMPLETED
            if workflow_status in {"completed", "partially_completed"}
            else WorkflowEventStatus.REJECTED
            if workflow_status == "rejected"
            else WorkflowEventStatus.ERROR
        )
        snapshot = state["sculpt_viewport_ui_snapshot"]
        if snapshot is None:
            raise WorkflowExecutionError(
                "Cannot finalize without Sculpt viewport UI snapshot"
            )
        ui_restore_response = self._restore_sculpt_viewport_ui_snapshot(
            snapshot
        )
        state_path = (
            self._workflow_dir(state) / "workflow-state.json"
        ).resolve()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text("{}", encoding="utf-8")
        artifact = create_artifact(
            run_id=state["run_id"],
            workflow_dir=self._workflow_dir(state),
            path=state_path,
            kind=WorkflowArtifactKind.WORKFLOW_STATE,
            label="Final workflow state",
        )
        emitter = WorkflowEventEmitter(state, node="finalize")
        emitter.artifacts(
            artifact_ids=[artifact.artifact_id],
            message="Final workflow state is available",
        )
        emitter.workflow(
            status=event_status,
            workflow_status=workflow_status,
            message=f"Workflow finished with status {workflow_status}",
        )
        emitter.node(
            status=WorkflowEventStatus.SUCCESS,
            phase="completed",
            message=(
                "Sculpt viewport UI restored, final state persisted, and "
                "Blender lease released"
            ),
        )
        completion_message = AIMessage(
            content=(
                "The Blender Sculpt workflow finished with status "
                f"{workflow_status}. "
                f"Processed {len(state['subtask_results'])} subtask results."
            )
        )
        event_update = emitter.state_update()
        artifacts = merge_artifacts(state["artifacts"], [artifact])
        final_state = dict(state)
        final_state.update(
            {
                "state_artifact_path": artifact.relative_path,
                "workflow_status": workflow_status,
                "blender_session_lease": None,
                "sculpt_viewport_ui_restore_response": ui_restore_response,
                "messages": [*state["messages"], completion_message],
                "artifacts": artifacts,
                **event_update,
            }
        )
        try:
            for _ in range(3):
                state_path.write_text(
                    json.dumps(
                        _json_safe_state(final_state),
                        ensure_ascii=False,
                        allow_nan=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                refreshed = create_artifact(
                    run_id=state["run_id"],
                    workflow_dir=self._workflow_dir(state),
                    path=state_path,
                    kind=WorkflowArtifactKind.WORKFLOW_STATE,
                    label="Final workflow state",
                ).model_copy(update={"created_at": artifact.created_at})
                refreshed_artifacts = merge_artifacts(
                    state["artifacts"],
                    [refreshed],
                )
                if refreshed_artifacts == final_state["artifacts"]:
                    break
                final_state["artifacts"] = refreshed_artifacts
        except (OSError, TypeError, ValueError) as error:
            raise WorkflowExecutionError(
                f"Cannot persist final workflow state to {state_path}"
            ) from error
        if not self._lease(state).release(state["run_id"]):
            raise WorkflowExecutionError(
                "Workflow finalized but no longer owned its Blender session "
                "lease"
            )
        return {
            "state_artifact_path": artifact.relative_path,
            "workflow_status": workflow_status,
            "blender_session_lease": None,
            "sculpt_viewport_ui_restore_response": ui_restore_response,
            "messages": [completion_message],
            "artifacts": final_state["artifacts"],
            **event_update,
        }

    def _capture_view_set(
        self,
        state: SculptWorkflowState,
        directory: Path,
        views: Sequence[str],
        *,
        emitter: WorkflowEventEmitter,
        label_prefix: str,
    ) -> tuple[dict[str, JsonValue], dict[str, str]]:
        directory.mkdir(parents=True, exist_ok=True)
        screenshots: dict[str, JsonValue] = {}
        paths: dict[str, str] = {}
        for index, view in enumerate(views, start=1):
            record = self._change_and_capture(
                state,
                view,
                directory / f"{index:02d}-{view.lower()}.png",
                label=f"{label_prefix} {view} view",
            )
            screenshots[view] = record
            path = record.get("path")
            if not isinstance(path, str):
                raise WorkflowExecutionError(
                    f"Screenshot record for {view} is missing path"
                )
            paths[view] = str(self._run_path(state, path))
            emitter.progress(
                label="Standard view capture",
                current=index,
                total=len(views),
                unit="view",
                message=f"Captured {view} view",
            )
        return screenshots, paths

    def _change_and_capture(
        self,
        state: SculptWorkflowState,
        view: str,
        path: Path,
        *,
        label: str,
    ) -> dict[str, JsonValue]:
        view_response = _invoke_tool(
            self.dependencies.tools.change_view,
            {"view": view, "frame": self.config.workflow.frame},
            label=f"change_view({view})",
        )
        _rpc_result(view_response, f"change_view({view})")
        record = self._capture_current_view(state, path, label=label)
        record["view"] = view
        record["view_response"] = view_response
        return record

    def _focus_segmented_execution(
        self,
        state: SculptWorkflowState,
        *,
        segment_input: dict[str, JsonValue],
        segmentation_result: dict[str, JsonValue],
        attempt_dir: Path,
        workflow_dir: Path,
        selected_view: str,
    ) -> tuple[
        dict[str, JsonValue],
        dict[str, JsonValue],
        dict[str, JsonValue],
        list[WorkflowArtifact],
    ]:
        """Focus one cleaned mask and remap it without rerunning SAM3."""
        segment_payload = _result_object(
            segmentation_result,
            "segment_with_sam3",
        )
        source_mask_value = segment_payload.get("cleaned_mask_path")
        if not isinstance(source_mask_value, str):
            raise WorkflowExecutionError(
                "ROI focus requires cleaned_mask_path"
            )
        source_mask = self._run_path(state, source_mask_value)
        roi = cleaned_mask_roi(source_mask)
        width, height = _screenshot_dimensions(segment_input)
        metadata = _required_mapping(
            segment_input,
            "metadata",
            label="Segmentation screenshot",
        )
        focus_input: dict[str, JsonValue] = {
            "roi": roi,
            "image_width": width,
            "image_height": height,
            "margin_ratio": self.config.workflow.roi_focus.margin_ratio,
            "maximum_zoom_factor": (
                self.config.workflow.roi_focus.maximum_zoom_factor
            ),
        }
        for key in ("window_index", "area_index"):
            value = metadata.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                focus_input[key] = value
        focus_response = _invoke_tool(
            self.dependencies.tools.focus_viewport_roi,
            focus_input,
            label="focus_blender_viewport_roi",
        )
        focus_payload = _json_object_copy(
            _rpc_result(focus_response, "focus_blender_viewport_roi")
        )
        if focus_payload.get("schema_version") != "viewport-roi-focus/v1":
            raise WorkflowExecutionError(
                "focus_blender_viewport_roi returned an unsupported schema"
            )

        focus_dir = (attempt_dir / "roi-focus").resolve()
        try:
            focused_input = self._capture_current_view(
                state,
                focus_dir / "focused-segment-input.png",
                label=(
                    f"Subtask {state['current_subtask_index'] + 1} "
                    f"attempt {state['current_attempt']} focused "
                    f"{selected_view} view"
                ),
            )
            transformed = warp_cleaned_mask_to_focused_view(
                source_mask_path=source_mask,
                focused_image_path=self._run_path(
                    state,
                    str(focused_input["path"]),
                ),
                focus_result=cast(Mapping[str, object], focus_payload),
                output_dir=focus_dir / "mask",
                overlay_opacity=self.config.workflow.sam3_overlay_opacity,
            )
        except (WorkflowExecutionError, ValueError) as error:
            message = f"Cannot prepare the focused mask: {error}"
            try:
                self._apply_viewport_snapshot(
                    focus_payload,
                    snapshot_name="before",
                )
            except WorkflowExecutionError as restore_error:
                message = _combine_errors(
                    message,
                    f"Cannot roll back ROI focus: {restore_error}",
                )
            raise WorkflowExecutionError(message) from error
        updated_payload = _json_object_copy(segment_payload)
        for key in (
            "mask_path",
            "cleaned_mask_path",
            "cleaned_overlay_path",
            "overlay_path",
        ):
            value = segment_payload.get(key)
            if isinstance(value, str):
                updated_payload[f"source_{key}"] = value
        updated_payload["cleaned_mask_path"] = transformed.cleaned_mask_path
        updated_payload["cleaned_overlay_path"] = (
            transformed.cleaned_overlay_path
        )
        # Trajectory overlays must exactly match the focused coordinate system.
        updated_payload["overlay_path"] = transformed.cleaned_overlay_path
        updated_payload["focused_mask_transform_path"] = (
            transformed.metadata_path
        )
        updated_payload["viewport_focus"] = transformed.as_payload()
        updated_segmentation = {
            **segmentation_result,
            "result": updated_payload,
        }
        focus_payload["source_screenshot"] = segment_input
        focus_payload["focused_screenshot"] = focused_input
        focus_payload["mask_transform"] = transformed.as_payload()
        artifacts = _screenshot_artifacts({"focused": focused_input})
        artifacts.extend(
            _segment_artifacts(
                state,
                workflow_dir,
                {
                    "result": {
                        "cleaned_mask_path": (
                            transformed.cleaned_mask_path
                        ),
                        "cleaned_overlay_path": (
                            transformed.cleaned_overlay_path
                        ),
                        "focused_mask_transform_path": (
                            transformed.metadata_path
                        ),
                    }
                },
                label_prefix="Focused",
                metadata={"coordinate_space": "focused_viewport"},
            )
        )
        return (
            focused_input,
            updated_segmentation,
            focus_payload,
            artifacts,
        )

    def _select_segment_component_if_needed(
        self,
        state: SculptWorkflowState,
        *,
        image_path: Path,
        segmentation_result: dict[str, JsonValue],
        part_description: str,
        output_dir: Path,
        workflow_dir: Path,
    ) -> tuple[dict[str, JsonValue], list[WorkflowArtifact]]:
        """Use the Translator VLM only for a multi-component cleaned mask."""
        segment_payload = _result_object(
            segmentation_result,
            "segment_with_sam3",
        )
        mask_value = segment_payload.get("cleaned_mask_path")
        if not isinstance(mask_value, str):
            raise WorkflowExecutionError(
                "Mask component selection requires cleaned_mask_path"
            )
        cleaned_mask_path = self._run_path(state, mask_value)
        try:
            component_count = foreground_component_count(cleaned_mask_path)
        except MaskComponentSelectionError as error:
            raise WorkflowExecutionError(
                f"Cannot inspect cleaned mask components: {error}"
            ) from error
        if component_count <= 1:
            return segmentation_result, []

        selector = self._llm_bound_tool(
            state,
            fallback=self.dependencies.tools.select_mask_component,
            factory=(
                self.dependencies.mask_component_selector_tool_factory
            ),
            call_site="mask_component_selector",
        )
        selection_response = _invoke_tool(
            selector,
            {
                "image_path": str(image_path.resolve()),
                "cleaned_mask_path": str(cleaned_mask_path),
                "part_description": part_description,
                "overlay_opacity": (
                    self.config.workflow.sam3_overlay_opacity
                ),
                "output_dir": str(output_dir.resolve()),
            },
            label="select_mask_component",
        )
        selection = _result_object(
            selection_response,
            "select_mask_component",
        )
        selected_paths: dict[str, Path] = {}
        for key, label in (
            ("selected_mask_path", "Selected component mask"),
            ("selected_overlay_path", "Selected component overlay"),
        ):
            value = selection.get(key)
            if not isinstance(value, str):
                raise WorkflowExecutionError(f"{label} path is missing")
            path = self._run_path(state, value)
            if not path.is_file():
                raise WorkflowExecutionError(
                    f"{label} is not a file: {path}"
                )
            selected_paths[key] = path
        selected_mask = selected_paths["selected_mask_path"]
        selected_overlay = selected_paths["selected_overlay_path"]
        try:
            selected_count = foreground_component_count(selected_mask)
        except MaskComponentSelectionError as error:
            raise WorkflowExecutionError(
                f"Cannot validate selected mask component: {error}"
            ) from error
        if selected_count != 1:
            raise WorkflowExecutionError(
                "select_mask_component must return exactly one foreground "
                f"region, got {selected_count}"
            )
        updated_payload = _json_object_copy(segment_payload)
        for key in (
            "cleaned_mask_path",
            "cleaned_overlay_path",
            "overlay_path",
        ):
            value = segment_payload.get(key)
            if isinstance(value, str):
                updated_payload[f"pre_component_selection_{key}"] = value
        updated_payload["cleaned_mask_path"] = str(selected_mask)
        updated_payload["cleaned_overlay_path"] = str(selected_overlay)
        updated_payload["overlay_path"] = str(selected_overlay)
        updated_payload["mask_component_selection"] = selection
        updated = {
            **segmentation_result,
            "result": updated_payload,
        }
        return (
            updated,
            _mask_component_selection_artifacts(
                state,
                workflow_dir,
                selection_response,
            ),
        )

    def _apply_viewport_snapshot(
        self,
        focus: Mapping[str, JsonValue],
        *,
        snapshot_name: str,
    ) -> JsonValue:
        """Apply one named ROI-focus snapshot with strict RPC validation."""
        snapshot = focus.get(snapshot_name)
        if not isinstance(snapshot, dict):
            raise WorkflowExecutionError(
                f"Viewport focus is missing {snapshot_name} snapshot"
            )
        response = _invoke_tool(
            self.dependencies.tools.restore_viewport_state,
            {
                "snapshot": snapshot,
                "require_region_match": True,
            },
            label=f"restore_blender_viewport_state({snapshot_name})",
        )
        result = _rpc_result(
            response,
            f"restore_blender_viewport_state({snapshot_name})",
        )
        if (
            result.get("schema_version") != "viewport-state-restore/v1"
            or result.get("status") != "restored"
        ):
            raise WorkflowExecutionError(
                "restore_blender_viewport_state did not confirm restoration"
            )
        return response

    def _capture_current_view(
        self,
        state: SculptWorkflowState,
        path: Path,
        *,
        label: str,
        redraw: bool = True,
    ) -> dict[str, JsonValue]:
        resolved = path.resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        screenshot_response = _invoke_tool(
            self.dependencies.tools.get_screenshot,
            {
                "output": "file",
                "filepath": str(resolved),
                "redraw": redraw,
            },
            label="get_screenshot",
        )
        metadata = _rpc_result(screenshot_response, "get_screenshot")
        if not resolved.is_file() or resolved.stat().st_size == 0:
            raise WorkflowExecutionError(
                f"Blender screenshot was not created at {resolved}"
            )
        artifact = create_artifact(
            run_id=state["run_id"],
            workflow_dir=self._workflow_dir(state),
            path=resolved,
            kind=WorkflowArtifactKind.SCREENSHOT,
            label=label,
        )
        metadata.pop("filepath", None)
        return {
            "path": artifact.relative_path,
            "artifact": artifact.model_dump(mode="json"),
            "metadata": metadata,
        }

    def _warm_current_view(self) -> None:
        """Advance redraws, then leave a stable framebuffer for capture."""
        redraw_count = (
            self.config.workflow.minimum_effect.capture_warmup_redraws
        )
        with tempfile.TemporaryDirectory(
            prefix="agentic_geometry_viewport_warmup_"
        ) as directory:
            root = Path(directory)
            for index in range(1, redraw_count + 1):
                path = root / f"redraw-{index:02d}.png"
                response = _invoke_tool(
                    self.dependencies.tools.get_screenshot,
                    {
                        "output": "file",
                        "filepath": str(path),
                        "redraw": True,
                    },
                    label="get_screenshot(viewport_warmup)",
                )
                _rpc_result(response, "get_screenshot(viewport_warmup)")
                if not path.is_file() or path.stat().st_size == 0:
                    raise WorkflowExecutionError(
                        "Viewport warmup screenshot was not created"
                    )

    def _attempt_evidence_images(
        self,
        state: SculptWorkflowState,
    ) -> tuple[list[str], list[str]]:
        """Return a stable, labeled image manifest for visual reviewers."""
        paths: list[str] = []
        labels: list[str] = []
        seen: set[Path] = set()

        def add(value: object, label: str) -> None:
            if not isinstance(value, str):
                return
            path = self._run_path(state, value)
            if not path.is_file() or path in seen:
                return
            seen.add(path)
            paths.append(str(path))
            labels.append(label)

        focused_before = _required_screenshot(
            state["baseline_screenshot_a"],
            "baseline A",
        )
        focused_after = _required_screenshot(
            state["after_screenshot"],
            "after",
        )
        full_before = (
            state["full_before_screenshot"]
            if isinstance(state["full_before_screenshot"], dict)
            else focused_before
        )
        full_after = (
            state["full_after_screenshot"]
            if isinstance(state["full_after_screenshot"], dict)
            else focused_after
        )
        add(full_before.get("path"), "Full before screenshot")
        add(full_after.get("path"), "Full after screenshot")
        add(focused_before.get("path"), "Focused before screenshot")
        add(focused_after.get("path"), "Focused after screenshot")

        minimum = state["minimum_effect_result"]
        if isinstance(minimum, dict):
            artifact_paths = minimum.get("artifact_paths")
            if isinstance(artifact_paths, dict):
                for key, label in (
                    ("before_roi_path", "Target-region before crop"),
                    ("after_roi_path", "Target-region after crop"),
                    (
                        "difference_heatmap_path",
                        "Target-only difference heatmap",
                    ),
                    (
                        "evaluation_mask_path",
                        "Stroke-footprint evaluation mask",
                    ),
                ):
                    add(artifact_paths.get(key), label)

        target_binding = state["drag_target_binding"]
        if isinstance(target_binding, dict):
            add(
                target_binding.get("anchor_overlay_path"),
                "Bound target component and mouse-down anchor",
            )
            add(
                target_binding.get("component_mask_path"),
                "Bound target component mask",
            )

        stroke_result = state["stroke_plan_result"]
        if isinstance(stroke_result, dict):
            result = stroke_result.get("result")
            if isinstance(result, dict):
                method = OperationMethod(
                    str(self._current_subtask(state)["operation_method"])
                )
                add(
                    result.get("visualization_path"),
                    (
                        "Actual Drag trajectory overlay"
                        if method is OperationMethod.DRAG
                        else "Actual Draw trajectory overlay"
                        if method is OperationMethod.DRAW
                        else "Actual Sculpt trajectory overlay"
                    ),
                )

        segment = _segment_context(
            state["segmentation_result"],
            state["stroke_plan_result"],
        )
        if segment is not None:
            for key, label in (
                ("overlay_path", "SAM3 segmentation overlay"),
                ("cleaned_mask_path", "Noise-cleaned SAM3 mask"),
                ("mask_path", "Raw SAM3 semantic mask"),
            ):
                add(segment.get(key), label)
        return paths, labels

    def _current_subtask(
        self,
        state: SculptWorkflowState,
    ) -> dict[str, JsonValue]:
        index = state["current_subtask_index"]
        try:
            value = state["subtasks"][index]
        except IndexError as error:
            raise WorkflowExecutionError(
                f"Missing subtask at index {index}"
            ) from error
        return dict(value)

    def _current_intent(
        self,
        state: SculptWorkflowState,
    ) -> dict[str, JsonValue]:
        index = state["current_subtask_index"]
        for translation in state["translations"]:
            if translation.get("subtask_index") != index:
                continue
            intent = translation.get("intent")
            if isinstance(intent, dict):
                return cast(dict[str, JsonValue], dict(intent))
        raise WorkflowExecutionError(
            f"Missing translated intent for subtask {index}"
        )

    def _attempt_dir(self, state: SculptWorkflowState) -> Path:
        return (
            self._workflow_dir(state)
            / "subtasks"
            / f"subtask-{state['current_subtask_index'] + 1:03d}"
            / f"attempt-{state['current_attempt']:02d}"
        ).resolve()

    @staticmethod
    def _route_prepare(state: SculptWorkflowState) -> str:
        return (
            "finish"
            if state["current_subtask_index"] >= len(state["subtasks"])
            else "save"
        )

    def _route_executor(self, state: SculptWorkflowState) -> str:
        method = OperationMethod(
            str(self._current_subtask(state)["operation_method"])
        )
        return (
            "effect"
            if method in {
                OperationMethod.SMEAR,
                OperationMethod.DRAG,
                OperationMethod.DRAW,
            }
            else "advance"
        )

    @staticmethod
    def _route_view_selector(state: SculptWorkflowState) -> str:
        if state["selected_view"] is not None:
            return "execute"
        intervention = state["view_selection_intervention"]
        if isinstance(intervention, dict) and intervention.get(
            "status"
        ) == "pending":
            return "manual"
        raise WorkflowExecutionError(
            "View Selector produced neither a selected view nor a manual "
            "selection request"
        )

    @staticmethod
    def _route_manual_view_selection(state: SculptWorkflowState) -> str:
        return "execute" if state["selected_view"] is not None else "advance"

    def _route_execution_dispatch(
        self,
        state: SculptWorkflowState,
    ) -> str:
        method = OperationMethod(
            str(self._current_subtask(state)["operation_method"])
        )
        if method is not OperationMethod.DRAG:
            return "execute"
        if (
            state["retry_scope"]
            in {_DRAG_RETRY_DOSE, _DRAG_RETRY_GESTURE}
            and isinstance(state["locked_drag_plan"], dict)
        ):
            return "execute"
        context = state["execution_preparation"]
        if not isinstance(context, dict):
            return "drag_context"
        identity = (
            context.get("subtask_index") == state["current_subtask_index"]
            and context.get("attempt") == state["current_attempt"]
            and context.get("selected_view") == state["selected_view"]
            and context.get("operation_method") == OperationMethod.DRAG.value
        )
        if not identity:
            return "drag_context"
        stage = context.get("stage")
        if stage == _DRAG_PREPARATION_CHANGED_PART:
            return "drag_changed_part"
        if stage == _DRAG_PREPARATION_QUADLOC:
            return "drag_quadloc"
        if stage in {
            _DRAG_PREPARATION_FINAL,
            _DRAG_PREPARATION_COMPLETED,
        }:
            return "execute"
        raise WorkflowExecutionError(
            f"Unknown Drag preparation stage {stage!r}"
        )

    @staticmethod
    def _route_drag_changed_part(state: SculptWorkflowState) -> str:
        if state["manual_mask_request"] is not None:
            return "manual_mask"
        if state["restore_action"] is not None:
            return "restore"
        return "quadloc"

    @staticmethod
    def _route_drag_quadloc(state: SculptWorkflowState) -> str:
        return "restore" if state["restore_action"] is not None else "execute"

    @staticmethod
    def _route_prepared_execution(state: SculptWorkflowState) -> str:
        if state["manual_mask_request"] is not None:
            return "manual_mask"
        return "restore" if state["restore_action"] else "approve"

    @staticmethod
    def _route_prepare_manual_mask(state: SculptWorkflowState) -> str:
        intervention = state["manual_mask_intervention"]
        return (
            "paint"
            if isinstance(intervention, dict)
            and intervention.get("status") == "awaiting_paint"
            else "advance"
        )

    @staticmethod
    def _route_manual_mask_input(state: SculptWorkflowState) -> str:
        intervention = state["manual_mask_intervention"]
        return (
            "review"
            if isinstance(intervention, dict)
            and intervention.get("status") == "awaiting_review"
            else "advance"
        )

    @staticmethod
    def _route_manual_mask_review(state: SculptWorkflowState) -> str:
        intervention = state["manual_mask_intervention"]
        if not isinstance(intervention, dict):
            return "advance"
        status = intervention.get("status")
        if status == "confirmed":
            return "execute"
        if status == "awaiting_paint":
            return "redraw"
        return "advance"

    @staticmethod
    def _route_approval(state: SculptWorkflowState) -> str:
        if state["execution_decision"] == "approve":
            return "execute"
        return "restore" if state["restore_action"] == "skip" else "advance"

    @staticmethod
    def _route_minimum_effect(state: SculptWorkflowState) -> str:
        result = state["minimum_effect_result"]
        return (
            "grade"
            if isinstance(result, dict) and result.get("verdict") == "VISIBLE"
            else "restore"
        )

    @staticmethod
    def _route_acceptance(state: SculptWorkflowState) -> str:
        decision = state["acceptance_decision"]
        if not isinstance(decision, dict):
            raise WorkflowExecutionError(
                "Acceptance Gate did not persist a decision"
            )
        action = decision.get("action")
        if action not in {"advance", "retry", "restore"}:
            raise WorkflowExecutionError(
                f"Unknown acceptance action {action!r}"
            )
        return str(action)

    @staticmethod
    def _route_restore(state: SculptWorkflowState) -> str:
        if state["restore_action"] != "retry":
            return "advance"
        return (
            "same_view"
            if state["retry_scope"]
            in {
                "same_view",
                _DRAG_RETRY_DOSE,
                _DRAG_RETRY_GESTURE,
                _DRAG_RETRY_LOCALIZE,
                _SURFACE_RETRY_RESEGMENT,
                _SURFACE_RETRY_REUSE_SEGMENTATION,
            }
            else "reselect"
        )


def _invoke_tool(
    tool: BaseTool,
    payload: dict[str, JsonValue],
    *,
    label: str,
) -> dict[str, JsonValue]:
    return _validate_tool_result(
        _invoke_raw_tool(tool, payload, label=label),
        label=label,
    )


def _invoke_sam3_tool(
    tool: BaseTool,
    payload: dict[str, JsonValue],
    *,
    label: str,
    call_site: str,
    image_path: str,
    prompt: str,
) -> dict[str, JsonValue]:
    result = _invoke_raw_tool(tool, payload, label=label)
    sam3_error = result.get("sam3_error")
    if _is_empty_sam3_error(sam3_error):
        raise _Sam3NoSegmentation(
            f"SAM3 produced no mask for {prompt}",
            call_site=call_site,
            image_path=image_path,
            prompt=prompt,
        )
    return _validate_tool_result(result, label=label)


def _invoke_quadloc_tool(
    tool: BaseTool,
    payload: dict[str, JsonValue],
    *,
    image_path: str,
) -> dict[str, JsonValue]:
    label = "quadloc"
    result = _invoke_raw_tool(tool, payload, label=label)
    error = result.get("quadloc_error")
    if isinstance(error, dict) and error.get("type") == "no_model_mask":
        raise _Sam3NoSegmentation(
            "QuadLoc could not segment the complete model",
            call_site=_QUADLOC_MODEL_MASK,
            image_path=image_path,
            prompt="3D model",
        )
    return _validate_tool_result(result, label=label)


def _invoke_part_segmentation_tool(
    tool: BaseTool,
    payload: dict[str, JsonValue],
    *,
    image_path: str,
    prompt: str,
) -> dict[str, JsonValue]:
    label = "part_segmentation_with_sam3"
    result = _invoke_raw_tool(tool, payload, label=label)
    error = result.get("part_segmentation_error")
    if (
        isinstance(error, dict)
        and error.get("type") == "parent_mask_unavailable"
    ):
        raise _Sam3NoSegmentation(
            f"SAM3 produced no changed-part mask for {prompt}",
            call_site=_MANUAL_MASK_CHANGED_PART,
            image_path=image_path,
            prompt=prompt,
        )
    return _validate_tool_result(result, label=label)


def _invoke_raw_tool(
    tool: BaseTool,
    payload: dict[str, JsonValue],
    *,
    label: str,
) -> dict[str, JsonValue]:
    try:
        value = tool.invoke(payload)
    except Exception as error:
        raise WorkflowExecutionError(f"{label} failed: {error}") from error
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as error:
            raise WorkflowExecutionError(
                f"{label} returned an unexpected string: {value[:1000]}"
            ) from error
        value = decoded
    if not isinstance(value, dict):
        raise WorkflowExecutionError(
            f"{label} returned a non-object result"
        )
    return cast(dict[str, JsonValue], value)


def _validate_tool_result(
    result: dict[str, JsonValue],
    *,
    label: str,
) -> dict[str, JsonValue]:
    for key in (
        "bridge_error",
        "sam3_error",
        "stroke_planning_error",
        "quadloc_error",
        "part_segmentation_error",
        "mask_component_selection_error",
        "generate_svg_pattern_error",
        "svg_trajectory_error",
        "trajectory_fit_error",
        "text_trajectory_error",
        "error",
    ):
        if key in result:
            raise WorkflowExecutionError(
                f"{label} failed: "
                f"{json.dumps(result[key], ensure_ascii=False)}"
            )
    if result.get("status") == "error":
        raise WorkflowExecutionError(
            f"{label} failed: {json.dumps(result, ensure_ascii=False)}"
        )
    return result


def _is_empty_sam3_error(value: JsonValue | None) -> bool:
    if not isinstance(value, dict):
        return False
    message = value.get("message")
    return (
        value.get("type") == "mask_cleanup_error"
        and isinstance(message, str)
        and "no foreground" in message.casefold()
    )


def _rpc_result(
    response: Mapping[str, JsonValue],
    label: str,
) -> dict[str, JsonValue]:
    result = response.get("result")
    if not isinstance(result, dict):
        raise WorkflowExecutionError(
            f"{label} response is missing an object result"
        )
    return cast(dict[str, JsonValue], dict(result))


def _result_object(
    tool_result: Mapping[str, JsonValue],
    label: str,
) -> dict[str, JsonValue]:
    result = tool_result.get("result")
    if not isinstance(result, dict):
        raise WorkflowExecutionError(
            f"{label} result is missing an object result"
        )
    return cast(dict[str, JsonValue], dict(result))


def _changed_part_segmentation_from_part_result(
    part_result: Mapping[str, JsonValue],
    *,
    image_path: str,
    part_description: str,
) -> dict[str, JsonValue]:
    """Expose the Pose parent mask through the canonical segmentation shape."""
    payload = _result_object(
        part_result,
        "part_segmentation_with_sam3",
    )
    parent = _required_mapping(
        payload,
        "parent",
        label="Part segmentation result",
    )
    cleaned_mask = parent.get("cleaned_mask_path")
    if not isinstance(cleaned_mask, str):
        raise WorkflowExecutionError(
            "Part segmentation parent is missing cleaned_mask_path"
        )
    segment_response = parent.get("segment_response")
    source_payload: dict[str, JsonValue] = {}
    service: JsonValue | None = None
    if isinstance(segment_response, dict):
        source_result = segment_response.get("result")
        if isinstance(source_result, dict):
            source_payload = _json_object_copy(source_result)
        service = segment_response.get("service")
    overlay = parent.get("sam3_overlay_path")
    result: dict[str, JsonValue] = {
        **source_payload,
        "source": "part_segmentation_parent",
        "image_path": image_path,
        "prompt": parent.get("sam3_prompt", part_description),
        "mask_path": source_payload.get("mask_path", cleaned_mask),
        "cleaned_mask_path": cleaned_mask,
        "component_count": parent.get("component_count"),
        "component_selection": parent.get("component_selection"),
        "foreground_pixels": parent.get("foreground_pixels"),
    }
    if isinstance(overlay, str):
        result["overlay_path"] = overlay
        result["cleaned_overlay_path"] = overlay
    response: dict[str, JsonValue] = {
        "input": {
            "image_path": image_path,
            "prompt": part_description,
        },
        "result": result,
    }
    if service is not None:
        response["service"] = service
    return response


def _json_object_copy(
    value: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    """Copy one JSON object without retaining nested mutable references."""
    return cast(
        dict[str, JsonValue],
        json.loads(json.dumps(dict(value), ensure_ascii=False)),
    )


def _json_object_field(
    payload: Mapping[str, JsonValue],
    key: str,
    *,
    label: str,
) -> dict[str, JsonValue]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise WorkflowExecutionError(f"{label} is missing object {key}")
    return _json_object_copy(cast(Mapping[str, JsonValue], value))


def _surface_segmentation_prompt(
    state: SculptWorkflowState,
    intent: Mapping[str, JsonValue],
) -> str:
    """Resolve the semantic prompt for one Smear or Draw attempt."""
    if state["retry_scope"] != _SURFACE_RETRY_RESEGMENT:
        prompt = intent.get("part_to_be_changed")
    else:
        feedback = state["retry_feedback"]
        prompt = (
            feedback.get("segmentation_prompt")
            if isinstance(feedback, dict)
            else None
        )
    if not isinstance(prompt, str) or not prompt.strip():
        raise WorkflowExecutionError(
            "Surface execution has no valid SAM3 segmentation prompt"
        )
    return " ".join(prompt.strip().split())


def _reused_surface_segmentation(
    state: SculptWorkflowState,
    *,
    image_path: str,
    prompt: str,
) -> dict[str, JsonValue] | None:
    """Restore the pre-focus mask when a retry keeps segmentation valid."""
    if state["retry_scope"] not in {
        "same_view",
        _SURFACE_RETRY_REUSE_SEGMENTATION,
    }:
        return None
    stored = state["segmentation_result"]
    if not isinstance(stored, dict):
        raise WorkflowExecutionError(
            "Surface retry requested mask reuse without segmentation state"
        )
    reused = _json_object_copy(stored)
    payload = _result_object(reused, "reused surface segmentation")
    for key in (
        "mask_path",
        "cleaned_mask_path",
        "cleaned_overlay_path",
        "overlay_path",
    ):
        source_value = payload.get(f"source_{key}")
        if isinstance(source_value, str):
            payload[key] = source_value
    payload.pop("focused_mask_transform_path", None)
    payload.pop("viewport_focus", None)
    payload["image_path"] = image_path
    payload["prompt"] = prompt
    payload["retry_reuse"] = {
        "source_attempt": state["current_attempt"] - 1,
        "coordinate_space": "selected_view_before_roi_focus",
    }
    reused["input"] = {
        "image_path": image_path,
        "prompt": prompt,
    }
    reused["result"] = payload
    return reused


def _reused_draw_pattern_source(
    state: SculptWorkflowState,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]] | None:
    """Reuse mask-independent SVG source trajectories when permitted."""
    feedback = state["retry_feedback"]
    if (
        isinstance(feedback, dict)
        and feedback.get("failed_attempt")
        == state["current_attempt"] - 1
        and feedback.get("regenerate_svg_pattern") is True
    ):
        return None
    pattern = state["draw_pattern_result"]
    trajectory = state["draw_trajectory_result"]
    if not isinstance(pattern, dict) or not isinstance(trajectory, dict):
        return None
    return _json_object_copy(pattern), _json_object_copy(trajectory)


def _required_mapping(
    payload: Mapping[str, JsonValue],
    key: str,
    *,
    label: str,
) -> Mapping[str, JsonValue]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise WorkflowExecutionError(f"{label} is missing object {key}")
    return cast(Mapping[str, JsonValue], value)


def _positive_number_field(
    payload: Mapping[str, JsonValue],
    key: str,
    *,
    label: str,
) -> float:
    value = payload.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise WorkflowExecutionError(
            f"{label} field {key} must be a positive number"
        )
    return float(value)


def _draw_retry_policy(
    directive: Mapping[str, JsonValue] | None,
    *,
    maximum_pass_count: int,
) -> tuple[float, int, float]:
    """Map one-sided effect repair into fixed-strength Draw execution."""
    values = {} if directive is None else directive

    def number(key: str, default: float) -> float:
        value = values.get(key, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 1.0
        ):
            raise WorkflowExecutionError(
                f"Draw retry field {key} must be a finite number >= 1"
            )
        return float(value)

    size_multiplier = number("size_multiplier", 1.0)
    dose_multiplier = number("dose_multiplier", 1.0)
    pass_value = number("pass_count", 1.0)
    if not float(pass_value).is_integer():
        raise WorkflowExecutionError(
            "Draw retry field pass_count must be an integer"
        )
    pass_count = min(
        maximum_pass_count,
        max(int(pass_value), math.ceil(dose_multiplier)),
    )
    return size_multiplier, pass_count, dose_multiplier


def _combine_errors(current: str | None, added: str) -> str:
    """Preserve the primary failure while appending cleanup failures."""
    return added if current is None else f"{current}; {added}"


def _write_workflow_text(path: Path, value: str) -> None:
    """Persist one UTF-8 Draw artifact with workflow-style errors."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
    except OSError as error:
        raise WorkflowExecutionError(
            f"Cannot persist Draw artifact at {path}"
        ) from error


def _write_workflow_json(
    path: Path,
    value: Mapping[str, JsonValue],
) -> None:
    """Persist deterministic Draw metadata without non-JSON values."""
    try:
        rendered = json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
    except (TypeError, ValueError) as error:
        raise WorkflowExecutionError(
            "Draw artifact contains a non-JSON value"
        ) from error
    _write_workflow_text(path, rendered)


def _relative_optional_object(
    value: Mapping[str, JsonValue] | None,
    workflow_dir: Path,
) -> dict[str, JsonValue] | None:
    if value is None:
        return None
    return cast(
        dict[str, JsonValue],
        _relativize_run_paths(dict(value), workflow_dir),
    )


def _operator_calls(
    plan_result: Mapping[str, JsonValue],
) -> list[dict[str, JsonValue]]:
    result = plan_result.get("result")
    if not isinstance(result, dict):
        raise WorkflowExecutionError(
            "plan_sculpt_strokes result is missing result"
        )
    stroke_plan = result.get("stroke_plan")
    if not isinstance(stroke_plan, dict):
        raise WorkflowExecutionError(
            "plan_sculpt_strokes result is missing stroke_plan"
        )
    calls = stroke_plan.get("operator_calls")
    if not isinstance(calls, list) or not calls:
        raise WorkflowExecutionError(
            "plan_sculpt_strokes generated no operator calls"
        )
    normalized: list[dict[str, JsonValue]] = []
    for call in calls:
        if not isinstance(call, dict):
            raise WorkflowExecutionError(
                "stroke_plan contains an invalid operator call"
            )
        normalized.append(cast(dict[str, JsonValue], dict(call)))
    return normalized


def _operator_tool_input(
    operator_call: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    kwargs = operator_call.get("kwargs")
    context = operator_call.get("context")
    if not isinstance(kwargs, dict) or not isinstance(context, dict):
        raise WorkflowExecutionError(
            "operator call is missing kwargs or context"
        )
    tool_input = cast(dict[str, JsonValue], dict(kwargs))
    for key in ("window_index", "area_index"):
        value = context.get(key)
        if value is not None:
            tool_input[key] = value
    return tool_input


def _draw_finishing_tool_input(
    operator_call: Mapping[str, JsonValue],
    *,
    brush_size: int,
) -> dict[str, JsonValue]:
    """Reuse one Draw gesture as a distinct low-strength Smooth gesture."""
    tool_input = _operator_tool_input(operator_call)
    stroke = tool_input.get("stroke")
    if not isinstance(stroke, list) or not stroke:
        raise WorkflowExecutionError(
            "Draw finishing smooth pass has no stroke elements"
        )
    smoothed_stroke: list[JsonValue] = []
    for element in stroke:
        if not isinstance(element, dict):
            raise WorkflowExecutionError(
                "Draw finishing smooth pass contains an invalid stroke"
            )
        updated = cast(dict[str, JsonValue], dict(element))
        original_name = updated.get("name")
        if not isinstance(original_name, str):
            raise WorkflowExecutionError(
                "Draw finishing smooth stroke has no name"
            )
        updated["name"] = f"smooth-finish-{original_name}"[:128]
        updated["size"] = float(brush_size)
        smoothed_stroke.append(updated)
    tool_input["stroke"] = smoothed_stroke
    return tool_input


def _required_screenshot(
    value: dict[str, JsonValue] | None,
    label: str,
) -> dict[str, JsonValue]:
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise WorkflowExecutionError(
            f"Workflow is missing the {label} screenshot"
        )
    return value


def _segment_context(
    segment_result: dict[str, JsonValue] | None,
    stroke_plan_result: dict[str, JsonValue] | None = None,
) -> dict[str, JsonValue] | None:
    if not isinstance(segment_result, dict):
        return None
    result = segment_result.get("result")
    if not isinstance(result, dict):
        return None
    context: dict[str, JsonValue] = {}
    for key in (
        "prompt",
        "overlay_path",
        "mask_path",
        "cleaned_mask_path",
        "cleaned_overlay_path",
        "source_cleaned_mask_path",
        "source_cleaned_overlay_path",
        "source_overlay_path",
        "pre_component_selection_cleaned_mask_path",
        "pre_component_selection_cleaned_overlay_path",
        "pre_component_selection_overlay_path",
        "mask_component_selection",
        "focused_mask_transform_path",
        "viewport_focus",
        "retry_reuse",
        "metadata_path",
        "metadata",
    ):
        if key in result:
            context[key] = result[key]
    if isinstance(stroke_plan_result, dict):
        plan_result = stroke_plan_result.get("result")
        if isinstance(plan_result, dict):
            if "stroke_plan_path" in plan_result:
                context["stroke_plan_path"] = plan_result[
                    "stroke_plan_path"
                ]
            stroke_plan = plan_result.get("stroke_plan")
            if isinstance(stroke_plan, dict) and "summary" in stroke_plan:
                context["stroke_summary"] = stroke_plan["summary"]
    return context or None


def _screenshot_paths(
    screenshots: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    paths: dict[str, JsonValue] = {}
    for view, value in screenshots.items():
        if isinstance(value, dict) and isinstance(value.get("path"), str):
            paths[view] = value["path"]
    return paths


def _screenshot_artifacts(
    screenshots: Mapping[str, JsonValue],
) -> list[WorkflowArtifact]:
    artifacts: list[WorkflowArtifact] = []
    for value in screenshots.values():
        if not isinstance(value, dict):
            continue
        artifact = value.get("artifact")
        if isinstance(artifact, dict):
            artifacts.append(WorkflowArtifact.model_validate(artifact))
    return artifacts


def _segment_artifacts(
    state: SculptWorkflowState,
    workflow_dir: Path,
    segment_result: Mapping[str, JsonValue],
    *,
    label_prefix: str | None = None,
    metadata: dict[str, JsonValue] | None = None,
) -> list[WorkflowArtifact]:
    result = segment_result.get("result")
    if not isinstance(result, dict):
        return []
    definitions = {
        "mask_path": (
            WorkflowArtifactKind.MASK,
            "SAM3 semantic mask",
        ),
        "cleaned_mask_path": (
            WorkflowArtifactKind.MASK,
            "Noise-cleaned Sculpt mask",
        ),
        "cleaned_overlay_path": (
            WorkflowArtifactKind.OVERLAY,
            "Noise-cleaned Sculpt mask overlay",
        ),
        "overlay_path": (
            WorkflowArtifactKind.OVERLAY,
            "SAM3 segmentation overlay",
        ),
        "metadata_path": (
            WorkflowArtifactKind.METADATA,
            "SAM3 segmentation metadata",
        ),
        "focused_mask_transform_path": (
            WorkflowArtifactKind.METADATA,
            "Mask affine transform",
        ),
    }
    artifacts: list[WorkflowArtifact] = []
    seen_paths: set[Path] = set()
    for key, (kind, label) in definitions.items():
        path_value = result.get(key)
        if not isinstance(path_value, str):
            continue
        path = Path(path_value)
        path = (
            path if path.is_absolute() else workflow_dir / path
        ).resolve()
        try:
            path.relative_to(workflow_dir.resolve())
        except ValueError:
            continue
        if not path.is_file() or path in seen_paths:
            continue
        artifacts.append(
            create_artifact(
                run_id=state["run_id"],
                workflow_dir=workflow_dir,
                path=path,
                kind=kind,
                label=(
                    f"{label_prefix} {label}"
                    if label_prefix is not None
                    else label
                ),
                metadata=metadata,
            )
        )
        seen_paths.add(path)
    return artifacts


def _manual_mask_artifacts(
    state: SculptWorkflowState,
    workflow_dir: Path,
    result: ManualMaskRasterResult,
    *,
    call_site: str,
    revision: int,
) -> list[WorkflowArtifact]:
    """Expose one paint mask, clipped mask, and review overlay safely."""
    definitions = (
        (
            Path(result.painted_mask_path),
            WorkflowArtifactKind.MASK,
            "User-painted mask",
        ),
        (
            Path(result.cleaned_mask_path),
            WorkflowArtifactKind.MASK,
            "Model-clipped manual mask",
        ),
        (
            Path(result.cleaned_overlay_path),
            WorkflowArtifactKind.OVERLAY,
            "Model-clipped manual mask overlay",
        ),
    )
    return [
        create_artifact(
            run_id=state["run_id"],
            workflow_dir=workflow_dir,
            path=path,
            kind=kind,
            label=label,
            metadata={
                "stage": "manual_mask",
                "call_site": call_site,
                "revision": revision,
            },
        )
        for path, kind, label in definitions
    ]


def _mask_component_selection_artifacts(
    state: SculptWorkflowState,
    workflow_dir: Path,
    selection_response: Mapping[str, JsonValue],
) -> list[WorkflowArtifact]:
    """Register the numbered decision image and selected mask artifacts."""
    result = selection_response.get("result")
    if not isinstance(result, dict):
        return []
    definitions = {
        "numbered_overlay_path": (
            WorkflowArtifactKind.OVERLAY,
            "Numbered mask components",
        ),
        "selected_mask_path": (
            WorkflowArtifactKind.MASK,
            "VLM-selected Sculpt mask component",
        ),
        "selected_overlay_path": (
            WorkflowArtifactKind.OVERLAY,
            "VLM-selected Sculpt mask overlay",
        ),
        "metadata_path": (
            WorkflowArtifactKind.METADATA,
            "Mask component selection metadata",
        ),
    }
    artifacts: list[WorkflowArtifact] = []
    for key, (kind, label) in definitions.items():
        value = result.get(key)
        if not isinstance(value, str):
            continue
        path = Path(value)
        path = (
            path if path.is_absolute() else workflow_dir / path
        ).resolve()
        try:
            path.relative_to(workflow_dir.resolve())
        except ValueError:
            continue
        if not path.is_file():
            continue
        artifacts.append(
            create_artifact(
                run_id=state["run_id"],
                workflow_dir=workflow_dir,
                path=path,
                kind=kind,
                label=label,
                metadata={
                    "stage": "mask_component_selection",
                    "subtask_index": state["current_subtask_index"],
                },
            )
        )
    return artifacts


def _part_segmentation_artifacts(
    state: SculptWorkflowState,
    workflow_dir: Path,
    part_result: Mapping[str, JsonValue],
) -> list[WorkflowArtifact]:
    """Register browser-facing overlays for every final Pose subpart mask."""
    result = part_result.get("result")
    if not isinstance(result, dict):
        return []
    subparts = result.get("subparts")
    if not isinstance(subparts, list):
        return []
    artifacts: list[WorkflowArtifact] = []
    seen_paths: set[Path] = set()
    for fallback_order, value in enumerate(subparts, start=1):
        if not isinstance(value, dict):
            continue
        path_value = value.get("sam3_overlay_path")
        if not isinstance(path_value, str):
            continue
        path = Path(path_value)
        path = (path if path.is_absolute() else workflow_dir / path).resolve()
        try:
            path.relative_to(workflow_dir.resolve())
        except ValueError:
            continue
        if not path.is_file() or path in seen_paths:
            continue
        raw_order = value.get("order")
        order = (
            raw_order
            if isinstance(raw_order, int) and not isinstance(raw_order, bool)
            else fallback_order
        )
        raw_label = value.get("label")
        label = (
            raw_label.strip()
            if isinstance(raw_label, str) and raw_label.strip()
            else f"Subpart {order}"
        )
        artifacts.append(
            create_artifact(
                run_id=state["run_id"],
                workflow_dir=workflow_dir,
                path=path,
                kind=WorkflowArtifactKind.OVERLAY,
                label=f"Pose subpart {order}: {label} mask overlay",
                metadata={
                    "stage": "pose_part_segmentation",
                    "visualization": "pose_subpart_mask_overlay",
                    "subtask_index": state["current_subtask_index"],
                    "subpart_order": order,
                    "subpart_label": label,
                },
            )
        )
        seen_paths.add(path)
    return artifacts


def _stroke_plan_artifacts(
    state: SculptWorkflowState,
    workflow_dir: Path,
    plan_result: Mapping[str, JsonValue],
) -> list[WorkflowArtifact]:
    result = plan_result.get("result")
    if not isinstance(result, dict):
        return []
    path_value = result.get("stroke_plan_path")
    if not isinstance(path_value, str):
        return []
    path = Path(path_value)
    path = (path if path.is_absolute() else workflow_dir / path).resolve()
    try:
        path.relative_to(workflow_dir.resolve())
    except ValueError:
        return []
    if not path.is_file():
        return []
    return [
        create_artifact(
            run_id=state["run_id"],
            workflow_dir=workflow_dir,
            path=path,
            kind=WorkflowArtifactKind.METADATA,
            label="Deterministic Sculpt stroke plan",
        )
    ]


def _stroke_trajectory_artifacts(
    state: SculptWorkflowState,
    workflow_dir: Path,
    visualization: SculptStrokePolylineVisualizationResult,
) -> list[WorkflowArtifact]:
    definitions = (
        (
            visualization.mask_visualization_path,
            "Sculpt mouse trajectories on cleaned mask",
            "cleaned_mask",
        ),
        (
            visualization.overlay_visualization_path,
            "Sculpt mouse trajectories on SAM3 overlay",
            "overlay",
        ),
    )
    artifacts: list[WorkflowArtifact] = []
    for path_value, label, background in definitions:
        path = Path(path_value).resolve()
        artifacts.append(
            create_artifact(
                run_id=state["run_id"],
                workflow_dir=workflow_dir,
                path=path,
                kind=WorkflowArtifactKind.TRAJECTORY,
                label=label,
                metadata={
                    "stage": "stroke_planning",
                    "visualization": "mouse_trajectories",
                    "background": background,
                    "view": state["selected_view"],
                },
            )
        )
    return artifacts


def _drag_trajectory_artifacts(
    state: SculptWorkflowState,
    workflow_dir: Path,
    *,
    visualization_path: str,
) -> list[WorkflowArtifact]:
    path = Path(visualization_path).resolve()
    try:
        path.relative_to(workflow_dir.resolve())
    except ValueError:
        return []
    if not path.is_file():
        return []
    return [
        create_artifact(
            run_id=state["run_id"],
            workflow_dir=workflow_dir,
            path=path,
            kind=WorkflowArtifactKind.TRAJECTORY,
            label="Drag trajectory overlay",
            metadata={
                "stage": "drag_planning",
                "visualization": "drag_trajectory",
                "representation": "solid_start_circle_and_arrow",
                "background": "selected_view_screenshot",
                "view": state["selected_view"],
            },
        )
    ]


def _draw_artifacts(
    state: SculptWorkflowState,
    workflow_dir: Path,
    *,
    pattern_result: Mapping[str, JsonValue] | None,
    trajectory_result: Mapping[str, JsonValue] | None,
    fitted_result: Mapping[str, JsonValue] | None,
    visualization_path: str,
) -> list[WorkflowArtifact]:
    """Expose compact Draw source, plan, and trajectory evidence."""
    definitions: list[tuple[object, WorkflowArtifactKind, str]] = []
    for result, key, kind, label in (
        (
            pattern_result,
            "svg_path",
            WorkflowArtifactKind.OTHER,
            "Generated Draw SVG pattern",
        ),
        (
            trajectory_result,
            "trajectory_plan_path",
            WorkflowArtifactKind.METADATA,
            "Source Draw mouse trajectories",
        ),
        (
            fitted_result,
            "trajectory_plan_path",
            WorkflowArtifactKind.METADATA,
            "Mask-fitted Draw mouse trajectories",
        ),
    ):
        payload = result.get("result") if isinstance(result, Mapping) else None
        value = payload.get(key) if isinstance(payload, Mapping) else None
        definitions.append((value, kind, label))
    definitions.append(
        (
            visualization_path,
            WorkflowArtifactKind.TRAJECTORY,
            "Draw trajectory overlay",
        )
    )
    artifacts: list[WorkflowArtifact] = []
    seen: set[Path] = set()
    for value, kind, label in definitions:
        if not isinstance(value, str):
            continue
        path = Path(value).resolve()
        try:
            path.relative_to(workflow_dir.resolve())
        except ValueError:
            continue
        if not path.is_file() or path in seen:
            continue
        metadata: dict[str, JsonValue] = {
            "stage": "draw_planning",
            "view": state["selected_view"],
        }
        if kind is WorkflowArtifactKind.TRAJECTORY:
            metadata.update(
                {
                    "visualization": "draw_trajectories",
                    "representation": "colored_polylines_and_points",
                    "background": "selected_view_screenshot",
                }
            )
        artifacts.append(
            create_artifact(
                run_id=state["run_id"],
                workflow_dir=workflow_dir,
                path=path,
                kind=kind,
                label=label,
                metadata=metadata,
            )
        )
        seen.add(path)
    return artifacts


def _drag_target_artifacts(
    state: SculptWorkflowState,
    workflow_dir: Path,
    *,
    binding: Mapping[str, JsonValue],
) -> list[WorkflowArtifact]:
    """Expose the bound component and anchor as stable review evidence."""
    definitions = (
        (
            "component_mask_path",
            WorkflowArtifactKind.MASK,
            "Bound Drag target component",
        ),
        (
            "anchor_overlay_path",
            WorkflowArtifactKind.OVERLAY,
            "Bound Drag target and anchor overlay",
        ),
    )
    artifacts: list[WorkflowArtifact] = []
    for key, kind, label in definitions:
        value = binding.get(key)
        if not isinstance(value, str):
            continue
        path = Path(value).resolve()
        try:
            path.relative_to(workflow_dir.resolve())
        except ValueError:
            continue
        if not path.is_file():
            continue
        artifacts.append(
            create_artifact(
                run_id=state["run_id"],
                workflow_dir=workflow_dir,
                path=path,
                kind=kind,
                label=label,
                metadata={
                    "stage": "drag_target_binding",
                    "view": state["selected_view"],
                    "target_identity": True,
                },
            )
        )
    return artifacts


def _minimum_effect_artifacts(
    state: SculptWorkflowState,
    workflow_dir: Path,
    payload: Mapping[str, JsonValue],
) -> list[WorkflowArtifact]:
    definitions = {
        "evaluation_mask_path": (
            WorkflowArtifactKind.MASK,
            "Minimum-effect evaluation mask",
        ),
        "before_roi_path": (
            WorkflowArtifactKind.SCREENSHOT,
            "Target-region before crop",
        ),
        "after_roi_path": (
            WorkflowArtifactKind.SCREENSHOT,
            "Target-region after crop",
        ),
        "difference_heatmap_path": (
            WorkflowArtifactKind.OVERLAY,
            "Target-only difference heatmap",
        ),
    }
    sources: list[Mapping[str, JsonValue]] = [payload]
    artifact_paths = payload.get("artifact_paths")
    if isinstance(artifact_paths, dict):
        sources.append(artifact_paths)
    artifacts: list[WorkflowArtifact] = []
    seen: set[Path] = set()
    for key, (kind, label) in definitions.items():
        value = next(
            (
                source[key]
                for source in sources
                if isinstance(source.get(key), str)
            ),
            None,
        )
        if not isinstance(value, str) or not value:
            continue
        path = Path(value)
        path = (path if path.is_absolute() else workflow_dir / path).resolve()
        try:
            path.relative_to(workflow_dir.resolve())
        except ValueError:
            continue
        if not path.is_file() or path in seen:
            continue
        seen.add(path)
        artifacts.append(
            create_artifact(
                run_id=state["run_id"],
                workflow_dir=workflow_dir,
                path=path,
                kind=kind,
                label=label,
            )
        )
    return artifacts


def _attempt_record(
    state: SculptWorkflowState,
    *,
    minimum_effect: Mapping[str, JsonValue],
    grader: Mapping[str, JsonValue] | None = None,
    acceptance: Mapping[str, JsonValue] | None = None,
) -> dict[str, JsonValue]:
    before_a = state["baseline_screenshot_a"]
    before_b = state["baseline_screenshot_b"]
    after = state["after_screenshot"]
    full_before = state["full_before_screenshot"]
    full_after = state["full_after_screenshot"]
    return {
        "subtask_index": state["current_subtask_index"],
        "attempt": state["current_attempt"],
        "selected_view": state["selected_view"],
        "view_selection_reason": state["view_selection_reason"],
        "valid_views": list(state["valid_views"]),
        "rejected_drag_anchor_views": list(
            state["rejected_drag_anchor_views"]
        ),
        "view_screenshots": _screenshot_paths(state["view_screenshots"]),
        "view_segmentation_results": state["view_segmentation_results"],
        "execution_preparation": state["execution_preparation"],
        "intent": _translation_intent(
            state["translations"],
            state["current_subtask_index"],
        ),
        "resolved_sculpt_plan": state["resolved_sculpt_plan"],
        "quadloc_result": state["quadloc_result"],
        "drag_target_binding": state["drag_target_binding"],
        "locked_drag_plan": state["locked_drag_plan"],
        "drag_direction_plan": state["drag_direction_plan"],
        "part_segmentation_result": state["part_segmentation_result"],
        "draw_pattern_result": state["draw_pattern_result"],
        "draw_trajectory_result": state["draw_trajectory_result"],
        "draw_fitted_trajectory_result": state[
            "draw_fitted_trajectory_result"
        ],
        "segment_context": _segment_context(
            state["segmentation_result"],
            state["stroke_plan_result"],
        ),
        "viewport_focus": state["viewport_focus"],
        "full_before_screenshot": (
            full_before.get("path")
            if isinstance(full_before, dict)
            else None
        ),
        "baseline_screenshot_a": (
            before_a.get("path") if isinstance(before_a, dict) else None
        ),
        "baseline_screenshot_b": (
            before_b.get("path") if isinstance(before_b, dict) else None
        ),
        "after_screenshot": (
            after.get("path") if isinstance(after, dict) else None
        ),
        "full_after_screenshot": (
            full_after.get("path")
            if isinstance(full_after, dict)
            else None
        ),
        "settings_result": state["settings_result"],
        "executed_stroke_count": len(state["execution_results"]),
        "execution_results": list(state["execution_results"]),
        "execution_error": state["execution_error"],
        "minimum_effect": dict(minimum_effect),
        "grader": None if grader is None else dict(grader),
        "acceptance": (
            None if acceptance is None else dict(acceptance)
        ),
    }


def _translation_intent(
    translations: Sequence[Mapping[str, JsonValue]],
    index: int,
) -> JsonValue:
    for translation in translations:
        if translation.get("subtask_index") == index:
            intent = translation.get("intent")
            if isinstance(intent, dict):
                return dict(intent)
    return None


def _capabilities_from_state(
    state: SculptWorkflowState,
) -> BlenderSculptCapabilities:
    """Read the validated runtime catalog from durable LangGraph State."""
    payload = state.get("sculpt_brush_capabilities")
    if not isinstance(payload, dict):
        raise WorkflowExecutionError(
            "Workflow State is missing Sculpt brush capabilities"
        )
    try:
        return BlenderSculptCapabilities.model_validate(payload)
    except ValueError as error:
        raise WorkflowExecutionError(
            f"Workflow State has invalid Sculpt brush capabilities: {error}"
        ) from error


def _validated_sculpt_intent(
    intent: SculptIntent,
    *,
    subtask: Mapping[str, JsonValue],
    capabilities: BlenderSculptCapabilities,
    role: str,
) -> SculptIntent:
    """Canonicalize one LLM intent against its subtask and local catalog."""
    description = subtask.get("description")
    if not isinstance(description, str):
        raise WorkflowExecutionError(
            "Subtask is missing its canonical description"
        )
    try:
        operation = OperationMethod(str(subtask.get("operation_method", "")))
        subtask_brush = capabilities.brush_from_subtask_description(
            description
        )
        intent_brush = capabilities.canonical_brush_name(
            intent.sculpt_brush
        )
        if intent_brush != subtask_brush:
            raise SculptCapabilityError(
                f"intent brush {intent_brush} does not match subtask brush "
                f"{subtask_brush}"
            )
        pose_drag = (
            operation is OperationMethod.DRAG
            and intent_brush.casefold() == "pose"
        )
        draw_operation = operation is OperationMethod.DRAW
        if intent_brush.casefold() == "pose" and not pose_drag:
            raise SculptCapabilityError(
                "Pose is only supported by the Drag workflow"
            )
        pattern_description = intent.draw_pattern_description
        draw_text = intent.draw_text
        draw_scale_tier = intent.draw_scale_tier
        if draw_operation:
            if intent_brush.casefold() != "draw":
                raise SculptCapabilityError(
                    "Draw operation must use the exact local Draw brush"
                )
            if (pattern_description is None) == (draw_text is None):
                raise SculptCapabilityError(
                    "Draw intent requires exactly one of "
                    "draw_pattern_description and draw_text"
                )
            if draw_scale_tier is None:
                draw_scale_tier = DrawScaleTier.MEDIUM
        elif (
            pattern_description is not None
            or draw_text is not None
            or draw_scale_tier is not None
        ):
            raise SculptCapabilityError(
                "Smear and Drag intents must omit Draw-only fields"
            )
        if pose_drag:
            if any(
                value is not None
                for value in (
                    intent.brush_scale,
                    intent.brush_strength,
                    intent.brush_direction,
                )
            ):
                raise SculptCapabilityError(
                    "Pose Drag intent must omit brush_scale, "
                    "brush_strength, and brush_direction"
                )
            direction = None
        elif draw_operation:
            if (
                intent.brush_scale is not None
                or intent.brush_strength is not None
            ):
                raise SculptCapabilityError(
                    "Draw intent must omit brush_scale and brush_strength"
                )
            direction = capabilities.normalize_direction(
                intent_brush,
                intent.brush_direction,
            )
            if direction not in {"ADD", "SUBTRACT"}:
                raise SculptCapabilityError(
                    "Draw brush Direction must be ADD or SUBTRACT"
                )
        else:
            if intent.brush_scale is None or intent.brush_strength is None:
                raise SculptCapabilityError(
                    "non-Pose intent requires brush_scale and brush_strength"
                )
            direction = capabilities.normalize_direction(
                intent_brush,
                intent.brush_direction,
            )
    except (SculptCapabilityError, ValueError) as error:
        raise WorkflowLlmError(
            f"{role} returned an invalid brush or Direction: {error}"
        ) from error
    payload = intent.model_dump(mode="json")
    payload["sculpt_brush"] = intent_brush
    payload["brush_direction"] = direction
    payload["draw_scale_tier"] = (
        None if draw_scale_tier is None else draw_scale_tier.value
    )
    return SculptIntent.model_validate(payload)


def _integer_coordinate(
    coordinate: Mapping[str, JsonValue],
    key: str,
) -> int:
    value = coordinate.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkflowExecutionError(
            f"QuadLoc coordinate {key} must be an integer"
        )
    return value


def _positive_result_integer(
    payload: Mapping[str, JsonValue],
    key: str,
) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise WorkflowExecutionError(
            f"Tool result {key} must be a positive integer"
        )
    return value


def _part_segmentation_overlay_path(
    payload: Mapping[str, JsonValue],
) -> str:
    artifacts = payload.get("artifacts")
    path = (
        artifacts.get("lasso_visualization_path")
        if isinstance(artifacts, dict)
        else None
    )
    if not isinstance(path, str) or not Path(path).is_file():
        raise WorkflowExecutionError(
            "Part segmentation is missing its kinematic lasso visualization"
        )
    return str(Path(path).resolve())


def _screenshot_dimensions(
    screenshot: Mapping[str, JsonValue],
) -> tuple[int, int]:
    metadata = screenshot.get("metadata")
    if not isinstance(metadata, dict):
        raise WorkflowExecutionError(
            "Drag screenshot is missing coordinate metadata"
        )
    dimensions: list[int] = []
    for key in ("width", "height"):
        value = metadata.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise WorkflowExecutionError(
                f"Drag screenshot metadata {key} must be positive"
            )
        dimensions.append(value)
    return dimensions[0], dimensions[1]


def _is_pose_drag(
    *,
    subtask: Mapping[str, JsonValue],
    intent: Mapping[str, JsonValue],
    pose_brush_name: str,
) -> bool:
    return (
        subtask.get("operation_method") == OperationMethod.DRAG.value
        and isinstance(intent.get("sculpt_brush"), str)
        and str(intent["sculpt_brush"]).casefold()
        == pose_brush_name.casefold()
    )


def _drag_distance_multiplier(
    directive: Mapping[str, JsonValue] | None,
) -> float:
    if directive is None:
        return 1.0
    value = directive.get("distance_multiplier", 1.0)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise WorkflowExecutionError(
            "Drag retry distance_multiplier must be positive"
        )
    return float(value)


def _drag_retry_settings(
    *,
    base_settings: Mapping[str, JsonValue],
    directive: Mapping[str, JsonValue] | None,
    pose_drag: bool,
    maximum_brush_size: int,
    maximum_brush_strength: float,
) -> dict[str, JsonValue]:
    """Apply a cumulative retry dose to immutable first-plan settings."""
    settings = _json_object_copy(base_settings)
    if pose_drag:
        return settings
    base_size = _positive_number_field(
        settings,
        "brush_size",
        label="Locked Drag settings",
    )
    base_strength = _positive_number_field(
        settings,
        "brush_strength",
        label="Locked Drag settings",
    )
    size_multiplier = _drag_directive_multiplier(
        directive,
        "size_multiplier",
    )
    strength_multiplier = _drag_directive_multiplier(
        directive,
        "strength_multiplier",
    )
    settings["brush_size"] = min(
        maximum_brush_size,
        max(1, round(base_size * size_multiplier)),
    )
    settings["brush_strength"] = round(
        min(maximum_brush_strength, base_strength * strength_multiplier),
        6,
    )
    return settings


def _next_drag_dose_directive(
    *,
    result: MinimumEffectResult,
    previous: Mapping[str, JsonValue] | None,
    retry_multiplier: float,
    pose_drag: bool,
) -> dict[str, JsonValue]:
    """Increase only visibility-deficient Drag dose dimensions."""
    distance = _drag_distance_multiplier(previous) * retry_multiplier
    strength = _drag_directive_multiplier(
        previous,
        "strength_multiplier",
    )
    size = _drag_directive_multiplier(previous, "size_multiplier")
    mean_met = (
        result.metrics.target_mean_abs_diff
        >= result.requirements.minimum_mean_abs_diff
    )
    fraction_met = (
        result.metrics.target_changed_fraction
        >= result.requirements.minimum_changed_fraction
    )
    if not pose_drag and not mean_met:
        strength *= retry_multiplier
    if not pose_drag and not fraction_met:
        size *= retry_multiplier
    return {
        "cause": result.verdict.value,
        "distance_multiplier": round(distance, 6),
        "strength_multiplier": round(strength, 6),
        "size_multiplier": round(size, 6),
    }


def _classify_drag_retry_scope(grader: GraderOutput) -> str:
    """Restart from the earliest Drag stage contradicted by evidence."""
    assessment = grader.drag_assessment
    if assessment is None:
        return _DRAG_RETRY_INTENT
    if (
        not assessment.target_identity_correct
        or grader.effect_appropriateness
        is EffectAppropriateness.WRONG_REGION
    ):
        return _DRAG_RETRY_LOCALIZE
    if not assessment.motion_direction_correct:
        return _DRAG_RETRY_GESTURE
    if (
        grader.effect_appropriateness
        is EffectAppropriateness.INCONCLUSIVE
    ):
        return _DRAG_RETRY_VIEW
    if grader.effect_appropriateness in {
        EffectAppropriateness.WRONG_EFFECT,
        EffectAppropriateness.EXCESSIVE_FOR_INSTRUCTION,
    }:
        return _DRAG_RETRY_INTENT
    if not assessment.non_target_geometry_stable:
        return _DRAG_RETRY_INTENT
    if (
        grader.effect_appropriateness
        is EffectAppropriateness.TOO_WEAK
        or not assessment.target_motion_visible
        or not assessment.spatial_goal_reached
    ):
        return _DRAG_RETRY_DOSE
    return _DRAG_RETRY_INTENT


def _force_drag_dose_directive(
    *,
    previous: Mapping[str, JsonValue] | None,
    retry_multiplier: float,
    pose_drag: bool,
    cause: str,
) -> dict[str, JsonValue]:
    """Escalate a visually underpowered Drag without relocalizing it."""
    strength = _drag_directive_multiplier(
        previous,
        "strength_multiplier",
    )
    if not pose_drag:
        strength *= retry_multiplier
    return {
        "cause": cause,
        "distance_multiplier": round(
            _drag_distance_multiplier(previous) * retry_multiplier,
            6,
        ),
        "strength_multiplier": round(strength, 6),
        "size_multiplier": round(
            _drag_directive_multiplier(previous, "size_multiplier"),
            6,
        ),
    }


def _drag_directive_multiplier(
    directive: Mapping[str, JsonValue] | None,
    key: str,
) -> float:
    if directive is None:
        return 1.0
    value = directive.get(key, 1.0)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 1.0
    ):
        raise WorkflowExecutionError(
            f"Drag retry {key} must be at least 1"
        )
    return float(value)


def _view_candidate_image_paths(
    screenshots: Mapping[str, str],
    overlays: Mapping[str, str],
) -> list[str]:
    paths: list[str] = []
    for view, screenshot in screenshots.items():
        paths.append(screenshot)
        overlay = overlays.get(view)
        if overlay is not None:
            paths.append(overlay)
    return paths


def _safe_anchor_retry_feedback(
    feedback: Mapping[str, JsonValue] | None,
    *,
    rejected_view: str,
    message: str,
    rejected_views: Sequence[str],
    standard_views: Sequence[str],
) -> dict[str, JsonValue]:
    rejected = set(rejected_views)
    rejected.add(rejected_view)
    payload = dict(feedback or {})
    payload.update(
        {
            "source": "safe_drag_anchor",
            "analysis": message,
            "previous_view": rejected_view,
            "rejected_views": [
                view for view in standard_views if view in rejected
            ],
            "required_change": (
                "Select a different SAM3-valid view with enough target-mask "
                "interior clearance for a reliable mouse-down point."
            ),
        }
    )
    return cast(dict[str, JsonValue], payload)


def _relativize_run_paths(
    value: JsonValue,
    workflow_dir: Path,
) -> JsonValue:
    """Replace run-owned absolute paths before persisting public State."""
    if isinstance(value, dict):
        return {
            str(key): _relativize_run_paths(item, workflow_dir)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _relativize_run_paths(item, workflow_dir) for item in value
        ]
    if isinstance(value, str):
        path = Path(value)
        if path.is_absolute():
            try:
                return path.resolve().relative_to(
                    workflow_dir.resolve()
                ).as_posix()
            except ValueError:
                return path.name
    return value


def _final_workflow_status(state: SculptWorkflowState) -> str:
    """Aggregate terminal subtask results into one truthful run status."""
    total = len(state["subtasks"])
    latest_by_index: dict[int, str] = {}
    for result in state["subtask_results"]:
        index = result.get("subtask_index")
        status = result.get("status")
        if isinstance(index, int) and isinstance(status, str):
            latest_by_index[index] = status
    statuses = list(latest_by_index.values())
    passed = sum(status == "passed" for status in statuses)
    if total > 0 and passed == total:
        return "completed"
    if passed > 0:
        return "partially_completed"
    if total > 0 and len(statuses) == total and all(
        status == "rejected_by_user" for status in statuses
    ):
        return "rejected"
    return "failed"


def _json_safe_state(state: Mapping[str, object]) -> dict[str, object]:
    payload = dict(state)
    messages = payload.get("messages")
    if isinstance(messages, list):
        payload["messages"] = [
            message_to_dict(message)
            if hasattr(message, "type")
            else message
            for message in messages
        ]
    return payload
