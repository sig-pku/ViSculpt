"""Versioned, node-independent events exposed to workflow clients."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Literal, Mapping

from langgraph.config import get_stream_writer
from pydantic import BaseModel, ConfigDict, Field, model_validator

from visculpt.bridge import JsonValue

WORKFLOW_EVENT_SCHEMA_VERSION = "2.0"


class WorkflowEventType(StrEnum):
    """Stable semantic event types understood by every client."""

    WORKFLOW_STATUS = "workflow.status"
    NODE_STATUS = "node.status"
    SERVICE_STATUS = "service.status"
    PROGRESS_UPDATE = "progress.update"
    ARTIFACT_CREATED = "artifact.created"
    SUBTASK_STATUS = "subtask.status"
    APPROVAL_STATUS = "approval.status"
    INTERVENTION_STATUS = "intervention.status"
    EFFECT_VISIBILITY = "effect.visibility"
    USAGE_UPDATED = "usage.updated"
    ERROR_RAISED = "error.raised"


class WorkflowEventStatus(StrEnum):
    """Finite lifecycle vocabulary shared by all event types."""

    CREATED = "created"
    RUNNING = "running"
    WAITING = "waiting"
    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"
    RETRYING = "retrying"
    PASSED = "passed"
    REJECTED = "rejected"
    COMPLETED = "completed"


class _PayloadBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorkflowStatusPayload(_PayloadBase):
    kind: Literal["workflow.status"] = "workflow.status"
    workflow_status: str = Field(min_length=1, max_length=64)


class NodeStatusPayload(_PayloadBase):
    kind: Literal["node.status"] = "node.status"
    phase: Literal["started", "completed"]


class ServiceStatusPayload(_PayloadBase):
    kind: Literal["service.status"] = "service.status"
    service: Literal["blender_rpc", "sam3"]
    ready: bool


class ProgressUpdatePayload(_PayloadBase):
    kind: Literal["progress.update"] = "progress.update"
    label: str = Field(min_length=1, max_length=160)
    current: int = Field(ge=0)
    total: int = Field(ge=1)
    unit: Literal["view", "stroke", "step", "artifact"]

    @model_validator(mode="after")
    def validate_progress(self) -> ProgressUpdatePayload:
        """Prevent progress counters from exceeding their declared total."""
        if self.current > self.total:
            raise ValueError("current cannot be greater than total")
        return self


class ArtifactCreatedPayload(_PayloadBase):
    kind: Literal["artifact.created"] = "artifact.created"
    artifact_ids: list[str] = Field(min_length=1, max_length=128)


class SubtaskStatusPayload(_PayloadBase):
    kind: Literal["subtask.status"] = "subtask.status"
    action: Literal["prepared", "executed", "graded", "advanced"]
    operation_method: Literal["Smear", "Drag", "Draw"] | None = None


class ApprovalStatusPayload(_PayloadBase):
    kind: Literal["approval.status"] = "approval.status"
    approval_id: str = Field(min_length=1, max_length=256)
    decision: Literal["pending", "approve", "reject"]


class InterventionStatusPayload(_PayloadBase):
    """Stable lifecycle event for bounded human-in-the-loop requests."""

    kind: Literal["intervention.status"] = "intervention.status"
    intervention_id: str = Field(min_length=1, max_length=256)
    intervention_type: Literal["manual_view_selection", "manual_mask"]
    decision: Literal[
        "pending",
        "selected",
        "painted",
        "redraw",
        "confirmed",
        "skipped",
    ]
    stage: Literal["paint", "review"] | None = None
    selected_view: str | None = Field(default=None, max_length=32)

    @model_validator(mode="after")
    def validate_selected_view(self) -> InterventionStatusPayload:
        """Require a view only for a completed selection."""
        if self.decision == "selected" and self.selected_view is None:
            raise ValueError("selected intervention requires selected_view")
        if self.decision != "selected" and self.selected_view is not None:
            raise ValueError(
                "non-selected intervention must omit selected_view"
            )
        if self.intervention_type == "manual_mask" and self.stage is None:
            raise ValueError("manual mask intervention requires a stage")
        if (
            self.intervention_type == "manual_view_selection"
            and self.stage is not None
        ):
            raise ValueError("manual view intervention must omit stage")
        return self


class EffectVisibilityPayload(_PayloadBase):
    """Compact one-sided visibility result for one Sculpt attempt."""

    kind: Literal["effect.visibility"] = "effect.visibility"
    verdict: Literal[
        "NO_EFFECT",
        "TOO_SUBTLE",
        "VISIBLE",
        "INCONCLUSIVE",
    ]
    target_mean_abs_diff: float = Field(ge=0.0, le=255.0)
    target_changed_fraction: float = Field(ge=0.0, le=1.0)


class TokenMetricsPayload(_PayloadBase):
    """Five stable metrics shared by State, HTTP, and live events."""

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)


class TokenMetricCoveragePayload(_PayloadBase):
    """Explain whether an aggregate is complete or provider-partial."""

    reported_calls: int = Field(ge=0)
    total_calls: int = Field(ge=0)
    partial: bool


class TokenCoveragePayload(_PayloadBase):
    input_tokens: TokenMetricCoveragePayload
    output_tokens: TokenMetricCoveragePayload
    total_tokens: TokenMetricCoveragePayload
    cached_input_tokens: TokenMetricCoveragePayload
    reasoning_tokens: TokenMetricCoveragePayload


class TokenUsageAggregatePayload(_PayloadBase):
    call_count: int = Field(ge=0)
    tokens: TokenMetricsPayload
    coverage: TokenCoveragePayload


class RoleTokenUsagePayload(_PayloadBase):
    key: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1, max_length=128)
    aggregate: TokenUsageAggregatePayload


class ModelTokenUsagePayload(_PayloadBase):
    key: str = Field(min_length=1, max_length=2048)
    provider: str = Field(min_length=1, max_length=128)
    provider_label: str = Field(min_length=1, max_length=128)
    base_url: str = Field(min_length=1, max_length=2048)
    model: str = Field(min_length=1, max_length=512)
    aggregate: TokenUsageAggregatePayload


class WorkflowTokenUsagePayload(_PayloadBase):
    run_id: str = Field(min_length=1, max_length=128)
    thread_id: str | None = Field(default=None, max_length=256)
    title: str | None = Field(default=None, max_length=4096)
    aggregate: TokenUsageAggregatePayload
    by_role: list[RoleTokenUsagePayload]
    by_model: list[ModelTokenUsagePayload]
    first_called_at: str | None = None
    last_called_at: str | None = None


class UsageUpdatedPayload(_PayloadBase):
    """Absolute usage snapshots emitted after one model call."""

    kind: Literal["usage.updated"] = "usage.updated"
    call_id: str = Field(min_length=1, max_length=128)
    outcome: Literal["success", "invalid_response", "transport_error"]
    role: str = Field(min_length=1, max_length=128)
    call_site: str = Field(min_length=1, max_length=128)
    model_key: str = Field(min_length=1, max_length=2048)
    workflow_summary: WorkflowTokenUsagePayload
    global_aggregate: TokenUsageAggregatePayload
    global_model: ModelTokenUsagePayload | None = None


class ErrorRaisedPayload(_PayloadBase):
    kind: Literal["error.raised"] = "error.raised"
    code: str = Field(min_length=1, max_length=128)
    retryable: bool


type WorkflowEventPayload = Annotated[
    WorkflowStatusPayload
    | NodeStatusPayload
    | ServiceStatusPayload
    | ProgressUpdatePayload
    | ArtifactCreatedPayload
    | SubtaskStatusPayload
    | ApprovalStatusPayload
    | InterventionStatusPayload
    | EffectVisibilityPayload
    | UsageUpdatedPayload
    | ErrorRaisedPayload,
    Field(discriminator="kind"),
]


class WorkflowEventRecord(BaseModel):
    """Canonical event envelope persisted in State and streamed live."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2.0"] = WORKFLOW_EVENT_SCHEMA_VERSION
    event_id: str = Field(min_length=1, max_length=256)
    sequence: int = Field(ge=1)
    timestamp: datetime
    run_id: str = Field(min_length=1, max_length=128)
    type: WorkflowEventType
    status: WorkflowEventStatus
    node: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=1000)
    subtask_index: int | None = Field(default=None, ge=0)
    attempt: int | None = Field(default=None, ge=0)
    payload: WorkflowEventPayload

    @model_validator(mode="after")
    def validate_type_matches_payload(self) -> WorkflowEventRecord:
        """Keep the envelope discriminator aligned with its payload."""
        if self.type.value != self.payload.kind:
            raise ValueError("event type must match payload kind")
        return self


class WorkflowEventEmitter:
    """Validate, persist, and live-stream events through one API."""

    def __init__(self, state: Mapping[str, object], *, node: str) -> None:
        self._state = state
        self._node = node
        self._run_id = str(state["run_id"])
        self._base_events = [
            dict(item)
            for item in state.get("events", [])
            if isinstance(item, Mapping)
        ]
        self._sequence = int(state.get("event_sequence", 0))
        self._pending: list[dict[str, JsonValue]] = []

    def workflow(
        self,
        *,
        status: WorkflowEventStatus,
        workflow_status: str,
        message: str,
    ) -> None:
        self._emit(
            event_type=WorkflowEventType.WORKFLOW_STATUS,
            status=status,
            message=message,
            payload=WorkflowStatusPayload(
                workflow_status=workflow_status,
            ),
        )

    def node(
        self,
        *,
        status: WorkflowEventStatus,
        phase: Literal["started", "completed"],
        message: str,
    ) -> None:
        self._emit(
            event_type=WorkflowEventType.NODE_STATUS,
            status=status,
            message=message,
            payload=NodeStatusPayload(phase=phase),
        )

    def service(self, *, service: str, ready: bool, message: str) -> None:
        self._emit(
            event_type=WorkflowEventType.SERVICE_STATUS,
            status=(
                WorkflowEventStatus.SUCCESS
                if ready
                else WorkflowEventStatus.ERROR
            ),
            message=message,
            payload=ServiceStatusPayload(
                service=service,  # type: ignore[arg-type]
                ready=ready,
            ),
        )

    def progress(
        self,
        *,
        label: str,
        current: int,
        total: int,
        unit: str,
        message: str,
    ) -> None:
        self._emit(
            event_type=WorkflowEventType.PROGRESS_UPDATE,
            status=(
                WorkflowEventStatus.SUCCESS
                if current == total
                else WorkflowEventStatus.RUNNING
            ),
            message=message,
            payload=ProgressUpdatePayload(
                label=label,
                current=current,
                total=total,
                unit=unit,  # type: ignore[arg-type]
            ),
        )

    def artifacts(self, *, artifact_ids: list[str], message: str) -> None:
        if not artifact_ids:
            return
        self._emit(
            event_type=WorkflowEventType.ARTIFACT_CREATED,
            status=WorkflowEventStatus.SUCCESS,
            message=message,
            payload=ArtifactCreatedPayload(artifact_ids=artifact_ids),
        )

    def subtask(
        self,
        *,
        status: WorkflowEventStatus,
        action: str,
        operation_method: str | None,
        message: str,
    ) -> None:
        self._emit(
            event_type=WorkflowEventType.SUBTASK_STATUS,
            status=status,
            message=message,
            payload=SubtaskStatusPayload(
                action=action,  # type: ignore[arg-type]
                operation_method=operation_method,  # type: ignore[arg-type]
            ),
        )

    def approval(
        self,
        *,
        approval_id: str,
        decision: Literal["pending", "approve", "reject"],
        message: str,
    ) -> None:
        status = {
            "pending": WorkflowEventStatus.WAITING,
            "approve": WorkflowEventStatus.SUCCESS,
            "reject": WorkflowEventStatus.REJECTED,
        }[decision]
        self._emit(
            event_type=WorkflowEventType.APPROVAL_STATUS,
            status=status,
            message=message,
            payload=ApprovalStatusPayload(
                approval_id=approval_id,
                decision=decision,
            ),
        )

    def intervention(
        self,
        *,
        intervention_id: str,
        decision: Literal[
            "pending",
            "selected",
            "painted",
            "redraw",
            "confirmed",
            "skipped",
        ],
        selected_view: str | None = None,
        intervention_type: Literal[
            "manual_view_selection",
            "manual_mask",
        ] = "manual_view_selection",
        stage: Literal["paint", "review"] | None = None,
        message: str,
    ) -> None:
        """Emit one bounded human-intervention lifecycle event."""
        status = {
            "pending": WorkflowEventStatus.WAITING,
            "selected": WorkflowEventStatus.SUCCESS,
            "painted": WorkflowEventStatus.SUCCESS,
            "redraw": WorkflowEventStatus.RETRYING,
            "confirmed": WorkflowEventStatus.SUCCESS,
            "skipped": WorkflowEventStatus.SKIPPED,
        }[decision]
        self._emit(
            event_type=WorkflowEventType.INTERVENTION_STATUS,
            status=status,
            message=message,
            payload=InterventionStatusPayload(
                intervention_id=intervention_id,
                intervention_type=intervention_type,
                decision=decision,
                stage=stage,
                selected_view=selected_view,
            ),
        )

    def effect_visibility(
        self,
        *,
        verdict: str,
        target_mean_abs_diff: float,
        target_changed_fraction: float,
        status: WorkflowEventStatus,
        message: str,
    ) -> None:
        """Emit one compact visibility summary per attempt."""
        self._emit(
            event_type=WorkflowEventType.EFFECT_VISIBILITY,
            status=status,
            message=message,
            payload=EffectVisibilityPayload(
                verdict=verdict,  # type: ignore[arg-type]
                target_mean_abs_diff=target_mean_abs_diff,
                target_changed_fraction=target_changed_fraction,
            ),
        )

    def error(
        self,
        *,
        code: str,
        retryable: bool,
        message: str,
    ) -> None:
        self._emit(
            event_type=WorkflowEventType.ERROR_RAISED,
            status=WorkflowEventStatus.ERROR,
            message=message,
            payload=ErrorRaisedPayload(
                code=code,
                retryable=retryable,
            ),
        )

    def state_update(self) -> dict[str, JsonValue]:
        """Return the two State keys controlled exclusively by this class."""
        return {
            "events": [*self._base_events, *self._pending],
            "event_sequence": self._sequence,
        }

    def _emit(
        self,
        *,
        event_type: WorkflowEventType,
        status: WorkflowEventStatus,
        message: str,
        payload: WorkflowEventPayload,
    ) -> None:
        self._sequence += 1
        subtask_index = self._state.get("current_subtask_index")
        attempt = self._state.get("current_attempt")
        record = WorkflowEventRecord(
            event_id=f"{self._run_id}:{self._sequence:06d}",
            sequence=self._sequence,
            timestamp=datetime.now(timezone.utc),
            run_id=self._run_id,
            type=event_type,
            status=status,
            node=self._node,
            message=message,
            subtask_index=(
                int(subtask_index)
                if isinstance(subtask_index, int)
                else None
            ),
            attempt=int(attempt) if isinstance(attempt, int) else None,
            payload=payload,
        )
        serialized = record.model_dump(mode="json")
        self._pending.append(serialized)
        try:
            writer = get_stream_writer()
        except (KeyError, RuntimeError):
            return
        try:
            writer(
                {
                    "name": "workflow.event",
                    "payload": serialized,
                }
            )
        except Exception:
            # The event stream is auxiliary; disconnection cannot stop the workflow.
            return


def stream_token_usage_update(
    state: Mapping[str, object],
    update: Mapping[str, JsonValue],
    *,
    node: str,
) -> None:
    """Stream a transient usage event without growing checkpoint history."""
    workflow_summary = update.get("workflow_summary")
    global_aggregate = update.get("global_aggregate")
    if not isinstance(workflow_summary, dict) or not isinstance(
        global_aggregate, dict
    ):
        return
    call_id = str(update["call_id"])
    run_id = str(state["run_id"])
    aggregate = workflow_summary.get("aggregate")
    call_count = (
        int(aggregate.get("call_count", 0))
        if isinstance(aggregate, dict)
        else 0
    )
    subtask_index = state.get("current_subtask_index")
    attempt = state.get("current_attempt")
    payload = UsageUpdatedPayload(
        call_id=call_id,
        outcome=str(update["outcome"]),  # type: ignore[arg-type]
        role=str(update["role"]),
        call_site=str(update["call_site"]),
        model_key=str(update["model_key"]),
        workflow_summary=workflow_summary,
        global_aggregate=global_aggregate,
        global_model=(
            update.get("global_model")
            if isinstance(update.get("global_model"), dict)
            else None
        ),
    )
    record = WorkflowEventRecord(
        event_id=f"{run_id}:usage:{call_id}",
        sequence=max(1, int(state.get("event_sequence", 0)) + call_count),
        timestamp=datetime.now(timezone.utc),
        run_id=run_id,
        type=WorkflowEventType.USAGE_UPDATED,
        status=(
            WorkflowEventStatus.SUCCESS
            if update.get("outcome") == "success"
            else WorkflowEventStatus.ERROR
        ),
        node=node,
        message="Token usage updated",
        subtask_index=(
            int(subtask_index) if isinstance(subtask_index, int) else None
        ),
        attempt=int(attempt) if isinstance(attempt, int) else None,
        payload=payload,
    )
    try:
        writer = get_stream_writer()
    except (KeyError, RuntimeError):
        return
    try:
        writer(
            {
                "name": "workflow.event",
                "payload": record.model_dump(mode="json"),
            }
        )
    except Exception:
        return


def workflow_event_json_schema() -> dict[str, JsonValue]:
    """Return the authoritative JSON Schema for frontend tooling."""
    return WorkflowEventRecord.model_json_schema(mode="serialization")
