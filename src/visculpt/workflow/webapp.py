"""Custom Agent Server routes for health, artifacts, and session control."""

from __future__ import annotations

import asyncio
import mimetypes
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import AsyncIterator, cast

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from visculpt.bridge import BlenderRpcClient, JsonValue
from visculpt.tools import (
    SculptViewportUiSnapshotInput,
    create_load_blender_state_tool,
    create_restore_sculpt_viewport_ui_tool,
)

from visculpt.workflow.artifacts import (
    WORKFLOW_ARTIFACT_SCHEMA_VERSION,
)
from visculpt.workflow.config import load_workflow_config
from visculpt.workflow.deletion import (
    WorkflowArtifactDeletionError,
    delete_workflow_run_artifacts,
)
from visculpt.workflow.events import (
    WORKFLOW_EVENT_SCHEMA_VERSION,
    workflow_event_json_schema,
)
from visculpt.workflow.errors import WorkflowConfigError
from visculpt.workflow.runtime_llm import (
    RuntimeLlmSettings,
    default_runtime_llm_settings,
    persist_runtime_llm_settings,
    public_runtime_llm_presets,
    resolve_runtime_llm_config,
    run_llm_api_test,
    runtime_llm_fingerprint,
)
from visculpt.workflow.runtime_services import (
    RuntimeServiceSettings,
    default_runtime_service_settings,
    persist_runtime_service_settings,
    resolve_runtime_service_config,
)
from visculpt.workflow.services import (
    check_blender_rpc_ready,
    check_sam3_ready,
)
from visculpt.workflow.session import (
    session_lease_for_artifact_root,
)
from visculpt.workflow.state import RUN_ID_PATTERN
from visculpt.workflow.thread_persistence import (
    ThreadPersistenceDeletionError,
    confirm_local_thread_deletion,
    purge_local_thread_checkpoints,
)
from visculpt.workflow.token_usage import (
    TokenUsageStore,
    default_token_usage_database_path,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG = load_workflow_config()
ARTIFACT_ROOT = CONFIG.artifact_root(PROJECT_ROOT)
CONFIG_WRITE_LOCK = Lock()
USAGE_STORE = TokenUsageStore(
    default_token_usage_database_path(PROJECT_ROOT)
)


class WorkflowArtifactDeletionRequest(BaseModel):
    """Validated set of semantic Workflow run IDs to remove."""

    model_config = ConfigDict(extra="forbid")

    run_ids: list[str]

    @field_validator("run_ids")
    @classmethod
    def validate_run_ids(cls, value: list[str]) -> list[str]:
        """Reject malformed paths before touching the filesystem."""
        invalid = [
            run_id
            for run_id in value
            if RUN_ID_PATTERN.fullmatch(run_id) is None
        ]
        if invalid:
            raise ValueError(f"Invalid workflow run IDs: {invalid}")
        return list(dict.fromkeys(value))


class ThreadDeletionConfirmationRequest(BaseModel):
    """Agent Server run IDs whose retry metadata must also be removed."""

    model_config = ConfigDict(extra="forbid")

    agent_server_run_ids: list[str]


def _cancel_cleanup_restore_tool() -> BaseTool:
    """Build the dedicated RPC Tool used only after cancellation."""
    return create_restore_sculpt_viewport_ui_tool(
        config=resolve_runtime_service_config(
            CONFIG,
            SERVICE_SETTINGS_CONTROLLER.settings(),
        ).blender_rpc_config()
    )


def _cancel_cleanup_load_tool() -> BaseTool:
    """Build the dedicated Blender rollback Tool used after cancellation."""
    return create_load_blender_state_tool(
        config=resolve_runtime_service_config(
            CONFIG,
            SERVICE_SETTINGS_CONTROLLER.settings(),
        ).blender_rpc_config()
    )


class LlmTestController:
    """Track only public connectivity-test state across HTTP requests."""

    def __init__(
        self,
        settings: RuntimeLlmSettings | None = None,
        *,
        usage_store: TokenUsageStore | None = None,
    ) -> None:
        self._generation = 0
        self._settings = settings or default_runtime_llm_settings(CONFIG)
        self._usage_store = usage_store
        self._state: dict[str, object] = {
            "status": "pending",
            "fingerprint": None,
            "tested_at": None,
            "error": None,
            "result": None,
        }

    def snapshot(self) -> dict[str, object]:
        """Return a detached status payload for one HTTP response."""
        return dict(self._state)

    def settings(self) -> RuntimeLlmSettings:
        """Return a detached copy of the latest accepted candidate."""
        return RuntimeLlmSettings.model_validate(
            self._settings.model_dump(mode="json")
        )

    def update_settings(
        self,
        settings: RuntimeLlmSettings,
    ) -> dict[str, object]:
        """Accept a new candidate and invalidate any earlier test result."""
        self._generation += 1
        self._settings = RuntimeLlmSettings.model_validate(
            settings.model_dump(mode="json")
        )
        self._state = {
            "status": "pending",
            "fingerprint": runtime_llm_fingerprint(settings),
            "tested_at": None,
            "error": None,
            "result": None,
        }
        return self.snapshot()

    async def test(
        self,
        settings: RuntimeLlmSettings,
    ) -> dict[str, object]:
        """Run a test without allowing stale requests to win a race."""
        self.update_settings(settings)
        generation = self._generation
        fingerprint = runtime_llm_fingerprint(settings)
        try:
            result = await asyncio.to_thread(
                run_llm_api_test,
                CONFIG,
                settings,
                workdir=PROJECT_ROOT,
                usage_store=self._usage_store,
            )
        except Exception as error:
            completed: dict[str, object] = {
                "status": "failure",
                "fingerprint": fingerprint,
                "tested_at": datetime.now(UTC).isoformat(),
                "error": str(error),
                "result": None,
            }
        else:
            completed = {
                "status": "success",
                "fingerprint": fingerprint,
                "tested_at": datetime.now(UTC).isoformat(),
                "error": None,
                "result": result,
            }
        if generation == self._generation:
            self._state = completed
        return completed


LLM_TEST_CONTROLLER = LlmTestController(
    default_runtime_llm_settings(CONFIG),
    usage_store=USAGE_STORE,
)


class ServiceSettingsController:
    """Keep the latest accepted local-service URLs in this server process."""

    def __init__(
        self,
        settings: RuntimeServiceSettings | None = None,
    ) -> None:
        self._settings = settings or default_runtime_service_settings(CONFIG)

    def settings(self) -> RuntimeServiceSettings:
        """Return a detached copy of the current settings."""
        return RuntimeServiceSettings.model_validate(
            self._settings.model_dump(mode="json")
        )

    def update(self, settings: RuntimeServiceSettings) -> None:
        """Replace the process-local settings after persistence succeeds."""
        self._settings = RuntimeServiceSettings.model_validate(
            settings.model_dump(mode="json")
        )


SERVICE_SETTINGS_CONTROLLER = ServiceSettingsController(
    default_runtime_service_settings(CONFIG)
)


def _apply_runtime_service_config(settings: RuntimeServiceSettings) -> None:
    """Retarget the already-compiled Agent Server workflow clients."""
    from visculpt.workflow.server import (
        apply_runtime_service_config,
    )

    apply_runtime_service_config(
        resolve_runtime_service_config(CONFIG, settings)
    )


def _probe_services(settings: RuntimeServiceSettings) -> dict[str, object]:
    """Probe exactly the URLs currently exposed in Settings."""
    service_config = resolve_runtime_service_config(CONFIG, settings)
    services: dict[str, object] = {}
    try:
        client = BlenderRpcClient(service_config.blender_rpc_config())
        services["blender_rpc"] = check_blender_rpc_ready(client)
    except Exception as error:
        services["blender_rpc"] = {
            "ready": False,
            "error": str(error),
        }
    try:
        services["sam3"] = check_sam3_ready(service_config.sam3_config())
    except Exception as error:
        services["sam3"] = {
            "ready": False,
            "error": str(error),
        }
    ready = all(
        isinstance(value, dict) and value.get("ready") is True
        for value in services.values()
    )
    return {"ready": ready, "services": services}


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Start the default LLM test without delaying HTTP availability."""
    startup_test = asyncio.create_task(
        LLM_TEST_CONTROLLER.test(LLM_TEST_CONTROLLER.settings())
    )
    try:
        yield
    finally:
        if not startup_test.done():
            startup_test.cancel()
        with suppress(asyncio.CancelledError):
            await startup_test

app = FastAPI(
    title="Agentic Geometry Editing custom routes",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)


@app.get("/healthz")
async def healthz() -> dict[str, object]:
    """Report that the custom HTTP application loaded successfully."""
    return {"ready": True, "component": "workflow-http-app"}


@app.get("/workflow/meta")
async def workflow_meta() -> dict[str, object]:
    """Return stable, non-secret frontend discovery metadata."""
    return {
        "graph_id": "sculpt_agent",
        "event_schema_version": WORKFLOW_EVENT_SCHEMA_VERSION,
        "artifact_schema_version": WORKFLOW_ARTIFACT_SCHEMA_VERSION,
        "require_execution_approval": (
            CONFIG.workflow.require_execution_approval
        ),
        "standard_views": list(CONFIG.workflow.standard_views),
        "max_subtask_attempts": CONFIG.workflow.max_subtask_attempts,
    }


@app.get("/workflow/events/schema")
async def workflow_events_schema() -> dict[str, object]:
    """Expose the authoritative versioned Workflow Event JSON Schema."""
    return {
        "schema_version": WORKFLOW_EVENT_SCHEMA_VERSION,
        "schema": workflow_event_json_schema(),
    }


@app.get("/workflow/services/health")
async def workflow_services_health() -> dict[str, object]:
    """Probe Blender RPC and SAM3 without invoking the editing graph."""
    settings = SERVICE_SETTINGS_CONTROLLER.settings()
    return await asyncio.to_thread(_probe_services, settings)


@app.get("/workflow/services/settings")
async def workflow_service_settings() -> dict[str, object]:
    """Expose the active local-service URLs without any credentials."""
    return {
        "settings": SERVICE_SETTINGS_CONTROLLER.settings().model_dump(
            mode="json"
        )
    }


@app.put("/workflow/services/settings")
async def save_workflow_service_settings(
    settings: RuntimeServiceSettings,
) -> dict[str, object]:
    """Persist and apply service URLs, then immediately probe them."""
    try:
        resolve_runtime_service_config(CONFIG, settings)
    except WorkflowConfigError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    lease = session_lease_for_artifact_root(
        ARTIFACT_ROOT,
        lease_seconds=CONFIG.workflow.blender_session_lease_seconds,
    )
    if lease.status().get("locked") is True:
        raise HTTPException(
            status_code=409,
            detail=(
                "Cannot change local service URLs while a workflow owns "
                "the Blender session"
            ),
        )

    previous = SERVICE_SETTINGS_CONTROLLER.settings()

    def persist_and_apply() -> None:
        with CONFIG_WRITE_LOCK:
            _apply_runtime_service_config(settings)
            try:
                persist_runtime_service_settings(CONFIG, settings)
            except Exception:
                _apply_runtime_service_config(previous)
                raise
            SERVICE_SETTINGS_CONTROLLER.update(settings)

    try:
        await asyncio.to_thread(persist_and_apply)
    except WorkflowConfigError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not save service settings: {error}",
        ) from error
    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=f"Could not apply service settings: {error}",
        ) from error

    health = await asyncio.to_thread(_probe_services, settings)
    return {
        "settings": settings.model_dump(mode="json"),
        "health": health,
        "persisted": True,
    }


@app.get("/workflow/llm/settings")
async def workflow_llm_settings() -> dict[str, object]:
    """Expose defaults, safe presets, and the latest test status."""
    return {
        "settings": LLM_TEST_CONTROLLER.settings().model_dump(mode="json"),
        "presets": public_runtime_llm_presets(CONFIG),
        "test": LLM_TEST_CONTROLLER.snapshot(),
    }


@app.put("/workflow/llm/settings")
async def save_workflow_llm_settings(
    settings: RuntimeLlmSettings,
) -> dict[str, object]:
    """Persist one validated candidate for the next server startup."""
    try:
        resolve_runtime_llm_config(CONFIG, settings)
    except WorkflowConfigError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    def persist() -> None:
        with CONFIG_WRITE_LOCK:
            persist_runtime_llm_settings(CONFIG, settings)

    try:
        await asyncio.to_thread(persist)
    except WorkflowConfigError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not save LLM settings: {error}",
        ) from error
    test_state = LLM_TEST_CONTROLLER.update_settings(settings)
    return {
        "settings": settings.model_dump(mode="json"),
        "presets": public_runtime_llm_presets(CONFIG),
        "test": test_state,
        "persisted": True,
    }


@app.post("/workflow/llm/test")
async def test_workflow_llm(
    settings: RuntimeLlmSettings,
) -> dict[str, object]:
    """Validate one candidate before the Web client enables Run."""
    result = await LLM_TEST_CONTROLLER.test(settings)
    return {"settings": settings.model_dump(mode="json"), "test": result}


@app.get("/workflow/usage")
async def workflow_token_usage() -> dict[str, JsonValue]:
    """Return global, model, Workflow, and API-test Token aggregates."""
    return await asyncio.to_thread(USAGE_STORE.overview)


@app.get("/workflow/usage/{run_id}")
async def workflow_run_token_usage(run_id: str) -> dict[str, JsonValue]:
    """Return role and model aggregates for one semantic Workflow run."""
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise HTTPException(status_code=400, detail="Invalid run_id")
    return await asyncio.to_thread(USAGE_STORE.workflow_summary, run_id)


@app.delete("/workflow/usage")
async def delete_workflow_token_usage(
    request: WorkflowArtifactDeletionRequest,
) -> dict[str, object]:
    """Delete usage records owned by semantic Workflow run IDs."""
    deleted = await asyncio.to_thread(
        USAGE_STORE.delete_runs,
        request.run_ids,
    )
    return {
        "run_ids": request.run_ids,
        "deleted_usage_calls": deleted,
    }


@app.get("/workflow/artifacts/{run_id}/{artifact_path:path}")
async def workflow_artifact(
    run_id: str,
    artifact_path: str,
) -> FileResponse:
    """Serve one run-owned artifact without accepting absolute paths."""
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise HTTPException(status_code=400, detail="Invalid run_id")
    relative = Path(artifact_path)
    if relative.is_absolute():
        raise HTTPException(status_code=400, detail="Invalid artifact path")
    run_root = (ARTIFACT_ROOT / f"run-{run_id}").resolve()
    target = (run_root / relative).resolve()
    try:
        target.relative_to(run_root)
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail="Artifact path escapes the workflow run",
        ) from error
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    media_type = mimetypes.guess_type(target.name)[0]
    disposition = (
        "inline"
        if media_type is not None and media_type.startswith("image/")
        else "attachment"
    )
    return FileResponse(
        target,
        filename=target.name,
        media_type=media_type,
        content_disposition_type=disposition,
        headers={"Cache-Control": "private, no-store"},
    )


@app.delete("/workflow/artifacts")
async def delete_workflow_artifacts(
    request: WorkflowArtifactDeletionRequest,
) -> dict[str, object]:
    """Delete every run-owned local artifact after its lease is released."""
    lease = session_lease_for_artifact_root(
        ARTIFACT_ROOT,
        lease_seconds=CONFIG.workflow.blender_session_lease_seconds,
    )
    lease_payload = lease.status().get("lease")
    lease_owner = (
        lease_payload.get("run_id")
        if isinstance(lease_payload, dict)
        else None
    )
    if isinstance(lease_owner, str) and lease_owner in request.run_ids:
        raise HTTPException(
            status_code=409,
            detail=(
                "Release the workflow's Blender session lease before "
                f"deleting artifacts for {lease_owner}"
            ),
        )
    try:
        result = await asyncio.to_thread(
            delete_workflow_run_artifacts,
            ARTIFACT_ROOT,
            request.run_ids,
        )
    except WorkflowArtifactDeletionError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return result.as_payload()


@app.delete("/workflow/threads/{thread_id}/local-checkpoints")
async def delete_local_thread_checkpoints(
    thread_id: str,
) -> dict[str, object]:
    """Remove local dev-checkpointer data before deleting its Thread row."""
    try:
        result = purge_local_thread_checkpoints(
            thread_id,
            project_root=PROJECT_ROOT,
        )
    except ThreadPersistenceDeletionError as error:
        status_code = (
            400 if "Invalid LangGraph thread ID" in str(error) else 409
        )
        raise HTTPException(status_code=status_code, detail=str(error)) from error
    return result.as_payload(thread_id)


@app.post("/workflow/threads/{thread_id}/deletion-confirmation")
async def confirm_thread_deletion(
    thread_id: str,
    request: ThreadDeletionConfirmationRequest,
) -> dict[str, object]:
    """Force local stores to disk and confirm no Thread data remains."""
    try:
        result = confirm_local_thread_deletion(
            thread_id,
            agent_server_run_ids=request.agent_server_run_ids,
            project_root=PROJECT_ROOT,
        )
    except ThreadPersistenceDeletionError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    deleted_usage_calls = await asyncio.to_thread(
        USAGE_STORE.delete_thread,
        thread_id,
    )
    return {
        **result.as_payload(thread_id),
        "deleted_usage_calls": deleted_usage_calls,
    }


@app.get("/workflow/session")
async def workflow_session() -> dict[str, object]:
    """Expose the single Blender session lease status."""
    lease = session_lease_for_artifact_root(
        ARTIFACT_ROOT,
        lease_seconds=CONFIG.workflow.blender_session_lease_seconds,
    )
    status = lease.status()
    status.pop("path", None)
    lease_payload = status.get("lease")
    if isinstance(lease_payload, dict):
        public_lease = dict(lease_payload)
        public_lease.pop("sculpt_viewport_ui_snapshot", None)
        public_lease.pop("checkpoint_path", None)
        status["lease"] = public_lease
    return status


@app.post("/workflow/session/{run_id}/cancel-cleanup")
async def cleanup_cancelled_workflow_session(
    run_id: str,
    request: dict[str, object],
) -> dict[str, object]:
    """Roll back Blender, restore UI, and release the cancelled run's lease."""
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise HTTPException(status_code=400, detail="Invalid run_id")
    unexpected = set(request) - {
        "sculpt_viewport_ui_snapshot",
        "checkpoint_path",
    }
    if unexpected:
        raise HTTPException(
            status_code=422,
            detail=f"Unexpected cancellation cleanup fields: {sorted(unexpected)}",
        )
    raw_snapshot = request.get("sculpt_viewport_ui_snapshot")
    raw_checkpoint = request.get("checkpoint_path")
    if raw_checkpoint is not None and not isinstance(raw_checkpoint, str):
        raise HTTPException(
            status_code=422,
            detail="checkpoint_path must be a string or null",
        )
    try:
        snapshot = (
            None
            if raw_snapshot is None
            else SculptViewportUiSnapshotInput.model_validate(raw_snapshot)
        )
    except ValidationError as error:
        raise HTTPException(
            status_code=422,
            detail=error.errors(include_url=False),
        ) from error

    lease = session_lease_for_artifact_root(
        ARTIFACT_ROOT,
        lease_seconds=CONFIG.workflow.blender_session_lease_seconds,
    )
    lease_state = lease.status()
    lease_payload = lease_state.get("lease")
    lease_owner = (
        lease_payload.get("run_id")
        if isinstance(lease_payload, dict)
        else None
    )
    if lease_owner is None:
        # Cancellation cleanup is idempotent after graph cleanup has completed.
        return {
            "run_id": run_id,
            "blender_state_restored": None,
            "blender_state_restore_response": None,
            "viewport_ui_restored": None,
            "viewport_ui_restore_response": None,
            "lease_released": False,
            "cleanup_error": None,
        }
    if lease_owner != run_id:
        return {
            "run_id": run_id,
            "blender_state_restored": False,
            "blender_state_restore_response": None,
            "viewport_ui_restored": False,
            "viewport_ui_restore_response": None,
            "lease_released": False,
            "cleanup_error": (
                "Cancelled workflow does not own the active Blender "
                f"session lease ({lease_owner})"
            ),
        }

    if snapshot is None and isinstance(lease_payload, dict):
        stored_snapshot = lease_payload.get("sculpt_viewport_ui_snapshot")
        if stored_snapshot is not None:
            try:
                snapshot = SculptViewportUiSnapshotInput.model_validate(
                    stored_snapshot
                )
            except ValidationError as error:
                snapshot = None
                stored_snapshot_error = error
            else:
                stored_snapshot_error = None
        else:
            stored_snapshot_error = None
    else:
        stored_snapshot_error = None

    checkpoint_value = raw_checkpoint
    if checkpoint_value is None and isinstance(lease_payload, dict):
        stored_checkpoint = lease_payload.get("checkpoint_path")
        if isinstance(stored_checkpoint, str):
            checkpoint_value = stored_checkpoint

    rollback_response: JsonValue = None
    rollback_restored: bool | None = None
    cleanup_errors: list[str] = []
    if checkpoint_value is not None:
        run_root = (ARTIFACT_ROOT / f"run-{run_id}").resolve()
        checkpoint = Path(checkpoint_value).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = run_root / checkpoint
        checkpoint = checkpoint.resolve()
        if not checkpoint.is_relative_to(run_root):
            cleanup_errors.append(
                "Cancellation checkpoint is outside the workflow run"
            )
            rollback_restored = False
        elif not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            cleanup_errors.append(
                f"Cancellation checkpoint is missing or empty: {checkpoint}"
            )
            rollback_restored = False
        else:
            try:
                rollback_response = cast(
                    JsonValue,
                    await _cancel_cleanup_load_tool().ainvoke(
                        {
                            "filepath": str(checkpoint),
                            "load_ui": CONFIG.workflow.load_ui_on_restore,
                        }
                    ),
                )
                rollback_restored = _blend_restore_succeeded(
                    rollback_response,
                    checkpoint,
                )
                if not rollback_restored:
                    cleanup_errors.append(
                        "Blender did not confirm cancellation rollback"
                    )
            except Exception as error:
                rollback_restored = False
                cleanup_errors.append(str(error))

    restore_response: JsonValue = None
    restored: bool | None = None
    if stored_snapshot_error is not None:
        cleanup_errors.append(
            "Stored Sculpt viewport UI snapshot is invalid: "
            f"{stored_snapshot_error}"
        )
    if snapshot is not None:
        try:
            restore_response = cast(
                JsonValue,
                await _cancel_cleanup_restore_tool().ainvoke(
                    {
                        "snapshot": snapshot.model_dump(
                            mode="json",
                            exclude_none=True,
                        )
                    }
                ),
            )
            restored = _viewport_restore_succeeded(restore_response)
            if not restored:
                cleanup_errors.append(
                    "Blender did not confirm Sculpt viewport UI restoration"
                )
        except Exception as error:
            restored = False
            cleanup_errors.append(str(error))

    released = lease.release(run_id)
    return {
        "run_id": run_id,
        "blender_state_restored": rollback_restored,
        "blender_state_restore_response": rollback_response,
        "viewport_ui_restored": restored,
        "viewport_ui_restore_response": restore_response,
        "lease_released": released,
        "cleanup_error": "; ".join(cleanup_errors) or None,
    }


@app.delete("/workflow/session/{run_id}")
async def release_workflow_session(run_id: str) -> dict[str, object]:
    """Release a cancelled run's own lease; never unlock another run."""
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise HTTPException(status_code=400, detail="Invalid run_id")
    lease = session_lease_for_artifact_root(
        ARTIFACT_ROOT,
        lease_seconds=CONFIG.workflow.blender_session_lease_seconds,
    )
    return {"released": lease.release(run_id), "run_id": run_id}


def _viewport_restore_succeeded(response: JsonValue) -> bool:
    """Return whether one complete JSON-RPC response confirms restoration."""
    if not isinstance(response, dict) or "error" in response:
        return False
    result = response.get("result")
    return isinstance(result, dict) and result.get("restored") is True


def _blend_restore_succeeded(
    response: JsonValue,
    checkpoint: Path,
) -> bool:
    """Validate the complete load_blend_file cancellation response."""
    if not isinstance(response, dict) or "error" in response:
        return False
    result = response.get("result")
    if not isinstance(result, dict):
        return False
    loaded = result.get("filepath")
    return (
        result.get("loaded") is True
        and isinstance(loaded, str)
        and Path(loaded).resolve() == checkpoint
    )
