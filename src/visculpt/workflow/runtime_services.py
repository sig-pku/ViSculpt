"""Validated runtime settings for the local Blender and SAM3 services."""

from __future__ import annotations

import os
import stat
import tempfile
from collections.abc import MutableMapping
from dataclasses import replace
from pathlib import Path

import tomlkit
from pydantic import BaseModel, ConfigDict, field_validator
from tomlkit.exceptions import TOMLKitError

from .config import (
    SculptWorkflowConfig,
    ServiceConfig,
    _loopback_port,
    load_workflow_config,
)
from .errors import WorkflowConfigError


class RuntimeServiceSettings(BaseModel):
    """Public loopback service URLs accepted from the Web client."""

    model_config = ConfigDict(extra="forbid")

    blender_rpc_url: str
    sam3_url: str

    @field_validator("blender_rpc_url")
    @classmethod
    def _valid_blender_rpc_url(cls, value: str) -> str:
        return _normalize_loopback_url(
            value,
            expected_path="/rpc",
            label="blender_rpc_url",
        )

    @field_validator("sam3_url")
    @classmethod
    def _valid_sam3_url(cls, value: str) -> str:
        return _normalize_loopback_url(
            value,
            expected_path="",
            label="sam3_url",
        )


def default_runtime_service_settings(
    config: SculptWorkflowConfig,
) -> RuntimeServiceSettings:
    """Translate the static service config into the public schema."""
    return RuntimeServiceSettings(
        blender_rpc_url=config.services.blender_rpc_url,
        sam3_url=config.services.sam3_url,
    )


def resolve_runtime_service_config(
    config: SculptWorkflowConfig,
    settings: RuntimeServiceSettings,
) -> ServiceConfig:
    """Build and validate a complete service config from public URLs."""
    resolved = replace(
        config.services,
        blender_rpc_url=settings.blender_rpc_url,
        sam3_url=settings.sam3_url,
    )
    resolved.blender_rpc_config()
    resolved.sam3_config()
    return resolved


def persist_runtime_service_settings(
    config: SculptWorkflowConfig,
    settings: RuntimeServiceSettings,
    *,
    path: str | Path | None = None,
) -> Path:
    """Atomically persist service URLs as the defaults for future starts."""
    resolve_runtime_service_config(config, settings)
    target = (
        config.source_path if path is None else Path(path).expanduser()
    ).resolve()
    try:
        source = target.read_text(encoding="utf-8")
        original_mode = stat.S_IMODE(target.stat().st_mode)
    except OSError as error:
        raise WorkflowConfigError(
            f"Cannot read workflow config {target}: {error}"
        ) from error
    try:
        document = tomlkit.parse(source)
    except TOMLKitError as error:
        raise WorkflowConfigError(
            f"Cannot parse workflow config {target}: {error}"
        ) from error

    services_section = document.get("services")
    if not isinstance(services_section, MutableMapping):
        raise WorkflowConfigError(
            f"Workflow config {target} is missing the [services] table"
        )
    services_section["blender_rpc_url"] = settings.blender_rpc_url
    services_section["sam3_url"] = settings.sam3_url

    temporary_path: Path | None = None
    try:
        # Atomically replace a sibling file to avoid invalid interrupted config.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(tomlkit.dumps(document))
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_path, original_mode)

        persisted = load_workflow_config(temporary_path)
        if default_runtime_service_settings(persisted) != settings:
            raise WorkflowConfigError(
                "Persisted service settings do not match the validated input"
            )
        os.replace(temporary_path, target)
        temporary_path = None
    except (OSError, WorkflowConfigError) as error:
        raise WorkflowConfigError(
            f"Cannot persist workflow config {target}: {error}"
        ) from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
    return target


def _normalize_loopback_url(
    value: str,
    *,
    expected_path: str,
    label: str,
) -> str:
    normalized = value.strip().rstrip("/")
    try:
        _loopback_port(
            normalized,
            expected_path=expected_path,
            label=label,
        )
    except WorkflowConfigError as error:
        raise ValueError(str(error)) from error
    return normalized
