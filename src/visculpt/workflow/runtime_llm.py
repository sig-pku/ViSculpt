"""Validated, secret-free per-run LLM settings and connectivity tests."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import MutableMapping
from dataclasses import replace
from pathlib import Path
from typing import Literal

import tomlkit
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from tomlkit.exceptions import TOMLKitError

from visculpt.bridge import JsonValue

from .config import (
    ANTHROPIC_COMPATIBLE_EFFORTS,
    LLM_ROLES,
    OPENAI_COMPATIBLE_EFFORTS,
    LlmConfig,
    SculptWorkflowConfig,
    _endpoint_path,
    _http_url,
    load_workflow_config,
)
from .errors import WorkflowConfigError
from .llm import (
    HttpStructuredLlm,
    LlmCallObserver,
    StructuredMultimodalLlm,
)
from .prompts import LLM_API_TEST_SYSTEM_PROMPT
from .token_usage import (
    TokenUsageContext,
    TokenUsageRecorder,
    TokenUsageStore,
)

LlmProvider = Literal["openai_compatible", "anthropic"]
LlmEffort = Literal[
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
]
CUSTOM_LLM_API_KEY_ENV = "CUSTOM_API_KEY"
CUSTOM_LLM_API_KEY_MODE = "if_present"


class RuntimeLlmModels(BaseModel):
    """Model IDs used by the seven workflow reasoning roles."""

    model_config = ConfigDict(extra="forbid")

    decomposer: str
    translator: str
    view_selector: str
    quadloc: str
    svg_pattern_generator: str
    grader: str
    retry_planner: str

    @field_validator("*")
    @classmethod
    def _non_empty_model_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("model IDs must not be empty")
        if len(stripped) > 512:
            raise ValueError("model IDs must contain at most 512 characters")
        return stripped


class RuntimeLlmSettings(BaseModel):
    """Public runtime settings accepted from Agent Server clients."""

    model_config = ConfigDict(extra="forbid")

    base_url: str
    provider: LlmProvider
    endpoint_path: str
    effort: LlmEffort | None = None
    models: RuntimeLlmModels

    @field_validator("base_url")
    @classmethod
    def _valid_base_url(cls, value: str) -> str:
        try:
            return _http_url(value.strip())
        except WorkflowConfigError as error:
            raise ValueError(str(error)) from error

    @field_validator("endpoint_path")
    @classmethod
    def _valid_endpoint_path(cls, value: str) -> str:
        try:
            return _endpoint_path(value.strip())
        except WorkflowConfigError as error:
            raise ValueError(str(error)) from error

    @model_validator(mode="after")
    def _valid_effort_for_provider(self) -> RuntimeLlmSettings:
        if self.effort is None:
            return self
        allowed = (
            ANTHROPIC_COMPATIBLE_EFFORTS
            if self.provider == "anthropic"
            else OPENAI_COMPATIBLE_EFFORTS
        )
        if self.effort not in allowed:
            choices = ", ".join(allowed)
            raise ValueError(
                f"effort must be one of {choices} for {self.provider}"
            )
        return self


class LlmApiTestOutput(BaseModel):
    """Minimal structured response expected from the multimodal test."""

    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1, max_length=4_096)


def default_runtime_llm_settings(
    config: SculptWorkflowConfig,
) -> RuntimeLlmSettings:
    """Translate the static default into the public runtime schema."""
    return RuntimeLlmSettings(
        base_url=config.llm.base_url,
        provider=config.llm.provider,
        endpoint_path=config.llm.endpoint_path,
        effort=config.llm.effort,
        models=RuntimeLlmModels.model_validate(config.llm.models),
    )


def persist_runtime_llm_settings(
    config: SculptWorkflowConfig,
    settings: RuntimeLlmSettings,
    *,
    path: str | Path | None = None,
) -> Path:
    """Atomically persist public runtime settings as the next defaults."""
    resolved = resolve_runtime_llm_config(config, settings)
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

    llm_section = document.get("llm")
    if not isinstance(llm_section, MutableMapping):
        raise WorkflowConfigError(
            f"Workflow config {target} is missing the [llm] table"
        )
    models_section = llm_section.get("models")
    if not isinstance(models_section, MutableMapping):
        raise WorkflowConfigError(
            f"Workflow config {target} is missing the [llm.models] table"
        )

    llm_section["provider"] = settings.provider
    llm_section["base_url"] = settings.base_url
    llm_section["endpoint_path"] = settings.endpoint_path
    llm_section["effort"] = settings.effort or ""
    llm_section["api_key_env"] = resolved.api_key_env
    llm_section["api_key_mode"] = resolved.api_key_mode
    llm_section["schema_profile"] = resolved.schema_profile
    for role, model_id in settings.models.model_dump().items():
        models_section[role] = model_id

    temporary_path: Path | None = None
    try:
        # Replace a sibling temporary file to avoid partial TOML after interruption.
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
        persisted_settings = default_runtime_llm_settings(persisted)
        if persisted_settings != settings:
            raise WorkflowConfigError(
                "Persisted LLM settings do not match the validated input"
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


def public_runtime_llm_presets(
    config: SculptWorkflowConfig,
) -> list[dict[str, JsonValue]]:
    """Return frontend choices without secret names or values."""
    return [
        {
            "id": preset.preset_id,
            "label": preset.label,
            "base_url": preset.base_url,
            "openai_endpoint_path": preset.openai_endpoint_path,
            "anthropic_endpoint_path": preset.anthropic_endpoint_path,
        }
        for preset in config.llm_runtime.presets
    ]


def resolve_runtime_llm_config(
    config: SculptWorkflowConfig,
    settings: RuntimeLlmSettings,
) -> LlmConfig:
    """Resolve public settings to a complete non-secret provider config."""
    preset = config.llm_runtime.preset_for_base_url(settings.base_url)
    if (
        settings.provider == "anthropic"
        and preset is not None
        and preset.anthropic_endpoint_path is None
    ):
        raise WorkflowConfigError(
            f"{preset.label} does not support the Anthropic-compatible "
            "endpoint"
        )
    return replace(
        config.llm,
        provider=settings.provider,
        base_url=settings.base_url,
        endpoint_path=settings.endpoint_path,
        effort=settings.effort,
        api_key_env=(
            preset.api_key_env
            if preset is not None
            else CUSTOM_LLM_API_KEY_ENV
        ),
        api_key_mode=(
            preset.api_key_mode
            if preset is not None
            else CUSTOM_LLM_API_KEY_MODE
        ),
        schema_profile=(
            preset.schema_profile
            if preset is not None
            else config.llm.schema_profile
        ),
        models=settings.models.model_dump(),
    )


def create_runtime_llm(
    config: SculptWorkflowConfig,
    settings: RuntimeLlmSettings,
    *,
    workdir: Path | None = None,
    call_observer: LlmCallObserver | None = None,
) -> StructuredMultimodalLlm:
    """Create an adapter from one immutable, secret-free snapshot."""
    llm_config = resolve_runtime_llm_config(config, settings)
    resolved_config = replace(config, llm=llm_config)
    return HttpStructuredLlm(
        llm_config,
        api_key=resolved_config.api_key(workdir=workdir),
        call_observer=call_observer,
    )


def runtime_llm_settings_from_runnable(
    config: SculptWorkflowConfig,
    configurable: object,
) -> tuple[RuntimeLlmSettings, bool]:
    """Read an optional runtime override from Runnable configurable data."""
    default = default_runtime_llm_settings(config)
    if not isinstance(configurable, dict) or "llm" not in configurable:
        return default, False
    settings = RuntimeLlmSettings.model_validate(configurable["llm"])
    # Validate the provider and known Base URL combination before updating State.
    resolve_runtime_llm_config(config, settings)
    return settings, True


def runtime_llm_fingerprint(settings: RuntimeLlmSettings) -> str:
    """Return a stable public identifier for one tested configuration."""
    serialized = json.dumps(
        settings.model_dump(mode="json"),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def run_llm_api_test(
    config: SculptWorkflowConfig,
    settings: RuntimeLlmSettings,
    *,
    workdir: Path,
    usage_store: TokenUsageStore | None = None,
) -> dict[str, JsonValue]:
    """Test every distinct configured model with the fixed text and image."""
    prompt_path = _resolve_test_asset(
        workdir,
        config.llm_runtime.test_prompt_path,
    )
    image_path = _resolve_test_asset(
        workdir,
        config.llm_runtime.test_image_path,
    )
    try:
        user_prompt = prompt_path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise WorkflowConfigError(
            f"Cannot read LLM API test prompt {prompt_path}: {error}"
        ) from error
    if not user_prompt:
        raise WorkflowConfigError("The LLM API test prompt is empty")

    recorder = (
        TokenUsageRecorder(
            usage_store,
            TokenUsageContext(
                scope="model_test",
                call_site="llm_api_test",
            ),
        )
        if usage_store is not None
        else None
    )
    llm = create_runtime_llm(
        config,
        settings,
        workdir=workdir,
        call_observer=recorder,
    )
    tested_models: set[str] = set()
    results: list[JsonValue] = []
    models = settings.models.model_dump()
    for role in LLM_ROLES:
        model_id = models[role]
        if model_id in tested_models:
            continue
        completion = llm.complete(
            role=role,
            system_prompt=LLM_API_TEST_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            image_paths=[image_path],
            response_model=LlmApiTestOutput,
        )
        output = completion.value
        if not isinstance(output, LlmApiTestOutput):
            raise RuntimeError("LLM API test returned an unexpected schema")
        results.append(
            {
                "model_id": model_id,
                "representative_role": role,
                "description": output.description,
            }
        )
        tested_models.add(model_id)
    return {
        "tested_model_count": len(tested_models),
        "results": results,
    }


def _resolve_test_asset(workdir: Path, configured_path: str) -> Path:
    """Resolve and verify one configured local test asset."""
    stored = Path(configured_path).expanduser()
    path = (stored if stored.is_absolute() else workdir / stored).resolve()
    if not path.is_file():
        raise WorkflowConfigError(f"LLM API test asset not found: {path}")
    return path
