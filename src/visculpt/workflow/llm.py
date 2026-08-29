"""Provider-neutral structured multimodal LLM adapters."""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import socket
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Literal, Protocol, TypeVar, cast
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from PIL import Image
from pydantic import BaseModel, ValidationError

from visculpt.bridge import JsonValue

from .config import LlmConfig
from .errors import WorkflowLlmError, WorkflowLlmTransientError

StructuredOutput = TypeVar("StructuredOutput", bound=BaseModel)
LOGGER = logging.getLogger(__name__)
HttpTransport = Callable[
    [str, Mapping[str, str], dict[str, JsonValue], float],
    dict[str, JsonValue],
]

_SENSITIVE_HEADER_NAMES = {"authorization", "x-api-key"}


class _SecretHeaderValue(str):
    """Behave as a header string while redacting diagnostic repr output."""

    def __repr__(self) -> str:
        return "'<redacted>'"


class _RedactedHeaders(Mapping[str, str]):
    """Mapping whose repr never exposes authentication header values."""

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, str]) -> None:
        self._values = {
            key: (
                _SecretHeaderValue(value)
                if key.casefold() in _SENSITIVE_HEADER_NAMES
                else value
            )
            for key, value in values.items()
        }

    def __getitem__(self, key: str) -> str:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        safe = {
            key: (
                "<redacted>"
                if key.casefold() in _SENSITIVE_HEADER_NAMES
                else value
            )
            for key, value in self._values.items()
        }
        return repr(safe)


@dataclass(frozen=True, slots=True)
class LlmCallResult:
    """Validated value plus secret-free request metadata."""

    value: BaseModel
    metadata: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class LlmCallObservation:
    """Raw response boundary used by provider-neutral usage accounting."""

    role: str
    provider: str
    base_url: str
    endpoint_path: str
    requested_model: str
    started_at: datetime
    completed_at: datetime
    outcome: Literal["success", "invalid_response", "transport_error"]
    response: dict[str, JsonValue] | None


LlmCallObserver = Callable[[LlmCallObservation], None]


class StructuredMultimodalLlm(Protocol):
    """Minimal provider-neutral interface consumed by graph nodes."""

    def complete(
        self,
        *,
        role: str,
        system_prompt: str,
        user_prompt: str,
        image_paths: Sequence[str | Path],
        response_model: type[StructuredOutput],
    ) -> LlmCallResult:
        """Return one schema-validated multimodal completion."""


class HttpStructuredLlm:
    """Call OpenAI-compatible or Anthropic structured-output endpoints."""

    def __init__(
        self,
        config: LlmConfig,
        *,
        api_key: str | None,
        transport: HttpTransport | None = None,
        call_observer: LlmCallObserver | None = None,
    ) -> None:
        key = (api_key or "").strip()
        if config.api_key_mode == "required" and not key:
            raise WorkflowLlmError("The configured LLM API key is empty")
        self.config = config
        self._api_key = (
            key if key and config.api_key_mode != "none" else None
        )
        self._transport = transport or _post_json
        self._call_observer = call_observer

    def with_call_observer(
        self,
        observer: LlmCallObserver | None,
    ) -> HttpStructuredLlm:
        """Clone the adapter while preserving its transport and secret."""
        return HttpStructuredLlm(
            self.config,
            api_key=self._api_key,
            transport=self._transport,
            call_observer=observer,
        )

    def complete(
        self,
        *,
        role: str,
        system_prompt: str,
        user_prompt: str,
        image_paths: Sequence[str | Path],
        response_model: type[StructuredOutput],
    ) -> LlmCallResult:
        """Send one request and validate the provider's JSON result."""
        model = self.config.model_for(role)
        started_at = datetime.now(UTC)
        response: dict[str, JsonValue] | None = None
        outcome: Literal[
            "success", "invalid_response", "transport_error"
        ] = "transport_error"
        images = [
            _encode_image(
                path,
                normalize_webp=(
                    self.config.provider == "openai_compatible"
                ),
            )
            for path in image_paths
        ]
        try:
            schema = cast(
                dict[str, JsonValue],
                response_model.model_json_schema(mode="serialization"),
            )
            provider_schema = _schema_for_provider(
                schema,
                profile=self.config.schema_profile,
            )
            if self.config.provider == "openai_compatible":
                endpoint, headers, payload = self._openai_request(
                    role=role,
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    images=images,
                    schema=provider_schema,
                )
                response = self._send_with_retry(
                    endpoint,
                    headers,
                    payload,
                )
                outcome = "invalid_response"
                content, provider_metadata = _parse_openai_response(response)
            elif self.config.provider == "anthropic":
                endpoint, headers, payload = self._anthropic_request(
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    images=images,
                    schema=provider_schema,
                )
                response = self._send_with_retry(
                    endpoint,
                    headers,
                    payload,
                )
                outcome = "invalid_response"
                content, provider_metadata = _parse_anthropic_response(
                    response
                )
            else:  # The loader validates this; guard direct construction too.
                raise WorkflowLlmError(
                    f"Unsupported LLM provider {self.config.provider}"
                )

            try:
                value = response_model.model_validate_json(
                    _strip_markdown_fence(content)
                )
            except ValidationError as error:
                raise WorkflowLlmError(
                    f"LLM response for {role} did not match "
                    f"{response_model.__name__}: {error}"
                ) from error
            except ValueError as error:
                raise WorkflowLlmError(
                    f"LLM response for {role} was not valid JSON: {error}"
                ) from error

            outcome = "success"
            metadata: dict[str, JsonValue] = {
                "role": role,
                "provider": self.config.provider,
                "model": model,
                "actual_model": response.get("model"),
                "image_count": len(images),
                **provider_metadata,
            }
            return LlmCallResult(value=value, metadata=metadata)
        finally:
            self._notify_call_observer(
                LlmCallObservation(
                    role=role,
                    provider=self.config.provider,
                    base_url=self.config.base_url,
                    endpoint_path=self.config.endpoint_path,
                    requested_model=model,
                    started_at=started_at,
                    completed_at=datetime.now(UTC),
                    outcome=outcome,
                    response=response,
                )
            )

    def _notify_call_observer(
        self,
        observation: LlmCallObservation,
    ) -> None:
        """Keep accounting failures from changing the model call result."""
        if self._call_observer is None:
            return
        try:
            self._call_observer(observation)
        except Exception:
            LOGGER.exception("Could not persist LLM Token usage")

    def _send_with_retry(
        self,
        endpoint: str,
        headers: Mapping[str, str],
        payload: dict[str, JsonValue],
    ) -> dict[str, JsonValue]:
        """Retry only explicitly classified transient transport failures."""
        for attempt in range(self.config.max_retries + 1):
            try:
                return self._transport(
                    endpoint,
                    headers,
                    payload,
                    self.config.timeout_seconds,
                )
            except WorkflowLlmTransientError as error:
                if attempt >= self.config.max_retries:
                    raise
                delay = self.config.retry_backoff_seconds * (2**attempt)
                if delay > 0:
                    time.sleep(delay)
        raise AssertionError("unreachable LLM retry state")

    def _openai_request(
        self,
        *,
        role: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        images: Sequence[tuple[str, str]],
        schema: dict[str, JsonValue],
    ) -> tuple[str, Mapping[str, str], dict[str, JsonValue]]:
        content: list[JsonValue] = [
            {"type": "text", "text": user_prompt}
        ]
        content.extend(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{encoded}"
                },
            }
            for media_type, encoded in images
        )
        payload: dict[str, JsonValue] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "max_tokens": self.config.max_output_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": f"{role}_output",
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        if self.config.effort is not None:
            payload["reasoning_effort"] = self.config.effort
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "agentic-geometry-editing-workflow/0.1",
        }
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return (
            f"{self.config.base_url}{self.config.endpoint_path}",
            _RedactedHeaders(headers),
            payload,
        )

    def _anthropic_request(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        images: Sequence[tuple[str, str]],
        schema: dict[str, JsonValue],
    ) -> tuple[str, Mapping[str, str], dict[str, JsonValue]]:
        content: list[JsonValue] = [
            {"type": "text", "text": user_prompt}
        ]
        content.extend(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": encoded,
                },
            }
            for media_type, encoded in images
        )
        output_config: dict[str, JsonValue] = {
            "format": {
                "type": "json_schema",
                "schema": schema,
            }
        }
        if self.config.effort is not None:
            output_config["effort"] = self.config.effort
        payload: dict[str, JsonValue] = {
            "model": model,
            "max_tokens": self.config.max_output_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": content}],
            "output_config": output_config,
        }
        headers = {
            "Accept": "application/json",
            "anthropic-version": self.config.anthropic_version,
            "Content-Type": "application/json",
            "User-Agent": "agentic-geometry-editing-workflow/0.1",
        }
        if self._api_key is not None:
            headers["x-api-key"] = self._api_key
        return (
            f"{self.config.base_url}{self.config.endpoint_path}",
            _RedactedHeaders(headers),
            payload,
        )


def _encode_image(
    path_value: str | Path,
    *,
    normalize_webp: bool = False,
) -> tuple[str, str]:
    """Encode and optionally normalize an image for multimodal APIs."""
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise WorkflowLlmError(f"LLM image does not exist: {path}")
    media_type = mimetypes.guess_type(path.name)[0]
    if media_type == "image/jpg":
        media_type = "image/jpeg"
    if media_type not in {
        "image/gif",
        "image/jpeg",
        "image/png",
        "image/webp",
    }:
        raise WorkflowLlmError(
            f"Unsupported LLM image type for {path.name}"
        )
    try:
        image_bytes = path.read_bytes()
        if normalize_webp and media_type == "image/webp":
            # Convert only the container format and preserve original dimensions.
            with Image.open(BytesIO(image_bytes)) as image:
                normalized = BytesIO()
                image.save(normalized, format="PNG", optimize=True)
            image_bytes = normalized.getvalue()
            media_type = "image/png"
        encoded = base64.b64encode(image_bytes).decode("ascii")
    except (OSError, ValueError) as error:
        raise WorkflowLlmError(f"Cannot read LLM image {path}") from error
    return media_type, encoded


def _schema_for_provider(
    schema: dict[str, JsonValue],
    *,
    profile: str,
) -> dict[str, JsonValue]:
    """Reduce generation constraints while retaining local validation."""
    if profile == "full":
        return schema
    if profile != "gemini_compatible":
        raise WorkflowLlmError(f"Unknown JSON Schema profile {profile}")
    pruned_keys = {
        "maximum",
        "maxItems",
        "maxLength",
        "minimum",
        "minItems",
        "minLength",
        "pattern",
    }

    def prune(value: JsonValue) -> JsonValue:
        if isinstance(value, dict):
            return {
                key: prune(item)
                for key, item in value.items()
                if key not in pruned_keys
            }
        if isinstance(value, list):
            return [prune(item) for item in value]
        return value

    return cast(dict[str, JsonValue], prune(schema))


def _post_json(
    endpoint: str,
    headers: Mapping[str, str],
    payload: dict[str, JsonValue],
    timeout: float,
) -> dict[str, JsonValue]:
    try:
        body = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise WorkflowLlmError(
            f"Cannot encode LLM request payload: {error}"
        ) from error
    request = Request(
        endpoint,
        data=body,
        headers=dict(headers),
        method="POST",
    )
    try:
        opener = build_opener(ProxyHandler({}))
        with opener.open(request, timeout=timeout) as response:
            response_body = response.read()
    except HTTPError as error:
        try:
            error_body = error.read(4097).decode(
                "utf-8", errors="replace"
            )
        finally:
            error.close()
        error_type = (
            WorkflowLlmTransientError
            if error.code == 429 or 500 <= error.code < 600
            else WorkflowLlmError
        )
        raise error_type(
            f"LLM provider returned HTTP {error.code}: "
            f"{_safe_provider_error(error_body)}"
        ) from error
    except (TimeoutError, socket.timeout) as error:
        raise WorkflowLlmTransientError(
            f"LLM request timed out after {timeout:g} seconds"
        ) from error
    except URLError as error:
        raise WorkflowLlmTransientError(
            f"Cannot connect to LLM endpoint {endpoint}: {error.reason}"
        ) from error
    except ConnectionError as error:
        raise WorkflowLlmTransientError(
            f"LLM connection to {endpoint} was interrupted: {error}"
        ) from error
    try:
        value = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WorkflowLlmError(
            "LLM provider returned an invalid JSON response"
        ) from error
    if not isinstance(value, dict):
        raise WorkflowLlmError(
            "LLM provider returned a non-object JSON response"
        )
    return cast(dict[str, JsonValue], value)


def _parse_openai_response(
    response: Mapping[str, JsonValue],
) -> tuple[str, dict[str, JsonValue]]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise WorkflowLlmError(
            "OpenAI-compatible response is missing choices"
        )
    first = choices[0]
    if not isinstance(first, dict):
        raise WorkflowLlmError(
            "OpenAI-compatible response has an invalid first choice"
        )
    message = first.get("message")
    if not isinstance(message, dict):
        raise WorkflowLlmError(
            "OpenAI-compatible response is missing choice.message"
        )
    content_source = "content"
    try:
        content = _text_content(message.get("content"))
    except WorkflowLlmError:
        # Some reasoning models place final structured JSON in this field.
        content = _text_content(message.get("reasoning_content"))
        content_source = "reasoning_content"
    metadata: dict[str, JsonValue] = {
        "request_id": response.get("id"),
        "usage": response.get("usage"),
        "finish_reason": first.get("finish_reason"),
        "content_source": content_source,
    }
    return content, metadata


def _parse_anthropic_response(
    response: Mapping[str, JsonValue],
) -> tuple[str, dict[str, JsonValue]]:
    content = _text_content(response.get("content"))
    metadata: dict[str, JsonValue] = {
        "request_id": response.get("id"),
        "usage": response.get("usage"),
        "stop_reason": response.get("stop_reason"),
    }
    return content, metadata


def _text_content(value: JsonValue) -> str:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, list):
        fragments: list[str] = []
        for block in value:
            if not isinstance(block, dict):
                continue
            text = block.get("text")
            if block.get("type") in {"text", "output_text"} and isinstance(
                text, str
            ):
                fragments.append(text)
        joined = "".join(fragments).strip()
        if joined:
            return joined
    raise WorkflowLlmError("LLM response does not contain text output")


def _strip_markdown_fence(value: str) -> str:
    stripped = value.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _safe_provider_error(raw_body: str) -> str:
    """Keep error diagnostics useful without echoing request secrets."""
    truncated = raw_body[:4096]
    try:
        payload = json.loads(truncated)
    except json.JSONDecodeError:
        return truncated or "empty response body"
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            status = error.get("status") or error.get("type")
            if isinstance(message, str):
                return f"{status}: {message}" if status else message
    return truncated or "empty response body"
