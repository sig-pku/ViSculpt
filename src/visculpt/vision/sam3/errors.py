"""Structured errors for the SAM3 Gradio client and Tool."""

from __future__ import annotations

import json

from visculpt.bridge import JsonValue


class Sam3ClientError(RuntimeError):
    """Base class for failures at the SAM3 service boundary."""

    error_type = "sam3_error"

    def details(self) -> dict[str, JsonValue]:
        """Return optional structured error details."""
        return {}

    def as_payload(self) -> dict[str, JsonValue]:
        """Return a stable JSON-compatible Tool error."""
        error: dict[str, JsonValue] = {
            "type": self.error_type,
            "message": str(self),
        }
        error.update(self.details())
        return {"sam3_error": error}

    def as_json(self) -> str:
        """Serialize the stable Tool error."""
        return json.dumps(
            self.as_payload(),
            ensure_ascii=False,
            separators=(",", ":"),
        )


class Sam3InputError(Sam3ClientError):
    """The local image, prompt, or inference option is invalid."""

    error_type = "invalid_input"


class Sam3TransportError(Sam3ClientError):
    """The local Gradio service could not be reached."""

    error_type = "transport_error"


class Sam3ServiceError(Sam3ClientError):
    """The Gradio service rejected or failed the inference request."""

    error_type = "service_error"


class Sam3ResponseError(Sam3ClientError):
    """The Gradio service returned an unusable response."""

    error_type = "invalid_response"


class Sam3OutputError(Sam3ClientError):
    """The segmentation outputs could not be persisted locally."""

    error_type = "output_error"
