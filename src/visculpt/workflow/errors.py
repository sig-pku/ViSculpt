"""Errors raised by the Sculpt Agent workflow."""

from __future__ import annotations


class SculptWorkflowError(RuntimeError):
    """Base error for deterministic workflow failures."""


class WorkflowConfigError(SculptWorkflowError):
    """Raised when the centralized workflow config is invalid."""


class WorkflowLlmError(SculptWorkflowError):
    """Raised when an LLM provider request or response is invalid."""


class WorkflowLlmTransientError(WorkflowLlmError):
    """Raised when a bounded retry may recover an LLM request."""


class WorkflowExecutionError(SculptWorkflowError):
    """Raised when a required service or Tool call fails."""
