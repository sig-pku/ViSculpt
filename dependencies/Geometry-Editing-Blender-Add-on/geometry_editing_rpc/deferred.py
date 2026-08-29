"""Deferred Blender main-thread calls executed across timer ticks."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DeferredMainThreadCall:
    """A generator whose yielded values are delays before the next step."""

    steps: Generator[float, None, Any]
    label: str
