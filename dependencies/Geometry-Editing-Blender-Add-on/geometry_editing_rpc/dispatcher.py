"""Dispatch RPC work from HTTP threads to Blender's main thread."""

from __future__ import annotations

import math
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import bpy

from .deferred import DeferredMainThreadCall


class DispatcherClosedError(RuntimeError):
    """Raised when work is submitted after shutdown."""


class DispatcherTimeoutError(TimeoutError):
    """Raised when Blender does not finish a request before its deadline."""

    def __init__(self, message: str, *, may_have_completed: bool) -> None:
        super().__init__(message)
        self.may_have_completed = may_have_completed


@dataclass
class _Task:
    callback: Callable[[], Any]
    done: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    state: str = "pending"
    cancel_requested: bool = False
    result: Any = None
    error: BaseException | None = None

    def claim(self) -> bool:
        with self.lock:
            if self.state != "pending":
                return False
            self.state = "running"
            return True

    def cancel(self) -> bool:
        with self.lock:
            if self.state == "pending":
                self.state = "cancelled"
                self.done.set()
                return True
            if self.state in {"running", "deferred"}:
                self.cancel_requested = True
            return False

    def defer(self) -> None:
        with self.lock:
            if self.state == "running":
                self.state = "deferred"

    def should_cancel(self) -> bool:
        with self.lock:
            return self.cancel_requested or self.state == "cancelled"

    def finish(
        self,
        *,
        result: Any = None,
        error: BaseException | None = None,
    ) -> None:
        with self.lock:
            self.result = result
            self.error = error
            self.state = "finished"
            self.done.set()


@dataclass
class _DeferredExecution:
    task: _Task
    call: DeferredMainThreadCall
    due_at: float


class BlenderMainThreadDispatcher:
    """Use a persistent Blender timer as a bounded main-thread work queue."""

    def __init__(
        self,
        *,
        max_tasks_per_tick: int = 16,
        idle_interval: float = 0.05,
    ) -> None:
        self._queue: queue.Queue[_Task] = queue.Queue()
        self._closed = False
        self._state_lock = threading.Lock()
        self._main_thread_id = threading.get_ident()
        self._max_tasks_per_tick = max_tasks_per_tick
        self._idle_interval = idle_interval
        self._active_deferred: _DeferredExecution | None = None
        self._timer_callback = self._drain
        bpy.app.timers.register(
            self._timer_callback,
            first_interval=0.0,
            persistent=True,
        )

    def submit(self, callback: Callable[[], Any], *, timeout: float) -> Any:
        """Run a callable on Blender's main thread and wait for its result."""
        if threading.get_ident() == self._main_thread_id:
            result = callback()
            if isinstance(result, DeferredMainThreadCall):
                raise RuntimeError(
                    "A deferred Blender call cannot be awaited on the "
                    "main thread"
                )
            return result

        with self._state_lock:
            if self._closed:
                raise DispatcherClosedError("RPC dispatcher is stopped")
            task = _Task(callback=callback)
            self._queue.put(task)

        if not task.done.wait(timeout):
            cancelled = task.cancel()
            raise DispatcherTimeoutError(
                "Timed out waiting for Blender's main thread",
                may_have_completed=not cancelled,
            )

        if task.error is not None:
            raise task.error
        return task.result

    def close(self) -> None:
        """Stop the timer and reject queued work."""
        with self._state_lock:
            if self._closed:
                return
            self._closed = True

        if bpy.app.timers.is_registered(self._timer_callback):
            bpy.app.timers.unregister(self._timer_callback)

        active = self._active_deferred
        self._active_deferred = None
        if active is not None:
            try:
                active.call.steps.close()
            except BaseException:  # noqa: BLE001
                pass
            active.task.finish(
                error=DispatcherClosedError("RPC dispatcher stopped")
            )

        while True:
            try:
                task = self._queue.get_nowait()
            except queue.Empty:
                break
            if task.claim():
                task.finish(error=DispatcherClosedError("RPC dispatcher stopped"))

    def _drain(self) -> float | None:
        with self._state_lock:
            if self._closed:
                return None

        active = self._active_deferred
        if active is not None:
            if active.task.should_cancel():
                self._advance_deferred()
                return (
                    0.001
                    if not self._queue.empty()
                    else self._idle_interval
                )
            remaining = active.due_at - time.monotonic()
            if remaining > 0.0:
                return max(remaining, 0.001)
            interval = self._advance_deferred()
            if self._active_deferred is not None:
                return interval
            return 0.001 if not self._queue.empty() else self._idle_interval

        processed = 0
        while processed < self._max_tasks_per_tick:
            try:
                task = self._queue.get_nowait()
            except queue.Empty:
                break
            if not task.claim():
                continue
            try:
                result = task.callback()
                if isinstance(result, DeferredMainThreadCall):
                    task.defer()
                    self._active_deferred = _DeferredExecution(
                        task=task,
                        call=result,
                        due_at=time.monotonic(),
                    )
                    interval = self._advance_deferred()
                    if self._active_deferred is not None:
                        return interval
                else:
                    task.finish(result=result)
            except BaseException as error:  # noqa: BLE001
                task.finish(error=error)
            processed += 1

        return 0.0 if not self._queue.empty() else self._idle_interval

    def _advance_deferred(self) -> float:
        execution = self._active_deferred
        if execution is None:
            return self._idle_interval
        if execution.task.should_cancel():
            try:
                execution.call.steps.close()
            except BaseException:  # noqa: BLE001
                pass
            execution.task.finish(
                error=DispatcherClosedError(
                    f"Deferred RPC call {execution.call.label} was cancelled"
                )
            )
            self._active_deferred = None
            return self._idle_interval

        try:
            delay = next(execution.call.steps)
        except StopIteration as completed:
            execution.task.finish(result=completed.value)
            self._active_deferred = None
            return self._idle_interval
        except BaseException as error:  # noqa: BLE001
            execution.task.finish(error=error)
            self._active_deferred = None
            return self._idle_interval

        if (
            not isinstance(delay, (int, float))
            or isinstance(delay, bool)
            or not math.isfinite(delay)
            or delay < 0.0
        ):
            execution.task.finish(
                error=TypeError(
                    f"Deferred RPC call {execution.call.label} yielded "
                    "an invalid delay"
                )
            )
            self._active_deferred = None
            return self._idle_interval

        interval = max(float(delay), 0.001)
        execution.due_at = time.monotonic() + interval
        return interval
