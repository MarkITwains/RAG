"""Bounded work admission, cooperative cancellation and one request deadline."""

from __future__ import annotations

import contextvars
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable


class DeadlineExceeded(TimeoutError):
    pass


class WorkRejected(RuntimeError):
    pass


@dataclass
class RequestControl:
    deadline: float
    revision: str = "legacy"
    cancelled: threading.Event = field(default_factory=threading.Event)
    timings: dict[str, float] = field(default_factory=dict)
    degraded: list[str] = field(default_factory=list)
    routes: dict[str, str] = field(default_factory=dict)
    usage: list[dict[str, Any]] = field(default_factory=list)
    usage_lock: threading.Lock = field(default_factory=threading.Lock)

    def remaining(self) -> float:
        if self.cancelled.is_set():
            raise DeadlineExceeded("request cancelled")
        left = self.deadline - time.monotonic()
        if left <= 0:
            self.cancelled.set()
            raise DeadlineExceeded("request deadline exceeded")
        return left


_control: contextvars.ContextVar[RequestControl | None] = contextvars.ContextVar("rag_request", default=None)
_in_worker: contextvars.ContextVar[bool] = contextvars.ContextVar("rag_io_worker", default=False)


def current_control() -> RequestControl | None:
    return _control.get()


@contextmanager
def request_scope(timeout: float = 120.0):
    existing = current_control()
    if existing is not None:
        existing.remaining()
        yield existing
        return
    control = RequestControl(time.monotonic() + max(0.001, timeout))
    token = _control.set(control)
    try:
        yield control
    finally:
        control.cancelled.set()
        _control.reset(token)


class BoundedExecutor:
    """Slots remain occupied until actual work finishes, including timed out IO.

    Python cannot terminate a running thread. Queued work is cancelled; running
    transports must also have their own timeout. Keeping their slots prevents
    disconnected clients from creating an unbounded backlog.
    """

    def __init__(self, workers: int = 8, queued: int = 16):
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rag-io")
        self._slots = threading.BoundedSemaphore(workers + queued)

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any):
        if not self._slots.acquire(blocking=False):
            raise WorkRejected("retrieval worker queue is full")
        ctx = contextvars.copy_context()

        def invoke():
            control = current_control()
            if control:
                control.remaining()
            token = _in_worker.set(True)
            try:
                return fn(*args, **kwargs)
            finally:
                _in_worker.reset(token)

        try:
            future = self._pool.submit(ctx.run, invoke)
        except BaseException:
            self._slots.release()
            raise
        future.add_done_callback(lambda _: self._slots.release())
        return future

    def close(self):
        self._pool.shutdown(wait=False, cancel_futures=True)


IO_EXECUTOR = BoundedExecutor()


def await_work(future, control: RequestControl | None = None, *, timeout: float | None = None):
    control = control or current_control()
    local_deadline = time.monotonic() + timeout if timeout is not None else None
    try:
        while True:
            left = control.remaining() if control else 120.0
            if local_deadline is not None:
                left = min(left, local_deadline - time.monotonic())
                if left <= 0:
                    if future.done():
                        return future.result()
                    raise DeadlineExceeded("stage deadline exceeded")
            try:
                return future.result(timeout=min(0.1, left))
            except FutureTimeout:
                if future.done():
                    raise
    except BaseException:
        future.cancel()
        raise


def run_stage(name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any):
    control = current_control()
    started = time.monotonic()
    try:
        if _in_worker.get():
            if control:
                control.remaining()
            return fn(*args, **kwargs)
        return await_work(IO_EXECUTOR.submit(fn, *args, **kwargs), control)
    finally:
        if control:
            control.timings[name] = control.timings.get(name, 0.0) + (time.monotonic() - started) * 1000
