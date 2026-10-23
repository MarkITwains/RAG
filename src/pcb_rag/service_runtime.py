"""ASGI request lifetime: bounded admission, identity, deadline and disconnects."""

from __future__ import annotations

import asyncio
import json
import os
import threading

from starlette.responses import JSONResponse

from pcb_rag import security
from pcb_rag.observability import trace_context
from pcb_rag.runtime import DeadlineExceeded, request_scope
from pcb_rag.sessions import SessionAccessDenied, principal_from_token


class RequestBoundaryMiddleware:
    def __init__(self, app, *, concurrent=None, queued=None, timeout=None):
        self.app = app
        self.running = threading.BoundedSemaphore(concurrent or int(os.getenv("RAG_MAX_CONCURRENT", "8")))
        self.waiting = threading.BoundedSemaphore(int(os.getenv("RAG_MAX_QUEUED", "16")) if queued is None else queued)
        self.timeout = float(timeout or os.getenv("RAG_REQUEST_TIMEOUT", "120"))

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.decode().lower(): value.decode() for key, value in scope.get("headers", [])}
        principal = principal_from_token(headers.get("authorization"))
        if principal is None:
            trusted = security.ACL_TRUST_HEADERS and security.ACL_TRUST_HEADERS_ACK
            principal = security.resolve_principal(headers if trusted else None)
        principal_token = security.set_request_principal(principal)
        expensive = scope.get("path", "").startswith(("/retrieval", "/api/ask", "/api/chat", "/api/graph"))
        admitted = False
        started = False
        disconnected = False
        queue_slot = False
        messages = asyncio.Queue(maxsize=8)
        pump = None
        with request_scope(self.timeout) as control, trace_context(headers.get("x-trace-id")) as trace:

            async def receive_pump():
                nonlocal disconnected
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        disconnected = True
                        control.cancelled.set()
                        await messages.put(message)
                        return
                    await messages.put(message)

            async def wrapped_send(message):
                nonlocal started
                if disconnected:
                    return
                if message["type"] == "http.response.start":
                    started = True
                    from pcb_rag.retrieval import get_pipeline

                    pipeline = get_pipeline()
                    extra = [
                        (b"x-trace-id", trace.encode()),
                        (b"x-kb-revision", control.revision.encode()),
                        (b"x-pipeline-fingerprint", pipeline.config.fingerprint.encode()),
                        (
                            b"x-rag-degraded",
                            ",".join(dict.fromkeys(control.degraded)).encode("ascii", errors="replace"),
                        ),
                    ]
                    message = dict(message, headers=list(message.get("headers", [])) + extra)
                await send(message)

            try:
                pump = asyncio.create_task(receive_pump())
                if expensive:
                    admitted = self.running.acquire(blocking=False)
                    if not admitted:
                        queue_slot = self.waiting.acquire(blocking=False)
                        if not queue_slot:
                            await JSONResponse(
                                {"detail": "request queue is full"}, status_code=429, headers={"Retry-After": "1"}
                            )(scope, receive, send)
                            return
                        while not admitted:
                            control.remaining()
                            await asyncio.sleep(0.01)
                            admitted = self.running.acquire(blocking=False)
                    if queue_slot:
                        self.waiting.release()
                        queue_slot = False
                await asyncio.wait_for(self.app(scope, messages.get, wrapped_send), timeout=control.remaining())
            except (DeadlineExceeded, asyncio.TimeoutError):
                control.cancelled.set()
                if disconnected:
                    pass
                elif not started:
                    await JSONResponse({"detail": "request deadline exceeded"}, status_code=504)(scope, receive, send)
                elif not scope.get("path", "").endswith("/stream"):
                    await send({"type": "http.response.body", "body": b"", "more_body": False})
                else:
                    body = "event: error\ndata: " + json.dumps({"message": "request deadline exceeded"}) + "\n\n"
                    await send({"type": "http.response.body", "body": body.encode(), "more_body": False})
            except SessionAccessDenied:
                if not started:
                    await JSONResponse({"detail": "session not found or identity required"}, status_code=404)(
                        scope, receive, send
                    )
                else:
                    raise
            finally:
                control.cancelled.set()
                if pump:
                    pump.cancel()
                    try:
                        await pump
                    except asyncio.CancelledError:
                        pass
                if admitted:
                    self.running.release()
                if queue_slot:
                    self.waiting.release()
                security.reset_request_principal(principal_token)
