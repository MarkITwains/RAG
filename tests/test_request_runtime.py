import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from pcb_rag.runtime import (
    BoundedExecutor,
    DeadlineExceeded,
    WorkRejected,
    await_work,
    current_control,
    request_scope,
    run_stage,
)
from pcb_rag.service_runtime import RequestBoundaryMiddleware
from pcb_rag.usage import UsageBudgetExceeded, model_complete


def test_cancelled_io_keeps_worker_slot_until_the_actual_call_finishes():
    executor = BoundedExecutor(1, 0)
    started, release = threading.Event(), threading.Event()

    def work():
        started.set()
        release.wait(2)

    try:
        with request_scope(0.03) as control:
            future = executor.submit(work)
            assert started.wait(1)
            with pytest.raises(DeadlineExceeded):
                await_work(future, control)
            with pytest.raises(WorkRejected):
                executor.submit(lambda: None)
        release.set()
        future.result(timeout=1)
        assert executor.submit(lambda: "released").result(timeout=1) == "released"
    finally:
        release.set()
        executor.close()


def test_nested_stages_share_control_and_do_not_deadlock():
    with request_scope(1) as control:
        assert run_stage("outer", lambda: run_stage("inner", lambda: current_control() is control))


def test_local_wait_budget_accepts_already_completed_work():
    executor = BoundedExecutor(1, 0)
    try:
        future = executor.submit(lambda: "done")
        assert future.result(timeout=1) == "done"
        with request_scope(1) as control:
            assert await_work(future, control, timeout=0) == "done"
    finally:
        executor.close()


def test_llm_transport_timeout_is_capped_by_client_and_request_deadline():
    from pcb_rag.usage import _call_options

    class LLM:
        timeout = 12.0
        __module__ = "llama_index.llms.openai.base"

    with request_scope(30):
        assert _call_options(LLM(), "query", {})["timeout"] <= 12.0
    with request_scope(0.01):
        assert _call_options(LLM(), "query", {})["timeout"] <= 0.01


def test_explicit_llm_timeout_is_not_replaced_by_request_deadline():
    from pcb_rag.usage import _call_options

    class LLM:
        timeout = 12.0
        __module__ = "llama_index.llms.openai.base"

    with request_scope(30):
        assert _call_options(LLM(), "query", {"timeout": 4.0})["timeout"] == 4.0


def test_model_budget_is_enforced_before_transport_and_actual_usage_is_distinct(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_MAX_MODEL_CALLS", "1")
    monkeypatch.setenv("RAG_USAGE_LEDGER", str(tmp_path / "usage.jsonl"))
    calls = []

    def complete(prompt):
        calls.append(prompt)
        return SimpleNamespace(text="answer", raw={"usage": {"total_tokens": 7}})

    with request_scope(1) as control:
        llm = SimpleNamespace(complete=complete, model="stub")
        model_complete(llm, "query")
        with pytest.raises(UsageBudgetExceeded):
            model_complete(llm, "second query")
        assert len(calls) == 1
        assert control.usage[0]["actual_tokens"] == 7
        assert control.usage[0]["input_estimated_tokens"] == 5
    import json

    events = [json.loads(line) for line in (tmp_path / "usage.jsonl").read_text().splitlines()]
    assert [event["phase"] for event in events] == ["reserved", "finished"]
    assert events[0]["id"] == events[1]["id"]


def test_late_llm_response_reconciles_usage_after_deadline():
    release, started, finished = threading.Event(), threading.Event(), threading.Event()

    def complete(prompt):
        started.set()
        release.wait(2)
        return SimpleNamespace(raw={"usage": {"total_tokens": 9}}, text="late")

    with request_scope(0.03) as control:
        with pytest.raises(DeadlineExceeded):
            model_complete(SimpleNamespace(complete=complete), "query")
        assert started.is_set()
        assert control.usage[0]["actual_tokens"] is None
    release.set()
    # Wait for accounting, with a bounded poll instead of racing the worker.
    until = time.monotonic() + 1
    while control.usage[0]["actual_tokens"] is None and time.monotonic() < until:
        finished.wait(0.005)
    assert control.usage[0]["actual_tokens"] == 9


def _scope():
    return {"type": "http", "path": "/api/ask", "headers": []}


def test_asgi_admission_queue_is_bounded():
    async def exercise():
        entered, release = asyncio.Event(), asyncio.Event()

        async def app(scope, receive, send):
            entered.set()
            await release.wait()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        boundary = RequestBoundaryMiddleware(app, concurrent=1, queued=0, timeout=2)
        queue = asyncio.Queue()
        sent1, sent2 = [], []

        async def save1(message):
            sent1.append(message)

        async def save2(message):
            sent2.append(message)

        first = asyncio.create_task(boundary(_scope(), queue.get, save1))
        await entered.wait()
        await boundary(_scope(), queue.get, save2)
        assert sent2[0]["status"] == 429
        release.set()
        await first
        assert sent1[0]["status"] == 200

    asyncio.run(exercise())


def test_disconnect_cancels_running_request_and_prevents_followup_work():
    async def exercise():
        queue = asyncio.Queue()
        entered = asyncio.Event()
        captured, calls = [], []

        async def app(scope, receive, send):
            control = current_control()
            captured.append(control)
            entered.set()
            while not control.cancelled.is_set():
                await asyncio.sleep(0.005)
            with pytest.raises(DeadlineExceeded):
                run_stage("should_not_run", lambda: calls.append("late"))

        boundary = RequestBoundaryMiddleware(app, timeout=1)

        async def send(message):
            pass

        task = asyncio.create_task(boundary(_scope(), queue.get, send))
        await entered.wait()
        await queue.put({"type": "http.disconnect"})
        await task
        assert captured[0].cancelled.is_set()
        assert not calls

    asyncio.run(exercise())


def test_stream_deadline_closes_the_response_with_error():
    async def exercise():
        queue, sent = asyncio.Queue(), []

        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"event: token\ndata: {}\n\n", "more_body": True})
            await asyncio.sleep(1)

        async def send(message):
            sent.append(message)

        boundary = RequestBoundaryMiddleware(app, timeout=0.03)
        await boundary(dict(_scope(), path="/api/ask/stream"), queue.get, send)
        assert sent[-1]["more_body"] is False
        assert b"event: error" in sent[-1]["body"]

    asyncio.run(exercise())
