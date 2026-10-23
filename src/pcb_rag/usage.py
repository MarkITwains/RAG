"""Request model usage, explicitly separating provider values and estimates."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path

from pcb_rag.context_budget import estimate_tokens
from pcb_rag.rerank_control import extract_usage
from pcb_rag.runtime import DeadlineExceeded, current_control, run_stage

_write_lock = threading.Lock()


class UsageBudgetExceeded(RuntimeError):
    pass


def request_limits(llm):
    metadata = getattr(llm, "metadata", None)
    window = int(getattr(metadata, "context_window", 0) or os.getenv("LLM_CONTEXT_WINDOW", "16384"))
    window = min(window, int(os.getenv("RAG_REQUEST_MAX_TOKENS", str(window))))
    reported = int(getattr(metadata, "num_output", 0) or getattr(llm, "max_tokens", 0) or 0)
    options = getattr(llm, "additional_kwargs", None) or {}
    output = (
        reported if reported > 0 else int(options.get("num_predict") or os.getenv("RAG_OUTPUT_RESERVE_TOKENS", "1024"))
    )
    reserve = max(output, int(os.getenv("RAG_OUTPUT_RESERVE_TOKENS", "1024")))
    envelope = int(os.getenv("RAG_MESSAGE_OVERHEAD_TOKENS", "128")) + estimate_tokens(
        str(getattr(llm, "system_prompt", "") or "")
    )
    tokenizer = getattr(llm, "tokenizer", None)
    count = (lambda value: len(tokenizer(value))) if callable(tokenizer) else estimate_tokens
    return window, output, reserve + envelope, count


def _call_options(llm, prompt, kwargs):
    window, output, reserve, count = request_limits(llm)
    if count(prompt) + reserve > window:
        raise ValueError("serialized model request exceeds its configured window")
    kwargs = dict(kwargs)
    if "openai" in type(llm).__module__:
        kwargs["max_tokens"] = min(int(kwargs.get("max_tokens") or output), output)
        control = current_control()
        if control:
            kwargs.setdefault("timeout", control.remaining())
    return kwargs


def _write_event(ticket, phase):
    ledger = os.getenv("RAG_USAGE_LEDGER", "")
    if ledger:
        path = Path(ledger)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _write_lock, path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(ticket, phase=phase), ensure_ascii=False) + "\n")


def reserve_usage(stage, texts, model=""):
    control = current_control()
    estimated = sum(estimate_tokens(text) for text in texts)
    ticket = {
        "id": uuid.uuid4().hex,
        "stage": stage,
        "model": model,
        "input_estimated_tokens": estimated,
        "actual_tokens": None,
        "usage_kind": "estimated",
        "status": "reserved",
        "timestamp": time.time(),
    }
    if control:
        control.remaining()
        with control.usage_lock:
            limit = int(os.getenv("RAG_" + stage.upper() + "_MAX_INPUT_TOKENS", "0"))
            used = sum(row["input_estimated_tokens"] for row in control.usage if row["stage"] == stage)
            calls = int(os.getenv("RAG_MAX_MODEL_CALLS", "0"))
            if (limit and used + estimated > limit) or (calls and len(control.usage) >= calls):
                raise UsageBudgetExceeded("configured model usage budget exhausted")
            control.usage.append(ticket)
    _write_event(ticket, "reserved")
    return ticket


def finish_usage(ticket, response=None, *, error=False):
    raw = getattr(response, "raw", None) or response
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump()
    if isinstance(raw, dict):
        tokens, source, usage = extract_usage(raw)
    else:
        tokens, source, usage = None, "unavailable", {}
    ticket.update(
        actual_tokens=tokens,
        usage_kind="actual" if tokens is not None else "estimated",
        usage_source=source,
        usage=usage,
        status="failed" if error else "complete",
    )
    _write_event(ticket, "finished")
    return ticket


def model_complete(llm, prompt, **kwargs):
    kwargs = _call_options(llm, prompt, kwargs)
    ticket = reserve_usage("llm", [prompt], str(getattr(llm, "model", "")))

    def invoke():
        ticket["status"] = "in_flight"
        try:
            response = llm.complete(prompt, **kwargs)
        except BaseException:
            finish_usage(ticket, error=True)
            raise
        finish_usage(ticket, response)
        return response

    try:
        return run_stage("llm", invoke)
    except DeadlineExceeded:
        # The worker may still receive a billable provider response. Its final
        # event reconciles this reservation instead of claiming zero usage.
        if ticket["status"] in {"reserved", "in_flight"}:
            ticket["status"] = "cancelled_before_send" if ticket["status"] == "reserved" else "cancelled_in_flight"
            _write_event(ticket, "cancelled")
        raise


def model_stream(llm, prompt):
    kwargs = _call_options(llm, prompt, {})
    ticket = reserve_usage("llm", [prompt], str(getattr(llm, "model", "")))
    stream = None
    last = None
    sentinel = object()
    try:
        stream = run_stage("llm_stream_open", llm.stream_complete, prompt, **kwargs)
        while True:
            chunk = run_stage("llm_stream", next, stream, sentinel)
            if chunk is sentinel:
                break
            last = chunk
            yield chunk
    except BaseException:
        finish_usage(ticket, last, error=True)
        raise
    finally:
        if stream is not None and hasattr(stream, "close"):
            try:
                stream.close()
            except (RuntimeError, ValueError):
                pass
    finish_usage(ticket, last)
