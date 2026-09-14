"""可观测性单元测试。

覆盖：span 生命周期与嵌套、异常标记、trace 上下文传播、指标计数与分位统计、
JSONL 落盘与读取、事件记录、关闭开关后的降级行为。
"""

import json

import pytest

import pcb_rag.observability as observability
from pcb_rag.observability import (
    MetricsSnapshot,
    counter,
    metrics_snapshot,
    new_trace_id,
    observe,
    record_event,
    reset_metrics,
    span,
    timer,
    trace_context,
    trace_id,
    traces_tail,
)


@pytest.fixture(autouse=True)
def _isolated_trace_file(tmp_path, monkeypatch):
    """把 trace 落盘重定向到临时目录，避免测试污染工作区。"""

    monkeypatch.setattr(observability, "TRACE_LOG_PATH", str(tmp_path / "traces.jsonl"))
    monkeypatch.setattr(observability, "TRACE_ENABLED", True)
    monkeypatch.setattr(observability, "METRICS_ENABLED", True)
    monkeypatch.setattr(observability, "OBSERVABILITY_ENABLED", True)
    reset_metrics()
    yield
    observability.get_tracer().flush()


# ---------------------------------------------------------------------------
# Span
# ---------------------------------------------------------------------------
class TestSpan:
    def test_span_records_duration_and_attributes(self, tmp_path):
        with span("retrieve", query="镀金层厚度") as sp:
            sp.set(recalled=12)
            sp.event("cache.miss")

        observability.get_tracer().flush()
        rows = traces_tail(limit=5)
        record = next(r for r in rows if r["name"] == "retrieve")
        assert record["status"] == "ok"
        assert record["attributes"]["query"] == "镀金层厚度"
        assert record["attributes"]["recalled"] == 12
        assert record["events"][0]["name"] == "cache.miss"
        assert record["duration_ms"] >= 0

    def test_nested_spans_share_trace_and_link_parent(self):
        with span("outer"):
            with span("inner"):
                pass

        observability.get_tracer().flush()
        rows = [r for r in traces_tail(limit=10) if r["name"] in {"outer", "inner"}]
        outer = next(r for r in rows if r["name"] == "outer")
        inner = next(r for r in rows if r["name"] == "inner")
        assert outer["trace_id"] == inner["trace_id"]
        assert inner["parent_id"] == outer["span_id"]

    def test_exception_marks_error_and_propagates(self):
        with pytest.raises(ValueError):
            with span("failing"):
                raise ValueError("boom")

        observability.get_tracer().flush()
        record = next(r for r in traces_tail(limit=10) if r["name"] == "failing")
        assert record["status"] == "error"
        assert "ValueError" in record["error"]

    def test_long_attribute_is_truncated(self, monkeypatch):
        monkeypatch.setattr(observability, "TRACE_MAX_ATTR_LEN", 20)
        with span("truncate", payload="x" * 200):
            pass
        observability.get_tracer().flush()
        record = next(r for r in traces_tail(limit=10) if r["name"] == "truncate")
        assert record["attributes"]["payload"].endswith("chars>")

    def test_none_attributes_are_dropped(self):
        with span("none-attr", value=None, kept=1):
            pass
        observability.get_tracer().flush()
        record = next(r for r in traces_tail(limit=10) if r["name"] == "none-attr")
        assert "value" not in record["attributes"]
        assert record["attributes"]["kept"] == 1

    def test_disabled_tracing_yields_noop(self, monkeypatch):
        monkeypatch.setattr(observability, "TRACE_ENABLED", False)
        with span("disabled") as sp:
            sp.set(any=1)
        observability.get_tracer().flush()
        assert all(r["name"] != "disabled" for r in traces_tail(limit=10))

    def test_slow_span_is_flagged(self, monkeypatch):
        monkeypatch.setattr(observability, "SLOW_SPAN_MS", 0.0)
        with span("slow-op"):
            pass
        observability.get_tracer().flush()
        record = next(r for r in traces_tail(limit=10) if r["name"] == "slow-op")
        assert record.get("slow") is True


# ---------------------------------------------------------------------------
# Trace 上下文
# ---------------------------------------------------------------------------
class TestTraceContext:
    def test_explicit_trace_id_is_reused(self):
        with trace_context("abc123") as tid:
            assert tid == "abc123"
            assert trace_id() == "abc123"
            with span("inner"):
                assert trace_id() == "abc123"
        assert trace_id() is None

    def test_generated_trace_id_is_hex(self):
        with trace_context() as tid:
            assert len(tid) == 32
            int(tid, 16)

    def test_new_trace_id_unique(self):
        assert new_trace_id() != new_trace_id()


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
class TestMetrics:
    def test_counter_accumulates(self):
        counter("cache.hit")
        counter("cache.hit", 2)
        snapshot = metrics_snapshot()
        assert snapshot["counters"]["cache.hit"] == 3.0

    def test_observe_computes_percentiles(self):
        for value in (10, 20, 30, 40, 50):
            observe("retrieve.latency_ms", value)
        hist = metrics_snapshot()["histograms"]["retrieve.latency_ms"]
        assert hist["count"] == 5
        assert hist["min"] == 10
        assert hist["max"] == 50
        assert hist["p50"] == 30

    def test_timer_records_observation(self):
        with timer("unit.timer"):
            pass
        hist = metrics_snapshot()["histograms"]["unit.timer"]
        assert hist["count"] == 1
        assert hist["max"] >= 0

    def test_snapshot_wrapper(self):
        counter("x.y", 5)
        wrapper = MetricsSnapshot()
        assert wrapper.counter_value("x.y") == 5.0
        assert wrapper.histogram("missing") == {}

    def test_reset_clears_metrics(self):
        counter("a.b")
        observe("c.d", 1)
        reset_metrics()
        snapshot = metrics_snapshot()
        assert snapshot["counters"] == {}
        assert snapshot["histograms"] == {}

    def test_disabled_metrics_are_noop(self, monkeypatch):
        monkeypatch.setattr(observability, "METRICS_ENABLED", False)
        counter("ignored")
        assert metrics_snapshot()["counters"] == {}


# ---------------------------------------------------------------------------
# 事件与落盘
# ---------------------------------------------------------------------------
class TestEvents:
    def test_record_event_written_as_jsonl(self):
        record_event("graph.retrieve_empty", query="无关问题")
        observability.get_tracer().flush()
        rows = traces_tail(limit=10)
        event = next(r for r in rows if r["name"] == "graph.retrieve_empty")
        assert event["attributes"]["query"] == "无关问题"
        assert event["level"] == "info"

    def test_traces_tail_missing_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr(observability, "TRACE_LOG_PATH", str(tmp_path / "none.jsonl"))
        assert traces_tail() == []

    def test_flush_writes_valid_json_lines(self):
        with span("json-check", value=1):
            pass
        observability.get_tracer().flush()
        with open(observability.TRACE_LOG_PATH, "r", encoding="utf-8") as fh:
            lines = [line for line in fh if line.strip()]
        assert lines
        for line in lines:
            json.loads(line)
