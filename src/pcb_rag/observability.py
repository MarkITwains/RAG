"""轻量可观测性模块：结构化 Tracing + 指标采集 + 事件记录。

设计目标
--------
- **零第三方依赖**：只用标准库，任何模块都能安全导入（不会拖入 torch / llama-index）
- **线程安全**：检索链路存在并发调用（Contextual Retrieval、批量评测）
- **默认低开销**：关闭或采样未命中时只有一次布尔判断
- **落地为 JSONL**：便于 grep / jq / 日志采集器直接消费，无需额外后端

三个能力
--------
1. ``span``：上下文管理器 / 装饰器，记录每个阶段耗时与属性，串成完整调用链
2. ``counter`` / ``observe`` / ``timer``：计数器与延迟直方图，供 ``/metrics`` 暴露
3. ``record_event``：关键业务事件（缓存命中、降级回退、检索为空）结构化落盘

用法::

    from pcb_rag.observability import span, counter, observe, metrics_snapshot

    with span("retrieve", query=query) as sp:
        nodes = retriever.retrieve(query)
        sp.set(recalled=len(nodes))

    counter("cache.hit")
    observe("retrieve.latency_ms", 12.5)
    print(metrics_snapshot())
"""

from __future__ import annotations

import atexit
import json
import os
import statistics
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional

# ---------------------------------------------------------------------------
# 1. 配置
# ---------------------------------------------------------------------------
def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


OBSERVABILITY_ENABLED = _env_bool("OBSERVABILITY_ENABLED", True)
TRACE_ENABLED = _env_bool("TRACE_ENABLED", True) and OBSERVABILITY_ENABLED
METRICS_ENABLED = _env_bool("METRICS_ENABLED", True) and OBSERVABILITY_ENABLED

TRACE_LOG_PATH = os.getenv("TRACE_LOG_PATH", "./data/traces.jsonl").strip()
# 采样率：1.0 = 全量记录；0.1 = 只记录 10% 的 trace（高 QPS 场景降开销）
TRACE_SAMPLE_RATE = max(0.0, min(1.0, float(os.getenv("TRACE_SAMPLE_RATE", "1.0"))))
# 单个属性值最大长度，避免把整篇文档写进日志
TRACE_MAX_ATTR_LEN = int(os.getenv("TRACE_MAX_ATTR_LEN", "600"))
# 慢 span 阈值（毫秒），超过则在日志中标记 slow=true
SLOW_SPAN_MS = float(os.getenv("SLOW_SPAN_MS", "3000"))
# 每个指标保留的样本数（直方图分位统计窗口）
METRIC_WINDOW = int(os.getenv("METRIC_WINDOW", "2048"))
# 落盘缓冲：累计多少条记录后刷盘
TRACE_FLUSH_EVERY = int(os.getenv("TRACE_FLUSH_EVERY", "32"))
TRACE_FLUSH_INTERVAL = float(os.getenv("TRACE_FLUSH_INTERVAL", "5"))

__all__ = [
    "OBSERVABILITY_ENABLED",
    "MetricsSnapshot",
    "Span",
    "counter",
    "get_metrics",
    "get_tracer",
    "metrics_snapshot",
    "new_trace_id",
    "observe",
    "record_event",
    "reset_metrics",
    "span",
    "timer",
    "trace_context",
    "trace_id",
    "traces_tail",
]

# ---------------------------------------------------------------------------
# 2. 上下文（ContextVar，兼容线程与 asyncio）
# ---------------------------------------------------------------------------
_current_trace: ContextVar[Optional[str]] = ContextVar("pcb_rag_trace_id", default=None)
_current_span: ContextVar[Optional[str]] = ContextVar("pcb_rag_span_id", default=None)
_trace_sampled: ContextVar[bool] = ContextVar("pcb_rag_trace_sampled", default=True)


def new_trace_id() -> str:
    """生成一个新的 trace id（32 位十六进制，便于与 W3C traceparent 对齐）。"""

    return uuid.uuid4().hex


def trace_id() -> Optional[str]:
    """返回当前上下文的 trace id；不在任何 trace 中时返回 ``None``。"""

    return _current_trace.get()


@contextmanager
def trace_context(trace: Optional[str] = None) -> Iterator[str]:
    """显式绑定一个 trace id（HTTP 请求入口 / 评测批处理时使用）。

    ``trace`` 为空时自动生成；调用方可把它回写到响应头，实现前后端链路对齐。
    """

    tid = trace or _current_trace.get() or new_trace_id()
    token_trace = _current_trace.set(tid)
    token_span = _current_span.set(None)
    token_sampled = _trace_sampled.set(True if TRACE_SAMPLE_RATE >= 1.0 else (uuid.uuid4().int % 1000) < TRACE_SAMPLE_RATE * 1000)
    try:
        yield tid
    finally:
        _current_trace.reset(token_trace)
        _current_span.reset(token_span)
        _trace_sampled.reset(token_sampled)


# ---------------------------------------------------------------------------
# 3. Span
# ---------------------------------------------------------------------------
def _truncate(value: Any, limit: Optional[int] = None) -> Any:
    """把属性值裁剪到可落盘的长度，避免日志体积失控。

    ``limit`` 缺省时**运行时**读取 ``TRACE_MAX_ATTR_LEN``（而不是在函数定义时绑定），
    这样配置在模块导入后仍可修改——测试与运行期热调整都依赖这一点。
    """

    max_len = TRACE_MAX_ATTR_LEN if limit is None else int(limit)

    if isinstance(value, str):
        return value if len(value) <= max_len else value[:max_len] + f"...<{len(value)} chars>"
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_truncate(v, max_len) for v in list(value)[:20]]
    if isinstance(value, dict):
        return {str(k): _truncate(v, max_len) for k, v in list(value.items())[:20]}
    return _truncate(str(value), max_len)


class Span:
    """一个可观测的时间段，记录名称、耗时、属性、状态与嵌套事件。"""

    __slots__ = (
        "name",
        "span_id",
        "parent_id",
        "trace",
        "attributes",
        "events",
        "status",
        "error",
        "_start",
        "_end",
        "_sampled",
    )

    def __init__(self, name: str, parent_id: Optional[str], trace: str, sampled: bool, **attrs: Any) -> None:
        self.name = name
        self.span_id = uuid.uuid4().hex[:16]
        self.parent_id = parent_id
        self.trace = trace
        self.attributes: Dict[str, Any] = {k: _truncate(v) for k, v in attrs.items() if v is not None}
        self.events: List[Dict[str, Any]] = []
        self.status = "ok"
        self.error: Optional[str] = None
        self._start = time.perf_counter()
        self._end: Optional[float] = None
        self._sampled = sampled

    # ---- 属性 / 事件 ----
    def set(self, **attrs: Any) -> "Span":
        """补充属性（None 值会被忽略，便于直接传可选参数）。"""

        for key, value in attrs.items():
            if value is not None:
                self.attributes[key] = _truncate(value)
        return self

    def event(self, name: str, **fields: Any) -> "Span":
        """在 span 内记录一个瞬时事件（如「缓存命中」「回退到纯向量检索」）。"""

        if self._sampled:
            self.events.append({"name": name, "t": round((time.perf_counter() - self._start) * 1000, 3), **{k: _truncate(v) for k, v in fields.items()}})
        return self

    def fail(self, exc: BaseException | str) -> "Span":
        """标记该 span 失败（异常仍由调用方决定是否继续抛出）。"""

        self.status = "error"
        self.error = f"{type(exc).__name__}: {exc}" if isinstance(exc, BaseException) else str(exc)
        return self

    @property
    def duration_ms(self) -> float:
        end = self._end if self._end is not None else time.perf_counter()
        return (end - self._start) * 1000.0

    def to_record(self) -> Dict[str, Any]:
        """转换为可 JSON 序列化的记录。"""

        record: Dict[str, Any] = {
            "ts": time.time(),
            "trace_id": self.trace,
            "span_id": self.span_id,
            "name": self.name,
            "duration_ms": round(self.duration_ms, 3),
            "status": self.status,
        }
        if self.parent_id:
            record["parent_id"] = self.parent_id
        if self.attributes:
            record["attributes"] = self.attributes
        if self.events:
            record["events"] = self.events
        if self.error:
            record["error"] = self.error
        if self.duration_ms >= SLOW_SPAN_MS:
            record["slow"] = True
        return record


# ---------------------------------------------------------------------------
# 4. Tracer（落盘）
# ---------------------------------------------------------------------------
class _Tracer:
    """把 span 记录按 JSONL 追加落盘；带缓冲与采样，失败时静默降级。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buffer: List[str] = []
        self._last_flush = time.time()
        self._disabled = not TRACE_ENABLED

    def emit(self, span: Span) -> None:
        if self._disabled or not span._sampled:
            return
        try:
            line = json.dumps(span.to_record(), ensure_ascii=False)
        except Exception:
            return

        with self._lock:
            self._buffer.append(line)
            should_flush = len(self._buffer) >= TRACE_FLUSH_EVERY or (time.time() - self._last_flush) >= TRACE_FLUSH_INTERVAL
            if should_flush:
                self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._buffer:
            return
        lines, self._buffer = self._buffer, []
        self._last_flush = time.time()
        try:
            path = os.path.abspath(TRACE_LOG_PATH)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
        except Exception:
            # 观测能力不应影响主流程：磁盘满 / 无权限时静默放弃这批记录
            pass

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()


_tracer = _Tracer()
atexit.register(_tracer.flush)


def get_tracer() -> _Tracer:
    return _tracer


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[Span]:
    """开启一个 span；异常时自动标记 error 并原样抛出。"""

    if not TRACE_ENABLED:
        yield _NoopSpan(name)
        return

    parent = _current_span.get()
    tid = _current_trace.get()
    if tid is None:
        tid = new_trace_id()
        token_trace = _current_trace.set(tid)
        token_sampled = _trace_sampled.set(True if TRACE_SAMPLE_RATE >= 1.0 else (uuid.uuid4().int % 1000) < TRACE_SAMPLE_RATE * 1000)
    else:
        token_trace = None
        token_sampled = None

    sp = Span(name, parent, tid, _trace_sampled.get(), **attrs)
    token_span = _current_span.set(sp.span_id)
    try:
        yield sp
    except BaseException as exc:  # noqa: BLE001 - 需要记录任意异常后再抛出
        sp.fail(exc)
        _tracer.emit(sp)
        raise
    else:
        _tracer.emit(sp)
    finally:
        _current_span.reset(token_span)
        if token_trace is not None:
            _current_trace.reset(token_trace)
        if token_sampled is not None:
            _trace_sampled.reset(token_sampled)


class _NoopSpan:
    """关闭观测时返回的空对象，接口与 ``Span`` 保持一致。"""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def set(self, **_attrs: Any) -> "_NoopSpan":
        return self

    def event(self, _name: str, **_fields: Any) -> "_NoopSpan":
        return self

    def fail(self, _exc: Any) -> "_NoopSpan":
        return self

    @property
    def duration_ms(self) -> float:
        return 0.0


def traced(name: Optional[str] = None, **attrs: Any) -> Callable:
    """装饰器版 span：``@traced("rerank")``。"""

    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with span(span_name, **attrs):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# 5. 指标
# ---------------------------------------------------------------------------
class _Metrics:
    """计数器 + 直方图（滑动窗口），线程安全，可随时快照为 JSON。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[str, float] = {}
        self._histograms: Dict[str, Deque[float]] = {}
        self._started = time.time()

    def counter(self, name: str, value: float = 1.0) -> None:
        if not METRICS_ENABLED:
            return
        with self._lock:
            self._counters[name] = self._counters.get(name, 0.0) + float(value)

    def observe(self, name: str, value: float) -> None:
        if not METRICS_ENABLED:
            return
        with self._lock:
            bucket = self._histograms.get(name)
            if bucket is None:
                bucket = deque(maxlen=METRIC_WINDOW)
                self._histograms[name] = bucket
            bucket.append(float(value))

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            counters = dict(self._counters)
            histograms = {k: list(v) for k, v in self._histograms.items()}

        summary: Dict[str, Any] = {}
        for name, samples in histograms.items():
            if not samples:
                continue
            ordered = sorted(samples)
            summary[name] = {
                "count": len(ordered),
                "mean": round(statistics.fmean(ordered), 3),
                "min": round(ordered[0], 3),
                "max": round(ordered[-1], 3),
                "p50": round(_percentile(ordered, 50), 3),
                "p95": round(_percentile(ordered, 95), 3),
                "p99": round(_percentile(ordered, 99), 3),
            }

        return {
            "enabled": OBSERVABILITY_ENABLED,
            "uptime_seconds": round(time.time() - self._started, 1),
            "counters": counters,
            "histograms": summary,
        }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._histograms.clear()


def _percentile(ordered: List[float], pct: float) -> float:
    """线性插值分位数（ordered 必须已升序）。"""

    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    pos = (pct / 100.0) * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


_metrics = _Metrics()


def get_metrics() -> _Metrics:
    return _metrics


def counter(name: str, value: float = 1.0) -> None:
    """累加一个计数器，如 ``counter("graph.triples_extracted", 12)``。"""

    _metrics.counter(name, value)


def observe(name: str, value: float) -> None:
    """记录一个观测值（通常是耗时或长度）。"""

    _metrics.observe(name, value)


@contextmanager
def timer(name: str) -> Iterator[None]:
    """计时上下文：``with timer("retrieve.latency_ms"): ...``。"""

    start = time.perf_counter()
    try:
        yield
    finally:
        _metrics.observe(name, (time.perf_counter() - start) * 1000.0)


def metrics_snapshot() -> Dict[str, Any]:
    """返回当前指标快照（可直接作为 ``/metrics`` 的 JSON 响应）。"""

    return _metrics.snapshot()


def reset_metrics() -> None:
    _metrics.reset()


# ---------------------------------------------------------------------------
# 6. 事件与日志辅助
# ---------------------------------------------------------------------------
def record_event(name: str, level: str = "info", **fields: Any) -> None:
    """记录一条独立业务事件（不依附于 span）。

    用于「缓存命中」「rerank 降级」「图检索无结果」这类值得统计但不需要耗时的点。
    """

    if not OBSERVABILITY_ENABLED:
        return
    try:
        record = {
            "ts": time.time(),
            "trace_id": _current_trace.get(),
            "span_id": _current_span.get(),
            "name": name,
            "level": level,
            "attributes": {k: _truncate(v) for k, v in fields.items() if v is not None},
        }
        line = json.dumps(record, ensure_ascii=False)
    except Exception:
        return

    with _tracer._lock:
        _tracer._buffer.append(line)
        if len(_tracer._buffer) >= TRACE_FLUSH_EVERY:
            _tracer._flush_locked()


def traces_tail(limit: int = 20) -> List[Dict[str, Any]]:
    """读取最近的 trace 记录（调试用；文件不存在或损坏时返回已解析的部分）。"""

    _tracer.flush()
    path = os.path.abspath(TRACE_LOG_PATH)
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return rows[-limit:]


def describe_observability() -> Dict[str, Any]:
    """返回观测配置摘要，供启动日志与 ``/health`` 展示。"""

    return {
        "enabled": OBSERVABILITY_ENABLED,
        "tracing": TRACE_ENABLED,
        "metrics": METRICS_ENABLED,
        "trace_log": TRACE_LOG_PATH,
        "sample_rate": TRACE_SAMPLE_RATE,
        "slow_span_ms": SLOW_SPAN_MS,
        "metric_window": METRIC_WINDOW,
    }


class MetricsSnapshot:
    """指标快照的轻量包装，便于在 FastAPI 端点里做增量对比。"""

    __slots__ = ("data",)

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        self.data = data if data is not None else metrics_snapshot()

    def counter_value(self, name: str) -> float:
        return float(self.data.get("counters", {}).get(name, 0.0))

    def histogram(self, name: str) -> Dict[str, Any]:
        return dict(self.data.get("histograms", {}).get(name, {}))

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return f"MetricsSnapshot(counters={len(self.data.get('counters', {}))}, histograms={len(self.data.get('histograms', {}))})"
