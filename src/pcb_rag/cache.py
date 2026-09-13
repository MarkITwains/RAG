"""PCB-RAG 语义缓存。

提供两级命中：

1. **精确匹配**：查询文本归一化（小写 + 折叠空白）后完全一致
2. **语义匹配**：查询向量与缓存条目的余弦相似度 >= ``SEMANTIC_CACHE_THRESHOLD``

带 TTL 过期与 LRU 容量上限，线程安全，纯内存实现（无需额外依赖）。

典型用法::

    from pcb_rag.cache import get_cache

    cache = get_cache("answer")
    hit = cache.get(query, embedding=query_vec)
    if hit is not None:
        return hit

    value = do_expensive_work(query)
    cache.set(query, value, embedding=query_vec)
    return value
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional, Sequence

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
SEMANTIC_CACHE_ENABLED = os.getenv("SEMANTIC_CACHE_ENABLED", "1") not in {"0", "false", "False"}
SEMANTIC_CACHE_MAX_SIZE = int(os.getenv("SEMANTIC_CACHE_MAX_SIZE", "512"))
SEMANTIC_CACHE_TTL_SECONDS = int(os.getenv("SEMANTIC_CACHE_TTL_SECONDS", "3600"))
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.95"))
# 语义检索时最多比较的条目数（避免缓存过大后线性扫描变慢）
SEMANTIC_CACHE_MAX_PROBE = int(os.getenv("SEMANTIC_CACHE_MAX_PROBE", "256"))


def normalize_query(query: str) -> str:
    """归一化查询文本，用于精确匹配。"""
    return " ".join((query or "").strip().lower().split())


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """纯 Python 余弦相似度（缓存规模小，无需引入 numpy 依赖）。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


@dataclass
class _Entry:
    value: Any
    created_at: float
    embedding: Optional[list[float]] = None
    query: str = ""


class SemanticCache:
    """线程安全的语义缓存。"""

    def __init__(
        self,
        max_size: int = SEMANTIC_CACHE_MAX_SIZE,
        ttl_seconds: int = SEMANTIC_CACHE_TTL_SECONDS,
        similarity_threshold: float = SEMANTIC_CACHE_THRESHOLD,
        enabled: bool = SEMANTIC_CACHE_ENABLED,
    ):
        self._store: "OrderedDict[str, _Entry]" = OrderedDict()
        self._max_size = max(1, int(max_size))
        self._ttl = max(0, int(ttl_seconds))
        self._threshold = float(similarity_threshold)
        self._enabled = bool(enabled)
        self._lock = threading.Lock()
        self._hits_exact = 0
        self._hits_semantic = 0
        self._misses = 0

    # ------------------------------------------------------------------ 内部
    def _is_expired(self, entry: _Entry, now: float) -> bool:
        return self._ttl > 0 and (now - entry.created_at) > self._ttl

    def _evict_locked(self) -> None:
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def _purge_expired_locked(self, now: float) -> None:
        if self._ttl <= 0:
            return
        expired = [k for k, e in self._store.items() if self._is_expired(e, now)]
        for k in expired:
            self._store.pop(k, None)

    # ------------------------------------------------------------------ 读写
    def get(self, query: str, embedding: Optional[Sequence[float]] = None) -> Optional[Any]:
        """命中返回缓存值，未命中返回 None。"""
        if not self._enabled or not query:
            return None

        key = normalize_query(query)
        now = time.time()

        with self._lock:
            self._purge_expired_locked(now)

            # 1) 精确匹配
            entry = self._store.get(key)
            if entry is not None and not self._is_expired(entry, now):
                self._store.move_to_end(key)
                self._hits_exact += 1
                return entry.value

            # 2) 语义匹配
            if embedding is not None:
                probe = 0
                best_key: Optional[str] = None
                best_score = 0.0
                for k in reversed(self._store.keys()):
                    probe += 1
                    if probe > SEMANTIC_CACHE_MAX_PROBE:
                        break
                    cand = self._store[k]
                    if cand.embedding is None:
                        continue
                    score = cosine_similarity(embedding, cand.embedding)
                    if score > best_score:
                        best_score = score
                        best_key = k

                if best_key is not None and best_score >= self._threshold:
                    self._store.move_to_end(best_key)
                    self._hits_semantic += 1
                    return self._store[best_key].value

            self._misses += 1
            return None

    def set(
        self,
        query: str,
        value: Any,
        embedding: Optional[Sequence[float]] = None,
    ) -> None:
        """写入缓存；``embedding`` 为空时只支持精确匹配。"""
        if not self._enabled or not query:
            return

        key = normalize_query(query)
        entry = _Entry(
            value=value,
            created_at=time.time(),
            embedding=list(embedding) if embedding is not None else None,
            query=query,
        )
        with self._lock:
            self._purge_expired_locked(time.time())
            self._store[key] = entry
            self._store.move_to_end(key)
            self._evict_locked()

    def clear(self) -> int:
        with self._lock:
            n = len(self._store)
            self._store.clear()
            return n

    # ------------------------------------------------------------------ 统计
    def stats(self) -> dict:
        with self._lock:
            total = self._hits_exact + self._hits_semantic + self._misses
            return {
                "enabled": self._enabled,
                "size": len(self._store),
                "max_size": self._max_size,
                "ttl_seconds": self._ttl,
                "similarity_threshold": self._threshold,
                "hits_exact": self._hits_exact,
                "hits_semantic": self._hits_semantic,
                "misses": self._misses,
                "hit_rate": round((self._hits_exact + self._hits_semantic) / total, 4) if total else 0.0,
            }


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------
_caches: dict[str, SemanticCache] = {}
_caches_lock = threading.Lock()


def get_cache(name: str = "default") -> SemanticCache:
    """获取（或创建）指定名称的缓存单例。"""
    with _caches_lock:
        cache = _caches.get(name)
        if cache is None:
            cache = SemanticCache()
            _caches[name] = cache
        return cache


def all_cache_stats() -> dict:
    """汇总所有缓存实例的统计信息（供 /health 展示）。"""
    with _caches_lock:
        return {name: cache.stats() for name, cache in _caches.items()}


def embed_query_for_cache(query: str) -> Optional[list[float]]:
    """用当前 Embedding 后端计算查询向量，失败返回 None（降级为精确匹配）。"""
    if not SEMANTIC_CACHE_ENABLED or not query:
        return None
    try:
        from llama_index.core import Settings

        if Settings.embed_model is None:
            return None
        vec = Settings.embed_model.get_query_embedding(query)
        return list(vec) if vec else None
    except Exception:
        return None
