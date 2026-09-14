"""``pcb_rag.cache`` 语义缓存单元测试。

缓存模块只依赖标准库，因此这些用例可在 CI 中脱离 Milvus / Ollama / 模型服务
独立运行，用于保护缓存命中、过期与淘汰逻辑不被回归破坏。
"""

import threading

import pytest

from pcb_rag.cache import (
    SemanticCache,
    all_cache_stats,
    cosine_similarity,
    get_cache,
    normalize_query,
)


class TestNormalizeQuery:
    def test_lowercases_and_collapses_whitespace(self):
        assert normalize_query("  PCB   阻抗控制 ") == "pcb 阻抗控制"

    def test_empty_inputs_are_safe(self):
        assert normalize_query("") == ""
        assert normalize_query("   ") == ""


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_dimension_mismatch_returns_zero(self):
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0

    def test_empty_or_zero_vectors_return_zero(self):
        assert cosine_similarity([], [1.0]) == 0.0
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


class TestExactMatch:
    def test_hit_after_normalization(self):
        cache = SemanticCache()
        cache.set("Hello  World", "answer")
        assert cache.get("hello world") == "answer"

    def test_miss_returns_none(self):
        cache = SemanticCache()
        assert cache.get("未写入的问题") is None

    def test_empty_query_is_ignored(self):
        cache = SemanticCache()
        cache.set("", "value")
        assert cache.get("") is None
        assert cache.stats()["size"] == 0

    def test_disabled_cache_never_stores(self):
        cache = SemanticCache(enabled=False)
        cache.set("q", "v")
        assert cache.get("q") is None
        assert cache.stats()["size"] == 0

    def test_clear_returns_removed_count(self):
        cache = SemanticCache()
        cache.set("a", 1)
        cache.set("b", 2)
        assert cache.clear() == 2
        assert cache.get("a") is None


class TestSemanticMatch:
    def test_near_vector_hits(self):
        cache = SemanticCache(similarity_threshold=0.95)
        cache.set("4 层板阻抗", "答案A", embedding=[1.0, 0.0])
        assert cache.get("4层板阻抗如何控制", embedding=[1.0, 0.001]) == "答案A"

    def test_far_vector_misses(self):
        cache = SemanticCache(similarity_threshold=0.95)
        cache.set("4 层板阻抗", "答案A", embedding=[1.0, 0.0])
        assert cache.get("无关问题", embedding=[0.0, 1.0]) is None

    def test_similarity_threshold_is_respected(self):
        cache = SemanticCache(similarity_threshold=0.999999)
        cache.set("q1", "v1", embedding=[1.0, 0.0])
        assert cache.get("q2", embedding=[1.0, 0.05]) is None

    def test_entry_without_embedding_is_not_matched_semantically(self):
        cache = SemanticCache(similarity_threshold=0.0)
        cache.set("q1", "v1")
        assert cache.get("q2", embedding=[1.0, 0.0]) is None

    def test_exact_match_takes_priority_over_semantic(self):
        cache = SemanticCache(similarity_threshold=0.0)
        cache.set("q1", "v1", embedding=[1.0, 0.0])
        cache.set("q2", "v2", embedding=[0.0, 1.0])
        # q2 精确命中，不应被语义更接近的 q1 抢占
        assert cache.get("q2", embedding=[1.0, 0.0]) == "v2"


class TestExpiryAndEviction:
    @staticmethod
    def _age_entries(cache: SemanticCache, seconds: float) -> None:
        for entry in cache._store.values():
            entry.created_at -= seconds

    def test_expired_entry_is_purged(self):
        cache = SemanticCache(ttl_seconds=60)
        cache.set("q", "v")
        self._age_entries(cache, 120)
        assert cache.get("q") is None
        assert cache.stats()["size"] == 0

    def test_zero_ttl_disables_expiry(self):
        cache = SemanticCache(ttl_seconds=0)
        cache.set("q", "v")
        self._age_entries(cache, 10**6)
        assert cache.get("q") == "v"

    def test_lru_eviction_keeps_most_recent(self):
        cache = SemanticCache(max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_recently_read_entry_survives_eviction(self):
        cache = SemanticCache(max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.get("a")  # a 变为最近使用
        cache.set("c", 3)  # 容量超限，淘汰 b
        assert cache.get("a") == 1
        assert cache.get("b") is None


class TestStats:
    def test_counters_and_hit_rate(self):
        cache = SemanticCache()
        cache.set("a", 1)
        cache.get("a")
        cache.get("miss")
        stats = cache.stats()
        assert stats["hits_exact"] == 1
        assert stats["hits_semantic"] == 0
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(0.5)
        assert stats["size"] == 1

    def test_semantic_hit_counter(self):
        cache = SemanticCache(similarity_threshold=0.9)
        cache.set("q1", "v1", embedding=[1.0, 0.0])
        cache.get("q2", embedding=[1.0, 0.001])
        assert cache.stats()["hits_semantic"] == 1

    def test_hit_rate_is_zero_without_traffic(self):
        assert SemanticCache().stats()["hit_rate"] == 0.0


class TestGlobalRegistry:
    def test_get_cache_returns_singleton(self):
        assert get_cache("unit-cache") is get_cache("unit-cache")

    def test_all_cache_stats_reports_registered_cache(self):
        cache = get_cache("unit-cache-stats")
        cache.set("q", "v")
        assert all_cache_stats()["unit-cache-stats"]["size"] >= 1


class TestConcurrency:
    def test_concurrent_access_keeps_state_consistent(self):
        cache = SemanticCache(max_size=64)
        errors = []

        def worker(offset: int) -> None:
            try:
                for i in range(50):
                    key = f"q-{offset}-{i}"
                    cache.set(key, i)
                    cache.get(key)
            except Exception as exc:  # pragma: no cover - 仅用于失败时暴露异常
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        assert cache.stats()["size"] <= 64
