"""Evaluation entrypoints must honor explicit CLI overrides and supported adapters."""

from types import SimpleNamespace

import pytest

pytest.importorskip("llama_index.core")
from eval import evaluate_recall as recall


class Retriever:
    def __init__(self, key):
        self.key = key

    def retrieve(self, query):
        return [SimpleNamespace(id_=self.key, text=self.key, metadata={}, score=1.0)]


def test_rrf_retriever_forwards_explicit_k(monkeypatch):
    seen = {}
    monkeypatch.setattr(recall, "_rrf_fuse", lambda *args, **kwargs: seen.update(kwargs) or [])
    retriever = recall._RRFRetriever(Retriever("vector"), Retriever("lexical"), 10, rrf_k=73)

    retriever.retrieve("query")

    assert seen["k"] == 73


def test_fusion_enhanced_forwards_explicit_k(monkeypatch):
    seen = {}
    monkeypatch.setattr(recall, "_rrf_fuse_multi", lambda *args, **kwargs: seen.update(kwargs) or [])
    retriever = recall._FusionEnhancedRetriever(Retriever("vector"), Retriever("lexical"), 10, use_hyde=False, rrf_k=73)

    retriever.retrieve("query")

    assert seen["k"] == 73


def test_plain_fusion_uses_supported_retriever(monkeypatch):
    monkeypatch.setattr(recall.rag_query, "_load_or_build_bm25", lambda _: object())
    monkeypatch.setattr(recall.rag_query, "LocalBM25Retriever", lambda bm25, similarity_top_k: Retriever("lexical"))
    index = SimpleNamespace(
        storage_context=SimpleNamespace(vector_store=object.__new__(recall.rag_query.MilvusVectorStore)),
        as_retriever=lambda similarity_top_k: Retriever("vector"),
    )

    retriever = recall._build_retriever(index, 10, mode="fusion", rrf_k=73)

    assert isinstance(retriever, recall._RRFRetriever)
    assert retriever._rrf_k == 73
