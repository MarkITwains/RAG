"""BM25 tuning must survive both production index-building paths."""

import json

import pytest

pytest.importorskip("llama_index.core")
from llama_index.core.schema import TextNode

from pcb_rag import query
from pcb_rag.retrieval import build_lexical


def test_snapshot_builder_honors_bm25_env(monkeypatch):
    monkeypatch.setenv("BM25_K1", "2.4")
    monkeypatch.setenv("BM25_B", "0.35")

    index = build_lexical([TextNode(text="铜厚 铜厚 铜厚")])

    assert index.k1 == pytest.approx(2.4)
    assert index.b == pytest.approx(0.35)


def test_legacy_cache_builder_honors_bm25_env(tmp_path, monkeypatch):
    monkeypatch.setenv("BM25_K1", "2.4")
    monkeypatch.setenv("BM25_B", "0.35")
    monkeypatch.setenv("LEXICAL_CACHE_TTL_HOURS", "0")
    cache = tmp_path / "lexical.jsonl"
    cache.write_text(
        json.dumps({"id": "copper", "text": "铜厚 铜厚 铜厚", "metadata": {}}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(query, "LEXICAL_CACHE_PATH", str(cache))
    monkeypatch.setattr(
        query,
        "_iter_milvus_rows",
        lambda _: pytest.fail("a valid lexical cache must not require Milvus"),
    )

    index = query._load_or_build_bm25(object())

    assert index.k1 == pytest.approx(2.4)
    assert index.b == pytest.approx(0.35)


def test_single_document_bm25_returns_exact_match():
    index = build_lexical([TextNode(id_="only", text="铜厚 35 μm")])

    result = query.LocalBM25Retriever(index).retrieve("铜厚")

    assert [item.node.id_ for item in result] == ["only"]


@pytest.mark.parametrize("micro", ["μ", "µ"])
def test_micro_sign_is_not_conflated_with_meter(micro):
    tokenized = query._tokenize(f"35 {micro}m")

    assert "um" in tokenized
    assert "m" not in tokenized
    assert tokenized != query._tokenize("35 m")
