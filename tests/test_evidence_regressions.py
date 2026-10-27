from types import SimpleNamespace

import pytest

from pcb_rag.context_budget import estimate_tokens, fit_evidence
from pcb_rag.evidence import build_evidence_bundle, build_generation_request


def doc(key, text="镀金层厚度应不小于 0.05 μm。", **metadata):
    return SimpleNamespace(
        node=SimpleNamespace(id_=key, text=text, metadata={"source_path": key + ".txt", **metadata}), score=0.9
    )


def test_citation_mapping_after_demotion_dedup_and_empty_filter():
    a, b = doc("A", low_value=True), doc("B")
    bundle = build_evidence_bundle([doc("empty", ""), a, a, b], 6000)
    assert [row["chunk_id"] for row in bundle.entries] == ["B", "A"]
    assert bundle.citations("要求[1]。")[0]["chunk_id"] == "B"
    assert bundle.citations("补充[2]。")[0]["chunk_id"] == "A"
    assert bundle.citations("错误[3]。") == []
    assert bundle.invalid_citations("错误[3][100]。") == [3, 100]
    assert bundle.citations("要求[1]。")[0]["support_status"] == "unverified"


@pytest.mark.parametrize("budget", [0, 1, 50, 400, 600, 6000])
def test_final_serialized_context_fits_long_paths_and_unicode(budget):
    docs = [
        doc(str(i), "要求铜厚不小于 35 μm。" * 100, source_path="中文长文件名" * 14 + f"{i}.txt") for i in range(20)
    ]
    bundle = build_evidence_bundle(docs, budget)
    assert estimate_tokens(bundle.context) <= budget
    assert [row["index"] for row in bundle.entries] == list(range(1, len(bundle.entries) + 1))
    for row in bundle.entries:
        assert row["text"] in bundle.context


def test_truncated_evidence_and_returned_node_are_exact_and_not_mutated():
    original = doc("A", "铜厚 35 μm。" * 500)
    bundle = build_evidence_bundle([original], 600)
    assert bundle.entries[0]["truncated"]
    assert bundle.nodes[0].node.text == bundle.entries[0]["text"]
    assert original.node.text != bundle.nodes[0].node.text
    assert bundle.citations("[1]")[0]["text"] == bundle.nodes[0].node.text


def test_request_budget_includes_prompt_history_question_and_output():
    req = build_generation_request(
        [doc("A", "证据" * 1000)],
        "系统要求\n{history}\n{context}\n问题：{query}",
        query="问题" * 40,
        context_budget=6000,
        request_limit=1800,
        output_reserve=300,
        values={"history": "历史" * 90},
    )
    assert estimate_tokens(req.prompt) + 300 <= 1800
    assert req.evidence.budget["request_tokens"] <= 1800
    with pytest.raises(ValueError, match="window"):
        build_generation_request(
            [], "{context}{query}", query="长问题" * 1000, context_budget=6000, request_limit=100, output_reserve=30
        )


def test_body_truncation_has_no_minimum_character_budget_overrun():
    rows = fit_evidence(["罕见𠮷铜厚" * 200], str, 125, min_chunk_tokens=10)
    assert sum(row["tokens"] for row in rows) <= 125


def test_configured_minimum_chunk_size_stops_tiny_truncated_tail():
    bundle = build_evidence_bundle([doc("A", "证据" * 100), doc("B")], 160, min_chunk_tokens=60)

    assert [row["chunk_id"] for row in bundle.entries] == ["A"]
    assert bundle.entries[0]["truncated"]


def test_minimum_chunk_limit_is_enforced_on_truncated_body_not_just_total_budget():
    bundle = build_evidence_bundle([doc("A", "证据" * 100), doc("B")], 160, min_chunk_tokens=60)

    assert all(row["tokens"] >= 60 for row in bundle.entries)


def test_minimum_chunk_limit_rejects_too_short_body_with_large_locator():
    bundle = build_evidence_bundle([doc("A", "证据" * 100, source_path="长路径" * 50)], 160, min_chunk_tokens=60)

    assert bundle.entries == ()


def test_zero_context_budget_uses_documented_legacy_document_limits(monkeypatch):
    from pcb_rag import query

    monkeypatch.setattr(query, "RAG_CONTEXT_MAX_TOKENS", 0)
    monkeypatch.setattr(query, "RAG_TOP_DOCS", 2)
    monkeypatch.setattr(query, "RAG_DOC_MAX_CHARS", 5)
    docs = [doc("A", "abcdefghij"), doc("B", "klmnopqrst"), doc("C", "uvwxyz")]

    bundle = query.build_context_bundle(docs)

    assert [row["chunk_id"] for row in bundle.entries] == ["A", "B"]
    assert [row["text"] for row in bundle.entries] == ["abcde...", "klmno..."]
    assert bundle.budget["legacy"] is True


def test_legacy_bundle_supports_sdk_evidence_mapping(monkeypatch):
    from pcb_rag import query
    from pcb_rag.engine import KnowledgeBase

    monkeypatch.setattr(query, "RAG_CONTEXT_MAX_TOKENS", 0)
    monkeypatch.setattr(query, "RAG_TOP_DOCS", 1)
    monkeypatch.setattr(query, "RAG_DOC_MAX_CHARS", 5)
    bundle = query.build_context_bundle([doc("A", "abcdefghij")])

    evidence = KnowledgeBase(auto_init=False, pipeline=object())._bundle_evidence(bundle)[0]

    assert evidence.text == "abcde..."
    assert evidence.tokens > 0
    assert evidence.truncated is True


def test_generation_uses_the_same_legacy_fallback(monkeypatch):
    from pcb_rag import query

    monkeypatch.setattr(query, "RAG_CONTEXT_MAX_TOKENS", 0)
    monkeypatch.setattr(query, "RAG_TOP_DOCS", 2)
    monkeypatch.setattr(query, "RAG_DOC_MAX_CHARS", 5)
    docs = [doc("A", "abcdefghij"), doc("B", "klmnopqrst"), doc("C", "uvwxyz")]
    llm = SimpleNamespace(tokenizer=None, model="test")

    request = query.prepare_generation("问题", docs, llm)

    assert [row["chunk_id"] for row in request.evidence.entries] == ["A", "B"]
    assert request.evidence.context in request.prompt
    assert "uvwxyz" not in request.prompt


def test_legacy_fallback_still_fits_a_smaller_model_window(monkeypatch):
    from pcb_rag import query

    monkeypatch.setattr(query, "RAG_CONTEXT_MAX_TOKENS", 0)
    monkeypatch.setattr(query, "RAG_TOP_DOCS", 2)
    monkeypatch.setattr(query, "RAG_DOC_MAX_CHARS", 5000)
    monkeypatch.setattr(query, "RAG_MIN_CHUNK_TOKENS", 0)
    llm = SimpleNamespace(tokenizer=None, model="test")

    request = query.prepare_generation("问题", [doc("A", "证据" * 500)], llm)

    assert request.evidence.entries
    assert request.evidence.context in request.prompt
    assert request.evidence.budget["request_tokens"] <= request.evidence.budget["request_limit"]


def test_legacy_fallback_rejects_a_window_too_small_for_prompt(monkeypatch):
    from pcb_rag import query

    monkeypatch.setattr(query, "RAG_CONTEXT_MAX_TOKENS", 0)
    bundle = query.build_context_bundle([doc("A", "证据" * 500)])

    with pytest.raises(ValueError, match="window"):
        build_generation_request(
            [doc("A")],
            "{context}{query}",
            query="问题" * 100,
            context_budget=0,
            request_limit=100,
            output_reserve=30,
            bundle=bundle,
        )


def test_legacy_fallback_is_trimmed_when_full_form_exceeds_window(monkeypatch):
    from pcb_rag import query

    monkeypatch.setattr(query, "RAG_CONTEXT_MAX_TOKENS", 0)
    monkeypatch.setattr(query, "RAG_TOP_DOCS", 2)
    monkeypatch.setattr(query, "RAG_DOC_MAX_CHARS", 5000)
    monkeypatch.setattr(query, "RAG_MIN_CHUNK_TOKENS", 0)
    bundle = query.build_context_bundle([doc("A", "证据" * 500)])

    request = build_generation_request(
        [doc("A", "证据" * 500)],
        "{context}{query}",
        query="q",
        context_budget=0,
        request_limit=800,
        output_reserve=30,
        bundle=bundle,
    )

    assert request.evidence.entries
    assert len(request.evidence.context) < len(bundle.context)
    assert request.evidence.budget["request_tokens"] <= 800
