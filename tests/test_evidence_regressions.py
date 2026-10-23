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
