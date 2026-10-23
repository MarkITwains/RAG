import json
from types import SimpleNamespace

import pytest
from llama_index.core.schema import Document, MetadataMode, NodeWithScore, TextNode

from pcb_rag.pcb_facts import detect_fact_conflicts, enrich_table_child, extract_qualified_facts, link_continued_tables
from pcb_rag.sources import expand_sources


TABLE = """IPC-6012E
| 类别 | 最小铜厚 (μm) | 孔径 (mm) |
| --- | --- | --- |
| 外层 | ≥35 | 0.20 |
| 内层 | ≥18 | 0.15 |
注：在规定的温度条件下测量。
"""


def test_ocr_cleanup_preserves_engineering_symbols_and_standard_letters(monkeypatch):
    from pcb_rag import ingest

    monkeypatch.setattr(ingest, "CLEAN_OCR_GARBAGE", True)
    text = "IPC-6012E 铜厚 ≥35 μm，阻抗 50 Ω。J-STD-001 温升 ΔT，电流 I。"
    cleaned = ingest._normalize_text_for_embed(text + "\x00\ufffd")
    assert cleaned == text
    facts = extract_qualified_facts(cleaned)
    assert any(f["value"] == "35" and f["unit"] == "μm" and f["edition"] == "E" for f in facts)
    assert any(f["value"] == "50" and f["unit"] == "Ω" for f in facts)


def test_units_in_headers_retain_row_conditions_and_footnotes():
    facts = extract_qualified_facts(TABLE, {"page_label": "2", "section": "3.4"})
    copper = next(f for f in facts if f["scope"] == "外层" and f["unit"] == "μm")
    assert copper["parameter"] == "最小铜厚"
    assert copper["value"] == "35" and copper["comparator"] == ">="
    assert copper["original_value"] == "≥35"
    assert copper["edition"] == "E" and copper["page"] == "2"
    assert "温度" in copper["table_footnotes"][0]
    assert copper["unit_source"] == "column_header"


def test_cut_table_row_recovers_its_header_and_footnote():
    text, metadata = enrich_table_child("| 内层 | ≥18 | 0.15 |", TABLE)
    assert "最小铜厚" in text and "注：" in text
    facts = extract_qualified_facts(text, metadata)
    assert any(f["value"] == "18" and f["unit"] == "μm" for f in facts)


def test_two_tables_do_not_mix_column_headers():
    second = "| 类型 | 阻抗 (Ω) |\n| --- | --- |\n| 差分 | 100 |\n注：阻抗测试说明。"
    text, metadata = enrich_table_child("| 差分 | 100 |", TABLE + "\n" + second)
    assert "铜厚" not in text
    assert "温度" not in text
    facts = extract_qualified_facts(text, metadata)
    assert facts[0]["unit"] == "Ω" and facts[0]["value"] == "100"


def test_edition_and_scope_variants_remain_separate():
    text = "IPC-6012D\n铜厚不小于 18 μm。\nIPC-6012E\n铜厚不小于 35 μm。"
    facts = extract_qualified_facts(text)
    assert [(f["edition"], f["value"]) for f in facts] == [("D", "18"), ("E", "35")]
    assert len(detect_fact_conflicts(facts)) == 1
    assert facts[0]["original_value"] == "不小于 18 μm"


def test_explicit_continued_table_preserves_source_page():
    docs = [
        Document(text=TABLE, metadata={"source_path": "spec.pdf", "page_label": "1"}),
        Document(text="续表\n| 外层 | ≥70 | 0.3 |", metadata={"source_path": "spec.pdf", "page_label": "2"}),
        Document(text="第三页正文", metadata={"source_path": "spec.pdf", "page_label": "3"}),
    ]
    link_continued_tables(docs)
    assert docs[1].metadata["table_context_page"] == "1"
    assert any(f["value"] == "70" for f in extract_qualified_facts(docs[1].text, docs[1].metadata))
    assert "table_context_page" not in docs[2].metadata


def test_pdf_pages_have_unique_chunks_and_real_parent_text(tmp_path, monkeypatch):
    import fitz
    from pcb_rag import ingest

    path = tmp_path / "IPC-6012E.pdf"
    pdf = fitz.open()
    text = "1 Copper requirements\n" + "Copper thickness has a test condition. " * 12 + "FINAL REQUIREMENT."
    for _ in range(2):
        page = pdf.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 550, 700), text, fontsize=11)
    pdf.save(path)
    pdf.close()
    monkeypatch.setenv("NORMALIZE_TEXT", "0")
    monkeypatch.setattr(ingest, "PARENT_MAX_SIZE", 1200)
    monkeypatch.setattr(ingest, "PARENT_MIN_SIZE", 20)
    monkeypatch.setattr(ingest, "CHILD_MAX_SIZE", 100)
    monkeypatch.setattr(ingest, "CHILD_MIN_SIZE", 20)
    monkeypatch.setattr(ingest, "SEMANTIC_SPLIT_THRESHOLD", 100000)
    docs = ingest._load_ingest_documents(tmp_path)
    assert len(docs) == 2 and len({doc.id_ for doc in docs}) == 2
    assert len({doc.metadata["doc_node_id"] for doc in docs}) == 1
    nodes = ingest._parent_child_parse(docs)
    assert len({node.id_ for node in nodes}) == len(nodes)
    children = [node for node in nodes if node.metadata["node_level"] == "child"]
    assert {node.metadata["page_label"] for node in children} == {"1", "2"}
    by_id = {node.id_: node for node in nodes}
    selected = children[1]
    expanded = expand_sources(
        [NodeWithScore(node=selected, score=0.8)], lambda ids: [by_id[key] for key in ids], neighbors=0
    )
    parent = next(item.node for item in expanded if item.node.metadata["node_level"] == "parent")
    assert parent.id_ == selected.metadata["parent_id"]
    assert len(parent.text) > len(selected.text)
    assert "FINAL REQUIREMENT" in parent.text


def test_oversized_passage_preserves_acl_and_neighbor_links():
    from pcb_rag.ingest import _truncate_text_nodes

    node = TextNode(
        id_="long",
        text="铜厚为 35 μm。" * 200,
        metadata={"doc_node_id": "doc", "tenant_id": "private", "parent_id": "parent", "next_id": "next"},
    )
    following = TextNode(id_="next", text="下一段", metadata={"doc_node_id": "doc", "prev_id": "long"})
    result = _truncate_text_nodes([node, following], max_bytes=800, target_bytes=400)
    parts = result[:-1]
    assert len(parts) > 1
    assert all(part.metadata["tenant_id"] == "private" and part.metadata["parent_id"] == "parent" for part in parts)
    assert all(len(part.text.encode()) <= 800 for part in parts)
    for left, right in zip(parts, parts[1:]):
        assert left.metadata["next_id"] == right.id_
        assert right.metadata["prev_id"] == left.id_
    assert result[-1].metadata["prev_id"] == parts[-1].id_
    assert "tenant_id" not in parts[0].get_content(metadata_mode=MetadataMode.EMBED)


def test_malformed_gleaning_fails_strict_graph_stage(monkeypatch):
    from pcb_rag import graph_rag as g

    monkeypatch.setattr(g, "GRAPH_LLM_GLEANING", 1)
    responses = iter([json.dumps([{"head": "铜厚", "relation": "要求", "tail": "35 μm"}]), "{broken json"])
    llm = SimpleNamespace(complete=lambda prompt: SimpleNamespace(text=next(responses)))
    with pytest.raises(ValueError, match="JSON"):
        g.extract_triples_llm("铜厚要求 35 μm", llm, strict=True)


def test_llm_qualifiers_are_not_overwritten_by_empty_metadata(monkeypatch):
    from pcb_rag import graph_rag as g

    monkeypatch.setattr(g, "GRAPH_LLM_GLEANING", 0)
    text = "IPC-6012E 规定在外层测量铜厚。"
    record = {
        "head": "IPC-6012E",
        "relation": "规定",
        "tail": "铜厚",
        "evidence": text,
        "qualifiers": {"edition": "E", "scope": "外层"},
    }
    llm = SimpleNamespace(complete=lambda prompt: SimpleNamespace(text=json.dumps([record])))
    graph = g.build_graph_from_nodes(
        [TextNode(id_="a", text=text, metadata={"doc_node_id": "d", "revision": "r"})], llm, backend="llm", strict=True
    )
    source = next(iter(graph.edges.values()))["contributions"][0]
    assert source["qualifiers"]["edition"] == "E"
    assert source["qualifiers"]["scope"] == "外层"
    assert source["revision"] == "r"
