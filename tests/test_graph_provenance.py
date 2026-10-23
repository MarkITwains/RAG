from pcb_rag.graph_rag import KnowledgeGraph, Triple


def fact(doc, chunk, **kwargs):
    return Triple(
        head="镀层",
        relation="采用",
        tail="测厚方法",
        doc_id=doc,
        chunk_id=chunk,
        source_path=doc + ".pdf",
        evidence=doc + " evidence",
        head_desc=doc + " description",
        **kwargs,
    )


def test_delete_shared_source_rebuilds_remaining_evidence_and_descriptions(tmp_path):
    graph = KnowledgeGraph()
    graph.add_triples([fact("A", "a"), fact("B", "b")])
    graph.communities = {"stale": {"summary": "A description"}}
    graph.remove_docs(["A"])
    edge = next(iter(graph.edges.values()))
    assert edge["d"] == ["B"] and edge["c"] == ["b"]
    assert edge["w"] == 1 and edge["s"] == "B.pdf"
    assert "A" not in edge["e"] and "A" not in graph.nodes["镀层"].desc
    assert graph.communities == {}
    path = tmp_path / "kg.json"
    graph.save(str(path))
    graph.remove_docs(["B"])
    graph.save(str(path))
    assert not KnowledgeGraph.load(str(path)).edges


def test_identified_retry_and_merge_are_idempotent_and_keep_all_sources(tmp_path):
    graph, other = KnowledgeGraph(), KnowledgeGraph()
    other.add_triples([fact("A", "a"), fact("B", "b")])
    other.add_triple(fact("B", "b"))
    graph.merge(other)
    graph.merge(other)
    edge = next(iter(graph.edges.values()))
    assert set(edge["d"]) == {"A", "B"} and edge["w"] == 2
    for i in range(30):
        graph.add_triple(fact("D" + str(i), "c" + str(i)))
    path = tmp_path / "graph.json"
    graph.save(str(path), compact=True)
    loaded = KnowledgeGraph.load(str(path))
    assert len(loaded.nodes["镀层"].chunk_ids) == 32


def test_chunk_delete_keeps_other_relation_connectivity_and_correct_doc_mapping():
    graph = KnowledgeGraph()
    graph.add_triples([fact("A", "a"), fact("B", "b")])
    graph.add_triple(Triple("镀层", "验证", "测厚方法", doc_id="C", chunk_id="c"))
    graph.remove_chunks(["a", "b"])
    assert graph.neighbors("镀层") == ["测厚方法"]
    assert graph.nodes["镀层"].doc_ids == ["C"]


def test_different_versions_and_conditions_remain_separate_contributions():
    graph = KnowledgeGraph()
    graph.add_triple(fact("A", "a", revision="v1", qualifiers={"edition": "2002", "value": "35", "unit": "μm"}))
    graph.add_triple(fact("B", "b", revision="v2", qualifiers={"edition": "2025", "value": "30", "unit": "μm"}))
    edge = next(iter(graph.edges.values()))
    assert [q["edition"] for q in edge["facts"]] == ["2002", "2025"]
