"""GraphRAG 单元测试。

覆盖范围：只测不依赖 llama-index / Milvus 的纯逻辑部分——
规则抽取、图存储与持久化、实体链接、图检索片段生成、社区检测、融合与扩展。
"""

from types import SimpleNamespace

from pcb_rag.graph_rag import (
    GraphRetriever,
    KnowledgeGraph,
    Triple,
    build_graph_from_nodes,
    detect_communities,
    expand_nodes_with_graph,
    extract_triples,
    extract_triples_rule,
    fuse_with_graph,
    link_entities,
    load_graph,
    normalize_entity,
)


def _node(text: str, node_id: str, doc_id: str = "doc1", source: str = "a.pdf") -> SimpleNamespace:
    return SimpleNamespace(text=text, node_id=node_id, metadata={"doc_node_id": doc_id, "source_path": source})


SENTENCE_A = "镀金层厚度应不小于 0.8 μm，符合 GB/T 4677 要求。"
SENTENCE_B = "沉金工艺采用 ENIG，IPC-6012 规定镍层厚度不低于 3 μm。"


def _build_sample_graph() -> KnowledgeGraph:
    return build_graph_from_nodes([_node(SENTENCE_A, "c1"), _node(SENTENCE_B, "c2")])


# ---------------------------------------------------------------------------
# 实体规范化
# ---------------------------------------------------------------------------
class TestNormalizeEntity:
    def test_strips_wrapping_punctuation(self):
        assert normalize_entity("  GB/T 4677 ") == "GB/T 4677"
        assert normalize_entity("（镀金层）") == "镀金层"

    def test_ascii_with_digits_upper_cased(self):
        assert normalize_entity("ipc-6012") == "IPC-6012"

    def test_empty_input(self):
        assert normalize_entity("") == ""
        assert normalize_entity("   ") == ""


# ---------------------------------------------------------------------------
# 规则抽取
# ---------------------------------------------------------------------------
class TestRuleExtraction:
    def test_extracts_limit_relation(self):
        triples = extract_triples_rule(SENTENCE_A, chunk_id="c1")
        pairs = {(t.head, t.tail) for t in triples}
        assert ("镀金层厚度", "0.8 μm") in pairs

    def test_extracts_adopt_relation(self):
        triples = extract_triples_rule(SENTENCE_B, chunk_id="c2")
        relations = {t.relation for t in triples}
        assert "采用" in relations

    def test_records_source_metadata(self):
        triples = extract_triples_rule(SENTENCE_A, chunk_id="c1", doc_id="d1", source_path="x.pdf")
        assert triples
        assert all(t.chunk_id == "c1" for t in triples)
        assert all(t.doc_id == "d1" for t in triples)
        assert all(t.source_path == "x.pdf" for t in triples)
        assert all(t.evidence for t in triples)

    def test_respects_max_triples(self):
        text = "。".join([SENTENCE_A] * 20)
        triples = extract_triples_rule(text, max_triples=5)
        assert len(triples) <= 5

    def test_ignores_blank_text(self):
        assert extract_triples_rule("") == []
        assert extract_triples("   ") == []

    def test_llm_backend_falls_back_to_rules(self):
        # llm=None 时应回退规则抽取，而不是返回空
        triples = extract_triples(SENTENCE_A, None, backend="llm")
        assert triples


# ---------------------------------------------------------------------------
# 知识图谱
# ---------------------------------------------------------------------------
class TestKnowledgeGraph:
    def test_add_triple_builds_nodes_and_adjacency(self):
        graph = KnowledgeGraph()
        assert graph.add_triple(Triple(head="镀金层", relation="应符合", tail="GB/T 4677")) is True
        assert set(graph.nodes) == {"镀金层", "GB/T 4677"}
        assert "GB/T 4677" in graph.adjacency["镀金层"]
        assert "GB/T 4677" in graph.neighbors("镀金层")

    def test_duplicate_triple_accumulates_weight(self):
        graph = KnowledgeGraph()
        graph.add_triple(Triple(head="镀金层", relation="应符合", tail="GB/T 4677"))
        graph.add_triple(Triple(head="镀金层", relation="应符合", tail="GB/T 4677"))
        assert len(graph.edges) == 1
        assert graph.edges[("镀金层", "应符合", "GB/T 4677")]["w"] == 2.0

    def test_rejects_self_loop_and_noise(self):
        graph = KnowledgeGraph()
        assert graph.add_triple(Triple(head="A", relation="rel", tail="A")) is False
        assert graph.add_triple(Triple(head="", relation="rel", tail="B")) is False
        assert graph.add_triple(Triple(head="A" * 60, relation="rel", tail="B")) is False
        assert graph.edges == {}

    def test_edges_of_returns_both_directions(self):
        graph = _build_sample_graph()
        anchor = next(iter(graph.nodes))
        edges = graph.edges_of(anchor)
        assert edges
        assert all(edge["h"] == anchor or edge["t"] == anchor for edge in edges)

    def test_pair_weight_sums_both_directions(self):
        graph = KnowledgeGraph()
        graph.add_triple(Triple(head="A", relation="r1", tail="B"))
        graph.add_triple(Triple(head="B", relation="r2", tail="A"))
        assert graph.pair_weight("A", "B") == 2.0

    def test_save_and_load_round_trip(self, tmp_path):
        graph = _build_sample_graph()
        target = tmp_path / "kg.json"
        graph.save(str(target))

        assert target.exists()
        restored = KnowledgeGraph.load(str(target))
        assert restored is not None
        assert restored.stats()["relations"] == graph.stats()["relations"]
        assert set(restored.nodes) == set(graph.nodes)
        assert restored.adjacency["GB/T 4677"]

    def test_load_missing_file_returns_none(self, tmp_path):
        assert KnowledgeGraph.load(str(tmp_path / "nope.json")) is None
        assert load_graph(str(tmp_path / "nope.json")) is None

    def test_load_corrupted_file_returns_none(self, tmp_path):
        broken = tmp_path / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        assert KnowledgeGraph.load(str(broken)) is None

    def test_remove_chunks_drops_orphan_edges(self):
        graph = _build_sample_graph()
        before = len(graph.edges)
        removed = graph.remove_chunks(["c1", "c2"])
        assert removed == before
        assert graph.edges == {}

    def test_merge_combines_graphs(self):
        left = KnowledgeGraph()
        left.add_triple(Triple(head="A", relation="r", tail="B"))
        right = KnowledgeGraph()
        right.add_triple(Triple(head="C", relation="r", tail="D"))
        left.merge(right)
        # 纯 ASCII 实体在规范化时统一小写
        assert set(left.nodes) >= {"a", "b", "c", "d"}

    def test_build_from_nodes_reports_stats(self):
        graph = _build_sample_graph()
        stats = graph.stats()
        assert stats["entities"] > 0
        assert stats["relations"] > 0
        assert "entity_types" in stats


# ---------------------------------------------------------------------------
# 实体链接
# ---------------------------------------------------------------------------
class TestEntityLinking:
    def test_full_name_hit(self):
        graph = _build_sample_graph()
        linked = {name for name, _ in link_entities("镀金层厚度要求是多少", graph)}
        assert "镀金层厚度" in linked

    def test_partial_entity_hit(self):
        graph = KnowledgeGraph()
        graph.add_triple(Triple(head="镀金层", relation="应符合", tail="GB/T 4677"))
        linked = {name for name, _ in link_entities("镀金的要求", graph)}
        assert "镀金层" in linked

    def test_no_match_returns_empty(self):
        graph = _build_sample_graph()
        assert link_entities("完全无关的问题", graph) == []

    def test_empty_graph_returns_empty(self):
        assert link_entities("镀金层厚度", KnowledgeGraph()) == []

    def test_scores_are_sorted_descending(self):
        graph = _build_sample_graph()
        scores = [score for _name, score in link_entities("镀金层厚度不低于多少", graph)]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# 图检索
# ---------------------------------------------------------------------------
class TestGraphRetriever:
    def test_build_passages_contains_relation_lines(self):
        graph = _build_sample_graph()
        passages = GraphRetriever(graph).build_passages("镀金层厚度要求")
        assert passages
        text = passages[0]["text"]
        assert "[知识图谱]" in text
        assert "-->" in text

    def test_build_passages_metadata_carries_chunk_ids(self):
        graph = _build_sample_graph()
        passages = GraphRetriever(graph).build_passages("镀金层厚度要求")
        metadata = passages[0]["metadata"]
        assert metadata["source_type"] == "knowledge_graph"
        assert metadata["chunk_ids"]

    def test_no_linked_entity_yields_no_passage(self):
        graph = _build_sample_graph()
        assert GraphRetriever(graph).build_passages("毫无关联的问题") == []

    def test_empty_query(self):
        graph = _build_sample_graph()
        assert GraphRetriever(graph).build_passages("") == []


# ---------------------------------------------------------------------------
# 社区检测
# ---------------------------------------------------------------------------
class TestCommunities:
    def test_small_cluster_detected(self):
        graph = KnowledgeGraph()
        for tail in ("B", "C", "D"):
            graph.add_triple(Triple(head="A", relation="r", tail=tail))
        communities = detect_communities(graph)
        assert communities
        members = set()
        for community in communities.values():
            members.update(community["entities"])
        assert {"a", "b", "c", "d"} <= members

    def test_empty_graph(self):
        assert detect_communities(KnowledgeGraph()) == {}


# ---------------------------------------------------------------------------
# 融合与扩展
# ---------------------------------------------------------------------------
class TestFusionAndExpansion:
    def _scored(self, text: str, score: float) -> SimpleNamespace:
        node = SimpleNamespace(node_id=text[:6], text=text, metadata={})
        return SimpleNamespace(node=node, score=score)

    def test_fuse_sorts_by_weighted_score(self):
        base = [self._scored("base item", 0.5)]
        graph_nodes = [self._scored("graph item", 1.0)]
        fused = fuse_with_graph(base, graph_nodes, graph_weight=0.5)
        assert len(fused) == 2
        # RRF 语义：主结果 rank1 = 1/(k+1)，图结果 rank1 = w/(k+1)，
        # 权重 0.5 让主结果排在前面，但两者都保留
        assert [item.node.text for item in fused] == ["base item", "graph item"]
        assert fused[0].score > fused[1].score

    def test_graph_score_scale_does_not_dominate(self):
        """量纲回归：图侧规则分（0.2~1.0）不能压过主结果的 RRF 分（~0.02）。

        旧实现是"图分 × weight 后与主结果同池按 score 排序"，而 0.35 × 0.55 = 0.19
        远大于 RRF 分 0.02 —— 图节点会无条件霸榜、把真正的检索结果挤下去。
        改成 rank-based RRF 后，两路量纲无关，主结果的榜首必然在图榜首之前
        （除非图权重 > 1）。
        """

        base = [self._scored(f"base{i}", 0.02) for i in range(3)]
        graph_nodes = [self._scored(f"graph{i}", 1.0) for i in range(3)]
        fused = fuse_with_graph(base, graph_nodes, graph_weight=0.35)
        order = [item.node.text for item in fused]
        assert order[:3] == ["base0", "base1", "base2"], (
            "主结果（RRF 分约 0.02）必须排在图表征（规则分约 1.0）之前，"
            f"实际顺序: {order}"
        )

    def test_graph_only_nodes_are_marked_as_evidence(self):
        """图证据必须带标记：调用方据此给它单独配额。

        否则它虽然排名被 RRF 正确压制，仍会凭自带的规则分（0.2~1.0）在
        "按 score 排序截断 top_k" 那一步挤掉真实检索结果。
        """
        base = [self._scored("base0", 0.02)]
        graph_nodes = [self._scored("graph0", 1.0)]
        fused = fuse_with_graph(base, graph_nodes, graph_weight=0.35)
        by_text = {n.node.text: n.node for n in fused}
        assert by_text["graph0"].metadata.get("graph_evidence") is True
        assert not by_text["base0"].metadata.get("graph_evidence")

    def test_node_hit_by_both_routes_is_not_marked_graph_only(self):
        """主结果里已存在的节点不算"纯图证据"，不应被限额。"""
        item = self._scored("dup", 0.02)
        other = self._scored("dup", 1.0)
        other.node.node_id = item.node.node_id
        fused = fuse_with_graph([item], [other])
        assert len(fused) == 1
        assert not fused[0].node.metadata.get("graph_evidence")

    def test_overwrite_scores_false_keeps_original_scores(self):
        """API 在 rerank 之后调用：只改顺序，不把 rerank 概率覆盖成 RRF 分。"""
        base = [self._scored("base0", 0.91), self._scored("base1", 0.55)]
        graph_nodes = [self._scored("graph0", 0.4)]
        fused = fuse_with_graph(base, graph_nodes, graph_weight=0.35, overwrite_scores=False)
        assert [n.score for n in fused[:2]] == [0.91, 0.55]

    def test_fuse_dedupes_same_node(self):
        item = self._scored("dup", 0.4)
        other = self._scored("dup", 0.9)
        other.node.node_id = item.node.node_id
        fused = fuse_with_graph([item], [other])
        assert len(fused) == 1

    def test_expand_appends_graph_nodes(self):
        graph = _build_sample_graph()
        nodes = [self._scored(SENTENCE_A, 0.7)]
        expanded = expand_nodes_with_graph(nodes, graph, max_extra=2)
        # 未安装 llama-index 时无法构造新节点，函数应安全返回原列表
        assert len(expanded) >= len(nodes)

    def test_expand_without_graph_is_noop(self):
        nodes = [self._scored(SENTENCE_A, 0.7)]
        assert expand_nodes_with_graph(nodes, None) == nodes
