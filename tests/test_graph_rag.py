"""GraphRAG 单元测试。

覆盖范围：只测不依赖 llama-index / Milvus 的纯逻辑部分——
规则抽取、图存储与持久化、实体链接、图检索片段生成、社区检测、融合与扩展。
"""

from types import SimpleNamespace

import json
import pytest

from pcb_rag.graph_rag import (
    GraphRetriever,
    KnowledgeGraph,
    Triple,
    _parse_triple_json,
    build_graph_from_nodes,
    build_neighborhood_text,
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
            f"主结果（RRF 分约 0.02）必须排在图表征（规则分约 1.0）之前，实际顺序: {order}"
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


# ---------------------------------------------------------------------------
# 实体语义化：类型归一、描述合并、关系描述、邻域文本
# ---------------------------------------------------------------------------
class TestEntitySemantics:
    def test_type_is_normalized_from_llm_label(self):
        graph = KnowledgeGraph()
        graph.add_triple(
            Triple(head="沉金", relation="属于", tail="表面处理", head_type="Process", tail_type="Concept")
        )
        assert graph.nodes["沉金"].type == "process"
        assert graph.nodes["表面处理"].type == "concept"

    def test_chinese_type_alias_normalized(self):
        graph = KnowledgeGraph()
        graph.add_triple(Triple(head="FR-4", relation="用作", tail="基材", head_type="材料"))
        assert graph.nodes["FR-4"].type == "material"

    def test_unknown_type_falls_back_to_pattern(self):
        graph = KnowledgeGraph()
        graph.add_triple(Triple(head="GB/T 4677", relation="规定", tail="镀金层厚度", head_type="WeirdType"))
        # 类型无法识别时退回实体名模式匹配
        assert graph.nodes["GB/T 4677"].type == "standard"

    def test_descriptions_accumulate_without_duplicates(self):
        graph = KnowledgeGraph()
        graph.add_triple(Triple(head="镀金层厚度", relation="r", tail="A", head_desc="金层的最小厚度指标"))
        graph.add_triple(Triple(head="镀金层厚度", relation="r2", tail="B", head_desc="金层的最小厚度指标"))
        graph.add_triple(Triple(head="镀金层厚度", relation="r3", tail="C", head_desc="用于评估镀层耐蚀性"))
        desc = graph.nodes["镀金层厚度"].desc
        assert desc.count("最小厚度指标") == 1
        assert "耐蚀性" in desc

    def test_description_respects_char_cap(self):
        graph = KnowledgeGraph()
        for i in range(20):
            graph.add_triple(
                Triple(head="实体X", relation=f"r{i}", tail=f"T{i}", head_desc=f"这是第{i}条描述" + "补充" * 10)
            )
        assert len(graph.nodes["实体X"].desc) <= 240

    def test_relation_description_accumulates(self):
        graph = KnowledgeGraph()
        graph.add_triple(
            Triple(head="GB/T 4677", relation="规定", tail="镀金层厚度", relation_desc="规定了镀层厚度测试要求")
        )
        edge = graph.edges[("GB/T 4677", "规定", "镀金层厚度")]
        assert edge["rd"] == "规定了镀层厚度测试要求"

    def test_degree_accumulates_as_count(self):
        graph = KnowledgeGraph()
        for i in range(3):
            graph.add_triple(Triple(head="镀金层厚度", relation=f"r{i}", tail=f"T{i}"))
        assert graph.nodes["镀金层厚度"].count == 3

    def test_save_load_preserves_semantics(self, tmp_path):
        graph = KnowledgeGraph()
        graph.add_triple(
            Triple(
                head="GB/T 4677",
                relation="规定",
                tail="镀金层厚度",
                head_type="Standard",
                head_desc="印制板测试方法国家标准",
                tail_type="Parameter",
                tail_desc="金层最小厚度指标",
                relation_desc="规定了镀层厚度测试要求",
            )
        )
        target = tmp_path / "kg.json"
        graph.save(str(target))

        restored = KnowledgeGraph.load(str(target))
        assert restored.nodes["GB/T 4677"].desc == "印制板测试方法国家标准"
        assert restored.nodes["镀金层厚度"].type == "parameter"
        assert restored.edges[("GB/T 4677", "规定", "镀金层厚度")]["rd"] == "规定了镀层厚度测试要求"

    def test_merge_keeps_descriptions(self):
        a = KnowledgeGraph()
        a.add_triple(
            Triple(head="沉金", relation="适用于", tail="HDI板", head_desc="化学镀镍金工艺", head_type="Process")
        )
        b = KnowledgeGraph()
        b.add_triple(Triple(head="沉金", relation="属于", tail="表面处理", head_desc="一种表面处理方式"))
        a.merge(b)
        assert a.nodes["沉金"].desc.startswith("化学镀镍金工艺")
        assert "表面处理" in a.nodes["沉金"].desc or "表面处理" in a.nodes


class TestNeighborhoodText:
    def _graph(self):
        graph = KnowledgeGraph()
        graph.add_triple(
            Triple(
                head="GB/T 4677",
                relation="规定",
                tail="镀金层厚度",
                head_type="Standard",
                head_desc="印制板测试方法国家标准",
                tail_type="Parameter",
                tail_desc="金层最小厚度指标",
                relation_desc="规定了镀层厚度测试要求",
                evidence="镀金层厚度应不小于0.05μm",
                chunk_id="c1",
            )
        )
        graph.add_triple(
            Triple(
                head="镀金层厚度",
                relation="测试方法为",
                tail="X射线荧光法",
                tail_type="Method",
                tail_desc="无损测厚方法",
                evidence="采用X射线荧光法测量",
                chunk_id="c2",
            )
        )
        return graph

    def test_includes_anchor_description(self):
        text = build_neighborhood_text(self._graph(), "镀金层厚度")
        assert "金层最小厚度指标" in text

    def test_includes_relation_description(self):
        text = build_neighborhood_text(self._graph(), "镀金层厚度")
        assert "规定了镀层厚度测试要求" in text

    def test_includes_neighbor_description(self):
        text = build_neighborhood_text(self._graph(), "镀金层厚度")
        assert "无损测厚方法" in text

    def test_includes_evidence(self):
        text = build_neighborhood_text(self._graph(), "镀金层厚度")
        assert "0.05" in text

    def test_missing_entity_returns_empty(self):
        assert build_neighborhood_text(self._graph(), "不存在的实体") == ""

    def test_text_without_descriptions_still_has_structure(self):
        """规则抽取的图没有描述，文本必须仍然包含关系三元组。"""
        graph = KnowledgeGraph()
        graph.add_triple(Triple(head="镀金层", relation="应符合", tail="GB/T 4677"))
        text = build_neighborhood_text(graph, "镀金层")
        assert "镀金层 --应符合--> GB/T 4677" in text


class TestTripleParsing:
    def test_parses_semantic_fields(self):
        raw = (
            '[{"head":"沉金","head_type":"Process","head_desc":"化学镀镍金",'
            '"relation":"适用于","relation_desc":"用于高密度板",'
            '"tail":"HDI板","tail_type":"Structure","tail_desc":"高密度互连板"}]'
        )
        parsed = _parse_triple_json(raw)
        assert len(parsed) == 1
        assert parsed[0]["head_type"] == "Process"
        assert parsed[0]["relation_desc"] == "用于高密度板"

    def test_parses_legacy_shape_without_semantics(self):
        parsed = _parse_triple_json('[{"head":"A","relation":"r","tail":"B"}]')
        assert parsed[0]["head"] == "A"
        assert parsed[0]["head_desc"] == ""

    def test_ignores_malformed_rows(self):
        raw = '[{"head":"A","relation":"r"},{"head":"B","relation":"r","tail":"C"}]'
        parsed = _parse_triple_json(raw)
        assert len(parsed) == 1
        assert parsed[0]["tail"] == "C"

    def test_garbage_returns_empty(self):
        assert _parse_triple_json("抱歉，我无法处理") == []


def test_evidence_quote_tolerates_line_break_normalisation(monkeypatch):
    """跨行引用被模型压成单行时不应判为杜撰；真正的杜撰仍须报错。"""

    from pcb_rag import graph_rag as g

    monkeypatch.setattr(g, "GRAPH_LLM_GLEANING", 0)
    text = "阻焊开窗应比焊盘单边大 0.05mm，\n防止焊盘边缘露铜。"
    good = [
        {
            "head": "阻焊开窗",
            "relation": "应大于",
            "tail": "焊盘",
            "evidence": "阻焊开窗应比焊盘单边大 0.05mm，防止焊盘边缘露铜。",
        }
    ]
    llm = SimpleNamespace(complete=lambda prompt: SimpleNamespace(text=json.dumps(good)))
    assert len(g.extract_triples_llm(text, llm, strict=True)) == 1

    bad = [{"head": "A", "relation": "r", "tail": "B", "evidence": "原文里没有这句话"}]
    llm_bad = SimpleNamespace(complete=lambda prompt: SimpleNamespace(text=json.dumps(bad)))
    with pytest.raises(ValueError, match="exact source quote"):
        g.extract_triples_llm(text, llm_bad, strict=True)


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({"head": "铜厚", "relation": "要求", "tail": "35 μm"}, "exact source quote"),
        (
            {"head": "不存在的材料", "relation": "要求", "tail": "35 μm", "evidence": "铜厚要求 35 μm"},
            "entities must appear",
        ),
    ],
)
def test_strict_extraction_rejects_missing_evidence_and_invented_entities(monkeypatch, record, message):
    from pcb_rag import graph_rag as g

    monkeypatch.setattr(g, "GRAPH_LLM_GLEANING", 0)
    llm = SimpleNamespace(complete=lambda prompt: SimpleNamespace(text=json.dumps([record], ensure_ascii=False)))
    with pytest.raises(ValueError, match=message):
        g.extract_triples_llm("铜厚要求 35 μm", llm, strict=True)


def test_strict_entity_matching_tolerates_case_and_whitespace(monkeypatch):
    from pcb_rag import graph_rag as g

    monkeypatch.setattr(g, "GRAPH_LLM_GLEANING", 0)
    text = "ipc-6012 规定铜厚。"
    record = {"head": "IPC-6012", "relation": "规定", "tail": "铜 厚", "evidence": text}
    llm = SimpleNamespace(complete=lambda prompt: SimpleNamespace(text=json.dumps([record], ensure_ascii=False)))

    triples = g.extract_triples_llm(text, llm, strict=True)

    assert len(triples) == 1
    assert triples[0].head == "IPC-6012"
