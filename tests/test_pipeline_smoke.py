"""跨模块集成冒烟测试（P1-2 + P2）。

验证「图检索 → 融合 → 权限复核 → 上下文压缩」这条新增链路能真正串起来，
以及各模块之间的数据契约（metadata 字段、节点对象形态）保持一致。

刻意不依赖 Milvus / llama-index：用 SimpleNamespace 模拟 NodeWithScore，
保证 CI（只装 pytest / ruff / chardet）也能完整跑通。
"""

from types import SimpleNamespace

import pcb_rag.compression as compression
import pcb_rag.graph_rag as graph_rag
import pcb_rag.security as security
from pcb_rag.compression import compress_pipeline, dedupe_nodes
from pcb_rag.graph_rag import (
    GraphRetriever,
    KnowledgeGraph,
    build_graph_from_nodes,
    expand_nodes_with_graph,
    fuse_with_graph,
)
from pcb_rag.security import Principal, filter_nodes, node_allowed


def _doc(text: str, node_id: str, doc_id: str, metadata: dict | None = None) -> SimpleNamespace:
    merged = {"doc_node_id": doc_id, "source_path": "GB_T_4677.pdf"}
    merged.update(metadata or {})
    return SimpleNamespace(text=text, node_id=node_id, metadata=merged)


def _scored(node: SimpleNamespace, score: float) -> SimpleNamespace:
    return SimpleNamespace(node=node, score=score)


SENTENCE_A = "镀金层厚度应不小于 0.8 μm，测试方法按 GB/T 4677 执行，取样位置为板面中心区域。"
SENTENCE_B = "沉金工艺采用 ENIG，IPC-6012 规定镍层厚度不低于 3 μm，试样需在 260 ℃ 下保持 10 s。"
NOISE = "目 录\n1 范围 …………………………………… 1\n2 规范性引用文件 ………………………… 2\n3 术语和定义 ……………………………… 3"


class TestGraphPipeline:
    def test_build_retrieve_fuse_and_expand(self, tmp_path):
        graph = build_graph_from_nodes(
            [
                _doc(SENTENCE_A, "c1", "doc1"),
                _doc(SENTENCE_B, "c2", "doc2"),
            ]
        )
        assert graph.nodes and graph.edges

        # 落盘 → 重新加载（模拟入库与查询分属两个进程）
        path = tmp_path / "kg.json"
        graph.save(str(path))
        reloaded = KnowledgeGraph.load(str(path))
        assert reloaded is not None

        retriever = GraphRetriever(reloaded)
        passages = retriever.build_passages("镀金层厚度要求")
        assert passages, "图检索应命中镀金层相关实体"

        # 图结果以 dict 形态给出，便于在无 llama-index 环境下校验契约
        assert passages[0]["metadata"]["source_type"] == "knowledge_graph"
        assert passages[0]["metadata"]["chunk_ids"], "图证据应携带来源 chunk id"

    def test_graph_nodes_fuse_into_base_results(self, tmp_path):
        graph = build_graph_from_nodes([_doc(SENTENCE_A, "c1", "doc1")])
        retriever = GraphRetriever(graph)
        passages = retriever.build_passages("镀金层厚度要求")
        assert passages

        graph_node = SimpleNamespace(text=passages[0]["text"], node_id="graph-1", metadata=passages[0]["metadata"])
        base_node = _doc(SENTENCE_A, "c1", "doc1")

        fused = fuse_with_graph([_scored(base_node, 0.9)], [_scored(graph_node, 0.8)], graph_weight=0.5)
        assert len(fused) == 2
        # 排序：主结果 0.9 > 图结果 0.8 × 0.5
        assert fused[0].node.node_id == "c1"

    def test_expand_is_safe_without_llama_index(self):
        graph = build_graph_from_nodes([_doc(SENTENCE_A, "c1", "doc1")])
        nodes = [_scored(_doc(SENTENCE_A, "c1", "doc1"), 0.7)]
        expanded = expand_nodes_with_graph(nodes, graph, max_extra=2)
        assert len(expanded) >= len(nodes)


class TestAclPipeline:
    def test_acl_filters_then_compresses(self, monkeypatch):
        monkeypatch.setattr(security, "ACL_ENABLED", True)
        monkeypatch.setattr(compression, "COMPRESSION_ENABLED", True)

        nodes = [
            _scored(_doc(SENTENCE_A, "c1", "doc1", {"tenant_id": "acme", "visibility": "private"}), 0.9),
            _scored(_doc(SENTENCE_B, "c2", "doc2", {"tenant_id": "other", "visibility": "private"}), 0.85),
            _scored(_doc("公开标准内容说明。", "c3", "doc3", {"tenant_id": "other", "visibility": "public"}), 0.8),
        ]

        principal = Principal(user_id="u1", tenant_id="acme", authenticated=True)
        visible = filter_nodes(nodes, principal)
        assert len(visible) == 2
        assert all("他租户" not in item.node.text for item in visible)

        compressed = compress_pipeline(visible, "镀金层厚度要求")
        assert 0 < len(compressed) <= len(visible)

    def test_admin_sees_cross_tenant(self, monkeypatch):
        monkeypatch.setattr(security, "ACL_ENABLED", True)
        nodes = [
            _scored(_doc(SENTENCE_B, "c2", "doc2", {"tenant_id": "other", "visibility": "private"}), 0.85),
        ]
        admin = Principal(user_id="root", tenant_id="root", roles=frozenset({"admin"}))
        assert len(filter_nodes(nodes, admin)) == 1

    def test_legacy_documents_survive_when_disabled(self, monkeypatch):
        monkeypatch.setattr(security, "ACL_ENABLED", False)
        assert node_allowed({}, Principal()) is True


class TestCompressionPipeline:
    def test_dedupe_then_compress_reduces_context(self, monkeypatch):
        monkeypatch.setattr(compression, "COMPRESSION_ENABLED", True)

        # 文本需超过 COMPRESSION_MIN_CHARS（默认 160），否则压缩器按「过短不处理」直接返回
        long_a = SENTENCE_A + "附加说明：判定依据与复检规则由供需双方另行约定并记录于质量协议。" * 6
        long_b = long_a + "！"
        nodes = [
            _scored(_doc(long_a, "c1", "doc1"), 0.9),
            _scored(_doc(long_b, "c2", "doc1"), 0.88),
            _scored(_doc(NOISE, "c3", "doc2"), 0.7),
        ]

        deduped = dedupe_nodes(nodes)
        assert len(deduped) == 2, "近似重复的 chunk 应被合并"

        # 压缩是就地改写节点文本，因此必须在压缩前统计原始长度
        total_before = sum(len(item.node.text) for item in deduped)
        compressed = compress_pipeline(deduped, "镀金层厚度要求")
        assert compressed
        total_after = sum(len(item.node.text) for item in compressed)
        assert total_after < total_before

    def test_pipeline_handles_empty_input(self):
        assert compress_pipeline([], "任意查询") == []
        assert dedupe_nodes([]) == []


class TestCrossModuleContracts:
    def test_graph_and_compression_agree_on_node_shape(self):
        """图检索产出的节点文本应能被压缩链路识别（同一 duck typing 契约）。"""

        graph = build_graph_from_nodes([_doc(SENTENCE_A, "c1", "doc1")])
        passages = GraphRetriever(graph).build_passages("镀金层厚度要求")
        assert passages

        node = SimpleNamespace(text=passages[0]["text"], metadata=dict(passages[0]["metadata"]))
        assert compression.iter_node_texts([SimpleNamespace(node=node)])[0].startswith("[知识图谱]")

    def test_describe_functions_are_serializable(self):
        """诊断函数返回值必须可 JSON 序列化，否则 /health 会 500。"""

        import json

        assert json.dumps(graph_rag.describe_graph()) is not None
        assert json.dumps(security.describe_acl()) is not None
        assert json.dumps(compression.describe_compression()) is not None
