"""知识库元件的单元测试。

元件层的价值不在"能不能检索"（那是 query.py 的事），而在：
- 返回的是结构化对象而不是拼好的字符串
- 降级状态显式暴露，不被静默吞掉
- tool 定义与分发一致

因此这里全部用假节点，不连 Milvus、不加载模型。
"""

import pytest

pytest.importorskip("llama_index.core")

from llama_index.core.llms import CompletionResponse, CustomLLM  # noqa: E402
from llama_index.core.schema import NodeWithScore, TextNode  # noqa: E402

from pcb_rag import dify_external_api  # noqa: E402
from pcb_rag.engine import (  # noqa: E402
    AnswerResult,
    Evidence,
    KnowledgeBase,
    RetrievalResult,
)


def _node(
    text: str,
    node_id: str = "c1",
    *,
    score: float = 0.88,
    source: str = "GBT4677-2002印制板测试方法.txt",
    section: str = "5.4 特性阻抗测试",
    doc_type: str = "national_standard",
    source_type: str = "standard_national",
):
    node = TextNode(
        text=text,
        id_=node_id,
        metadata={
            "source_path": source,
            "parent_title": section,
            "doc_type": doc_type,
            "source_type": source_type,
        },
    )
    return NodeWithScore(node=node, score=score)


def _ready_kb(**kwargs) -> KnowledgeBase:
    """构造一个"已就绪"的元件，底层状态用假对象占位。"""
    kb = KnowledgeBase(auto_init=False, **kwargs)
    kb._state = lambda: {  # type: ignore[method-assign]
        "index": object(),
        "bm25_index": object(),
        "rerank": object(),
        "graph": None,
        "recall_k": 200,
    }
    kb._ready = True
    return kb


class _FakeLLM(CustomLLM):
    """llama-index 的 Settings.llm 有类型校验，假对象必须是真的 BaseLLM 子类。"""

    model_name: str = "fake-llm"
    context_window: int = 8192
    num_output: int = 1024

    @property
    def metadata(self):
        from llama_index.core.base.llms.types import LLMMetadata

        return LLMMetadata(
            model_name=self.model_name,
            context_window=self.context_window,
            num_output=self.num_output,
        )

    def complete(self, prompt, formatted: bool = False, **kwargs):
        return CompletionResponse(text="采用时域反射法测量[1]。")

    def stream_complete(self, prompt, formatted: bool = False, **kwargs):
        yield CompletionResponse(text="采用时域反射法测量[1]。", delta="采用时域反射法测量[1]。")


SPEC_TEXT = "特性阻抗采用时域反射法测量，探头上升时间应不大于 35 ps。"


class TestHealth:
    def test_reports_all_degraded_when_uninitialized(self):
        kb = KnowledgeBase(auto_init=False)
        kb._state = lambda: {}  # type: ignore[method-assign]
        health = kb.health()
        assert health["ready"] is False
        assert "vector_index_missing" in health["degraded"]
        assert "graph_unavailable" in health["degraded"]

    def test_reports_graph_missing_while_index_ready(self):
        kb = _ready_kb()
        health = kb.health()
        assert health["ready"] is True
        assert health["degraded"] == ["graph_unavailable"]


class TestRetrieve:
    def test_returns_structured_evidence(self, monkeypatch):
        kb = _ready_kb()
        monkeypatch.setattr(
            dify_external_api, "_retrieve_nodes", lambda **kw: [_node(SPEC_TEXT)]
        )

        result = kb.retrieve("阻抗怎么测", top_k=5)

        assert isinstance(result, RetrievalResult)
        assert len(result.evidence) == 1
        ev = result.evidence[0]
        assert isinstance(ev, Evidence)
        assert ev.index == 1
        assert ev.source_name == "GBT4677-2002印制板测试方法"
        assert ev.section == "5.4 特性阻抗测试"
        assert ev.score == pytest.approx(0.88)
        assert ev.tokens > 0
        assert SPEC_TEXT in ev.text

    def test_empty_query_short_circuits(self):
        kb = _ready_kb()
        result = kb.retrieve("   ")
        assert result.evidence == []
        assert "empty_query" in result.degraded

    def test_retrieval_error_is_reported_not_raised(self, monkeypatch):
        kb = _ready_kb()

        def boom(**kw):
            raise RuntimeError("Milvus 断开")

        monkeypatch.setattr(dify_external_api, "_retrieve_nodes", boom)
        result = kb.retrieve("任意问题")

        assert result.evidence == []
        assert any(d.startswith("retrieval_error") for d in result.degraded)

    def test_degraded_flags_propagate(self, monkeypatch):
        """图不可用要传到结果里，agent 才知道少了一路证据。"""
        kb = _ready_kb()
        monkeypatch.setattr(dify_external_api, "_retrieve_nodes", lambda **kw: [_node(SPEC_TEXT)])

        result = kb.retrieve("阻抗怎么测")
        assert "graph_unavailable" in result.degraded

    def test_full_text_omitted_when_disabled(self, monkeypatch):
        kb = _ready_kb(include_full_text=False)
        monkeypatch.setattr(dify_external_api, "_retrieve_nodes", lambda **kw: [_node(SPEC_TEXT)])

        result = kb.retrieve("阻抗怎么测")
        ev = result.evidence[0]
        assert ev.text == ""
        assert ev.snippet  # 摘要仍在，用于展示

    def test_top_k_forwarded(self, monkeypatch):
        captured = {}

        def fake(**kw):
            captured.update(kw)
            return []

        kb = _ready_kb()
        monkeypatch.setattr(dify_external_api, "_retrieve_nodes", fake)
        kb.retrieve("问题", top_k=7, score_threshold=0.3)

        assert captured["top_k"] == 7
        assert captured["score_threshold"] == pytest.approx(0.3)

    def test_to_dict_is_json_serializable(self, monkeypatch):
        import json

        kb = _ready_kb()
        monkeypatch.setattr(dify_external_api, "_retrieve_nodes", lambda **kw: [_node(SPEC_TEXT)])
        payload = kb.retrieve("阻抗怎么测").to_dict()

        json.dumps(payload, ensure_ascii=False)  # 不抛异常即通过
        assert payload["is_empty"] is False
        assert payload["evidence"][0]["section"] == "5.4 特性阻抗测试"


class TestAnswer:
    def test_no_evidence_returns_explicit_message(self, monkeypatch):
        kb = _ready_kb()
        monkeypatch.setattr(dify_external_api, "_retrieve_nodes", lambda **kw: [])

        result = kb.answer("知识库里没有的问题")
        assert isinstance(result, AnswerResult)
        assert "未检索到" in result.answer
        assert "no_evidence" in result.degraded

    def test_generation_failure_is_reported(self, monkeypatch):
        kb = _ready_kb()
        monkeypatch.setattr(dify_external_api, "_retrieve_nodes", lambda **kw: [_node(SPEC_TEXT)])

        class BrokenLLM(_FakeLLM):
            def complete(self, prompt, **kwargs):
                raise RuntimeError("模型超时")

        import llama_index.core as li_core

        monkeypatch.setattr(li_core.Settings, "_llm", BrokenLLM(), raising=False)

        result = kb.answer("阻抗怎么测")
        assert result.answer == ""
        assert any(d.startswith("generation_error") for d in result.degraded)

    def test_answer_returns_citations(self, monkeypatch):
        kb = _ready_kb()
        monkeypatch.setattr(dify_external_api, "_retrieve_nodes", lambda **kw: [_node(SPEC_TEXT)])

        import llama_index.core as li_core

        monkeypatch.setattr(li_core.Settings, "_llm", _FakeLLM(), raising=False)

        result = kb.answer("阻抗怎么测")
        assert "时域反射法" in result.answer
        assert result.evidence  # 证据随答案返回


class TestGraphNeighbors:
    def test_missing_graph_degrades(self):
        kb = _ready_kb()
        result = kb.graph_neighbors("镀金层厚度")
        assert result.found is False
        assert "graph_unavailable" in result.degraded

    def test_found_entity_returns_relations(self):
        from pcb_rag.graph_rag import KnowledgeGraph, Triple

        graph = KnowledgeGraph()
        graph.add_triple(
            Triple(
                head="镀金层厚度",
                relation="测试方法为",
                tail="X射线荧光法",
                head_desc="金层最小厚度指标",
                tail_desc="无损测厚方法",
                relation_desc="规定了测量手段",
                evidence="采用X射线荧光法测量",
            )
        )

        kb = KnowledgeBase(auto_init=False)
        kb._state = lambda: {"index": object(), "graph": graph}  # type: ignore[method-assign]
        kb._ready = True

        result = kb.graph_neighbors("镀金层厚度")
        assert result.found is True
        assert result.description == "金层最小厚度指标"
        assert result.relations[0]["relation"] == "测试方法为"
        assert "规定了测量手段" in result.text

    def test_unknown_entity_returns_not_found(self):
        from pcb_rag.graph_rag import KnowledgeGraph, Triple

        graph = KnowledgeGraph()
        graph.add_triple(Triple(head="镀金层厚度", relation="r", tail="X射线荧光法"))

        kb = KnowledgeBase(auto_init=False)
        kb._state = lambda: {"index": object(), "graph": graph}  # type: ignore[method-assign]
        kb._ready = True

        result = kb.graph_neighbors("完全无关的词组")
        assert result.found is False


class TestToolSpecs:
    def test_four_tools_with_valid_schema(self):
        kb = KnowledgeBase(auto_init=False)
        specs = kb.tool_specs()

        assert len(specs) == 4
        for spec in specs:
            assert spec["type"] == "function"
            fn = spec["function"]
            assert fn["name"]
            assert fn["description"]
            assert fn["parameters"]["type"] == "object"

    def test_search_knowledge_requires_query(self):
        kb = KnowledgeBase(auto_init=False)
        spec = next(s for s in kb.tool_specs() if s["function"]["name"] == "search_knowledge")
        assert spec["function"]["parameters"]["required"] == ["query"]

    def test_unknown_tool_returns_error(self):
        kb = KnowledgeBase(auto_init=False)
        assert "error" in kb.call_tool("nope", {})

    def test_knowledge_status_dispatch(self):
        kb = _ready_kb()
        result = kb.call_tool("knowledge_status", {})
        assert result["ready"] is True

    def test_search_knowledge_dispatch_drops_context(self, monkeypatch):
        kb = _ready_kb()
        monkeypatch.setattr(dify_external_api, "_retrieve_nodes", lambda **kw: [_node(SPEC_TEXT)])

        result = kb.call_tool("search_knowledge", {"query": "阻抗", "top_k": 3})
        assert result["evidence"]
        assert "context" not in result

    def test_top_k_clamped_to_max(self, monkeypatch):
        captured = {}

        def fake(**kw):
            captured.update(kw)
            return []

        kb = _ready_kb()
        monkeypatch.setattr(dify_external_api, "_retrieve_nodes", fake)
        kb.call_tool("search_knowledge", {"query": "x", "top_k": 999})

        assert captured["top_k"] == 30


class TestAsyncWrappers:
    def test_aretrieve_runs_in_thread(self, monkeypatch):
        import asyncio

        kb = _ready_kb()
        monkeypatch.setattr(dify_external_api, "_retrieve_nodes", lambda **kw: [_node(SPEC_TEXT)])

        result = asyncio.run(kb.aretrieve("阻抗怎么测"))
        assert len(result.evidence) == 1
