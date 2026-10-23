"""Unit tests must not inherit a developer's deployment file or credentials."""

import os

# Run before test collection imports API/query. Explicit env-loader tests can
# still pass a fixture file; production startup retains its existing behavior.
os.environ["PCB_RAG_LOAD_DOTENV"] = "0"
os.environ.setdefault("ACL_TRUST_HEADERS", "0")


import pytest


@pytest.fixture
def offline_pipeline(tmp_path, monkeypatch):
    """Real pipeline and real LlamaIndex nodes, deterministic IO boundaries."""
    import copy
    from types import SimpleNamespace
    from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
    from llama_index.core.bridge.pydantic import PrivateAttr
    from llama_index.core.schema import TextNode, NodeWithScore
    from pcb_rag.retrieval import PipelineConfig, RetrievalPipeline, build_lexical
    from pcb_rag.revisions import RevisionStore

    class LLM(CustomLLM):
        _prompts: list = PrivateAttr(default_factory=list)

        @property
        def metadata(self):
            return LLMMetadata(context_window=8192, num_output=1024, model_name="deterministic-test")

        def complete(self, prompt, **kwargs):
            self._prompts.append(prompt)
            return CompletionResponse(text="测试结论来自原文[1]。", raw={"usage": {"total_tokens": 20}})

        def stream_complete(self, prompt, **kwargs):
            self._prompts.append(prompt)
            text = ""
            for delta in ("测试结论", "来自原文[1]。"):
                text += delta
                yield CompletionResponse(text=text, delta=delta)

    class Index:
        def __init__(self):
            self.calls = []
            self.rows = []

        def as_retriever(self, **kwargs):
            self.filters = kwargs.get("filters")
            return self

        def retrieve(self, query):
            self.calls.append(query)
            return copy.deepcopy(self.rows(query) if callable(self.rows) else self.rows)

    def node(key, text="铜厚不小于 35 μm。", **metadata):
        return TextNode(
            id_=key, text=text, metadata={"source_path": key + ".txt", "source_type": "technical_doc", **metadata}
        )

    index, llm = Index(), LLM()
    docs = [node("low", "目录\n铜厚", low_value=True), node("real")]
    index.rows = [
        NodeWithScore(node=docs[0], score=0.9),
        NodeWithScore(node=docs[1], score=0.8),
        NodeWithScore(node=docs[1], score=0.7),
    ]
    pipeline = RetrievalPipeline(
        PipelineConfig(routing=False, lexical_enabled=False, rerank_enabled=False, expand=False),
        state={"index": index, "llm": llm, "revision": "legacy", "source_nodes": {n.id_: n for n in docs}},
        revision_store=RevisionStore(tmp_path / "state.json"),
    )
    from pcb_rag import retrieval

    monkeypatch.setattr(retrieval, "_default", pipeline)
    return SimpleNamespace(
        pipeline=pipeline,
        index=index,
        llm=llm,
        node=node,
        docs=docs,
        score=lambda n, score=0.8: NodeWithScore(node=n, score=score),
        build_lexical=build_lexical,
    )
