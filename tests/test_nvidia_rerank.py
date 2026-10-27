"""Wire-level tests for the NVIDIA NIM reranking adapter (no external requests).

The NIM retrieval endpoint is not wire-compatible with Jina/SiliconFlow: the
query and every passage are objects, ``top_n`` is rejected, and the response is
``rankings[].logit``. These tests pin all three differences down.
"""

import math

import pytest

from pcb_rag import api_clients
from pcb_rag.api_clients import (
    ApiReranker,
    NvidiaApiReranker,
    build_api_reranker,
    resolve_rerank_provider,
)
from pcb_rag.rerank_control import RerankStopped

NVIDIA_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia/nemotron-3.5-rerank-8b/reranking"
MODEL = "nvidia/nemotron-3.5-rerank-8b"


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.headers = {}

    def json(self):
        return self._payload


class RecordingPost:
    """Stand in for ``requests.post`` and remember the outgoing body."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def __call__(self, url, json=None, headers=None, timeout=None, allow_redirects=None):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return FakeResponse(self.payload)

    @property
    def body(self):
        return self.calls[-1]["json"]


@pytest.fixture
def control_db(tmp_path, monkeypatch):
    monkeypatch.setenv("RERANK_CONTROL_DB", str(tmp_path / "control.sqlite3"))
    monkeypatch.setenv("RERANK_MIN_INTERVAL", "0")
    monkeypatch.delenv("RAG_USAGE_LEDGER", raising=False)


def make_reranker(**overrides):
    params = {
        "api_url": NVIDIA_URL,
        "api_key": "test-key",
        "model": MODEL,
        "top_n": 2,
        "max_candidates": 50,
    }
    params.update(overrides)
    return NvidiaApiReranker(**params)


def test_payload_uses_query_and_passage_objects_without_top_n(control_db, monkeypatch):
    post = RecordingPost({"rankings": [{"index": 0, "logit": -1.955}, {"index": 1, "logit": -3.4}]})
    monkeypatch.setattr("requests.post", post)

    make_reranker()._request_scores("阻焊开窗", ["正文 A", "正文 B"])

    assert post.body == {
        "model": MODEL,
        "query": {"text": "阻焊开窗"},
        "passages": [{"text": "正文 A"}, {"text": "正文 B"}],
        "truncate": "END",
    }
    assert "top_n" not in post.body and "documents" not in post.body
    assert post.calls[-1]["headers"]["Authorization"] == "Bearer test-key"


def test_logits_are_mapped_by_index_and_normalised_to_probability(control_db, monkeypatch):
    # Deliberately out of order: the API promises an index, not a position.
    post = RecordingPost({"rankings": [{"index": 1, "logit": -3.3998916}, {"index": 0, "logit": -1.9551046}]})
    monkeypatch.setattr("requests.post", post)

    scores = make_reranker()._request_scores("q", ["relevant", "irrelevant"])

    assert scores[0] == pytest.approx(1 / (1 + math.exp(1.9551046)))
    assert scores[1] == pytest.approx(1 / (1 + math.exp(3.3998916)))
    assert 0.0 < scores[0] < 1.0 and scores[0] > scores[1]


@pytest.mark.parametrize(
    "rankings",
    [
        None,
        [],
        [{"index": 0, "logit": -2.0}],  # fewer rankings than passages
        [{"index": 5, "logit": -2.0}, {"index": 1, "logit": -2.0}],  # index out of range
        [{"index": 0, "logit": -2.0}, {"index": 0, "logit": -3.0}],  # duplicate index
        [{"index": 0, "logit": float("nan")}, {"index": 1, "logit": -3.0}],  # nonfinite logit
    ],
)
def test_malformed_rankings_are_rejected(control_db, monkeypatch, rankings):
    payload = {} if rankings is None else {"rankings": rankings}
    monkeypatch.setattr("requests.post", RecordingPost(payload))

    # Invalid scores trip the circuit: the pipeline must not fall back to raw
    # recall order while pretending the reranker had run.
    with pytest.raises(RerankStopped):
        make_reranker()._request_scores("q", ["a", "b"])


def test_passage_cap_and_truncate_are_enforced_before_the_request(control_db, monkeypatch):
    post = RecordingPost({"rankings": []})
    monkeypatch.setattr("requests.post", post)

    with pytest.raises(ValueError, match="512"):
        make_reranker(max_candidates=1000)._request_scores("q", ["x"] * 513)
    with pytest.raises(ValueError, match="RERANK_TRUNCATE"):
        make_reranker(truncate="START")._request_scores("q", ["x"])
    assert post.calls == []


def test_postprocess_orders_and_trims_with_probability_scores(control_db, monkeypatch):
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

    post = RecordingPost(
        {"rankings": [{"index": 0, "logit": -6.0}, {"index": 1, "logit": 2.0}, {"index": 2, "logit": -0.5}]}
    )
    monkeypatch.setattr("requests.post", post)
    nodes = [
        NodeWithScore(node=TextNode(text="first", id_="1")),
        NodeWithScore(node=TextNode(text="second", id_="2")),
        NodeWithScore(node=TextNode(text="third", id_="3")),
    ]

    result = make_reranker(top_n=2)._postprocess_nodes(nodes, query_bundle=QueryBundle(query_str="q"))

    assert [node.node.id_ for node in result] == ["2", "3"]
    assert all(0.0 <= node.score <= 1.0 for node in result)


@pytest.mark.parametrize(
    "setting,url,expected",
    [
        ("auto", NVIDIA_URL, "nvidia"),
        ("auto", "https://ai.api.nvidia.com/v1/retrieval/x", "openai"),  # path suffix missing
        ("auto", "https://ai.api.nvidia.com.evil.invalid/v1/x/reranking", "openai"),
        ("auto", "https://api.siliconflow.cn/v1/rerank", "openai"),
        ("nvidia", "https://nim.internal/v1/reranking", "nvidia"),
    ],
)
def test_provider_resolution(setting, url, expected):
    assert resolve_rerank_provider(setting, url) == expected


def test_provider_rejects_unknown_values():
    with pytest.raises(ValueError, match="RERANK_API_PROVIDER"):
        resolve_rerank_provider("jina", NVIDIA_URL)


def test_factory_selects_wire_format_from_the_endpoint(monkeypatch):
    monkeypatch.setattr(api_clients, "RERANK_API_URL", NVIDIA_URL)
    monkeypatch.setattr(api_clients, "RERANK_API_MODEL", MODEL)
    monkeypatch.setattr(api_clients, "RERANK_API_PROVIDER", "auto")
    assert isinstance(build_api_reranker(top_n=10), NvidiaApiReranker)

    monkeypatch.setattr(api_clients, "RERANK_API_URL", "https://api.siliconflow.cn/v1/rerank")
    jina_style = build_api_reranker(top_n=10)
    assert isinstance(jina_style, ApiReranker) and not isinstance(jina_style, NvidiaApiReranker)


def test_factory_requires_a_url(monkeypatch):
    monkeypatch.setattr(api_clients, "RERANK_API_URL", "")
    with pytest.raises(RuntimeError, match="RERANK_API_URL"):
        build_api_reranker(top_n=10)
