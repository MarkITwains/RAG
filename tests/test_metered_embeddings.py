import asyncio
import json

import httpx
import pytest
from llama_index.embeddings.openai import OpenAIEmbedding

from pcb_rag.metered_embeddings import EmbeddingUsageMixin
from pcb_rag.runtime import request_scope
from pcb_rag.usage import UsageBudgetExceeded


class MeteredEmbedding(EmbeddingUsageMixin, OpenAIEmbedding):
    pass


def make_model(handler):
    transport = httpx.MockTransport(handler)
    return MeteredEmbedding(
        model_name="test-embedding",
        api_base="https://embedding.invalid/v1",
        api_key="test-key",
        max_retries=0,
        embed_batch_size=2,
        http_client=httpx.Client(transport=transport),
        async_http_client=httpx.AsyncClient(transport=transport),
    )


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("kind", ["query", "text", "batch"])
def test_all_paths_account_each_http_call_once(kind, is_async):
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(
            200,
            json={
                "data": [{"index": i, "embedding": [0.5, 0.1]} for i in range(len(body["input"]))],
                "usage": {"total_tokens": 7},
            },
        )

    model = make_model(handler)
    name = {"query": "get_query_embedding", "text": "get_text_embedding", "batch": "get_text_embedding_batch"}[kind]
    value = ["a", "b", "c"] if kind == "batch" else "text"
    with request_scope(10) as control:
        result = asyncio.run(getattr(model, "a" + name)(value)) if is_async else getattr(model, name)(value)
    assert result and len(control.usage) == len(requests) == (2 if kind == "batch" else 1)
    assert all(row["status"] == "complete" for row in control.usage)
    # The upstream vector-only interface discards provider usage: never label estimates as actual.
    assert all(row["actual_tokens"] is None and row["usage_kind"] == "estimated" for row in control.usage)
    assert sum(row["input_estimated_tokens"] for row in control.usage) == (3 if kind == "batch" else 4)


@pytest.mark.parametrize("is_async", [False, True])
def test_failed_call_is_recorded_without_hidden_retries(is_async):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(500, json={"error": {"message": "test failure"}})

    model = make_model(handler)
    with request_scope(10) as control:
        with pytest.raises(Exception, match="test failure"):
            if is_async:
                asyncio.run(model.aget_query_embedding("query"))
            else:
                model.get_query_embedding("query")
    assert len(calls) == len(control.usage) == 1
    assert control.usage[0]["status"] == "failed"


def test_usage_budget_rejects_embedding_before_http(monkeypatch):
    calls = []
    model = make_model(lambda request: calls.append(request))
    monkeypatch.setenv("RAG_EMBEDDING_MAX_INPUT_TOKENS", "1")
    with request_scope(10), pytest.raises(UsageBudgetExceeded):
        model.get_query_embedding("long query")
    assert not calls
