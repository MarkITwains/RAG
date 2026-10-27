"""Exercise the HTTP payload of every embedding path without external requests."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from pcb_rag import api_clients
from pcb_rag.embedding_contract import (
    PROFILE_KEY,
    embedding_profile,
    ensure_ingest_embedding_profile,
    resolve_embedding_provider,
    validate_embedding_profile,
)
from pcb_rag.nvidia_embeddings import NvidiaApiEmbedding


def make_model(handler):
    transport = httpx.MockTransport(handler)
    return NvidiaApiEmbedding(
        model_name="nvidia/test-embedding",
        api_base="https://embedding.invalid/v1",
        api_key="test-key",
        expected_dimension=2,
        max_retries=0,
        http_client=httpx.Client(transport=transport),
        async_http_client=httpx.AsyncClient(transport=transport),
    )


def responder(request):
    body = json.loads(request.content)
    value = 1.0 if body["input_type"] == "query" else 2.0
    return httpx.Response(
        200,
        json={"data": [{"index": i, "embedding": [value, float(i)]} for i in range(len(body["input"]))]},
    )


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("kind", ["query", "text", "batch"])
def test_modes_reach_http_body_for_all_paths(kind, is_async):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return responder(request)

    model = make_model(handler)
    if kind == "query":
        result = (
            asyncio.run(model.aget_query_embedding("line1\nline2"))
            if is_async
            else model.get_query_embedding("line1\nline2")
        )
    elif kind == "text":
        result = asyncio.run(model.aget_text_embedding("text")) if is_async else model.get_text_embedding("text")
    else:
        result = (
            asyncio.run(model.aget_text_embedding_batch(["a", "b"]))
            if is_async
            else model.get_text_embedding_batch(["a", "b"])
        )
    assert result
    assert bodies[0]["input_type"] == ("query" if kind == "query" else "passage")
    assert bodies[0]["encoding_format"] == "float"
    assert bodies[0]["truncate"] == "END"
    if kind == "query":
        assert bodies[0]["input"] == ["line1 line2"]
    assert model.additional_kwargs == {}


def test_async_embedding_retries_retryable_failures(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(503, json={"error": "temporary"})
        return responder(request)

    model = make_model(handler)
    sleeps = []
    monkeypatch.setattr("asyncio.sleep", lambda seconds: _async_sleep(sleeps, seconds))

    result = asyncio.run(model.aget_query_embedding("query"))

    assert result
    assert len(calls) == 2
    assert sleeps == [1.5]


async def _async_sleep(records, seconds):
    records.append(seconds)


def test_concurrent_query_and_document_modes_do_not_leak():
    model = make_model(responder)
    with ThreadPoolExecutor(max_workers=8) as pool:
        pending = [
            pool.submit(model.get_query_embedding if i % 2 == 0 else model.get_text_embedding, "test")
            for i in range(24)
        ]
        assert [future.result()[0] for future in pending] == [1.0 if i % 2 == 0 else 2.0 for i in range(24)]


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"index": 1, "embedding": [1.0, 2.0]}],
        [{"index": 0, "embedding": [1.0]}],
        [{"index": 0, "embedding": []}],
        [{"index": 0, "embedding": "base64"}],
    ],
)
def test_malformed_vectors_are_rejected(rows):
    model = make_model(lambda request: httpx.Response(200, json={"data": rows}))
    with pytest.raises(ValueError):
        model.get_query_embedding("query")


def test_batch_response_is_reordered_by_input_index():
    rows = [{"index": 1, "embedding": [2.0, 0.0]}, {"index": 0, "embedding": [1.0, 0.0]}]
    model = make_model(lambda request: httpx.Response(200, json={"data": rows}))
    assert model.get_text_embedding_batch(["first", "second"]) == [[1.0, 0.0], [2.0, 0.0]]


def test_factory_automatically_selects_nvidia_for_official_host(monkeypatch):
    monkeypatch.setattr(api_clients, "EMBED_BASE_URL", "https://integrate.api.nvidia.com/v1")
    monkeypatch.setattr(api_clients, "EMBED_MODEL", "nvidia/test-embedding")
    monkeypatch.setattr(api_clients, "EMBED_API_PROVIDER", "auto")
    assert isinstance(api_clients.build_embed_model(backend="api"), NvidiaApiEmbedding)


def test_configured_dimension_is_sent_as_a_real_request_parameter():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return responder(request)

    make_model(handler).get_query_embedding("q")
    assert bodies[0]["dimensions"] == 2


def test_native_dimension_is_requested_without_dimensions():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return responder(request)

    model = NvidiaApiEmbedding(
        model_name="nvidia/test-embedding",
        api_base="https://embedding.invalid/v1",
        api_key="test-key",
        expected_dimension=0,
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    model.get_query_embedding("q")
    assert "dimensions" not in bodies[0]


def test_factory_rejects_unsupported_truncate_and_dimensions(monkeypatch):
    monkeypatch.setattr(api_clients, "EMBED_BASE_URL", "https://integrate.api.nvidia.com/v1")
    monkeypatch.setattr(api_clients, "EMBED_MODEL", "nvidia/nemotron-3.5-embed-8b")
    monkeypatch.setattr(api_clients, "EMBED_API_PROVIDER", "nvidia")

    monkeypatch.setattr(api_clients, "EMBED_TRUNCATE", "START")
    monkeypatch.setattr(api_clients, "EMBED_DIM", 4096)
    with pytest.raises(ValueError, match="END or NONE"):
        api_clients.build_embed_model(backend="api")

    monkeypatch.setattr(api_clients, "EMBED_TRUNCATE", "END")
    monkeypatch.setattr(api_clients, "EMBED_DIM", 4097)
    with pytest.raises(ValueError, match="4096"):
        api_clients.build_embed_model(backend="api")


def test_factory_leaves_other_openai_compatible_providers_unchanged(monkeypatch):
    monkeypatch.setattr(api_clients, "EMBED_BASE_URL", "https://other.invalid/v1")
    monkeypatch.setattr(api_clients, "EMBED_MODEL", "other-embedding")
    monkeypatch.setattr(api_clients, "EMBED_API_PROVIDER", "auto")
    from llama_index.embeddings.openai import OpenAIEmbedding

    model = api_clients.build_embed_model(backend="api")
    assert isinstance(model, OpenAIEmbedding) and not isinstance(model, NvidiaApiEmbedding)


def test_self_hosted_nvidia_requires_explicit_provider():
    assert resolve_embedding_provider("nvidia", "https://nim.internal/v1") == "nvidia"
    assert resolve_embedding_provider("auto", "https://integrate.api.nvidia.com.evil.invalid/v1") == "openai"
    with pytest.raises(ValueError):
        resolve_embedding_provider("typo", "https://embedding.invalid/v1")


def test_collection_profile_blocks_wrong_provider_and_legacy_nvidia_vectors():
    profile = embedding_profile(provider="nvidia", api_base="https://api.invalid/v1", model="test", dimension=2)
    validate_embedding_profile(json.dumps({PROFILE_KEY: profile}), profile)
    with pytest.raises(RuntimeError, match="re-embedded"):
        validate_embedding_profile("", profile)
    with pytest.raises(RuntimeError, match="differs"):
        validate_embedding_profile(json.dumps({PROFILE_KEY: profile}), {**profile, "model": "different"})


def test_legacy_openai_collection_remains_supported():
    profile = embedding_profile(provider="openai", api_base="https://api.invalid/v1", model="test", dimension=2)
    validate_embedding_profile("", profile)


@pytest.mark.parametrize("row_count", [0, 1])
def test_ingest_only_registers_empty_collections(row_count):
    profile = embedding_profile(provider="nvidia", api_base="https://api.invalid/v1", model="test", dimension=2)

    class Client:
        description = ""
        writes = 0

        def describe_collection(self, name):
            return {"description": self.description}

        def get_collection_stats(self, name):
            return {"row_count": row_count}

        def alter_collection_properties(self, *, collection_name, properties):
            self.writes += 1
            self.description = properties["collection.description"]

    client = Client()
    if row_count:
        with pytest.raises(RuntimeError, match="re-embedded"):
            ensure_ingest_embedding_profile(client, "test", profile)
        assert client.writes == 0
    else:
        ensure_ingest_embedding_profile(client, "test", profile)
        validate_embedding_profile(client.description, profile)
        assert client.writes == 1
