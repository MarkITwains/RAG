"""Model factories must propagate configuration to their real clients."""

import pytest

pytest.importorskip("llama_index.core")
from pcb_rag import api_clients


@pytest.mark.parametrize("backend", ["local", "api"])
def test_build_llm_uses_configured_temperature(monkeypatch, backend):
    monkeypatch.setattr(api_clients, "LLM_TEMPERATURE", 0.13)
    monkeypatch.setattr(api_clients, "LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setattr(api_clients, "LLM_MODEL", "configured-model")

    llm = api_clients.build_llm(backend=backend)

    actual = llm.temperature if backend == "api" else llm.additional_kwargs["temperature"]
    assert actual == pytest.approx(0.13)


@pytest.mark.parametrize("backend", ["local", "api"])
def test_explicit_temperature_overrides_environment(monkeypatch, backend):
    monkeypatch.setattr(api_clients, "LLM_TEMPERATURE", 0.13)
    monkeypatch.setattr(api_clients, "LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setattr(api_clients, "LLM_MODEL", "configured-model")

    llm = api_clients.build_llm(backend=backend, temperature=0.0)

    actual = llm.temperature if backend == "api" else llm.additional_kwargs["temperature"]
    assert actual == 0.0


def test_local_embedding_uses_ollama_timeout(monkeypatch):
    monkeypatch.setattr(api_clients, "OLLAMA_TIMEOUT", 17.0)

    embed = api_clients.build_embed_model(backend="local")

    timeout = embed._client._client.timeout
    assert timeout.connect == 17.0
    assert timeout.read == 17.0


def test_legacy_ollama_embedding_constructor_also_applies_timeout(monkeypatch):
    """0.4.x creates clients before a subclass can supply client_kwargs."""
    from types import SimpleNamespace

    monkeypatch.setattr(api_clients, "OLLAMA_TIMEOUT", 17.0)

    class HTTPClient:
        def __init__(self, timeout=None, **_):
            self.timeout = SimpleNamespace(connect=timeout, read=timeout)

        def close(self):
            pass

    class Client:
        def __init__(self, host=None, timeout=None, **_):
            self._client = HTTPClient(timeout=timeout)

    class AsyncClient(Client):
        pass

    class LegacyOllamaEmbedding:
        def __init__(self, **kwargs):
            assert "client_kwargs" not in kwargs
            self._client = Client(host=kwargs["base_url"])
            self._async_client = AsyncClient(host=kwargs["base_url"])

    monkeypatch.setattr("ollama.Client", Client)
    monkeypatch.setattr("ollama.AsyncClient", AsyncClient)
    monkeypatch.setattr("httpx.Client", HTTPClient)
    monkeypatch.setattr("httpx.AsyncClient", HTTPClient)

    embed = api_clients.build_embed_model(backend="local")

    assert embed._client._client.timeout.connect == 17.0
    assert embed._async_client._client.timeout.connect == 17.0


def test_attention_splitter_uses_embedding_context_window(monkeypatch):
    from pcb_rag import ingest

    monkeypatch.setattr(ingest, "EMBED_BACKEND", "local")
    monkeypatch.setattr(ingest, "ATTENTION_USE_OLLAMA", True)
    monkeypatch.setenv("OLLAMA_NUM_CTX", "8192")
    monkeypatch.setenv("EMBED_NUM_CTX", "333")
    splitter = ingest.AttentionSemanticSplitter(model_name="qwen3-embedding:8b")

    splitter._load_model()

    assert splitter._ollama_embed.ollama_additional_kwargs["num_ctx"] == 333
