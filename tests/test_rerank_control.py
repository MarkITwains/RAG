"""Rerank inputs, accounting and shared stops must be enforced before spending."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from pcb_rag.rerank_control import (
    RerankBudgetExceeded,
    RerankControl,
    RerankStopped,
    estimate_tokens,
    extract_usage,
)


@pytest.mark.parametrize(
    "data,expected,source",
    [
        ({"meta": {"billed_units": {"input_tokens": 88, "output_tokens": 0}}}, 88, "meta.billed_units"),
        ({"meta": {"tokens": {"input_tokens": 90, "output_tokens": 2}}}, 92, "meta.tokens"),
        ({"usage": {"total_tokens": 42, "prompt_tokens": 40}}, 42, "usage"),
        ({"usage": {"prompt_tokens": 40, "completion_tokens": 2}}, 42, "usage"),
        ({"usage": {"total_tokens": -1}}, None, "unavailable"),
        ({"usage": {"total_tokens": True}}, None, "unavailable"),
        ({}, None, "unavailable"),
    ],
)
def test_usage_sources_and_unavailable_values(data, expected, source):
    tokens, actual_source, _ = extract_usage(data)
    assert tokens == expected and actual_source == source


def reserve(control, **overrides):
    args = dict(
        provider="account",
        model="rerank",
        query="",
        documents=["a"],
        interval=0,
        phase="test",
        activity="measured",
        current_time=100,
    )
    return control.reserve(**{**args, **overrides})


def test_budget_reserves_inflight_calls_atomically(tmp_path):
    control = RerankControl(tmp_path / "budget.sqlite3", "run", 130, 100)

    def attempt(_):
        try:
            return reserve(control)
        except RerankStopped:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        tickets = [ticket for ticket in pool.map(attempt, range(8)) if ticket]
    assert len(tickets) == 2
    state = control.snapshot()
    assert state["reserved_tokens"] == 130 and state["attempts"] == 2
    assert state["circuit_reason"] == "token_budget_exhausted"


def test_unknown_usage_is_conservatively_accounted_and_not_called_reported(tmp_path):
    control = RerankControl(tmp_path / "usage.sqlite3", "run", 1000, 10)
    ticket = reserve(control)
    control.finish(ticket, status="transport_error")
    state = control.snapshot()
    assert state["accounted_tokens"] == estimate_tokens("", ["a"])
    assert state["reported_tokens"] == 0 and state["unknown_usage_requests"] == 1
    assert state["reserved_tokens"] == 0


def test_reported_usage_settles_reservations_and_cannot_be_settled_twice(tmp_path):
    control = RerankControl(tmp_path / "usage.sqlite3", "run", 1000, 10)
    ticket = reserve(control)
    control.finish(ticket, status="success", actual_tokens=10, usage_source="meta.billed_units")
    state = control.snapshot()
    assert state["accounted_tokens"] == 10 and state["reported_tokens"] == 10
    assert state["reserved_tokens"] == 0
    with pytest.raises(RuntimeError, match="already settled"):
        control.finish(ticket, status="success", actual_tokens=10)


def test_rate_slots_are_shared_between_scopes_and_zero_disables_spacing(tmp_path):
    path = tmp_path / "shared.sqlite3"
    a = RerankControl(path, "a", 1000, 10)
    b = RerankControl(path, "b", 1000, 10)
    assert reserve(a, interval=6.2).scheduled_at == 100
    assert reserve(b, interval=6.2).scheduled_at == 106.2
    assert reserve(a, interval=0).scheduled_at == 100
    assert reserve(a, provider="unlimited", interval=0).scheduled_at == 100
    assert reserve(b, provider="unlimited", interval=0).scheduled_at == 100


def test_shared_circuit_and_request_budget_prevent_new_calls(tmp_path):
    path = tmp_path / "shared.sqlite3"
    a = RerankControl(path, "run", 1000, 1)
    b = RerankControl(path, "run", 1000, 1)
    reserve(a)
    with pytest.raises(RerankBudgetExceeded, match="request_budget"):
        reserve(b)
    with pytest.raises(RerankStopped):
        a.check()
    assert a.snapshot()["attempts"] == 1


def test_zero_limits_keep_accounting_without_total_token_or_request_cutoff(tmp_path):
    control = RerankControl(tmp_path / "unlimited.sqlite3", "run", 0, 0)
    with control.connect() as db:
        db.execute("UPDATE budgets SET accounted_tokens=100000000,attempts=3000 WHERE scope='run'")
    ticket = reserve(control)
    control.finish(ticket, status="success", actual_tokens=123, usage_source="usage")
    state = control.snapshot()
    assert state["accounted_tokens"] == 100000123 and state["attempts"] == 3001
    assert state["circuit_reason"] is None
    assert state["token_limit_enabled"] is False and state["request_limit_enabled"] is False
    control.check()


@pytest.mark.parametrize("tokens,requests", [(-1, 0), (0, -1)])
def test_negative_limits_remain_invalid(tmp_path, tokens, requests):
    with pytest.raises(ValueError, match="negative"):
        RerankControl(tmp_path / "negative.sqlite3", "run", tokens, requests)


@pytest.fixture
def api_client(monkeypatch, tmp_path):
    pytest.importorskip("llama_index.core")
    from pcb_rag import api_clients as api

    monkeypatch.setenv("RERANK_CONTROL_DB", str(tmp_path / "client.sqlite3"))
    monkeypatch.setenv("RERANK_BUDGET_SCOPE", "test-client")
    monkeypatch.setenv("RERANK_TOKEN_BUDGET", "10000000")
    monkeypatch.setenv("RERANK_REQUEST_BUDGET", "100")
    monkeypatch.setattr(api, "RERANK_MIN_INTERVAL", 0)
    monkeypatch.setattr(api, "RERANK_429_RETRIES", 0)
    return api


class Response:
    def __init__(self, data, status=200, headers=None):
        self.data = data
        self.status_code = status
        self.headers = headers or {}

    def json(self):
        return self.data


def test_wire_input_cap_plain_text_and_output_limit(api_client, monkeypatch):
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
    import requests

    captured = []

    def post(url, **kwargs):
        payload = kwargs["json"]
        captured.append(payload)
        return Response(
            {
                "results": [{"index": i, "relevance_score": float(i)} for i in range(10)],
                "meta": {"billed_units": {"input_tokens": 1234, "output_tokens": 0}},
            }
        )

    monkeypatch.setattr(requests, "post", post)
    client = api_client.ApiReranker(
        api_url="https://example.invalid/rerank", top_n=10, max_candidates=50, max_document_chars=8, max_query_chars=4
    )
    nodes = [
        NodeWithScore(node=TextNode(text=f"body-{i}-long", metadata={"file_path": "DO_NOT_SEND"}), score=1000)
        for i in range(200)
    ]
    result = client._postprocess_nodes(nodes, QueryBundle(query_str="query-long"))
    assert len(captured[0]["documents"]) == 50
    assert captured[0]["top_n"] == 10 and captured[0]["query"] == "quer"
    assert all(len(doc) <= 8 and "DO_NOT_SEND" not in doc for doc in captured[0]["documents"])
    assert len(result) == 10 and result[0].node.text == "body-9-long"
    assert RerankControl.from_env().snapshot()["reported_tokens"] == 1234


def test_partial_scores_do_not_promote_unreturned_nodes(api_client, monkeypatch):
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
    import requests

    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **k: Response(
            {
                "results": [{"index": 1, "relevance_score": -0.2}, {"index": 2, "relevance_score": -0.3}],
                "usage": {"total_tokens": 10},
            }
        ),
    )
    client = api_client.ApiReranker(api_url="https://example.invalid/rerank", top_n=2)
    nodes = [NodeWithScore(node=TextNode(text=str(i)), score=100) for i in range(3)]
    result = client._postprocess_nodes(nodes, QueryBundle(query_str="q"))
    assert [node.node.text for node in result] == ["1", "2"]


def test_payment_failure_opens_shared_circuit_without_fallback_or_retry(api_client, monkeypatch):
    import requests

    calls = []
    monkeypatch.setattr(requests, "post", lambda *a, **k: calls.append(k) or Response({}, 402))
    a = api_client.ApiReranker(api_url="https://example.invalid/rerank", top_n=1)
    b = api_client.ApiReranker(api_url="https://example.invalid/rerank", top_n=1)
    with pytest.raises(RerankStopped, match="402"):
        a._request_scores("q", ["doc"])
    with pytest.raises(RerankStopped, match="402"):
        b._request_scores("q", ["doc"])
    assert len(calls) == 1
    assert RerankControl.from_env().snapshot()["circuit_reason"] == "http_402"


def test_unlimited_total_does_not_disable_provider_payment_stop(api_client, monkeypatch):
    import requests

    monkeypatch.setenv("RERANK_TOKEN_BUDGET", "0")
    monkeypatch.setenv("RERANK_REQUEST_BUDGET", "0")
    monkeypatch.setattr(requests, "post", lambda *a, **k: Response({}, 402))
    client = api_client.ApiReranker(api_url="https://example.invalid/rerank", top_n=1)
    with pytest.raises(RerankStopped, match="402"):
        client._request_scores("q", ["doc"])
    state = RerankControl.from_env().snapshot()
    assert state["token_limit"] == state["request_limit"] == 0
    assert state["circuit_reason"] == "http_402"


def test_budget_exhaustion_blocks_network_before_sending(api_client, monkeypatch):
    import requests

    monkeypatch.setenv("RERANK_TOKEN_BUDGET", "1")
    monkeypatch.setattr(requests, "post", lambda *a, **k: pytest.fail("must not send"))
    client = api_client.ApiReranker(api_url="https://example.invalid/rerank", top_n=1)
    with pytest.raises(RerankBudgetExceeded):
        client._request_scores("q", ["doc"])


@pytest.mark.parametrize(
    "results",
    [
        [{"index": 0, "relevance_score": float("nan")}],
        [{"index": 5, "relevance_score": 1}],
        [],
    ],
)
def test_bad_scores_fail_closed(api_client, monkeypatch, results):
    import requests

    monkeypatch.setattr(requests, "post", lambda *a, **k: Response({"results": results, "usage": {"total_tokens": 5}}))
    client = api_client.ApiReranker(api_url="https://example.invalid/rerank", top_n=1)
    with pytest.raises(RerankStopped, match="response invalid"):
        client._request_scores("q", ["doc"])
    assert RerankControl.from_env().snapshot()["reported_tokens"] == 5


def test_429_retry_is_bounded_and_every_attempt_is_recorded(api_client, monkeypatch):
    import requests

    monkeypatch.setattr(api_client, "RERANK_429_RETRIES", 1)
    responses = iter(
        [
            Response({}, 429, {"Retry-After": "0"}),
            Response(
                {
                    "results": [{"index": 0, "relevance_score": 0.9}],
                    "usage": {"total_tokens": 10},
                }
            ),
        ]
    )
    monkeypatch.setattr(requests, "post", lambda *a, **k: next(responses))
    client = api_client.ApiReranker(api_url="https://example.invalid/rerank", top_n=1)
    assert client._request_scores("q", ["doc"]) == [0.9]
    state = RerankControl.from_env().snapshot()
    assert state["attempts"] == 2 and state["reported_tokens"] == 10
    assert state["unknown_usage_requests"] == 1


def test_tpm_rejection_without_retry_after_waits_for_a_token_window(api_client, monkeypatch):
    import requests

    monkeypatch.setattr(api_client, "RERANK_429_RETRIES", 1)
    monkeypatch.setattr(api_client, "RERANK_429_BACKOFF_SECONDS", 60)
    elapsed = [0.0]
    monkeypatch.setattr(api_client.time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(api_client.time, "sleep", lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds))
    responses = iter(
        [
            Response({"message": "TPM limit reached"}, 429),
            Response({"results": [{"index": 0, "relevance_score": 0.9}], "usage": {"total_tokens": 10}}),
        ]
    )
    monkeypatch.setattr(requests, "post", lambda *a, **k: next(responses))
    client = api_client.ApiReranker(api_url="https://example.invalid/rerank", top_n=1)
    assert client._request_scores("q", ["doc"]) == [0.9]
    assert elapsed[0] == 60
    assert RerankControl.from_env().snapshot()["circuit_reason"] is None


def test_timeout_is_not_retried_and_keeps_unknown_billing(api_client, monkeypatch):
    import requests

    calls = []

    def post(*a, **k):
        calls.append(1)
        raise requests.Timeout("timeout")

    monkeypatch.setattr(requests, "post", post)
    client = api_client.ApiReranker(api_url="https://example.invalid/rerank", top_n=1)
    with pytest.raises(RerankStopped, match="billing is unknown"):
        client._request_scores("q", ["doc"])
    state = RerankControl.from_env().snapshot()
    assert len(calls) == 1 and state["unknown_usage_requests"] == 1
    assert state["accounted_tokens"] > 0 and state["reserved_tokens"] == 0
