import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from pcb_rag import security
from pcb_rag.sessions import SessionManager


@pytest.fixture
def api_client(offline_pipeline, monkeypatch, tmp_path):
    from pcb_rag import dify_external_api as api

    monkeypatch.setattr(api, "API_TOKEN", "isolated-service-token")
    monkeypatch.setattr(api, "_app_state", offline_pipeline.pipeline.state)
    monkeypatch.setattr(api, "get_pipeline", lambda: offline_pipeline.pipeline)
    monkeypatch.setattr(api, "embed_query_for_cache", lambda query: None)
    monkeypatch.setattr(security, "ACL_TRUST_HEADERS", False)
    monkeypatch.setattr(security, "ACL_TRUST_HEADERS_ACK", False)
    token_file = tmp_path / "identities.json"
    token_file.write_text(
        json.dumps(
            {
                hashlib.sha256(token.encode()).hexdigest(): identity
                for token, identity in {
                    "alice": {"tenant_id": "A", "user_id": "alice"},
                    "bob": {"tenant_id": "A", "user_id": "bob"},
                    "alice-b": {"tenant_id": "B", "user_id": "alice"},
                    "admin": {"tenant_id": "A", "user_id": "admin", "roles": ["admin"]},
                }.items()
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AUTH_TOKEN_PRINCIPALS_FILE", str(token_file))
    manager = SessionManager(store="sqlite", db_path=tmp_path / "sessions.db")
    monkeypatch.setattr(api, "_session_manager", manager)
    with TestClient(api.app) as client:
        client.headers["authorization"] = "Bearer alice"
        yield client, api, offline_pipeline
    manager.close()


def _events(body):
    result = []
    for event in body.split("\n\n"):
        if not event.strip():
            continue
        kind, value = event.split("\n", 1)
        result.append((kind.removeprefix("event: "), json.loads(value.removeprefix("data: "))))
    return result


def test_api_sdk_and_sse_use_the_prompt_evidence(api_client):
    from pcb_rag.engine import KnowledgeBase

    client, api, fixture = api_client
    normal = client.post("/api/ask", json={"query": "铜厚", "top_k": 5})
    assert normal.status_code == 200, normal.text
    data = normal.json()
    sdk = KnowledgeBase(auto_init=False, pipeline=fixture.pipeline).answer("铜厚", top_k=5)
    assert (
        [row["metadata"]["chunk_id"] for row in data["records"]]
        == [row.chunk_id for row in sdk.evidence]
        == ["real", "low"]
    )
    assert data["citations"][0]["chunk_id"] == "real"
    assert data["budget"]["request_tokens"] <= data["budget"]["request_limit"]
    assert data["usage"][0]["actual_tokens"] == 20
    streamed = client.post("/api/ask/stream", json={"query": "铜厚"})
    assert streamed.status_code == 200
    events = dict(_events(streamed.text))
    assert events["sources"]["citations"][0]["chunk_id"] == "real"
    assert events["citations"]["citations"][0]["chunk_id"] == "real"
    assert events["done"]["budget"] == data["budget"]
    assert fixture.llm._prompts[0] == fixture.llm._prompts[-1]
    assert normal.headers["x-kb-revision"] == "legacy"


@pytest.mark.parametrize("stranger", ["bob", "alice-b"])
def test_session_endpoints_enforce_owner(api_client, stranger):
    client, api, _ = api_client
    first = client.post("/api/chat", json={"query": "铜厚"})
    assert first.status_code == 200, first.text
    sid = first.json()["session_id"]
    private_headers = {
        "authorization": "Bearer " + stranger,
        "X-User-Id": "alice",
        "X-Tenant-Id": "A",
        "X-User-Roles": "admin",
    }
    assert client.get("/api/sessions", headers=private_headers).json()["total_count"] == 0
    for method, path, body in [
        ("get", "/api/session/" + sid, None),
        ("delete", "/api/session/" + sid, None),
        ("post", "/api/session/" + sid + "/clear", None),
        ("post", "/api/chat", {"session_id": sid, "query": "它的条件？"}),
    ]:
        response = client.request(method, path, headers=private_headers, **({"json": body} if body else {}))
        assert response.status_code == 404, response.text
    assert client.get("/api/session/" + sid).json()["message_count"] == 2


def test_management_requires_admin_and_exposes_stage_failure(api_client):
    client, api, fixture = api_client
    assert client.get("/manage").status_code == 200
    assert client.get("/api/admin/status").status_code == 403
    store = fixture.pipeline.revisions
    store.begin({"doc": {"path": "A.txt", "revision": "r"}}, {}, "test")
    with pytest.raises(ValueError):
        store.stage("graph", lambda: (_ for _ in ()).throw(ValueError("extraction failed")))
    headers = {"authorization": "Bearer admin"}
    response = client.get("/api/admin/status", headers=headers)
    assert response.status_code == 200, response.text
    assert response.json()["stages"]["graph"]["status"] == "failed"
    assert response.json()["retry_available"]
    docs = client.get("/api/admin/documents", headers=headers).json()
    assert docs["documents"][0]["source_path"] == "A.txt"
    assert docs["documents"][0]["stages"]["graph"] == "failed"
    assert client.post("/api/admin/retry").status_code == 403


def test_deadline_is_not_translated_to_http_500(api_client, monkeypatch):
    from pcb_rag.runtime import DeadlineExceeded

    client, api, _ = api_client

    def expired(*args, **kwargs):
        raise DeadlineExceeded("test timeout")

    monkeypatch.setattr(api, "_retrieve_nodes", expired)
    response = client.post("/api/ask", json={"query": "铜厚"})
    assert response.status_code == 504, response.text


def test_empty_retrieval_does_not_call_generation(api_client):
    client, _, fixture = api_client
    fixture.index.rows = []
    response = client.post("/api/ask/stream", json={"query": "不存在的参数"})
    assert response.status_code == 200
    events = dict(_events(response.text))
    assert events["sources"]["citations"] == []
    assert "无法回答" in events["token"]["text"]
    assert fixture.llm._prompts == []


def test_api_self_rag_retrieves_the_planned_query(api_client, monkeypatch):
    from pcb_rag import query as q

    client, api, fixture = api_client
    monkeypatch.setattr(api, "SELF_RAG_ENABLED", True)
    monkeypatch.setattr(q, "SELF_RAG_ENABLED", True)
    monkeypatch.setattr(q, "SELF_RAG_MAX_ITERATIONS", 3)
    monkeypatch.setattr(
        q,
        "_agentic_plan",
        lambda query, docs, llm, used: {
            "sufficient": len(used) >= 2,
            "next_query": "铜厚测量条件",
            "reason": "补充条件",
        },
    )
    monkeypatch.setattr(q, "self_rag_evaluate", lambda *args: {"overall": 4})
    response = client.post("/api/ask", json={"query": "铜厚", "use_self_rag": True})
    assert response.status_code == 200, response.text
    assert fixture.index.calls == ["铜厚", "铜厚测量条件"]
