"""Artifact-affecting configuration must invalidate the correct pipeline stage."""

import pytest

from pcb_rag.revisions import pipeline_fingerprints


@pytest.mark.parametrize(
    "key,value",
    [
        ("PARENT_MIN_SECTION_KEEP", "120"),
        ("STRUCTURE_MAX_CHUNK_SIZE", "2000"),
        ("SEMANTIC_BUFFER_SIZE", "1"),
        ("SEMANTIC_BREAKPOINT_THRESHOLD", "90"),
        ("SEMANTIC_CHUNK_SIZE", "3000"),
        ("HIERARCHICAL_CHUNK_SIZES", "1024,384,128"),
        ("ATTENTION_MODEL", "test-embed"),
        ("ATTENTION_THRESHOLD", "0.45"),
        ("ATTENTION_OVERLAP", "220"),
        ("AUTO_FIX_ENCODING", "1"),
        ("STRUCTURED_FLATTEN", "1"),
        ("ANCHOR_FIGURE_TABLE", "1"),
        ("MULTIMODAL_VLM_MODEL", "test-vlm"),
        ("MULTIMODAL_MAX_TOKENS", "512"),
        ("MULTIMODAL_MAX_IMAGES_PER_DOC", "20"),
        ("MULTIMODAL_MIN_IMAGE_PX", "120"),
        ("MULTIMODAL_MIN_IMAGE_BYTES", "4096"),
        ("MULTIMODAL_DESC_MAX_CHARS", "500"),
        ("MULTIMODAL_IMAGE_PROMPT", "Describe image"),
        ("TABLE_MIN_ROWS", "2"),
        ("TABLE_MIN_COLS", "2"),
    ],
)
def test_chunk_artifact_settings_change_chunk_fingerprint(key, value):
    assert pipeline_fingerprints({})["chunks"] != pipeline_fingerprints({key: value})["chunks"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("GRAPH_LLM_DESC_CHARS", "120"),
        ("GRAPH_NODE_DESC_CHARS", "240"),
        ("GRAPH_EVIDENCE_CHARS", "240"),
        ("GRAPH_COMMUNITY_MIN_SIZE", "3"),
        ("GRAPH_COMMUNITY_MAX_SUMMARY_CHARS", "600"),
    ],
)
def test_graph_artifact_settings_change_graph_fingerprint(key, value):
    assert pipeline_fingerprints({})["graph"] != pipeline_fingerprints({key: value})["graph"]


@pytest.mark.parametrize("key", ["CONTEXTUAL_LLM_WORKERS", "GRAPH_LLM_WORKERS", "RAG_REQUEST_TIMEOUT"])
def test_execution_only_settings_do_not_force_reingestion(key):
    assert pipeline_fingerprints({}) == pipeline_fingerprints({key: "different"})


def test_chunk_embedding_identity_invalidates_semantic_chunks():
    from pcb_rag import ingest

    base = ingest.artifact_config({"EMBEDDING_PROFILE": {"model": "a"}})
    other = ingest.artifact_config({"EMBEDDING_PROFILE": {"model": "b"}})

    assert pipeline_fingerprints(base)["chunks"] != pipeline_fingerprints(other)["chunks"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("ACL_ENABLED", "1"),
        ("ACL_TENANT_FIELD", "tenant_id"),
        ("ACL_GROUPS_FIELD", "acl_groups"),
        ("ACL_VISIBILITY_FIELD", "visibility"),
        ("ACL_OWNER_FIELD", "owner_id"),
        ("ACL_DEFAULT_TENANT", "tenant"),
        ("ACL_ANONYMOUS_TENANT", "public"),
        ("ACL_DEFAULT_VISIBILITY", "private"),
        ("ACL_DEFAULT_OWNER", "owner"),
        ("ACL_PUBLIC_VALUE", "public"),
    ],
)
def test_acl_ingest_settings_change_chunk_fingerprint(key, value):
    assert pipeline_fingerprints({})["chunks"] != pipeline_fingerprints({key: value})["chunks"]


def test_acl_legacy_read_policy_does_not_force_reingestion():
    assert pipeline_fingerprints({}) == pipeline_fingerprints({"ACL_STRICT_LEGACY": "1"})


def test_graph_model_backend_and_local_model_change_graph_fingerprint():
    base = pipeline_fingerprints({"LLM_BACKEND": "local", "OLLAMA_LLM_MODEL": "old"})
    changed = pipeline_fingerprints({"LLM_BACKEND": "local", "OLLAMA_LLM_MODEL": "new"})

    assert base["graph"] != changed["graph"]
