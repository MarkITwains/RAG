"""Protect corpus identity and metric scope during embedding-only migrations."""

import json

from llama_index.core.schema import MetadataMode, TextNode
from llama_index.core.vector_stores.utils import node_to_metadata_dict

from scripts.repair_embeddings import corpus_digest, embedding_text, summarize_ranks


def test_corpus_digest_ignores_vectors_but_checks_all_other_fields():
    rows = [{"id": "a", "text": "unchanged", "embedding": [1.0], "prev_id": "b"}]
    expected = corpus_digest(rows, "embedding")
    assert corpus_digest([{**rows[0], "embedding": [2.0]}], "embedding") == expected
    for key in ["id", "text", "prev_id"]:
        assert corpus_digest([{**rows[0], key: "changed"}], "embedding") != expected


def test_migration_restores_the_exact_ingestion_text_and_metadata():
    node = TextNode(
        text="chunk body", id_="stable-id", metadata={"source_path": "source.txt", "parent_title": "section"}
    )
    metadata = node_to_metadata_dict(node, remove_text=True)
    assert json.loads(metadata["_node_content"])["text"] == ""
    row = {"id": node.node_id, "text": node.text, **metadata}
    assert embedding_text(row) == node.get_content(metadata_mode=MetadataMode.EMBED)


def test_metrics_use_the_reported_cutoff_and_count_misses():
    result = summarize_ranks([1, 2, 20, 21, 0])
    assert result["n"] == 5
    assert result["mrr@20"] == 0.31
    assert result["mrr@10"] == 0.3
    assert result["hit@1"] == 0.2
    assert result["hit@20"] == 0.6
    assert result["hit@200"] == 0.8
