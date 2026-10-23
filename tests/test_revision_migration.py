import json
from types import SimpleNamespace

import pytest
from llama_index.core.schema import TextNode

from pcb_rag.ingestion import run_ingestion
from pcb_rag.migrate import activate_migration, migration_report
from pcb_rag.revisions import RevisionStore, pipeline_fingerprints
from pcb_rag.retrieval import read_nodes


def publish(store, collection="new", text="铜厚不小于 35 μm。"):
    rows = {}

    def verify(ids):
        if not all(key in rows for key in ids):
            raise RuntimeError("missing vector")

    return run_ingestion(
        [SimpleNamespace(id_="doc", text=text, hash=text, metadata={"doc_node_id": "doc", "file_path": "A.txt"})],
        pipeline=pipeline_fingerprints({}),
        collection=collection,
        store=store,
        parse=lambda docs: [
            TextNode(
                id_="child",
                text=doc.text,
                metadata={
                    "source_path": "A.txt",
                    "source_type": "technical_doc",
                    "doc_node_id": "doc",
                    "section": "3.4",
                    "page_label": "1",
                },
            )
            for doc in docs
        ],
        insert=lambda batch: rows.update({node.id_: node for node in batch}),
        verify=verify,
    )


def test_validated_collection_switch_keeps_old_snapshot_for_rollback(tmp_path):
    source, target = RevisionStore(tmp_path / "source.json"), RevisionStore(tmp_path / "target.json")
    old = publish(source, "old")
    new = publish(target)
    report = migration_report(new, old_nodes=read_nodes(old["artifacts"]["chunks"]))
    assert report["valid"], report["errors"]
    assert report["mapping_counts"] == {"mapped": 1}
    assert source.read()["active"]["collection"] == "old"
    activate_migration(source, target, report, expected_revision=old["revision"])
    assert source.read()["active"]["collection"] == "new"
    assert source.read()["previous"]["revision"] == old["revision"]


def test_unmapped_truth_ids_block_activation(tmp_path):
    source, target = RevisionStore(tmp_path / "source.json"), RevisionStore(tmp_path / "target.json")
    active = publish(target)
    report = migration_report(active, required_ids=["unmapped-old-answer-id"])
    assert not report["valid"]
    with pytest.raises(ValueError, match="validation"):
        activate_migration(source, target, report, expected_revision=None)
    assert not source.read()["active"]


def test_artifact_change_after_validation_blocks_activation(tmp_path):
    from pathlib import Path

    source, target = RevisionStore(tmp_path / "source.json"), RevisionStore(tmp_path / "target.json")
    active = publish(target)
    report = migration_report(active)
    with Path(active["artifacts"]["chunks"]).open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="artifacts changed"):
        activate_migration(source, target, report, expected_revision=None)
    assert not source.read()["active"]


def test_publishing_refuses_a_concurrently_changed_live_revision(tmp_path):
    source, target = RevisionStore(tmp_path / "source.json"), RevisionStore(tmp_path / "target.json")
    publish(source, "old")
    new = publish(target)
    with pytest.raises(RuntimeError, match="live revision changed"):
        activate_migration(source, target, migration_report(new), expected_revision="stale")
    assert source.read()["active"]["collection"] == "old"


def test_retry_does_not_overwrite_pending_when_source_config_changed(tmp_path, monkeypatch):
    store = RevisionStore(tmp_path / "state.json")
    store.begin({}, {}, "pending")
    before = store.read()
    monkeypatch.setenv("INGEST_EXPECTED_REVISION", before["pending"]["revision"])
    with pytest.raises(ValueError, match="source/configuration changed"):
        publish(store)
    assert store.read() == before


def test_snapshot_keeps_embedding_input_free_of_identity_fields(tmp_path):
    active = publish(RevisionStore(tmp_path / "state.json"))
    node = read_nodes(active["artifacts"]["chunks"])[0]
    assert node.get_content(metadata_mode="embed") == "铜厚不小于 35 μm。"
    assert node.metadata["revision"]
    assert json.dumps(node.metadata)


def test_legacy_lexical_export_is_accepted_for_exact_id_mapping(tmp_path):
    active = publish(RevisionStore(tmp_path / "state.json"))
    legacy = tmp_path / "lexical.jsonl"
    legacy.write_text(
        json.dumps(
            {
                "id": "old-id",
                "doc_id": "legacy-doc",
                "text": "铜厚不小于 35 μm。",
                "metadata": {"source_path": "A.txt", "source_type": "unknown"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    report = migration_report(active, old_nodes=read_nodes(legacy), required_ids=["old-id"])
    assert report["valid"] and report["mapping_counts"] == {"mapped": 1}
