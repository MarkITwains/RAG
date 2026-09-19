"""增量入库纯逻辑测试（不依赖 llama-index / pymilvus / torch）。

对应两个真实缺陷的回归：

1. **指纹用 mtime**：``git clone`` / ``checkout`` / ``cp -r`` 都会改写 mtime，
   导致"没改过"的文档被判为变更 → 全量重嵌入。现改为内容哈希。
2. **INGEST_OVERWRITE + manifest 冲突**：重建集合后未忽略清单，指纹全部命中，
   ``to_process`` 为空 → 库被清空、什么都没插进去，而清单还声称数据都在，
   后续每次增量入库都继续空转，只能手删 manifest。这是最接近生产事故的一个。
"""

from types import SimpleNamespace

import pytest

from pcb_rag import incremental


def _doc(path: str, doc_id: str = "d1"):
    return SimpleNamespace(id_=doc_id, hash="", metadata={"file_path": path})


@pytest.fixture
def doc_file(tmp_path):
    p = tmp_path / "spec.txt"
    p.write_text("镀金层厚度应不小于 0.8 μm。", encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# 指纹
# ---------------------------------------------------------------------------
class TestFingerprint:
    def test_content_change_changes_fingerprint(self, tmp_path, doc_file):
        before = incremental.doc_fingerprint_of(_doc(str(doc_file)))
        doc_file.write_text("镀金层厚度应不小于 1.0 μm。", encoding="utf-8")
        after = incremental.doc_fingerprint_of(_doc(str(doc_file)))
        assert before and after and before != after

    def test_rewriting_mtime_does_not_change_fingerprint(self, doc_file):
        """核心回归：git clone / cp -r 改写 mtime 不应造成"变更"。"""
        import os

        before = incremental.doc_fingerprint_of(_doc(str(doc_file)))
        st = doc_file.stat()
        os.utime(doc_file, (st.st_atime, st.st_mtime + 10_000))
        after = incremental.doc_fingerprint_of(_doc(str(doc_file)))
        assert before == after

    def test_same_content_different_path_same_fingerprint(self, tmp_path):
        """路径不进指纹：内容相同即视为未变更（跨目录复制不再触发重嵌入）。"""
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("same", encoding="utf-8")
        b.write_text("same", encoding="utf-8")
        assert incremental.file_fingerprint(a) == incremental.file_fingerprint(b)

    def test_missing_file_is_empty_fingerprint(self, tmp_path):
        assert incremental.file_fingerprint(tmp_path / "nope.txt") == ""

    def test_missing_source_falls_back_to_doc_hash(self):
        doc = SimpleNamespace(id_="d", hash="abc123", metadata={})
        assert incremental.doc_fingerprint_of(doc) == "abc123"


# ---------------------------------------------------------------------------
# 文档比对
# ---------------------------------------------------------------------------
class TestDiffDocuments:
    def test_new_document_is_processed(self, doc_file):
        to_process, stale, current = incremental.diff_documents(
            [_doc(str(doc_file))], {"version": 2, "docs": {}}
        )
        assert len(to_process) == 1
        assert stale == []
        assert len(current) == 1

    def test_unchanged_document_is_skipped(self, doc_file):
        manifest = {"version": 2, "docs": incremental.snapshot_of([_doc(str(doc_file))])}
        to_process, stale, _ = incremental.diff_documents([_doc(str(doc_file))], manifest)
        assert to_process == []
        assert stale == []

    def test_changed_document_is_marked_stale(self, doc_file):
        manifest = {"version": 2, "docs": incremental.snapshot_of([_doc(str(doc_file))])}
        doc_file.write_text("改了内容", encoding="utf-8")
        to_process, stale, _ = incremental.diff_documents([_doc(str(doc_file))], manifest)
        assert len(to_process) == 1
        assert len(stale) == 1

    def test_removed_document_is_marked_stale(self, doc_file):
        manifest = {"version": 2, "docs": incremental.snapshot_of([_doc(str(doc_file))])}
        to_process, stale, current = incremental.diff_documents([], manifest)
        assert to_process == []
        assert len(stale) == 1
        assert current == {}


# ---------------------------------------------------------------------------
# 入库计划（OVERWRITE 回归）
# ---------------------------------------------------------------------------
class TestPlanIngest:
    def test_overwrite_forces_full_reprocess(self, doc_file):
        """核心回归：OVERWRITE 必须忽略并重写清单，否则重建后是空库。"""
        docs = [_doc(str(doc_file))]
        manifest = {"version": 2, "docs": incremental.snapshot_of(docs)}
        to_process, stale, current, did_diff = incremental.plan_ingest(
            docs, manifest, incremental=True, overwrite=True
        )
        assert len(to_process) == 1, "重建集合后必须全量处理，否则库会是空的"
        assert stale == []
        assert did_diff is False
        # 清单必须写入真实指纹，否则下一次增量又会全量重嵌入
        assert all(entry["hash"] for entry in current.values())

    def test_non_incremental_forces_full_reprocess(self, doc_file):
        docs = [_doc(str(doc_file))]
        to_process, _, current, did_diff = incremental.plan_ingest(
            docs, {"version": 2, "docs": {}}, incremental=False, overwrite=False
        )
        assert len(to_process) == 1
        assert did_diff is False
        assert all(entry["hash"] for entry in current.values())

    def test_incremental_uses_diff(self, doc_file):
        docs = [_doc(str(doc_file))]
        manifest = {"version": 2, "docs": incremental.snapshot_of(docs)}
        to_process, stale, _, did_diff = incremental.plan_ingest(
            docs, manifest, incremental=True, overwrite=False
        )
        assert to_process == []
        assert did_diff is True

    def test_full_reprocess_fingerprint_is_real(self, doc_file):
        """全量场景写入的指纹必须与增量比对口径一致。

        早先这里写 hash=""，于是下一次增量比对把「空指纹 ≠ 真实指纹」判成
        全部变更，导致每次入库都全量重嵌入。
        """
        docs = [_doc(str(doc_file))]
        _, _, current, _ = incremental.plan_ingest(
            docs, {"version": 2, "docs": {}}, incremental=True, overwrite=True
        )
        to_process, stale, _, _ = incremental.plan_ingest(
            docs, {"version": 2, "docs": current}, incremental=True, overwrite=False
        )
        assert to_process == [], "全量写入的指纹应与随后的增量比对自洽"
        assert stale == []


# ---------------------------------------------------------------------------
# 清单读写与版本
# ---------------------------------------------------------------------------
class TestManifest:
    def test_roundtrip(self, tmp_path, doc_file):
        path = str(tmp_path / "m.json")
        snapshot = incremental.snapshot_of([_doc(str(doc_file))])
        incremental.save_manifest(path, snapshot)
        loaded = incremental.load_manifest(path)
        assert loaded["version"] == incremental.MANIFEST_VERSION
        assert loaded["docs"] == snapshot

    def test_missing_manifest(self, tmp_path):
        loaded = incremental.load_manifest(str(tmp_path / "nope.json"))
        assert loaded == {"version": incremental.MANIFEST_VERSION, "docs": {}}

    def test_corrupt_manifest_is_treated_as_empty(self, tmp_path):
        p = tmp_path / "m.json"
        p.write_text("{ not json", encoding="utf-8")
        assert incremental.load_manifest(str(p))["docs"] == {}

    def test_old_version_needs_full_reprocess(self):
        assert incremental.needs_full_reprocess({"version": 1, "docs": {}}) is True
        assert incremental.needs_full_reprocess({"version": 2, "docs": {}}) is False

    def test_atomic_write_leaves_no_tmp_file(self, tmp_path, doc_file):
        path = tmp_path / "m.json"
        incremental.save_manifest(str(path), incremental.snapshot_of([_doc(str(doc_file))]))
        assert path.exists()
        assert not (tmp_path / "m.json.tmp").exists()
