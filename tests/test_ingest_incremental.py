"""增量入库回归测试。

覆盖两个曾经存在的硬伤：

1. ``_delete_chunks_by_doc_node_id`` 对同步 ``MilvusClient.delete()`` 的返回值做 ``await``，
   异常被兜底吞掉 → 删除从未真正执行过。
2. 主流程「先 insert 再按 doc_node_id 删除」→ 变更文档刚写入的新 chunk 被一起删掉。
   现在必须是「先删旧、再插新」。

注意：本文件里的删除 / 顺序测试需要 ``pcb_rag.ingest``（依赖 llama-index），
因此**逐测试**用 importorskip 跳过，而不是在模块顶层整文件跳过 ——
模块级的 importorskip 会让整个文件在 CI 里静默消失，"260 passed" 其实可能是
"200 passed, 60 skipped"，而 skipped 不会让流水线变红。

不依赖重型组件的指纹 / 比对 / 清单逻辑已抽到 ``pcb_rag.incremental``，
其测试见 ``tests/test_incremental.py``（在任何环境都会真跑）。
"""

import hashlib
from types import SimpleNamespace

import pytest


class _FakeSyncClient:
    """模拟 pymilvus.MilvusClient：delete 是同步方法，返回 {"delete_count": n}。"""

    def __init__(self, log):
        self.log = log

    def delete(self, collection_name, filter):
        self.log.append(("delete", filter))
        return {"delete_count": 3}


class _FakeStore:
    def __init__(self, log):
        self.client = _FakeSyncClient(log)


@pytest.fixture
def ingest():
    return pytest.importorskip("pcb_rag.ingest")


class TestDeleteHelper:
    def test_uses_sync_client_and_returns_count(self, ingest):
        log = []
        n = ingest._delete_chunks_by_doc_node_id(_FakeStore(log), "abc123")
        assert n == 3
        assert log == [("delete", 'doc_node_id == "abc123"')]

    def test_empty_id_is_noop(self, ingest):
        log = []
        assert ingest._delete_chunks_by_doc_node_id(_FakeStore(log), "") == 0
        assert log == []

    def test_missing_client_is_noop(self, ingest):
        assert ingest._delete_chunks_by_doc_node_id(SimpleNamespace(), "x") == 0


class TestEnsureDocNodeId:
    def test_stamps_missing_doc_node_id_from_ref_doc_id(self, ingest):
        doc = SimpleNamespace(id_="d1", metadata={"file_path": "a.txt"})
        expected = hashlib.md5(b"doc-d1", usedforsecurity=False).hexdigest()
        n_missing = SimpleNamespace(metadata={}, ref_doc_id="d1")
        n_has = SimpleNamespace(metadata={"doc_node_id": "keep-me"}, ref_doc_id="d1")
        n_unknown = SimpleNamespace(metadata={}, ref_doc_id="nope")
        ingest._ensure_doc_node_id([n_missing, n_has, n_unknown], [doc])
        assert n_missing.metadata["doc_node_id"] == expected
        assert n_has.metadata["doc_node_id"] == "keep-me"
        assert "doc_node_id" not in n_unknown.metadata

    def test_consistent_with_doc_node_id_of(self, ingest):
        doc = SimpleNamespace(id_="d1", metadata={"file_path": "a.txt"})
        n = SimpleNamespace(metadata={}, ref_doc_id="d1")
        ingest._ensure_doc_node_id([n], [doc])
        assert n.metadata["doc_node_id"] == ingest._doc_node_id_of(doc)


class TestIncrementalOrder:
    """从源码层面锁定「删除在 insert 之前」——避免以后重构时又调回去。"""

    def test_delete_happens_before_insert(self, ingest):
        import inspect

        src = inspect.getsource(ingest.main) if hasattr(ingest, "main") else inspect.getsource(ingest)
        i_del = src.find("_delete_chunks_by_doc_node_id(vector_store, nid)")
        i_ins = src.find("index.insert_nodes(nodes)")
        assert i_del != -1 and i_ins != -1
        assert i_del < i_ins, "增量入库必须先删旧 chunk 再插入新 chunk"


class TestIngestHelpersDelegateToIncremental:
    """ingest 的私有包装必须与纯模块保持同一口径（防止两处实现再次漂移）。"""

    def test_fingerprint_matches_pure_module(self, ingest, tmp_path):
        from pcb_rag import incremental

        p = tmp_path / "a.txt"
        p.write_text("content", encoding="utf-8")
        assert ingest._file_fingerprint(p) == incremental.file_fingerprint(p)

    def test_doc_node_id_matches_pure_module(self, ingest):
        from pcb_rag import incremental

        doc = SimpleNamespace(id_="d1", metadata={"file_path": "a.txt"})
        assert ingest._doc_node_id_of(doc) == incremental.doc_node_id_of(doc)
