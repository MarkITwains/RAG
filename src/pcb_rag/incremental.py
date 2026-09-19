"""增量入库的纯逻辑：文件指纹、文档比对、入库清单读写。

为什么单独拆一个模块
--------------------
这段逻辑原先长在 `ingest.py` 里，而 `ingest.py` 顶层 import torch / llama-index /
pymilvus —— 于是 tests/test_ingest_incremental.py 只能用
``pytest.importorskip("pcb_rag.ingest")`` 整文件跳过，CI 里"260 passed"实际上
可能是"200 passed, 60 skipped"，而 skipped 不会让流水线变红。
"先删旧再插新"这种顺序回归就是这么被漏掉的。

这里的函数只依赖标准库，可以在任何环境里直接单测。

设计要点
--------
- **指纹用内容哈希，不用 mtime**：``git clone`` / ``checkout`` / ``cp -r`` 都会改写
  mtime，会把"没改过"的文档判成变更并触发全量重嵌入；反之只改内容不改 mtime 会漏检。
- **路径不进指纹**：路径由 ``doc_node_id`` 单独承载，挪目录不应改变"内容是否变化"。
- manifest 带 ``version``：指纹算法换代时显式提示"本次全量重嵌入一次"，而不是
  让使用者看到一堆莫名其妙的变更。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

#: manifest 结构版本。v1 = path+size+mtime 指纹，v2 = 内容哈希。
MANIFEST_VERSION = 2


def content_hash(path: Path) -> str:
    """文件内容哈希（blake2b，16 字节）。读不到文件时返回空串。"""
    try:
        with path.open("rb") as f:
            return hashlib.blake2b(f.read(), digest_size=16).hexdigest()
    except OSError:
        return ""


def file_fingerprint(path: Path) -> str:
    """文件指纹 = size 快筛 + 内容哈希。

    先比 size（不同必然变更），再算内容哈希。
    """
    try:
        st = path.stat()
    except OSError:
        return ""
    digest = content_hash(path)
    return hashlib.md5(
        f"{st.st_size}|{digest}".encode("utf-8"), usedforsecurity=False
    ).hexdigest()


def doc_source_of(doc: Any) -> str:
    """从文档对象里取源文件路径（不同 reader 的字段名不一致）。"""
    md = getattr(doc, "metadata", None) or {}
    return str(md.get("file_path") or md.get("filename") or getattr(doc, "hash", "") or "")


def doc_node_id_of(doc: Any) -> str:
    """文档级节点 id（与 ingest 切块时写入 chunk metadata 的 ``doc_node_id`` 一致）。

    注意：它由**绝对路径**参与计算，因此换目录会导致 node_id 变化 —— 旧 id 会被
    当作"已删除文档"清掉、新 id 重新入库。功能上自愈，代价是一次全量重嵌入。
    """
    doc_id = getattr(doc, "id_", None) or hashlib.md5(
        doc_source_of(doc).encode("utf-8"), usedforsecurity=False
    ).hexdigest()
    return hashlib.md5(f"doc-{doc_id}".encode("utf-8"), usedforsecurity=False).hexdigest()


def doc_fingerprint_of(doc: Any) -> str:
    """文档指纹：优先按源文件内容哈希，缺少源文件时退回 ``doc.hash``。"""
    src = doc_source_of(doc)
    if src:
        fp = file_fingerprint(Path(src))
        if fp:
            return fp
    return str(getattr(doc, "hash", "") or "")


def diff_documents(documents: List[Any], manifest: Dict[str, Any]) -> Tuple[List[Any], List[str], Dict[str, dict]]:
    """比对文档与清单。

    Returns:
        (待处理文档列表, 需先清理旧 chunk 的 doc_node_id 列表, 新的清单条目)
    """
    old_docs = (manifest or {}).get("docs", {}) or {}
    to_process: List[Any] = []
    stale_ids: List[str] = []
    current: Dict[str, dict] = {}

    for d in documents:
        node_id = doc_node_id_of(d)
        src = doc_source_of(d)
        fingerprint = doc_fingerprint_of(d)
        current[node_id] = {"path": src, "hash": fingerprint}

        old = old_docs.get(node_id)
        if old is None:
            to_process.append(d)  # 新增
        elif old.get("hash") != fingerprint:
            stale_ids.append(node_id)  # 变更：需先删除旧 chunk
            to_process.append(d)

    # 清单中存在、但数据目录已移除的文档
    stale_ids.extend(nid for nid in old_docs if nid not in current)
    return to_process, stale_ids, current


def snapshot_of(documents: List[Any]) -> Dict[str, dict]:
    """生成"全量处理"场景下的清单条目。

    必须在全量 / 重建场景下也写入**真实指纹**：早先这里写 ``hash=""``，
    于是下一次增量比对会把「空指纹 ≠ 真实指纹」判成全部变更，导致每次都全量重嵌入。
    """
    return {
        doc_node_id_of(d): {"path": doc_source_of(d), "hash": doc_fingerprint_of(d)}
        for d in documents
    }


def load_manifest(path: str, *, default_version: int = MANIFEST_VERSION) -> Dict[str, Any]:
    """读取入库清单；不存在或损坏时返回空清单。"""
    p = Path(path)
    if not p.exists():
        return {"version": default_version, "docs": {}}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("docs"), dict):
            return data
    except Exception:
        pass
    return {"version": default_version, "docs": {}}


def needs_full_reprocess(manifest: Dict[str, Any]) -> bool:
    """清单是否为旧版本（指纹算法换代）→ 需要显式全量重嵌入一次。"""
    try:
        return int((manifest or {}).get("version", 1) or 1) < MANIFEST_VERSION
    except (TypeError, ValueError):
        return True


def save_manifest(path: str, docs_map: Dict[str, dict]) -> None:
    """写回入库清单（原子替换：先写临时文件再 rename，避免半截文件）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps({"version": MANIFEST_VERSION, "docs": docs_map}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(p)


def plan_ingest(
    documents: List[Any],
    manifest: Dict[str, Any],
    *,
    incremental: bool,
    overwrite: bool,
) -> Tuple[List[Any], List[str], Dict[str, dict], bool]:
    """决定"本次入库要处理哪些文档"。

    ``overwrite=True`` 时集合会被清空重建，此时 manifest 里的指纹全部失效 ——
    必须忽略清单、全量处理并重写清单。否则 ``diff_documents`` 会因为指纹完全一致
    返回空的 to_process：库被清空了、什么都没插进去，而清单还声称数据都在，
    之后每次增量入库都继续匹配、继续空转（只能手删 manifest 才能恢复）。

    Returns:
        (待处理文档, 待清理旧 chunk 的 doc_node_id, 新清单条目, 是否走了增量比对)
    """
    if overwrite or not incremental:
        return list(documents), [], snapshot_of(documents), False
    to_process, stale_ids, current = diff_documents(documents, manifest)
    return to_process, stale_ids, current, True
