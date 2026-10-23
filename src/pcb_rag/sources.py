"""Authorized source lookup and parent/neighbor expansion with full metadata."""

from __future__ import annotations

import copy
import json


def fetch_source_nodes(store, ids):
    from llama_index.core.schema import TextNode

    if not ids:
        return []
    collection = getattr(store, "collection_name", None)
    if not collection:
        raise ValueError("source store does not identify its collection")
    rows = store.client.get(collection_name=collection, ids=list(dict.fromkeys(ids)), output_fields=["*"], timeout=30)
    result = []
    for row in rows:
        metadata = dict(row)
        serialized = row.get("_node_content")
        if serialized:
            parsed = json.loads(serialized)
            metadata = dict(parsed.get("metadata") or {}, **metadata)
        text = row.get("text") or (parsed.get("text") if serialized else "")
        if not text:
            continue
        for key in ("text", "embedding", "_node_content", "_node_type"):
            metadata.pop(key, None)
        result.append(TextNode(id_=str(row["id"]), text=text, metadata=metadata))
    return result


def expand_sources(nodes, fetch, *, neighbors=1, parents=True, max_extra=5, allowed=lambda nodes: nodes):
    from llama_index.core.schema import NodeWithScore

    result = list(nodes)
    seen = {item.node.id_ for item in result}
    frontier = list(nodes)
    for depth in range(max(1 if parents else 0, neighbors)):
        ids = []
        for item in frontier:
            metadata = item.node.metadata
            if depth < neighbors:
                ids.extend([metadata.get("prev_id"), metadata.get("next_id")])
            if parents and depth == 0:
                ids.insert(0, metadata.get("parent_id"))
        ids = list(dict.fromkeys(value for value in ids if value and value not in seen))
        if not ids:
            break
        fetched = {node.id_: node for node in fetch(ids)}
        additions = []
        score = min((float(item.score or 0) for item in nodes), default=0) * 0.95
        for key in ids:
            if key not in fetched:
                continue
            node = copy.deepcopy(fetched[key])
            node.metadata["is_context_expansion"] = True
            additions.append(NodeWithScore(node=node, score=score))
        additions = allowed(additions)
        for item in additions:
            if len(result) - len(nodes) >= max_extra:
                return result
            seen.add(item.node.id_)
            result.append(item)
        frontier = additions
    return result
