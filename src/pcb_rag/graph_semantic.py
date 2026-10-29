"""Entity, relation and community vectors that resolve back to original evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Protocol

from pcb_rag.graph_rag import GRAPH_HOP, link_entities
from pcb_rag.revisions import atomic_json, fingerprint

#: 图路内部各子路（实体名 / 实体描述 / 关系描述 / 社区摘要）按名次合并用的 RRF k
GRAPH_RRF_K = 60


def embedding_identity(model):
    """Include retrieval mode and endpoint, never credentials."""
    return {
        key: getattr(model, key, None)
        for key in (
            "model_name",
            "api_base",
            "base_url",
            "dimensions",
            "expected_dimension",
            "truncate",
            "query_instruction",
            "text_instruction",
        )
    }


def authorized_graph(graph, fetch, allowed, *, entities=None):
    """Rebuild aggregate descriptions using only authorized source contributions."""
    import copy
    from pcb_rag.graph_rag import KnowledgeGraph

    selected = None if entities is None else {
        key for name in entities for key in graph.incident.get(name, set())
    }
    keys = list(graph.edges) if selected is None else [key for key in graph.edges if key in selected]
    ids = list(dict.fromkeys(
        source["chunk_id"] for key in keys for source in _source_records(graph.edges[key])
    ))
    nodes = {node.id_: node for node in allowed(fetch(ids))} if ids else {}
    result = KnowledgeGraph()
    result.revision = graph.revision
    for key in keys:
        edge = graph.edges[key]
        sources = []
        for source in edge.get("contributions", []):
            node = nodes.get(source.get("chunk_id"))
            if node is None or source.get("needs_backfill"):
                continue
            if source.get("doc_id") and node.metadata.get("doc_node_id") != source["doc_id"]:
                continue
            if source.get("revision", "") != node.metadata.get("revision", ""):
                continue
            sources.append(copy.deepcopy(source))
        if sources:
            result.edges[key] = dict(copy.deepcopy(edge), contributions=sources)
    result._rebuild_derived()
    return result


class DescriptionRetriever(Protocol):
    def search(self, vector: list[float], top_k: int) -> list[dict]: ...


class VectorDescriptionRetriever:
    def __init__(self, records, *, client=None, collection=""):
        self.records = records
        self.client = client
        self.collection = collection

    def search(self, vector, top_k):
        if self.client is not None and self.collection:
            matches = self.client.search(
                collection_name=self.collection,
                data=[vector],
                anns_field="vector",
                limit=top_k,
                output_fields=["key"],
                search_params={"metric_type": "COSINE"},
                timeout=30,
            )
            mapping = {record["key"]: record for record in self.records}
            return [
                dict(mapping[hit["entity"]["key"]], score=float(hit["distance"]))
                for hit in (matches[0] if matches else [])
                if hit["entity"]["key"] in mapping
            ]
        norm = math.sqrt(sum(value * value for value in vector))
        scored = []
        for record in self.records:
            other = record.get("vector", [])
            if len(vector) != len(other):
                raise ValueError("graph/query embedding dimension mismatch; rebuild graph vectors")
            denominator = norm * math.sqrt(sum(value * value for value in other))
            score = sum(a * b for a, b in zip(vector, other)) / denominator if denominator else 0
            scored.append(dict(record, score=score))
        return sorted(scored, key=lambda item: (-item["score"], item["key"]))[:top_k]


class EntityRetriever(VectorDescriptionRetriever):
    pass


class RelationRetriever(VectorDescriptionRetriever):
    pass


class CommunityRetriever(VectorDescriptionRetriever):
    pass


def _source_records(edge):
    return [
        {
            "chunk_id": source["chunk_id"],
            "doc_id": source.get("doc_id", ""),
            "revision": source.get("revision", ""),
            "qualifiers": source.get("qualifiers", {}),
        }
        for source in edge.get("contributions", [])
        if source.get("chunk_id") and not source.get("needs_backfill")
    ]


def build_graph_vectors(graph, embed_model, path, *, client=None, collection_prefix="pcb_graph", backend="local"):
    """Build immutable description indexes using the existing passage embedder."""
    entities, relations, communities = [], [], []
    for name, node in graph.nodes.items():
        sources = [source for edge in graph.edges_of(name) for source in _source_records(edge)]
        if not sources:
            continue
        entities.append(
            {
                "key": fingerprint(["entity", name])[:32],
                "name": name,
                "text": " ".join([name, node.type, node.desc, *node.aliases]),
                "sources": sources,
            }
        )
    for edge in graph.edges.values():
        for contribution in edge.get("contributions", []):
            sources = _source_records({"contributions": [contribution]})
            if not sources:
                continue
            relations.append(
                {
                    "key": fingerprint([edge["h"], edge["r"], edge["t"], contribution["id"]])[:32],
                    "text": " ".join(
                        [
                            edge["h"],
                            edge["r"],
                            edge["t"],
                            contribution.get("relation_desc", ""),
                            json.dumps(contribution.get("qualifiers", {}), ensure_ascii=False),
                        ]
                    ),
                    "sources": sources,
                }
            )
    for label, community in graph.communities.items():
        summary = str(community.get("summary") or "").strip()
        if not summary:
            continue
        members = set(community.get("entities", []))
        sources = list({
            source["chunk_id"]: source
            for edge in graph.edges.values()
            if edge["h"] in members and edge["t"] in members
            for source in _source_records(edge)
        }.values())
        if not sources:
            continue
        communities.append(
            {
                "key": fingerprint(["community", label])[:32],
                "text": " ".join([label, summary, "、".join(community.get("entities", [])[:30])]),
                "sources": sources,
            }
        )
    records = entities + relations + communities
    for start in range(0, len(records), 32):
        batch = records[start : start + 32]
        if hasattr(embed_model, "get_text_embedding_batch"):
            vectors = embed_model.get_text_embedding_batch([record["text"] for record in batch])
        else:
            vectors = [embed_model.get_text_embedding(record["text"]) for record in batch]
        if len(vectors) != len(batch):
            raise ValueError("graph embedding response count mismatch")
        for record, vector in zip(batch, vectors):
            record["vector"] = [float(value) for value in vector]
    collections = {}
    if backend == "milvus" and records:
        if client is None:
            raise ValueError("Milvus graph backend requires a client")
        for name, batch in (("entities", entities), ("relations", relations), ("communities", communities)):
            if not batch:
                continue
            collection = f"{collection_prefix[:180]}_{name}_{fingerprint(batch)[:12]}"
            if not client.has_collection(collection_name=collection):
                client.create_collection(
                    collection_name=collection,
                    dimension=len(batch[0]["vector"]),
                    primary_field_name="id",
                    id_type="string",
                    max_length=64,
                    vector_field_name="vector",
                    metric_type="COSINE",
                    enable_dynamic_field=True,
                    auto_id=False,
                    consistency_level="Strong",
                )
            for start in range(0, len(batch), 128):
                client.upsert(
                    collection_name=collection,
                    data=[
                        dict(id=record["key"], key=record["key"], vector=record["vector"])
                        for record in batch[start : start + 128]
                    ],
                )
            collections[name] = collection
    payload = {
        "version": 2,
        "revision": graph.revision,
        "backend": backend,
        "entities": entities,
        "relations": relations,
        "communities": communities,
        "collections": collections,
        "embedding_model": str(getattr(embed_model, "model_name", "")),
        "embedding_identity": embedding_identity(embed_model),
        "usage": {
            "kind": "estimated",
            "passages": len(records),
            "input_utf8_bytes": sum(len(record["text"].encode()) for record in records),
        },
    }
    atomic_json(Path(path), payload)
    return payload


class GraphEvidenceRetriever:
    def __init__(self, graph, embed_model=None, *, entities=None, relations=None, communities=None):
        self.graph = graph
        self.embed_model = embed_model
        self.entities = entities
        self.relations = relations
        self.communities = communities

    @property
    def semantic_ready(self):
        return self.embed_model is not None and any(
            getattr(retriever, "records", []) for retriever in (self.entities, self.relations, self.communities)
        )

    @classmethod
    def load(cls, graph, path, embed_model, *, client=None):
        if not Path(path).is_file():
            return cls(graph)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if str(payload.get("revision", "")) != graph.revision:
            return cls(graph)
        model = str(getattr(embed_model, "model_name", ""))
        if payload.get("embedding_model") and payload["embedding_model"] != model:
            return cls(graph)
        if payload.get("embedding_identity") != embedding_identity(embed_model):
            return cls(graph)
        names = payload.get("collections", {})
        return cls(
            graph,
            embed_model,
            entities=EntityRetriever(payload.get("entities", []), client=client, collection=names.get("entities", "")),
            relations=RelationRetriever(payload.get("relations", []), client=client, collection=names.get("relations", "")),
            communities=CommunityRetriever(
                payload.get("communities", []), client=client, collection=names.get("communities", "")
            ),
        )

    def retrieve_sources(self, query, fetch, *, top_k=5, allowed=None):
        from llama_index.core.schema import NodeWithScore

        if top_k <= 0:
            return []
        # 每条子路各自排好名次，再按 RRF 合并。实体名链接分（0.34~1.0）和描述向量的
        # 余弦分标尺不同，早先混在一起按分数排序：链接分最高的那个实体的所有边
        # 拿同一个分、按 chunk_id 字母序占满 top_k，语义三路几乎进不来。
        runs = []
        entity_hits = []
        for name, score in link_entities(query, self.graph):
            for edge in self.graph.related_edges([name], hops=GRAPH_HOP):
                weight = float(edge.get("w", 0.0) or 0.0)
                entity_hits.extend(
                    (score, weight, dict(source, route="entity_name")) for source in _source_records(edge)
                )
        # 同一实体内按边权重排序，不再让所有边并列同分
        entity_hits.sort(key=lambda hit: (-hit[0], -hit[1], hit[2]["chunk_id"]))
        runs.append([hit for _score, _weight, hit in entity_hits])
        if self.semantic_ready:
            vector = self.embed_model.get_query_embedding(query)
            routes = (
                ("entity_description", self.entities),
                ("relation_description", self.relations),
                ("community_summary", self.communities),
            )
            for route, retriever in routes:
                if retriever is not None:
                    run = []
                    for match in retriever.search(vector, top_k):
                        if match["score"] <= 0:
                            continue
                        run.extend(dict(source, route=route) for source in match["sources"])
                    runs.append(run)
        by_id = {}
        fused = {}
        for run in runs:
            rank = 0
            seen = set()
            for hit in run:
                chunk_id = hit["chunk_id"]
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                rank += 1
                contribution = 1.0 / (GRAPH_RRF_K + rank)
                fused[chunk_id] = fused.get(chunk_id, 0.0) + contribution
                # 元数据取该块排名最靠前的那条子路
                if chunk_id not in by_id or contribution > by_id[chunk_id][0]:
                    by_id[chunk_id] = (contribution, hit)
        by_id = {chunk_id: dict(hit, score=fused[chunk_id]) for chunk_id, (_c, hit) in by_id.items()}
        ids = sorted(by_id, key=lambda chunk_id: (-by_id[chunk_id]["score"], chunk_id))
        batch_size = max(top_k * 4, 20)
        results = []
        for start in range(0, len(ids), batch_size):
            batch = ids[start : start + batch_size]
            requested = set(batch)
            candidates = []
            for node in fetch(batch):
                hit = by_id.get(node.id_)
                if not hit or node.id_ not in requested:
                    continue
                if hit["revision"] and str(node.metadata.get("revision", "")) != hit["revision"]:
                    continue
                if hit["doc_id"] and str(node.metadata.get("doc_node_id", "")) != hit["doc_id"]:
                    continue
                candidates.append(node)
            if allowed is not None:
                authorized = {node.id_ for node in allowed(candidates)}
                candidates = [node for node in candidates if node.id_ in authorized]
            for node in candidates:
                hit = by_id[node.id_]
                node.metadata = dict(node.metadata, graph_route=hit["route"], graph_qualifiers=hit["qualifiers"])
                results.append(NodeWithScore(node=node, score=hit["score"]))
            if len(results) >= top_k:
                break
        return sorted(results, key=lambda item: (-float(item.score), item.node.id_))[:top_k]
