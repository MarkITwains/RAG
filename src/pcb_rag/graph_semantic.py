"""Entity/relation description vectors that resolve back to original evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Protocol

from pcb_rag.graph_rag import GRAPH_HOP, link_entities
from pcb_rag.revisions import atomic_json, fingerprint


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


def authorized_graph(graph, fetch, allowed):
    """Rebuild aggregate descriptions using only authorized source contributions."""
    import copy
    from pcb_rag.graph_rag import KnowledgeGraph

    ids = list(dict.fromkeys(source["chunk_id"] for edge in graph.edges.values() for source in _source_records(edge)))
    nodes = {node.id_: node for node in allowed(fetch(ids))} if ids else {}
    result = KnowledgeGraph()
    result.revision = graph.revision
    for key, edge in graph.edges.items():
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
    entities, relations = [], []
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
    records = entities + relations
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
        for name, batch in (("entities", entities), ("relations", relations)):
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
        "version": 1,
        "revision": graph.revision,
        "backend": backend,
        "entities": entities,
        "relations": relations,
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
    def __init__(self, graph, embed_model=None, *, entities=None, relations=None):
        self.graph = graph
        self.embed_model = embed_model
        self.entities = entities
        self.relations = relations

    @property
    def semantic_ready(self):
        return self.embed_model is not None and any(
            getattr(retriever, "records", []) for retriever in (self.entities, self.relations)
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
            entities=EntityRetriever(payload["entities"], client=client, collection=names.get("entities", "")),
            relations=RelationRetriever(payload["relations"], client=client, collection=names.get("relations", "")),
        )

    def retrieve_sources(self, query, fetch, *, top_k=5):
        from llama_index.core.schema import NodeWithScore

        hits = []
        for name, score in link_entities(query, self.graph):
            for edge in self.graph.related_edges([name], hops=GRAPH_HOP):
                hits.extend(dict(source, score=score, route="entity_name") for source in _source_records(edge))
        if self.semantic_ready:
            vector = self.embed_model.get_query_embedding(query)
            for route, retriever in (("entity_description", self.entities), ("relation_description", self.relations)):
                if retriever is not None:
                    for match in retriever.search(vector, top_k):
                        if match["score"] <= 0:
                            continue
                        hits.extend(dict(source, score=match["score"], route=route) for source in match["sources"])
        hits.sort(key=lambda source: (-source["score"], source["chunk_id"]))
        by_id = {}
        for hit in hits:
            by_id.setdefault(hit["chunk_id"], hit)
        # Fetch more than the display top_k, so ACL filtering can preserve recall.
        ids = list(by_id)[: max(top_k * 4, 20)]
        results = []
        for node in fetch(ids):
            hit = by_id.get(node.id_)
            if not hit:
                continue
            if hit["revision"] and str(node.metadata.get("revision", "")) != hit["revision"]:
                continue
            if hit["doc_id"] and str(node.metadata.get("doc_node_id", "")) != hit["doc_id"]:
                continue
            node.metadata = dict(node.metadata, graph_route=hit["route"], graph_qualifiers=hit["qualifiers"])
            results.append(NodeWithScore(node=node, score=hit["score"]))
        return sorted(results, key=lambda item: (-float(item.score), item.node.id_))[:top_k]
