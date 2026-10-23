"""Vector-space identity and collection validation, without model dependencies."""

from __future__ import annotations

import hashlib
import json
from urllib.parse import urlsplit

PROFILE_KEY = "pcb_rag_embedding"


def resolve_embedding_provider(setting: str, api_base: str) -> str:
    provider = setting.strip().lower()
    if provider not in {"auto", "openai", "nvidia"}:
        raise ValueError("EMBED_API_PROVIDER must be auto, openai, or nvidia")
    if provider != "auto":
        return provider
    host = (urlsplit(api_base).hostname or "").lower()
    return "nvidia" if host in {"integrate.api.nvidia.com", "ai.api.nvidia.com"} else "openai"


def embedding_profile(*, provider: str, api_base: str, model: str, dimension: int, truncate: str = "END",
                      document_text: str = "llama_index_metadata_embed_v1") -> dict:
    profile = {
        "version": 1,
        "provider": provider,
        "api_base": api_base.rstrip("/"),
        "model": model,
        "dimension": dimension,
        "document_text": document_text,
    }
    if provider == "nvidia":
        profile.update(query_input_type="query", document_input_type="passage", truncate=truncate)
    return profile


def profile_digest(profile: dict) -> str:
    return hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_embedding_profile(description: str, expected: dict, *, query_only: bool = False) -> None:
    try:
        stored = json.loads(description or "{}").get(PROFILE_KEY)
    except (ValueError, AttributeError):
        stored = None
    if stored is None:
        if expected["provider"] == "nvidia":
            raise RuntimeError(
                "The NVIDIA query/passage client requires a re-embedded collection with a vector-space profile. "
                "Use scripts/repair_embeddings.py to migrate; do not query an unverified legacy collection."
            )
        return
    if query_only and stored.get("document_text") in {"llama_index_metadata_embed_v1", "passage_text_only_v2"}:
        # Queries have no passage metadata. Both documented passage encodings
        # share the same query/vector-space contract; ingestion stays strict.
        expected = dict(expected, document_text=stored["document_text"])
    if stored != expected:
        raise RuntimeError("Embedding configuration differs from the collection's vector-space profile; re-embed first")


def ensure_ingest_embedding_profile(client, collection: str, expected: dict) -> None:
    """Register a new empty collection; reject mixing modes in populated ones."""
    info = client.describe_collection(collection)
    description = info.get("description", "")
    try:
        stored = json.loads(description or "{}").get(PROFILE_KEY)
    except (ValueError, AttributeError):
        stored = None
    if expected["provider"] == "nvidia" and stored is None:
        count = int(client.get_collection_stats(collection).get("row_count", 0))
        if count:
            validate_embedding_profile(description, expected)
        metadata = {PROFILE_KEY: expected}
        if description:
            metadata["original_description"] = description
        client.alter_collection_properties(
            collection_name=collection, properties={"collection.description": json.dumps(metadata)}
        )
        description = client.describe_collection(collection).get("description", "")
    validate_embedding_profile(description, expected)
