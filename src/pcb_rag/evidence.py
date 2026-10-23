"""Final evidence selection, citation identity and serialized request budgets.

This module has no model, database or HTTP imports. Every consumer receives the
same ordered entries and the exact text the model saw, including truncation.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Callable, Sequence

from pcb_rag.context_budget import dedupe_by_key, estimate_tokens, render_sections


@dataclass(frozen=True)
class EvidenceBundle:
    context: str = ""
    entries: tuple[dict[str, Any], ...] = ()
    budget: dict[str, Any] = field(default_factory=dict)

    @property
    def sources(self) -> list[dict[str, Any]]:
        return [{k: v for k, v in row.items() if k not in {"node", "item", "metadata"}} for row in self.entries]

    @property
    def nodes(self) -> list[Any]:
        """Copies carrying only selected text; never mutate shared index nodes."""
        result = []
        for entry in self.entries:
            item = copy.deepcopy(entry["item"])
            node = getattr(item, "node", item)
            if hasattr(node, "text"):
                node.text = entry["text"]
            node.metadata = dict(entry["metadata"], citation_id=entry["index"])
            result.append(item)
        return result

    def citations(self, answer: str) -> list[dict[str, Any]]:
        ids = {int(value) for value in re.findall(r"\[(\d+)\]", answer)}
        return [
            dict(
                entry,
                source=entry["source_path"],
                matched_text=f"[{entry['index']}]",
                citation_valid=True,
                support_status="unverified",
                # Kept for older clients; a valid number does not prove support.
                confidence="unverified",
            )
            for entry in self.sources
            if entry["index"] in ids
        ]

    def invalid_citations(self, answer: str) -> list[int]:
        ids = {int(value) for value in re.findall(r"\[(\d+)\]", answer)}
        return sorted(ids - {entry["index"] for entry in self.entries})


def _entry(item: Any) -> dict[str, Any]:
    node = getattr(item, "node", item)
    metadata = dict(getattr(node, "metadata", None) or {})
    text = getattr(node, "text", None)
    if text is None:
        try:
            text = node.get_content(metadata_mode="none")
        except (AttributeError, TypeError):
            text = node.get_content() if hasattr(node, "get_content") else str(node)
    source = str(metadata.get("source_path") or metadata.get("file_path") or "未知来源")
    return {
        "item": item,
        "node": node,
        "metadata": metadata,
        "text": str(text or "").strip(),
        "source_path": source,
        "source_name": PurePosixPath(source.replace("\\", "/")).stem,
        "section": str(metadata.get("parent_title") or metadata.get("chunk_section") or metadata.get("section") or ""),
        "page": str(metadata.get("page_label") or metadata.get("page") or ""),
        "table_context_page": str(metadata.get("table_context_page") or ""),
        "chunk_id": str(getattr(node, "id_", "") or metadata.get("chunk_id") or ""),
        "doc_id": str(metadata.get("doc_node_id") or ""),
        "revision": str(metadata.get("revision") or metadata.get("doc_revision") or ""),
        "score": float(getattr(item, "score", 0) or 0),
        "truncated": False,
    }


def build_evidence_bundle(
    items: Sequence[Any],
    budget: int,
    *,
    facts_ratio: float = 0.22,
    demote_low_value: bool = True,
    count: Callable[[str], int] = estimate_tokens,
    counter_name: str = "utf8_bytes_upper_bound",
) -> EvidenceBundle:
    """Select against the complete serialization, including labels and sources."""
    budget = max(0, int(budget))
    rows = [_entry(item) for item in items]
    rows = dedupe_by_key(rows, lambda row: row["chunk_id"] or id(row["item"]))
    rows = [row for row in rows if row["text"]]
    if demote_low_value:
        rows.sort(key=lambda row: bool(row["metadata"].get("low_value")))
    facts_budget = max(0, int(budget * min(0.8, max(0.0, facts_ratio))))
    selected: list[dict[str, Any]] = []
    for row in rows:
        candidate = selected + [row]
        if count(render_sections(candidate, facts_budget)) <= budget:
            selected.append(row)
            continue
        # A useful original passage takes priority over a duplicate fact summary.
        if not selected and count(render_sections(candidate, 0)) <= budget:
            facts_budget = 0
            selected.append(row)
            continue
        original = row["text"]
        lo, hi = 0, len(original)
        # Removing summaries can rescue a first large passage with long locators.
        if not selected:
            facts_budget = 0
        while lo < hi:
            mid = (lo + hi + 1) // 2
            row["text"] = original[:mid].rstrip() + "…"
            if count(render_sections(candidate, facts_budget)) <= budget:
                lo = mid
            else:
                hi = mid - 1
        row["text"] = original[:lo].rstrip() + "…"
        if lo and count(render_sections(candidate, facts_budget)) <= budget:
            row["truncated"] = True
            selected.append(row)
            break
        # An oversized source name must not prevent a later short entry fitting.
    context = render_sections(selected, facts_budget)
    for index, row in enumerate(selected, 1):
        row.update(
            index=index,
            citation_id=index,
            tokens=count(row["text"]),
            char_count=len(row["text"]),
            snippet=row["text"][:200],
        )
    used = count(context)
    if used > budget:
        raise ValueError("serialized evidence exceeds its token budget")
    return EvidenceBundle(
        context,
        tuple(selected),
        {
            "context_limit": budget,
            "context_tokens": used,
            "counter": counter_name,
            "usage_kind": "estimated" if count is estimate_tokens else "tokenized",
        },
    )


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    evidence: EvidenceBundle


def build_generation_request(
    items: Sequence[Any],
    template: str,
    *,
    query: str,
    context_budget: int,
    request_limit: int,
    output_reserve: int,
    values: dict[str, Any] | None = None,
    count: Callable[[str], int] = estimate_tokens,
    **bundle_options: Any,
) -> GenerationRequest:
    """Budget the exact template, query, history and output reserve together.

    Inputs are never silently shortened. Oversized question/history is an
    explicit error, so a caller can deliberately trim history or reject input.
    """
    values = dict(values or {}, query=query)
    fixed = template.format(**dict(values, context=""))
    available = int(request_limit) - max(0, int(output_reserve)) - count(fixed)
    if available < 0:
        raise ValueError("question/history and output reserve exceed the request window")
    allowed = min(max(0, context_budget), available)
    bundle = build_evidence_bundle(items, allowed, count=count, **bundle_options)
    prompt = template.format(**dict(values, context=bundle.context))
    # Some tokenizers merge across context/template boundaries differently.
    while count(prompt) + output_reserve > request_limit and bundle.entries:
        allowed = max(0, allowed - max(1, count(prompt) + output_reserve - request_limit))
        bundle = build_evidence_bundle(items, allowed, count=count, **bundle_options)
        prompt = template.format(**dict(values, context=bundle.context))
    if count(prompt) + output_reserve > request_limit:
        raise ValueError("serialized request exceeds the model window")
    stats = dict(
        bundle.budget,
        request_limit=request_limit,
        prompt_tokens=count(prompt),
        output_reserve=output_reserve,
        request_tokens=count(prompt) + output_reserve,
    )
    return GenerationRequest(prompt, EvidenceBundle(bundle.context, bundle.entries, stats))
