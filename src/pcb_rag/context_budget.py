"""上下文预算分配与三段式组装（纯逻辑，不依赖 llama-index / torch）。

为什么单独拆出来
----------------
上下文构建是"检索质量"到"答案质量"的最后一道关口：召回的前 k 条怎么塞进
模型的上下文窗口，直接决定模型能看到多少证据。原实现是
``固定 5 条 × 每条 1000 字符``，与语料实际形态无关 —— 语料 chunk 的 p90
已达 868 字符，单块几乎吃满单槽且被截断。

本模块把「预算怎么切」「每条放多少」「三段怎么拼」抽成纯函数，便于单测与
跨调用点复用（CLI / API / 评测脚本走同一套逻辑）。

设计参照 LightRAG 的 ``_build_context_str``：总量预算先行，再按段分配，
chunk 段拿剩余额度并逐条填充。
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

# 事实句识别：含「数值+单位」「标准号」「规范性措辞」的句子。
# PCB 规范类问答的落点几乎总是这三类，定义性长句对定位帮助有限。
_FACT_SENTENCE_RE = re.compile(
    r"[^。！？；\n]*(?:"
    r"\d+(?:\.\d+)?\s*(?:mm|um|μm|mil|oz|nm|N/mm|N/m|MPa|GPa|℃|°C|%|A|V|W|GHz|MHz|kHz)"
    r"|GB\s*/?\s*T?\s*\d|IPC[-\s]?[A-Z]*\d|SJ\s*/?\s*T?\s*\d|ISO\s*\d"
    r"|应(?:不|大|小)|不得|必须|宜"
    r")[^。！？；\n]*[。！？；]?"
)


def estimate_tokens(text: str, chars_per_token: float = 1.5) -> int:
    """粗估 token 数：中文按字/token 系数折算，其余按 4 字符/token。

    不引入 tiktoken —— 它需要按模型下载词表，而这里只用于预算分配。
    宁可高估（中文取 1.5 字/token，实际 qwen 系约 1.5~1.7）也不要低估导致超窗。
    """
    if not text:
        return 0
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    other = max(len(text) - cjk, 0)
    return int(cjk / max(chars_per_token, 0.5) + other / 4.0) + 1


def extract_fact_sentences(text: str, limit: int = 2, max_chars: int = 220) -> str:
    """从 chunk 中挑出最"像事实"的句子。

    优先保留含数值+单位、标准号、或规范性措辞（应/不得/必须）的句子。
    一条都没命中时退回首句，至少给出主题。
    """
    if not text:
        return ""

    picked: List[str] = []
    seen: set = set()
    for match in _FACT_SENTENCE_RE.finditer(text):
        sent = match.group(0).strip(" \t\r\n-·•")
        if len(sent) < 8 or sent in seen:
            continue
        seen.add(sent)
        picked.append(sent)
        if len(picked) >= limit:
            break

    if not picked:
        head = re.split(r"[。！？；\n]", text.strip(), maxsplit=1)[0].strip()
        picked = [head] if head else []

    return " / ".join(picked)[:max_chars]


def dedupe_by_key(
    items: Iterable[Any],
    key_fn: Callable[[Any], Any],
) -> List[Any]:
    """按 key 去重，保留首次出现（调用方保证输入已按分数降序）。

    多路召回会让同一 chunk 重复出现。不去重的话，预算紧张时等于用一半窗口
    重复同一段话。
    """
    seen: set = set()
    out: List[Any] = []
    for item in items:
        try:
            key = key_fn(item)
        except Exception:
            key = id(item)
        if key is None:
            key = id(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def plan_budget(
    total_tokens: int,
    *,
    facts_ratio: float = 0.22,
    min_chunk_tokens: int = 120,
    ref_ratio: float = 0.06,
    max_ref_tokens: int = 200,
    overhead_tokens: int = 60,
) -> Dict[str, int]:
    """把总预算切成「引用表 / 事实摘要 / 原文证据 / 固定开销」四份。

    原文证据拿剩余额度（而不是固定比例）—— 事实段与引用表的体量随条数
    线性增长但有上限，剩余部分全给证据才符合"证据越多越好"的取向。
    预算过小时放弃事实段，把窗口全部让给原文。
    """
    total = max(int(total_tokens), 0)
    if total <= 0:
        return {"total": 0, "ref": 0, "facts": 0, "chunks": 0, "overhead": 0}

    ref = min(int(total * max(ref_ratio, 0.0)), max(int(max_ref_tokens), 0))
    overhead = min(int(overhead_tokens), max(total // 10, 0))
    facts = int(total * min(max(facts_ratio, 0.0), 0.8))
    chunks = total - ref - facts - overhead

    if chunks < int(min_chunk_tokens) * 2:
        facts = 0
        chunks = total - ref - overhead

    return {
        "total": total,
        "ref": max(ref, 0),
        "facts": max(facts, 0),
        "chunks": max(chunks, 0),
        "overhead": max(overhead, 0),
    }


def fit_evidence(
    items: Sequence[Any],
    text_of: Callable[[Any], str],
    chunk_budget: int,
    *,
    min_chunk_tokens: int = 120,
    chars_per_token: float = 1.5,
) -> List[Dict[str, Any]]:
    """逐条填充原文证据，直到预算耗尽。

    最后一条放不下时按剩余额度比例截断（保留开头 —— 结论、条款通常在前），
    剩余额度低于 ``min_chunk_tokens`` 则直接停止，避免塞进半句话。
    """
    budget = max(int(chunk_budget), 0)
    if budget <= 0:
        return []

    fitted: List[Dict[str, Any]] = []
    used = 0
    for item in items:
        try:
            text = (text_of(item) or "").strip()
        except Exception:
            continue
        if not text:
            continue

        cost = estimate_tokens(text, chars_per_token)
        truncated = False

        if used + cost > budget:
            remain = budget - used
            if remain < int(min_chunk_tokens):
                break
            keep_chars = max(int(len(text) * (remain / max(cost, 1))), 80)
            text = text[:keep_chars].rstrip() + "…"
            cost = estimate_tokens(text, chars_per_token)
            truncated = True

        fitted.append({"item": item, "text": text, "tokens": cost, "truncated": truncated})
        used += cost
        if used >= budget:
            break

    return fitted


def render_sections(
    entries: Sequence[Dict[str, Any]],
    facts_budget: int,
    *,
    facts_ratio_guard: bool = True,
) -> str:
    """拼装三段式上下文。

    ``entries`` 每项需含：source_name / source_path / section / text，
    以及可选 metadata（用于取 doc_type、standards）与 page。

    编号在事实段、证据段、引用表三处保持一致 —— 模型引用 [n] 时不会错位。
    """
    if not entries:
        return ""

    # ── 事实摘要 ────────────────────────────────────────────────────────
    fact_blocks: List[str] = []
    fact_used = 0
    for idx, entry in enumerate(entries, 1):
        if facts_budget <= 0:
            break

        md = entry.get("metadata") or {}
        locator_parts: List[str] = []
        if entry.get("source_name"):
            locator_parts.append(str(entry["source_name"]))
        if entry.get("section"):
            locator_parts.append(str(entry["section"]))
        for key in ("doc_type", "source_type"):
            val = md.get(key)
            if isinstance(val, str) and val and val not in {"general", "unknown"}:
                locator_parts.append(val)
                break
        standards = md.get("standards")
        if isinstance(standards, (list, tuple)) and standards:
            locator_parts.append("/".join(str(s) for s in list(standards)[:3]))

        locator = " · ".join(p for p in locator_parts if p)
        fact = extract_fact_sentences(entry.get("text") or "")

        block = f"[{idx}] {locator}"
        if fact:
            block += f"\n    {fact}"

        cost = estimate_tokens(block)
        if fact_used + cost > facts_budget:
            if fact_used > 0:
                break
            block = f"[{idx}] {locator}"
            cost = estimate_tokens(block)
            if fact_used + cost > facts_budget:
                break

        fact_blocks.append(block)
        fact_used += cost

    # ── 原文证据 ────────────────────────────────────────────────────────
    evidence_blocks: List[str] = []
    for idx, entry in enumerate(entries, 1):
        label = f"[{idx}] {entry.get('source_name') or '未知来源'}"
        if entry.get("section"):
            label += f" · {entry['section']}"
        evidence_blocks.append(f"{label}\n{entry.get('text') or ''}")

    # ── 引用来源 ────────────────────────────────────────────────────────
    ref_lines: List[str] = []
    for idx, entry in enumerate(entries, 1):
        line = f"[{idx}] {entry.get('source_path') or '未知来源'}"
        if entry.get("page"):
            line += f"（第 {entry['page']} 页）"
        ref_lines.append(line)

    sections: List[str] = []
    if fact_blocks:
        sections.append("【事实摘要】\n" + "\n".join(fact_blocks))
    sections.append("【原文证据】\n" + "\n\n---\n\n".join(evidence_blocks))
    if ref_lines:
        sections.append("【引用来源】\n" + "\n".join(ref_lines))

    return "\n\n".join(sections)
