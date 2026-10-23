"""chunk 信息密度评估（纯逻辑，不依赖任何第三方库）。

为什么要做
----------
切块器的目标是"按结构切开"，它不判断切开后的内容有没有检索价值。PCB 规范
文档里存在大量这样的片段：

    <socket>语句包含以下内容：
    <default>
    <port>
    <portgroup>
    ...

这类块 60% 以上是裸标签，语义空洞。它们被向量化后接近噪声：既拉低了同主题
块的相对相似度，又会在 BM25 里贡献大量无区分度的高频词（``<default>`` 在
几百个块里都出现，IDF 接近 0，但 term 数量稀释了 avgdl）。

本模块只做**判定**，不改变切块结果。是否丢弃、降权或合并，由调用方按
``CHUNK_QUALITY_MODE`` 决定 —— 默认只标记（``flag``），因为误删的代价
（丢失条款）远高于误留（多一条低价值候选）。
"""

from __future__ import annotations

import re
from typing import Dict, List

# 整行只有一个标签，例如 "<default>"、"<port_assignement>"
_BARE_TAG_LINE_RE = re.compile(r"^<[a-zA-Z_][\w:.\-]*/?>$")

# 有效字符：中文、拉丁字母、数字。标点、括号、空白不计入
_EFFECTIVE_CHAR_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")

# 表格分隔行（|---|:--:|---|）单独判定：它是结构而非内容
_TABLE_RULE_LINE_RE = re.compile(r"^\s*\|?[\s:\-|]{6,}\|?\s*$")


def assess_chunk_quality(text: str) -> Dict[str, object]:
    """评估单个 chunk 的信息密度。

    Returns:
        {
            "score": 0.0~1.0（越高越有检索价值）,
            "is_low_value": bool,
            "flags": [命中的问题标签],
            "metrics": {各项原始指标，便于阈值调参},
        }
    """
    if not text or not text.strip():
        return {
            "score": 0.0,
            "is_low_value": True,
            "flags": ["empty"],
            "metrics": {"chars": 0, "lines": 0},
        }

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    n_lines = max(len(lines), 1)
    n_chars = len(text)

    bare_tags = sum(1 for line in lines if _BARE_TAG_LINE_RE.match(line))
    table_rules = sum(1 for line in lines if _TABLE_RULE_LINE_RE.match(line))
    effective = len(_EFFECTIVE_CHAR_RE.findall(text))

    density = effective / max(n_chars, 1)
    unique_ratio = len(set(text)) / max(n_chars, 1)
    bare_ratio = bare_tags / n_lines

    flags: List[str] = []
    score = 1.0

    # 裸标签罗列：标签行占比高且数量足够
    if bare_tags >= 3 and bare_ratio >= 0.4:
        flags.append("bare_tag_list")
        score -= 0.55

    # 有效字符密度过低：多为符号、分隔线、空白
    if density < 0.35:
        flags.append("low_density")
        score -= 0.30

    # 过短：不足一句完整表述
    if n_chars < 40:
        flags.append("too_short")
        score -= 0.30

    # 高度重复：同一字符反复出现（表格填充、OCR 残留）
    if unique_ratio < 0.08:
        flags.append("repetitive")
        score -= 0.40

    # 纯表格分隔线为主
    if table_rules >= 2 and table_rules / n_lines >= 0.5:
        flags.append("table_rule_only")
        score -= 0.35

    score = max(0.0, min(1.0, score))

    return {
        "score": round(score, 3),
        "is_low_value": score < 0.5,
        "flags": flags,
        "metrics": {
            "chars": n_chars,
            "lines": n_lines,
            "bare_tag_lines": bare_tags,
            "bare_ratio": round(bare_ratio, 3),
            "density": round(density, 3),
            "unique_ratio": round(unique_ratio, 3),
        },
    }


def annotate_metadata(metadata: Dict[str, object], text: str) -> Dict[str, object]:
    """把评估结果写入 metadata（原地更新并返回）。

    只在低价值时写标记，正常块不增加字段 —— 避免给每个 chunk 都塞三个键，
    它们会一并进入 Milvus 的 metadata 与 BM25 的词法语料。
    """
    result = assess_chunk_quality(text)
    if result["is_low_value"]:
        metadata["chunk_quality"] = result["score"]
        metadata["low_value"] = True
        metadata["quality_flags"] = ",".join(result["flags"])
    return metadata
