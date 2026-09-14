"""上下文压缩与去重：用更少的 token 装进更多有效信息。

为什么需要
----------
本项目的召回链路很长（向量 + BM25 + HyDE + Chunk 扩展 + 父块回填），
单个查询最终可能带 30~80 个 chunk 进上下文，而 PCB 标准文档里
「目录 / 修订历史 / 前言 / 引用文件」这类低信息密度内容占比很高，
直接拼接既浪费 token，也容易稀释生成模型对关键条款的注意力。

三种模式
--------
- ``extractive``（默认，零 LLM 成本）：按查询相关性给句子打分，保留高分句并按原文顺序重组
- ``llm``：把候选片段交给模型压成要点（保留标准号、数值、单位），压缩率最高但有一次调用延迟
- ``hybrid``：先抽取式压缩，若压缩后仍超出预算再走 LLM

配套能力：``dedupe_nodes`` 用字符 shingle 的 Jaccard 相似度去掉近似重复块
（父块回填与邻居扩展很容易带进重复内容）。

设计取舍
--------
- 不导入 llama-index，靠 duck typing 操作节点，因此可被单元测试轻量引用
- 打分刻意偏「保守」：宁可多留一句，也不要把关键数值句删掉
- 数字 / 标准号 / 单位句有额外加权——PCB 领域「0.8 mm」「GB/T 4677」「260 ℃」这类
  片段一旦被删，答案就废了

用法::

    from pcb_rag.compression import compress_nodes, dedupe_nodes

    nodes = dedupe_nodes(nodes)
    nodes = compress_nodes(nodes, query=query, mode="extractive")
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = [
    "COMPRESSION_ENABLED",
    "compress_nodes",
    "compress_text_extractive",
    "dedupe_nodes",
    "describe_compression",
    "similarity",
    "split_sentences",
]


# ---------------------------------------------------------------------------
# 1. 配置
# ---------------------------------------------------------------------------
def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


COMPRESSION_ENABLED = _env_bool("CONTEXT_COMPRESSION_ENABLED", False)
COMPRESSION_MODE = os.getenv("COMPRESSION_MODE", "extractive").strip().lower()
# 目标压缩率：保留句子的数量占比上限（0.5 = 最多保留一半）
COMPRESSION_TARGET_RATIO = float(os.getenv("COMPRESSION_TARGET_RATIO", "0.6"))
# 单个片段压缩后的最小 / 最大字符数
COMPRESSION_MIN_CHARS = int(os.getenv("COMPRESSION_MIN_CHARS", "160"))
COMPRESSION_MAX_CHARS = int(os.getenv("COMPRESSION_MAX_CHARS", "900"))
# 进入上下文的字符预算（超出后按相关性截断尾部）
COMPRESSION_BUDGET_CHARS = int(os.getenv("COMPRESSION_BUDGET_CHARS", "12000"))
# 句子过滤阈值
MIN_SENTENCE_LEN = int(os.getenv("COMPRESSION_MIN_SENTENCE_LEN", "6"))
MAX_SENTENCE_LEN = int(os.getenv("COMPRESSION_MAX_SENTENCE_LEN", "400"))
# 去重
DEDUPE_ENABLED = _env_bool("CONTEXT_DEDUPE_ENABLED", True)
DEDUPE_THRESHOLD = float(os.getenv("CONTEXT_DEDUPE_THRESHOLD", "0.85"))
DEDUPE_MIN_CHARS = int(os.getenv("CONTEXT_DEDUPE_MIN_CHARS", "80"))

_SENT_TAIL = "。！？!?；;"
# 数值 / 标准号 / 单位：PCB 文档中这些片段信息密度最高
_VALUE_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:mm|cm|um|μm|nm|mil|mil|oz|℃|°C|%|MPa|N/mm|kV|V|mA|A|MHz|GHz|kHz|s|min|h|g|kg|kg/m|µm)",
    re.IGNORECASE,
)
_STANDARD_RE = re.compile(r"(?:GB/T|GB|SJ/T|SJ|IPC|JIS|IEC|ISO|MIL|ASTM|UL)\s*[-–]?\s*\d+", re.IGNORECASE)
_CLAUSE_RE = re.compile(r"第\s*[\d一二三四五六七八九十]+\s*[章节条]|\b\d+(?:\.\d+){1,3}\b")
_ASCII_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-\.]{1,}")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_NOISE_PREFIXES = ("目录", "前言", "修订记录", "版本历史", "目 录")


# ---------------------------------------------------------------------------
# 2. 分句
# ---------------------------------------------------------------------------
def split_sentences(text: str, *, min_len: int = MIN_SENTENCE_LEN, max_len: int = MAX_SENTENCE_LEN) -> List[str]:
    """把文本切成句子（中文标点 + 换行 + 长句逗号回退）。

    返回的句子保持原有顺序，且已去除空串与过短片段。
    """

    if not text:
        return []

    sentences: List[str] = []
    for line in re.split(r"\n+", text):
        line = line.strip()
        if not line:
            continue

        # 按句末标点切分，保留标点
        pieces = re.findall(rf"[^{re.escape(_SENT_TAIL)}]+[{re.escape(_SENT_TAIL)}]?", line)
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            if len(piece) <= max_len:
                if len(piece) >= min_len:
                    sentences.append(piece)
                continue
            # 超长句：退化为按逗号 / 分号切分
            buffer = ""
            for sub in re.split(r"(?<=[，,、])", piece):
                if len(buffer) + len(sub) <= max_len:
                    buffer += sub
                else:
                    if len(buffer.strip()) >= min_len:
                        sentences.append(buffer.strip())
                    buffer = sub
            if len(buffer.strip()) >= min_len:
                sentences.append(buffer.strip())

    return sentences


# ---------------------------------------------------------------------------
# 3. 查询词与打分
# ---------------------------------------------------------------------------
def extract_terms(query: str, max_terms: int = 40) -> List[str]:
    """从查询中抽取匹配用的词元（英文单词 + 中文 2/3-gram）。

    刻意不用 jieba：压缩发生在每次查询的热路径上，
    引入分词器会把查询延迟从毫秒级推到几十毫秒，而 2/3-gram 对短查询已经够用。
    """

    if not query:
        return []

    terms: List[str] = []
    seen: Set[str] = set()

    def _add(token: str) -> None:
        token = token.strip().lower()
        if len(token) < 2 or token in seen:
            return
        seen.add(token)
        terms.append(token)

    for word in _ASCII_WORD_RE.findall(query):
        _add(word)
    for std in _STANDARD_RE.findall(query):
        _add(std)
    for num in re.findall(r"\d+(?:\.\d+)?", query):
        _add(num)

    cjk_runs = re.findall(r"[\u4e00-\u9fff]+", query)
    for run in cjk_runs:
        if len(run) <= 4:
            _add(run)
        for i in range(len(run) - 1):
            _add(run[i : i + 2])
        for i in range(len(run) - 2):
            _add(run[i : i + 3])

    # 长词优先（更具体），同时保证数量可控
    terms.sort(key=lambda t: (-len(t), t))
    return terms[:max_terms]


def score_sentence(sentence: str, terms: Sequence[str], position: int, total: int) -> float:
    """给单句打分：查询覆盖度为主，位置与数值密度为辅。"""

    if not sentence:
        return 0.0

    lowered = sentence.lower()
    hit_weight = 0.0
    hit_count = 0
    for term in terms:
        if term in lowered:
            hit_count += 1
            # 长词命中更有信息量
            hit_weight += 1.0 + 0.2 * (len(term) - 2)
    coverage = hit_weight / (len(terms) + 1e-6) if terms else 0.0
    if coverage > 1.0:
        coverage = 1.0

    score = 0.62 * coverage + 0.06 * (1.0 if hit_count else 0.0)

    # 位置加成：开头 25% 常为主题句 / 适用范围
    if total > 1:
        ratio = position / (total - 1)
        score += 0.14 * (1.0 - ratio) ** 2

    # 数值 / 标准号 / 条款号密度
    if _VALUE_RE.search(sentence):
        score += 0.12
    if _STANDARD_RE.search(sentence) or _CLAUSE_RE.search(sentence):
        score += 0.08

    # 噪声段落降权（目录 / 前言 / 修订历史）
    stripped = sentence.strip()
    if stripped.startswith(_NOISE_PREFIXES) or stripped.count("…") >= 3 or stripped.count(".") >= 12:
        score -= 0.25

    # 过长/过短的句子轻微降权
    length = len(sentence)
    if length < 12:
        score -= 0.08
    elif length > 220:
        score -= 0.05

    return score


# ---------------------------------------------------------------------------
# 4. 抽取式压缩
# ---------------------------------------------------------------------------
def compress_text_extractive(
    text: str,
    query: str = "",
    *,
    max_chars: int = COMPRESSION_MAX_CHARS,
    min_chars: int = COMPRESSION_MIN_CHARS,
    target_ratio: float = COMPRESSION_TARGET_RATIO,
    terms: Optional[Sequence[str]] = None,
) -> str:
    """按查询相关性抽取关键句，返回压缩后的文本（保持原文顺序）。"""

    if not text:
        return ""
    if len(text) <= min_chars:
        return text

    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return text[:max_chars]

    terms = list(terms) if terms is not None else extract_terms(query)
    scored = [(score_sentence(s, terms, i, len(sentences)), i, s) for i, s in enumerate(sentences)]

    # 目标预算：压缩率与 max_chars 双重约束
    budget = min(max_chars, max(min_chars, int(len(text) * max(0.15, min(1.0, target_ratio)))))

    chosen: List[Tuple[int, str]] = []
    used = 0
    for score, idx, sentence in sorted(scored, key=lambda x: (-x[0], x[1])):
        if used >= budget and chosen:
            break
        # 允许首句略微超预算，避免一个句子都留不下
        if chosen and used + len(sentence) > budget:
            continue
        chosen.append((idx, sentence))
        used += len(sentence)

    if not chosen:
        return text[:budget]

    # 恢复原文顺序，保证连贯性
    chosen.sort(key=lambda x: x[0])
    compressed = "".join(s for _, s in chosen).strip()

    # 若头部被裁掉，补一个省略标记，让生成模型知道上下文不完整
    if chosen[0][0] > 0 and not compressed.startswith(sentences[0][:10]):
        compressed = "…" + compressed
    if chosen[-1][0] < len(sentences) - 1:
        compressed = compressed + "…"

    return compressed


def llm_compress_text(text: str, query: str, llm: Any, *, max_chars: int = COMPRESSION_MAX_CHARS) -> str:
    """用 LLM 把片段压成与问题相关的要点（失败时回退抽取式压缩）。"""

    if llm is None or not text.strip():
        return compress_text_extractive(text, query, max_chars=max_chars)

    prompt = (
        "你是资料压缩助手。下面是一段从 PCB 标准/规范中检索到的原文。\n"
        f"问题：{query}\n"
        "请只保留与问题直接相关的内容，压缩为精炼要点：\n"
        "1) 保留标准号、条款号、具体数值与单位，不得改写或臆造数字；\n"
        "2) 删除目录、修订记录、前言、套话；\n"
        f"3) 输出不超过 {max(80, max_chars)} 字，不要添加解释或结论。\n\n"
        f"原文：\n{text}"
    )

    try:
        response = llm.complete(prompt)
        compressed = str(getattr(response, "text", response)).strip()
    except Exception:
        return compress_text_extractive(text, query, max_chars=max_chars)

    if not compressed:
        return compress_text_extractive(text, query, max_chars=max_chars)
    return compressed[: max_chars * 2]


# ---------------------------------------------------------------------------
# 5. 节点级 API
# ---------------------------------------------------------------------------
def _node_get_text(item: Any) -> str:
    node = getattr(item, "node", item)
    getter = getattr(node, "get_content", None)
    if callable(getter):
        try:
            return str(getter())
        except Exception:
            pass
    return str(getattr(node, "text", "") or "")


def _node_set_text(item: Any, text: str) -> None:
    node = getattr(item, "node", item)
    setter = getattr(node, "set_content", None)
    if callable(setter):
        try:
            setter(text)
            return
        except Exception:
            pass
    try:
        node.text = text
    except Exception:
        pass


def _node_metadata(item: Any) -> Dict[str, Any]:
    node = getattr(item, "node", item)
    metadata = getattr(node, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def compress_nodes(
    nodes: Optional[Sequence[Any]],
    query: str = "",
    *,
    mode: Optional[str] = None,
    max_chars: int = COMPRESSION_MAX_CHARS,
    target_ratio: float = COMPRESSION_TARGET_RATIO,
    llm: Any = None,
) -> List[Any]:
    """就地压缩节点文本。

    - ``mode="extractive"``：逐块抽取关键句（不改变节点数量与顺序）
    - ``mode="llm"``：逐块调用模型压缩（成本高，建议只对 top_n 使用）
    - 关闭压缩时原样返回
    """

    if not nodes:
        return []
    if not COMPRESSION_ENABLED:
        return list(nodes)

    resolved_mode = (mode or COMPRESSION_MODE or "extractive").strip().lower()
    if resolved_mode not in {"extractive", "llm", "hybrid"}:
        resolved_mode = "extractive"

    terms = extract_terms(query)
    result: List[Any] = []

    for item in nodes:
        text = _node_get_text(item)
        if not text:
            result.append(item)
            continue

        original_len = len(text)
        if resolved_mode in {"extractive", "hybrid"}:
            compressed = compress_text_extractive(
                text, query, max_chars=max_chars, target_ratio=target_ratio, terms=terms
            )
        else:
            compressed = ""

        if resolved_mode in {"llm", "hybrid"} and (not compressed or len(compressed) > max_chars):
            if llm is not None:
                compressed = llm_compress_text(text, query, llm, max_chars=max_chars)
        if not compressed:
            compressed = compress_text_extractive(text, query, max_chars=max_chars, terms=terms)

        if compressed and len(compressed) < original_len:
            _node_set_text(item, compressed)
            metadata = _node_metadata(item)
            if metadata is not None:
                metadata["compressed"] = True
                metadata["original_chars"] = original_len
                metadata["compressed_chars"] = len(compressed)

        result.append(item)

    return result


def apply_context_budget(
    nodes: Optional[Sequence[Any]],
    *,
    budget_chars: int = COMPRESSION_BUDGET_CHARS,
) -> List[Any]:
    """按字符预算截断上下文：保留靠前的节点（已按相关性排序），超出预算的丢弃。"""

    if not nodes:
        return []
    if budget_chars <= 0:
        return list(nodes)

    kept: List[Any] = []
    used = 0
    for item in nodes:
        length = len(_node_get_text(item))
        if kept and used + length > budget_chars:
            continue
        kept.append(item)
        used += length

    if len(kept) < len(nodes):
        try:
            from pcb_rag.observability import counter

            counter("compression.dropped_nodes", len(nodes) - len(kept))
        except Exception:
            pass

    return kept


# ---------------------------------------------------------------------------
# 6. 去重
# ---------------------------------------------------------------------------
_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\u4e00-\u9fff]+")


def _normalize_for_similarity(text: str) -> str:
    text = _PUNCT_RE.sub("", _WS_RE.sub("", text.lower()))
    return text


def _shingles(text: str, k: int = 4) -> Set[str]:
    normalized = _normalize_for_similarity(text)
    if len(normalized) <= k:
        return {normalized} if normalized else set()
    return {normalized[i : i + k] for i in range(len(normalized) - k + 1)}


def similarity(a: str, b: str, *, k: int = 4) -> float:
    """两段文本的字符 shingle Jaccard 相似度（0~1）。"""

    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    set_a, set_b = _shingles(a, k), _shingles(b, k)
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def dedupe_nodes(
    nodes: Optional[Sequence[Any]],
    *,
    threshold: float = DEDUPE_THRESHOLD,
    min_chars: int = DEDUPE_MIN_CHARS,
) -> List[Any]:
    """去掉内容近似重复的节点，保留先出现（通常分数更高）的那个。

    先按文本签名分桶（长度 + 首尾片段）再用 Jaccard 精确比对，
    避免 O(n²) 全量比较带来的延迟。
    """

    if not nodes:
        return []
    if not DEDUPE_ENABLED or len(nodes) < 2:
        return list(nodes)

    kept: List[Any] = []
    kept_texts: List[str] = []
    kept_shingles: List[Set[str]] = []
    buckets: Dict[str, List[int]] = {}

    for item in nodes:
        text = _node_get_text(item)
        if not text:
            continue
        if len(text) < min_chars:
            kept.append(item)
            continue

        signature = _bucket_key(text)
        candidate_indexes = buckets.get(signature, [])
        shingle_set: Optional[Set[str]] = None
        duplicate = False

        for idx in candidate_indexes:
            other = kept_texts[idx]
            # 长度差过大直接跳过，避免无效的集合运算
            if max(len(text), len(other)) > 1.6 * min(len(text), len(other)):
                continue
            if shingle_set is None:
                shingle_set = _shingles(text)
            other_set = kept_shingles[idx]
            if not shingle_set or not other_set:
                continue
            inter = len(shingle_set & other_set)
            union = len(shingle_set | other_set)
            if union and inter / union >= threshold:
                duplicate = True
                break

        if duplicate:
            continue

        if shingle_set is None:
            shingle_set = _shingles(text)
        kept.append(item)
        kept_texts.append(text)
        kept_shingles.append(shingle_set)
        buckets.setdefault(signature, []).append(len(kept) - 1)

    removed = len(nodes) - len(kept)
    if removed:
        try:
            from pcb_rag.observability import counter

            counter("compression.deduped_nodes", removed)
        except Exception:
            pass

    return kept


def _bucket_key(text: str, prefix: int = 24) -> str:
    """粗粒度分桶键：长度档位 + 去除标点后的前后缀片段。"""

    normalized = _normalize_for_similarity(text)
    length_bucket = len(normalized) // 64
    head = normalized[:prefix]
    tail = normalized[-prefix:] if len(normalized) > prefix else ""
    return f"{length_bucket}|{head}|{tail}"


# ---------------------------------------------------------------------------
# 7. 诊断
# ---------------------------------------------------------------------------
def describe_compression() -> Dict[str, Any]:
    """返回压缩配置摘要，供启动日志与 ``/health`` 展示。"""

    return {
        "enabled": COMPRESSION_ENABLED,
        "mode": COMPRESSION_MODE,
        "target_ratio": COMPRESSION_TARGET_RATIO,
        "max_chars": COMPRESSION_MAX_CHARS,
        "budget_chars": COMPRESSION_BUDGET_CHARS,
        "dedupe": DEDUPE_ENABLED,
        "dedupe_threshold": DEDUPE_THRESHOLD,
    }


def compress_pipeline(
    nodes: Optional[Sequence[Any]],
    query: str = "",
    *,
    llm: Any = None,
    mode: Optional[str] = None,
) -> List[Any]:
    """完整压缩链路：去重 → 抽取/LLM 压缩 → 字符预算裁剪。"""

    if not nodes:
        return []
    result = dedupe_nodes(nodes)
    result = compress_nodes(result, query, mode=mode, llm=llm)
    result = apply_context_budget(result)
    return result


def summarize_compression(nodes: Optional[Sequence[Any]]) -> Dict[str, Any]:
    """统计压缩效果（供评测与日志使用）。"""

    if not nodes:
        return {"nodes": 0, "chars": 0}
    original = 0
    compressed = 0
    for item in nodes:
        metadata = _node_metadata(item)
        text_len = len(_node_get_text(item))
        compressed += text_len
        original += int(metadata.get("original_chars") or text_len)
    ratio = (compressed / original) if original else 1.0
    return {"nodes": len(nodes), "chars": compressed, "original_chars": original, "ratio": round(ratio, 4)}


def iter_node_texts(nodes: Optional[Iterable[Any]]) -> List[str]:
    """调试辅助：取出节点文本列表。"""

    return [_node_get_text(n) for n in (nodes or [])]
