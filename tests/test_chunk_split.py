"""``_split_by_length`` 切块质量回归测试。

对应 2026-09 切块质量体检发现的两个实现缺陷：

1. **overlap 被句号截没**：旧实现取末 ``overlap`` 字后又 ``rfind('。')`` 向前截断，
   实际重叠近 0（全库实测同父块相邻命中率 1.1%，配置承诺 80 字）。
2. **超长"句"整块塞入**：无句末标点的超长段（XML 片段 / 连排图注，实测单"句"
   1901 字）无法按句切，整段进一个 chunk，远超 child 上限 800。

测试目标：
- 相邻块 overlap 真实存在（≥ overlap 且从句边界开始）；
- 所有块 ≤ max_size（超长句在句内标点处硬切兜底）；
- 内容无损：每块都是原文的连续子串（去空白后）；
- ``_hard_split`` / ``_overlap_tail`` 的单元行为。

不依赖 Milvus / torch / embedding：只测纯字符串逻辑。
"""

import re

from pcb_rag.ingest import HierarchicalStructureSplitter

MAX_SIZE = 200
MIN_SIZE = 50
OVERLAP = 80


def _make_sentences(n: int, span: tuple = (30, 90)) -> str:
    """生成 n 个长度在 span 内、以。结尾的句子。"""
    import random

    rng = random.Random(42)
    words = "印制板 阻抗 层压 敷铜 过孔 阻焊 丝印 基板 蚀刻 电镀".split()
    out = []
    for i in range(n):
        k = rng.randint(*span)
        body = "".join(rng.choice(words) for _ in range(max(1, k // 3)))
        out.append(f"第{i}段{body}。")
    return "".join(out)


def _splitter() -> HierarchicalStructureSplitter:
    return HierarchicalStructureSplitter(
        child_max_size=MAX_SIZE,
        child_min_size=MIN_SIZE,
        child_overlap=OVERLAP,
        semantic_split_threshold=10**9,  # 测试里不触发语义路径
    )


def _norm(t: str) -> str:
    return re.sub(r"\s+", "", t or "")


class TestOverlapRealized:
    def test_adjacent_chunks_share_tail(self):
        """相邻块必须以前一块的末 overlap 字开头（配置承诺 80 字）。"""
        text = _make_sentences(40)
        chunks = _splitter()._split_by_length(text, MAX_SIZE, MIN_SIZE, OVERLAP)
        assert len(chunks) >= 3, f"样本应产生多块，实际 {len(chunks)}"

        hit = 0
        for prev, nxt in zip(chunks, chunks[1:]):
            tail = _norm(prev)[-OVERLAP:]
            # 重叠要么精确出现在下一块开头附近，要么因放不下 overlap 而被放弃
            if tail and tail in _norm(nxt)[:200]:
                hit += 1
        assert hit >= len(chunks) - 2, (
            f"相邻块 overlap 命中 {hit}/{len(chunks)-1}："
            "旧实现的 bug 是 overlap 被句号截没（实测 1.1%），修复后应绝大多数命中"
        )

    def test_overlap_tail_length_and_sentence_aligned(self):
        chunk = _make_sentences(8, span=(60, 90))
        carry = HierarchicalStructureSplitter._overlap_tail(chunk, OVERLAP)
        assert len(carry) >= OVERLAP, f"实际重叠 {len(carry)} < 配置 {OVERLAP}"
        assert len(carry) < 2 * OVERLAP, "向前补齐不应携带整个长句"
        assert chunk.endswith(carry), "carry 必须是块尾的连续后缀"

        # 对齐语义：carry 的前一个字符不是句号（= 从句中开始）当且仅当
        # 窗口前 OVERLAP 字内找不到句首（文档化兜底）。
        # 对齐成功时前一个字符本来就是句号，这是合法情形。
        start = len(chunk) - len(carry)
        if start > 0 and chunk[start - 1] not in "。.":
            assert "。" not in chunk[max(0, start - OVERLAP):start], (
                "窗口内明明有句首却从句中开始，说明对齐逻辑失效"
            )


class TestHardCap:
    def test_no_chunk_exceeds_max_size(self):
        text = _make_sentences(60)
        chunks = _splitter()._split_by_length(text, MAX_SIZE, MIN_SIZE, OVERLAP)
        over = [c for c in chunks if len(c) > MAX_SIZE]
        assert not over, f"{len(over)} 块超过 max_size={MAX_SIZE}，最长 {max(map(len, over))}"

    def test_oversized_punctuated_run_is_hard_split(self):
        """复现实测 max=1901 的场景：单个无句末标点的超长段。"""
        blob = "阻抗控制参数" * 240  # 1440 字，无 。！？
        assert len(blob) == 1440
        text = _make_sentences(5) + blob + _make_sentences(5)
        chunks = _splitter()._split_by_length(text, MAX_SIZE, MIN_SIZE, OVERLAP)
        assert all(len(c) <= MAX_SIZE for c in chunks), (
            f"超长段应被硬切，实际最长 {max(map(len, chunks))}"
        )

    def test_hard_split_covers_input(self):
        blob = "参数A，参数B；参数C：参数D" * 60
        parts = HierarchicalStructureSplitter._hard_split(blob, MAX_SIZE)
        assert all(len(p) <= MAX_SIZE for p in parts)
        assert "".join(parts) == blob, "硬切不得丢失或改写内容"


class TestContentIntegrity:
    def test_every_chunk_is_contiguous_substring(self):
        text = _make_sentences(50)
        chunks = _splitter()._split_by_length(text, MAX_SIZE, MIN_SIZE, OVERLAP)
        norm_text = _norm(text)
        for c in chunks:
            assert _norm(c) in norm_text, f"切块不是原文连续子串: {c[:40]!r}..."

    def test_short_text_returned_as_is(self):
        text = "只有一句话。"
        assert _splitter()._split_by_length(text, MAX_SIZE, MIN_SIZE, OVERLAP) == [text]
