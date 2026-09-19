"""加权 RRF 融合的单元测试。

这些测试之所以写得出来，是因为融合逻辑已经从 `query.py` 下沉到
`pcb_rag/fusion.py` —— 不再随 `query.py` 一起被 torch / llama-index / pymilvus
的重型 import 绑住，因此在轻量 CI 里也能真跑（而不是被 importorskip 静默跳过）。

覆盖的语义点：
- 多路命中累加（RRF 的"投票"语义）
- 单路内部的 rank 顺序保持
- 权重 <= 0 的路被跳过
- rrf_k 的平滑语义（k 越小头部越激进）
- 分路贡献归因
"""

from types import SimpleNamespace

from pcb_rag.fusion import (
    DEFAULT_RRF_K,
    describe_fusion,
    node_id_key,
    rank_of,
    route_contribution,
    weighted_rrf_fuse,
)


def _item(node_id: str, score: float = 0.0):
    return SimpleNamespace(node=SimpleNamespace(node_id=node_id), score=score)


# ---------------------------------------------------------------------------
# 基础语义
# ---------------------------------------------------------------------------
class TestWeightedRrfFuse:
    def test_single_route_keeps_rank_order(self):
        nodes = [_item("a"), _item("b"), _item("c")]
        out = weighted_rrf_fuse([(nodes, 1.0)], top_n=10, rrf_k=60)
        assert [n.node.node_id for n in out] == ["a", "b", "c"]
        # score = 1/(k+rank)
        assert out[0].score == 1.0 / 61
        assert out[2].score == 1.0 / 63

    def test_multi_route_hit_accumulates(self):
        """同一节点被两路召回 → 分数累加，而不是取最大值。"""
        r1 = [_item("a"), _item("b")]
        r2 = [_item("b"), _item("c")]
        out = weighted_rrf_fuse([(r1, 1.0), (r2, 1.0)], top_n=10, rrf_k=60)
        assert out[0].node.node_id == "b"
        assert out[0].score == (1.0 / 62) + (1.0 / 61)

    def test_zero_or_negative_weight_route_is_skipped(self):
        r1 = [_item("a")]
        r2 = [_item("b")]
        out = weighted_rrf_fuse([(r1, 1.0), (r2, 0.0)], top_n=10, rrf_k=60)
        assert [n.node.node_id for n in out] == ["a"]

        out2 = weighted_rrf_fuse([(r1, 1.0), (r2, -1.0)], top_n=10, rrf_k=60)
        assert [n.node.node_id for n in out2] == ["a"]

    def test_route_weight_changes_ordering(self):
        """权重决定哪一路的榜首最终排第一。"""
        r1 = [_item("vec")]
        r2 = [_item("bm25")]
        out = weighted_rrf_fuse([(r1, 0.1), (r2, 1.0)], top_n=10, rrf_k=60)
        assert out[0].node.node_id == "bm25"

    def test_smaller_k_makes_head_more_aggressive(self):
        """k 越小，rank1 与 rank2 的差距越大（头部更激进）。

        这是"40 是调出来的还是随手写的"这类追问唯一能站住的论证基础：
        k 的语义必须可被量化。

        注意：融合会把分数写回 ``item.score`` 并复用入参对象，因此每次调用
        都必须用**新建的**节点，否则后一次调用会覆盖前一次的结果。
        """

        def gap_at(k: int) -> float:
            nodes = [_item("first"), _item("second")]
            out = weighted_rrf_fuse([(nodes, 1.0)], top_n=2, rrf_k=k)
            return out[0].score - out[1].score

        gaps = {k: gap_at(k) for k in (10, 20, 40, 60, 80)}
        # 单调递减：k 越大，头部与次席的差距越小
        assert gaps[10] > gaps[20] > gaps[40] > gaps[60] > gaps[80]
        # k=10 时 rank1 与 rank2 相对差距约 9%，k=80 时仅约 1.2%
        assert gaps[10] / gaps[80] > 5

    def test_top_n_truncates(self):
        nodes = [_item(f"n{i}") for i in range(10)]
        out = weighted_rrf_fuse([(nodes, 1.0)], top_n=3, rrf_k=60)
        assert len(out) == 3
        assert [n.node.node_id for n in out] == ["n0", "n1", "n2"]

    def test_top_n_zero_returns_empty(self):
        assert weighted_rrf_fuse([([_item("a")], 1.0)], top_n=0) == []

    def test_empty_routes(self):
        assert weighted_rrf_fuse([], top_n=10) == []
        assert weighted_rrf_fuse([([], 1.0)], top_n=10) == []

    def test_duplicate_within_same_route_counted_once_per_rank(self):
        """同一路里重复出现的节点按第一次出现的 rank 计分（不重复累加）。"""
        nodes = [_item("a"), _item("a")]
        out = weighted_rrf_fuse([(nodes, 1.0)], top_n=10, rrf_k=60)
        assert len(out) == 1
        assert out[0].score == (1.0 / 61) + (1.0 / 62)

    def test_default_k_is_documented_value(self):
        assert DEFAULT_RRF_K == 60

    def test_node_without_id_still_works(self):
        """取不到 node_id 时退回对象 id，不应抛异常。"""
        node = SimpleNamespace(node=SimpleNamespace(node_id=None), score=0.0)
        out = weighted_rrf_fuse([([node], 1.0)], top_n=10, rrf_k=60)
        assert len(out) == 1


# ---------------------------------------------------------------------------
# 归因工具
# ---------------------------------------------------------------------------
class TestAttribution:
    def test_node_id_key_prefers_node_id(self):
        assert node_id_key(_item("abc")) == "abc"

    def test_rank_of(self):
        assert rank_of([_item("a"), _item("b")]) == {"a": 1, "b": 2}

    def test_route_contribution_reports_single_route_hits(self):
        """能回答"这 10 条里有多少条只靠某一路进来"。"""
        r1 = [_item("a"), _item("b")]
        r2 = [_item("b"), _item("c")]
        rows = route_contribution([(r1, 1.0), (r2, 1.0)], rrf_k=60, top_k=10)
        by_id = {r["node_id"]: r for r in rows}
        assert by_id["b"]["n_routes"] == 2
        assert by_id["a"]["n_routes"] == 1
        assert by_id["c"]["n_routes"] == 1

    def test_describe_fusion(self):
        info = describe_fusion(40)
        assert info["algorithm"] == "weighted_rrf"
        assert info["rrf_k"] == 40
