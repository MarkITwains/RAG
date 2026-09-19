"""多路召回融合（纯函数，不依赖 llama-index / Milvus / torch）。

为什么单独拆一个模块
--------------------
融合逻辑（RRF）是本项目检索质量的核心，却长期和 `query.py` 里的重型 import
绑在一起 —— `query.py` 顶层 import torch / llama-index / pymilvus，导致这段
纯逻辑无法在轻量 CI 里被测试，于是同一套 RRF 在仓库里长出了好几份互不相同的
实现（加权三路、MultiPath 无权重版、评测脚本的 overlap boost 版……），
每一份都改动过，每一份都"看起来对"。

把纯逻辑下沉到这里后：
- 单测不需要任何重型依赖（见 tests/test_fusion.py）
- CLI / API / 评测脚本可以共用同一份实现，修一处即修全部

本模块只接受「有 ``.node`` 和 ``.score`` 属性的对象」，不去 import
``NodeWithScore`` 做 isinstance 判断 —— 那是唯一会把依赖引回来的地方。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

#: 默认 RRF 平滑参数。RRF 原论文的经验值，也是本仓库 MultiPath 路沿用值。
#: 注意线上主融合（query.py: FUSION_RRF_K）默认是 40 —— 两者从未对齐，
#: 复现线上行为必须显式传 rrf_k=40。
DEFAULT_RRF_K = 60


def node_id_key(item: Any) -> str:
    """返回用于去重的稳定 key。

    优先用 ``node.node_id``；取不到时退回对象 id（此时同一 node 的不同包装
    不会被视为同一份结果，属于可接受的保守行为）。
    """
    try:
        nid = getattr(getattr(item, "node", None), "node_id", None)
        if nid:
            return str(nid)
    except Exception:
        pass
    return str(id(item))


def weighted_rrf_fuse(
    routes: Sequence[Tuple[Sequence[Any], float]],
    top_n: int,
    rrf_k: int = DEFAULT_RRF_K,
) -> List[Any]:
    """加权 RRF 融合多路召回结果。

    ``score(d) = Σ_route w_route / (rrf_k + rank_route(d))``

    - ``rrf_k`` 越大，对 rank 越不敏感（k=60 时 rank1→10 只从 1/61 变 1/70），
      各路趋于均匀、长尾相对受益、头部区分度被压平；k 越小头部越激进。
    - ``w_route <= 0`` 的路直接跳过（支持"临时关掉某一路"）。
    - 同一 node 在多路命中时分数累加（这正是 RRF 的"多路投票"语义）。
    - 返回值按融合分降序，并把融合分写回 ``item.score``（调用方依赖这一约定）。

    Args:
        routes: [(节点列表, 该路权重), ...]；列表顺序即各路的 rank 顺序。
        top_n: 截断长度；<=0 时返回空列表。
        rrf_k: 平滑参数。

    Returns:
        融合后的节点列表（复用入参对象，不复制）。
    """

    scores: Dict[str, float] = {}
    nodes_map: Dict[str, Any] = {}

    for nodes, route_weight in routes:
        weight = float(route_weight)
        if weight <= 0:
            continue
        for rank, n in enumerate(nodes, 1):
            key = node_id_key(n)
            nodes_map.setdefault(key, n)
            scores[key] = scores.get(key, 0.0) + weight / float(rrf_k + rank)

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    out: List[Any] = []
    limit = max(0, int(top_n))
    for key, score in ranked[:limit]:
        node = nodes_map.get(key)
        if node is None:
            continue
        try:
            node.score = float(score)
        except Exception:
            pass
        out.append(node)
    return out


def rank_of(nodes: Sequence[Any]) -> Dict[str, int]:
    """把一路结果转成 ``{node_key: rank}``（rank 从 1 开始）。

    供"看某条结果在各路里分别排第几"的诊断 / 归因使用：判断某条被融合压下去的
    正确文档，究竟是"某一路没召回"还是"多路都召回但排名靠后"。
    """
    return {node_id_key(n): i for i, n in enumerate(nodes, 1)}


def route_contribution(
    routes: Sequence[Tuple[Sequence[Any], float]],
    rrf_k: int = DEFAULT_RRF_K,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """分路贡献归因：返回前 ``top_k`` 条结果分别由哪些路贡献、各贡献多少分。

    用途：在没有分路 ablation 的情况下，至少能回答"这 10 条里有多少条只靠
    HyDE 路进来"这类问题，而不是只能给一个整体开关对比。
    """
    per_route: Dict[str, List[float]] = {}
    names = [f"route{i}" for i in range(len(routes))]
    total: Dict[str, float] = {}
    merged: Dict[str, Any] = {}

    for name, (nodes, weight) in zip(names, routes):
        w = float(weight)
        if w <= 0:
            continue
        for rank, n in enumerate(nodes, 1):
            key = node_id_key(n)
            merged.setdefault(key, n)
            contribution = w / float(rrf_k + rank)
            total[key] = total.get(key, 0.0) + contribution
            per_route.setdefault(key, [])
            per_route[key].append(contribution)

    order = sorted(total.items(), key=lambda x: -x[1])[: max(0, int(top_k))]
    result: List[Dict[str, Any]] = []
    for key, score in order:
        contributions = per_route.get(key, [])
        result.append(
            {
                "node_id": key,
                "score": score,
                "n_routes": len(contributions),
                "by_route": {f"route{i}": c for i, c in enumerate(contributions)},
            }
        )
    return result


def describe_fusion(rrf_k: Optional[int] = None) -> Dict[str, Any]:
    """融合配置摘要，供启动日志与 ``/health/detail`` 展示。"""
    return {
        "algorithm": "weighted_rrf",
        "formula": "score(d) = Σ w_route / (k + rank_route(d))",
        "rrf_k": DEFAULT_RRF_K if rrf_k is None else int(rrf_k),
        "note": "k 越大头部区分度越小、长尾相对受益；k 越小头部越激进",
    }
