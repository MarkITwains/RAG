"""GraphRAG：实体关系图构建、社区摘要与图检索。

为什么需要
----------
纯向量 / BM25 检索擅长「找到与问题相似的段落」，但对**多跳与聚合类问题**很弱：

- 「GB/T 4677 里对镀金层厚度是怎么要求的，和 IPC-6012 有什么差异？」需要跨文档连接
- 「本厂常用的几种表面处理工艺各自适用场景？」需要跨段落聚合

GraphRAG 的思路是先把语料抽成「实体—关系—实体」三元组，形成一张可遍历的图，
再在查询时做**实体链接 → 邻域扩展**，把散落在不同文档里的相关事实一次带齐。

四个组成部分
------------
1. **抽取**（``extract_triples``）：``rule`` 规则版（零 LLM 成本）/ ``llm`` 版 / ``hybrid``
2. **图存储**（``KnowledgeGraph``）：纯 Python 邻接结构 + JSON 原子落盘，
   不引入 neo4j / networkx，部署零负担
3. **社区摘要**（``detect_communities`` + ``summarize_communities``）：
   标签传播划分主题簇，可选 LLM 生成簇摘要，用于「全局性」问题
4. **检索**（``GraphRetriever`` / ``expand_nodes_with_graph``）：
   实体链接 → 邻域三元组 → 组织成证据文本，作为独立一路召回参与融合

设计取舍
--------
- 图上**自带证据片段**（``evidence``）：检索时无需回查 Milvus，零额外 IO；
  代价是图文件更大，因此对证据做长度截断并限制每实体保留条数
- 不导入 llama-index（延迟导入），可被单元测试轻量引用
- 标签传播而非 Louvain：纯 Python 实现简单、结果确定（tie-break 用字典序），
  在万级边的图上毫秒级完成

用法::

    from pcb_rag.graph_rag import KnowledgeGraph, build_graph_from_nodes, GraphRetriever

    graph = build_graph_from_nodes(text_nodes, llm=None)   # 入库阶段
    graph.save("./data/graph/kg.json")

    graph = KnowledgeGraph.load("./data/graph/kg.json")     # 查询阶段
    nodes = GraphRetriever(graph).retrieve("镀金层厚度要求", top_k=5)
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import threading
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# 复用上下文压缩模块的轻量词元抽取（纯标准库，不会引入重型依赖）
from pcb_rag.compression import extract_terms
from pcb_rag.config import env_bool

__all__ = [
    "GRAPH_RAG_ENABLED",
    "GraphRetriever",
    "KnowledgeGraph",
    "Triple",
    "build_graph_from_nodes",
    "describe_graph",
    "detect_communities",
    "expand_nodes_with_graph",
    "extract_triples",
    "link_entities",
    "load_graph",
    "normalize_entity",
    "summarize_communities",
]


# ---------------------------------------------------------------------------
# 1. 配置
# ---------------------------------------------------------------------------
def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


GRAPH_RAG_ENABLED = _env_bool("GRAPH_RAG_ENABLED", False)
GRAPH_PATH = os.getenv("GRAPH_PATH", "./data/graph/kg.json").strip() or "./data/graph/kg.json"
# rule = 句式规则 + 实体共现（零 LLM）；llm = 模型抽取；hybrid = 规则优先，LLM 补充
GRAPH_EXTRACT_BACKEND = os.getenv("GRAPH_EXTRACT_BACKEND", "rule").strip().lower()
GRAPH_MAX_TRIPLES_PER_CHUNK = int(os.getenv("GRAPH_MAX_TRIPLES_PER_CHUNK", "12"))
# LLM 抽取时并发度与单块最大字符
GRAPH_LLM_WORKERS = int(os.getenv("GRAPH_LLM_WORKERS", "4"))
GRAPH_LLM_MAX_CHARS = int(os.getenv("GRAPH_LLM_MAX_CHARS", "1600"))

# 图检索
GRAPH_HOP = int(os.getenv("GRAPH_HOP", "1"))
GRAPH_TOP_K = int(os.getenv("GRAPH_TOP_K", "5"))
# 图路召回在最终结果中的权重（作为一路 RRF 的权重，见 fuse_with_graph）
GRAPH_WEIGHT = float(os.getenv("GRAPH_WEIGHT", "0.35"))
# 图路 RRF 平滑参数 k（与主融合 FUSION_RRF_K 同量级，便于横向比较）
GRAPH_RRF_K = int(os.getenv("GRAPH_RRF_K", "60"))
GRAPH_MAX_ENTITIES_PER_QUERY = int(os.getenv("GRAPH_MAX_ENTITIES_PER_QUERY", "8"))
GRAPH_MAX_NEIGHBORS = int(os.getenv("GRAPH_MAX_NEIGHBORS", "24"))
GRAPH_EVIDENCE_CHARS = int(os.getenv("GRAPH_EVIDENCE_CHARS", "240"))

# ---- 节点语义化（对齐 LightRAG 的实体描述与跨块归并）----
# 单个实体描述的字符上限。描述会进图文件、并在检索时直接进上下文，
# 太长会把 top_k 名额挤掉，太短则失去"命中实体即拿到一句解释"的价值。
GRAPH_NODE_DESC_CHARS = int(os.getenv("GRAPH_NODE_DESC_CHARS", "240"))
# 单条实体/关系描述的字符上限（LLM 抽取时的输出约束）
GRAPH_LLM_DESC_CHARS = int(os.getenv("GRAPH_LLM_DESC_CHARS", "120"))
# gleaning 轮数：首轮抽取后把已有结果回灌，再问一轮"还有遗漏吗"。
# 0 = 关闭（省一次 LLM 调用）；1 = 常见配置，召回提升明显而成本可控。
GRAPH_LLM_GLEANING = int(os.getenv("GRAPH_LLM_GLEANING", "1"))

# 社区摘要
GRAPH_COMMUNITY_ENABLED = _env_bool("GRAPH_COMMUNITY_ENABLED", False)
GRAPH_COMMUNITY_MIN_SIZE = int(os.getenv("GRAPH_COMMUNITY_MIN_SIZE", "3"))
GRAPH_COMMUNITY_MAX_SUMMARY_CHARS = int(os.getenv("GRAPH_COMMUNITY_MAX_SUMMARY_CHARS", "600"))

_SCHEMA_VERSION = 2


# ---------------------------------------------------------------------------
# 2. PCB 实体模式（与 ingest 的 extract_pcb_entities 保持同源思路，但独立定义，
#    避免 graph_rag 反向依赖重型的 ingest 模块）
# ---------------------------------------------------------------------------
PCB_ENTITY_PATTERNS: Dict[str, str] = {
    "standard": r"(?:GB/T|GB|SJ/T|SJ|IPC|JIS|IEC|ISO|ASTM|UL|MIL)\s*[-–]?\s*\d+(?:[.\-]\d+)*",
    "material": r"(?:FR-?4|FR-?5|CEM-?\d|PI|PTFE|铝基板|陶瓷基板|覆铜板|半固化片|铜箔|聚酰亚胺|环氧树脂|无卤素|Rogers\s*\d+)",
    "process": r"(?:沉金|镀金|喷锡|沉银|OSP|化金|ENIG|HASL|蚀刻|电镀|阻焊|丝印|钻孔|层压|回流焊|波峰焊|激光钻孔|机械钻孔)",
    "structure": r"(?:焊盘|过孔|盲孔|埋孔|通孔|孔环|导线|走线|阻焊层|丝印层|内层|外层|层压结构|微带线|带状线)",
    "parameter": r"(?:线宽|线距|板厚|铜厚|阻抗|介质层厚度|孔壁铜厚|剥离强度|翘曲度|玻璃化转变温度|Tg|CTE|介电常数|损耗因子|Dk|Df)",
    "test": r"(?:热冲击|耐热性|耐湿性|盐雾试验|可焊性|拉脱强度|金相切片|离子污染|绝缘电阻|耐电压|阻燃等级|UL94|剥离试验)",
    "defect": r"(?:开路|短路|空洞|分层|起泡|白斑|黑盘|露铜|虚焊|立碑|偏移|毛刺|残铜)",
    "equipment": r"(?:AOI|X-Ray|AOI设备|飞针测试机|测试针床|回流炉|曝光机|蚀刻线)",
}

_COMPILED_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    (label, re.compile(pattern, re.IGNORECASE)) for label, pattern in PCB_ENTITY_PATTERNS.items()
]

# 句式规则：从陈述句中直接抽取关系（覆盖标准文档高频表述）
_RULE_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?P<h>[^，。；：\s]{2,24}?)(?:应|须|必须|需)(?:符合|满足|执行|遵循|采用)(?P<t>[^，。；]{2,32})"), "应符合"),
    (re.compile(r"(?P<h>[^，。；：\s]{2,24}?)(?:的)?(?:要求|规定)(?:为|是|如下)(?P<t>[^，。；]{2,32})"), "要求"),
    (re.compile(r"(?P<h>[^，。；：\s]{2,24}?)(?:采用|使用|选用)(?P<t>[^，。；]{2,32})"), "采用"),
    (re.compile(r"(?P<h>[^，。；：\s]{2,24}?)(?:包含|包括|分为)(?P<t>[^，。；]{2,32})"), "包含"),
    (re.compile(r"(?P<h>[^，。；：\s]{2,24}?)(?:适用于|用于)(?P<t>[^，。；]{2,32})"), "适用于"),
    (re.compile(r"(?P<h>[^，。；：\s]{2,24}?)(?:不低于|不小于|不大于|不超过|≥|≤)(?P<t>[^，。；]{1,24})"), "限值"),
]

_SENTENCE_SPLIT = re.compile(r"(?<=[。！？；!?;])")
_WS_RE = re.compile(r"\s+")
# 实体名两侧可能包裹的字符（含全角括号与中英文引号）
_WRAP_CHARS = r"\s\-\*\u2022\(\)\[\]【】「」《》〈〉（）\"'`“”‘’"
_ENTITY_CLEAN_RE = re.compile(rf"^[{_WRAP_CHARS}]+|[{_WRAP_CHARS}：:，,。.；;]+$")
# 规则抽取的 head 末尾常带助动词（「镀金层厚度应」→「镀金层厚度」），统一剥离
_TRAILING_AUX_RE = re.compile(r"(?:应|须|必须|需|宜|可|要|的)+$")


def normalize_entity(name: str) -> str:
    """实体名规范化：去空白与包裹标点，英文统一小写，标准号统一大写。"""

    if not name:
        return ""
    text = _WS_RE.sub(" ", str(name)).strip()
    text = _ENTITY_CLEAN_RE.sub("", text)
    if not text:
        return ""
    if re.fullmatch(r"[A-Za-z0-9\.\-/%\s]+", text):
        # 标准号保持大写，便于与查询中的写法对齐
        return text.upper() if re.search(r"\d", text) else text.lower()
    return text


def _entity_type(name: str) -> str:
    for label, pattern in _COMPILED_PATTERNS:
        if pattern.fullmatch(name) or pattern.match(name):
            return label
    return "concept"


# LLM 输出的类型标签 → 内部类型名。
# 提示词里给的是带示例的英文标签（Standard / Process / ...），模型也可能直接
# 回中文或小写变体，这里统一映射，避免同一类实体在图里碎成多种 type —— type
# 参与图谱统计与检索时的类型偏置，碎掉就失去意义。
_ENTITY_TYPE_ALIASES: Dict[str, str] = {
    "standard": "standard",
    "标准": "standard",
    "specification": "standard",
    "material": "material",
    "材料": "material",
    "process": "process",
    "工艺": "process",
    "procedure": "process",
    "structure": "structure",
    "结构": "structure",
    "layer": "structure",
    "parameter": "parameter",
    "参数": "parameter",
    "metric": "parameter",
    "equipment": "equipment",
    "设备": "equipment",
    "tool": "equipment",
    "software": "software",
    "eda": "software",
    "软件": "software",
    "method": "method",
    "测试方法": "method",
    "test": "method",
    "defect": "defect",
    "缺陷": "defect",
    "failure": "defect",
    "component": "component",
    "元件": "component",
    "器件": "component",
    "concept": "concept",
    "概念": "concept",
    "other": "concept",
    "其他": "concept",
}


def _normalize_entity_type(raw: str) -> str:
    """把模型给的类型标签归一到内部类型名；无法识别时返回空串。"""

    key = _WS_RE.sub(" ", str(raw or "")).strip().lower()
    if not key:
        return ""
    if key in _ENTITY_TYPE_ALIASES:
        return _ENTITY_TYPE_ALIASES[key]
    # 允许 "Process Step" 这类短语：取首个可识别的词
    for token in re.split(r"[\s/、,，]+", key):
        if token in _ENTITY_TYPE_ALIASES:
            return _ENTITY_TYPE_ALIASES[token]
    return ""


def _merge_node_desc(node: "GraphNode", desc: str) -> None:
    """把新描述并入节点摘要，去重并按上限截断。

    为什么不用简单拼接：同一实体在几十个 chunk 里出现时，拼接会迅速膨胀成
    一篇长文，既撑大图文件又挤占检索时的上下文名额。这里只保留「未被已有
    摘要覆盖」的新表述，超出上限即停止累加 —— 摘要是用来判断相关性的，
    不是证据全集（证据在边的 ``e`` 字段与 chunk 原文里）。
    """

    text = _WS_RE.sub(" ", str(desc or "")).strip()
    if not text:
        return

    # 已经包含该表述（或与之高度重叠）就不再追加
    if text in node.desc:
        return
    if node.desc and node.desc in text:
        # 新描述覆盖了旧摘要，直接用更完整的版本
        node.desc = text[:GRAPH_NODE_DESC_CHARS]
        return

    if not node.desc:
        node.desc = text[:GRAPH_NODE_DESC_CHARS]
        return

    remain = GRAPH_NODE_DESC_CHARS - len(node.desc) - 3
    if remain < 12:
        return
    node.desc = node.desc + "；" + text[:remain]



# ---------------------------------------------------------------------------
# 3. 三元组与图
# ---------------------------------------------------------------------------
@dataclass
class Triple:
    """一条实体关系：``head -[relation]-> tail``，附带来源证据。

    ``head_type`` / ``head_desc`` 等语义字段为可选：规则抽取不产出它们，
    LLM 抽取会填。它们的作用是让图节点带上「这是什么」的说明 —— 检索命中
    一个实体时，除了邻接关系还能拿到一段可直接进上下文的描述。
    """

    head: str
    relation: str
    tail: str
    chunk_id: str = ""
    doc_id: str = ""
    source_path: str = ""
    evidence: str = ""
    weight: float = 1.0
    head_type: str = ""
    tail_type: str = ""
    head_desc: str = ""
    tail_desc: str = ""
    relation_desc: str = ""
    revision: str = ""
    qualifiers: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)

    def key(self) -> Tuple[str, str, str]:
        return (self.head, self.relation, self.tail)


@dataclass
class GraphNode:
    """图节点（实体）及其出处。"""

    name: str
    type: str = "concept"
    count: int = 0
    desc: str = ""
    chunk_ids: List[str] = field(default_factory=list)
    doc_ids: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"t": self.type, "n": self.count}
        if self.desc:
            data["s"] = self.desc[:GRAPH_NODE_DESC_CHARS]
        if self.chunk_ids:
            data["c"] = list(self.chunk_ids)
        if self.doc_ids:
            data["d"] = list(self.doc_ids)
        if self.aliases:
            data["a"] = self.aliases[:10]
        return data


class KnowledgeGraph:
    """轻量知识图谱：邻接表 + 边表，可原子落盘为 JSON。"""

    def __init__(self) -> None:
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)
        # 实体 -> 关联边的 key 集合：避免每次查邻居/邻边都全量扫描边表
        self.incident: Dict[str, Set[Tuple[str, str, str]]] = defaultdict(set)
        self.communities: Dict[str, Dict[str, Any]] = {}
        self.built_at: float = 0.0
        self._lock = threading.RLock()
        self.revision: str = ""

    # ---- 构建 ----
    @staticmethod
    def _contributions(edge):
        if "contributions" in edge:
            return edge["contributions"]
        # Old files did not retain the doc/chunk association. Preserve that
        # uncertainty explicitly; never invent a zip or Cartesian product.
        import hashlib
        identity = json.dumps(edge, sort_keys=True, ensure_ascii=False)
        return [{"id": "legacy-" + hashlib.sha256(identity.encode()).hexdigest(),
                 "legacy": True, "needs_backfill": True,
                 "doc_ids": list(edge.get("d", [])), "chunk_ids": list(edge.get("c", [])),
                 "doc_id": "", "chunk_id": "", "weight": float(edge.get("w", 1)),
                 "evidence": edge.get("e", ""), "source_path": edge.get("s", ""),
                 "relation_desc": edge.get("rd", ""), "qualifiers": {}}]

    def _refresh_edge(self, edge):
        contributions = edge["contributions"]
        edge["w"] = sum(float(c.get("weight", 1)) for c in contributions)
        edge["c"] = list(dict.fromkeys(cid for c in contributions
                          for cid in ([c.get("chunk_id")] if c.get("chunk_id") else c.get("chunk_ids", [])) if cid))
        edge["d"] = list(dict.fromkeys(did for c in contributions
                          for did in ([c.get("doc_id")] if c.get("doc_id") else c.get("doc_ids", [])) if did))
        for dst, src, limit in (
            ("e", "evidence", GRAPH_EVIDENCE_CHARS),
            ("rd", "relation_desc", GRAPH_LLM_DESC_CHARS),
        ):
            merged = ""
            for value in dict.fromkeys(str(c.get(src, "")) for c in contributions if c.get(src)):
                if not merged:
                    merged = value[: max(0, limit)]
                elif len(merged) + 1 + len(value) <= limit:
                    merged += "；" + value
                if len(merged) >= limit:
                    break
            edge[dst] = merged
        edge["s"] = next((c.get("source_path", "") for c in contributions if c.get("source_path")), "")
        edge["facts"] = [dict(c.get("qualifiers") or {}, doc_id=c.get("doc_id", ""),
                              chunk_id=c.get("chunk_id", ""), revision=c.get("revision", ""))
                         for c in contributions if c.get("qualifiers")]
        edge["needs_backfill"] = any(c.get("needs_backfill", False) for c in contributions)

    def _refresh_nodes(self, names):
        for name in names:
            keys = self.incident.get(name, set())
            if not keys:
                self.nodes.pop(name, None)
                self.adjacency.pop(name, None)
                self.incident.pop(name, None)
                continue
            node = GraphNode(name=name, type=_entity_type(name))
            descriptions = []
            explicit_type = False
            for key in sorted(keys, key=lambda key: self.edges[key].get("order", 0)):
                edge = self.edges[key]
                side = "head" if edge["h"] == name else "tail"
                for c in edge["contributions"]:
                    node.count += 1
                    resolved = _normalize_entity_type(c.get(side + "_type", ""))
                    if resolved and not explicit_type:
                        node.type = resolved
                        explicit_type = True
                    desc = c.get(side + "_desc", "")
                    if desc and desc not in descriptions:
                        descriptions.append(desc)
                    node.aliases.extend(a for a in c.get("aliases", []) if a not in node.aliases)
                node.chunk_ids.extend(cid for cid in edge["c"] if cid not in node.chunk_ids)
                node.doc_ids.extend(did for did in edge["d"] if did not in node.doc_ids)
            node.desc = "；".join(descriptions)[:GRAPH_NODE_DESC_CHARS]
            self.nodes[name] = node

    def _rebuild_derived(self):
        self.adjacency = defaultdict(set)
        self.incident = defaultdict(set)
        previous_names = set(self.nodes)
        for key, edge in self.edges.items():
            self._refresh_edge(edge)
            self.adjacency[key[0]].add(key[2])
            self.adjacency[key[2]].add(key[0])
            self.incident[key[0]].add(key)
            self.incident[key[2]].add(key)
        self._refresh_nodes(previous_names | set(self.incident))

    def add_triple(self, triple: Triple) -> bool:
        """Upsert identified source contributions; anonymous occurrences accumulate."""
        import hashlib
        import uuid
        head, tail = normalize_entity(triple.head), normalize_entity(triple.tail)
        relation = _WS_RE.sub(" ", (triple.relation or "").strip())
        if not head or not tail or not relation or head == tail:
            return False
        if len(head) > 40 or len(tail) > 40 or len(relation) > 24:
            return False
        contribution = asdict(triple)
        identity = json.dumps([triple.doc_id, triple.chunk_id, triple.revision, triple.qualifiers],
                              sort_keys=True, ensure_ascii=False)
        contribution["id"] = (hashlib.sha256(identity.encode()).hexdigest()
                              if triple.doc_id or triple.chunk_id else uuid.uuid4().hex)
        with self._lock:
            key = (head, relation, tail)
            edge = self.edges.setdefault(key, {"h": head, "r": relation, "t": tail, "contributions": [], "order": len(self.edges)})
            contributions = {c["id"]: c for c in self._contributions(edge)}
            contributions[contribution["id"]] = contribution
            edge["contributions"] = list(contributions.values())
            self._refresh_edge(edge)
            self.adjacency[head].add(tail)
            self.adjacency[tail].add(head)
            self.incident[head].add(key)
            self.incident[tail].add(key)
            self._refresh_nodes({head, tail})
            self.communities = {}
        return True

    def add_triples(self, triples: Iterable[Triple]) -> int:
        return sum(1 for t in triples if self.add_triple(t))

    def merge(self, other: "KnowledgeGraph") -> None:
        """Merge every contribution without multiplying identified-source weight."""
        import copy
        if other is None or other is self:
            return
        with self._lock:
            for key, source in other.edges.items():
                target = self.edges.setdefault(key, {"h": key[0], "r": key[1], "t": key[2], "contributions": [], "order": len(self.edges)})
                contributions = {c["id"]: c for c in self._contributions(target)}
                contributions.update({c["id"]: copy.deepcopy(c) for c in self._contributions(source)})
                target["contributions"] = list(contributions.values())
            self._rebuild_derived()
            self.communities = {}

    def _remove_sources(self, targets, field):
        targets = {value for value in targets if value}
        if not targets:
            return 0
        removed = 0
        with self._lock:
            for key, edge in list(self.edges.items()):
                remaining = []
                for contribution in self._contributions(edge):
                    if contribution.get(field) in targets:
                        continue
                    plural = field + "s"
                    legacy_ids = contribution.get(plural, [])
                    if contribution.get("legacy") and set(legacy_ids) & targets:
                        kept = [value for value in legacy_ids if value not in targets]
                        if not kept:
                            continue
                        contribution = dict(contribution, **{plural: kept}, needs_backfill=True)
                        # Aggregated legacy text may come from the deleted source.
                        for attr in ("evidence", "source_path", "head_desc", "tail_desc", "relation_desc"):
                            contribution[attr] = ""
                        contribution["chunk_ids" if field == "doc_id" else "doc_ids"] = []
                    remaining.append(contribution)
                if remaining:
                    edge["contributions"] = remaining
                else:
                    del self.edges[key]
                    removed += 1
            self._rebuild_derived()
            self.communities = {}
        return removed

    def remove_docs(self, doc_ids: Iterable[str]) -> int:
        return self._remove_sources(doc_ids, "doc_id")

    def remove_chunks(self, chunk_ids: Iterable[str]) -> int:
        return self._remove_sources(chunk_ids, "chunk_id")

    # ---- 查询 ----
    def neighbors(self, entity: str, hops: int = 1, limit: int = GRAPH_MAX_NEIGHBORS) -> List[str]:
        """返回实体在 ``hops`` 跳内的邻居（按边权重降序）。"""

        start = normalize_entity(entity)
        if start not in self.nodes:
            return []

        visited: Set[str] = {start}
        frontier: deque[Tuple[str, int]] = deque([(start, 0)])
        scored: List[Tuple[float, str]] = []

        while frontier:
            current, depth = frontier.popleft()
            if depth >= max(1, hops):
                continue
            for neighbor in self.adjacency.get(current, ()):  # type: ignore[arg-type]
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                scored.append((self.pair_weight(current, neighbor), neighbor))
                frontier.append((neighbor, depth + 1))

        scored.sort(key=lambda x: (-x[0], x[1]))
        return [name for _, name in scored[:limit]]

    def pair_weight(self, a: str, b: str) -> float:
        """两个实体之间所有直接关系的权重之和（正向 + 反向）。"""

        left, right = normalize_entity(a), normalize_entity(b)
        total = 0.0
        for key in self.incident.get(left, ()):  # type: ignore[union-attr]
            if right not in (key[0], key[2]):
                continue
            edge = self.edges.get(key)
            if edge is not None:
                total += float(edge.get("w", 0.0))
        return total

    def edges_of(self, entity: str) -> List[Dict[str, Any]]:
        """返回与实体直接相连的所有边（出边 + 入边）。"""

        name = normalize_entity(entity)
        result: List[Dict[str, Any]] = []
        for key in self.incident.get(name, ()):  # type: ignore[union-attr]
            edge = self.edges.get(key)
            if edge is not None:
                result.append(edge)
        result.sort(key=lambda e: (-float(e.get("w", 0.0)), e["h"], e["t"]))
        return result

    def related_edges(
        self,
        entities: Sequence[str],
        *,
        hops: int = 1,
        max_edges: int = 60,
    ) -> List[Dict[str, Any]]:
        """收集一组实体邻域内的边，按权重降序返回。"""

        seeds = {normalized for entity in entities if (normalized := normalize_entity(entity)) in self.nodes}
        visited = set(seeds)
        frontier = set(seeds)
        collected: List[Dict[str, Any]] = []
        seen_keys: Set[Tuple[str, str, str]] = set()
        for _ in range(max(0, hops)):
            next_frontier = set()
            for name in sorted(frontier):
                for key in self.incident.get(name, ()):  # type: ignore[union-attr]
                    edge = self.edges.get(key)
                    if edge is None:
                        continue
                    if key not in seen_keys:
                        seen_keys.add(key)
                        collected.append(edge)
                    other = edge["t"] if edge["h"] == name else edge["h"]
                    if other not in visited:
                        visited.add(other)
                        next_frontier.add(other)
            frontier = next_frontier

        if not seeds:
            return []

        collected.sort(key=lambda e: (-float(e.get("w", 0.0)), e["h"], e["t"]))
        return collected[:max_edges]

    def entity_chunks(self, entity: str) -> List[str]:
        node = self.nodes.get(normalize_entity(entity))
        return list(node.chunk_ids) if node else []

    # ---- 持久化 ----
    def to_dict(self) -> Dict[str, Any]:
        edges = sorted(self.edges.values(), key=lambda e: (e["h"], e["r"], e["t"]))
        return {
            "version": _SCHEMA_VERSION,
            "built_at": self.built_at,
            "revision": self.revision,
            "nodes": {name: node.to_dict() for name, node in sorted(self.nodes.items())},
            "edges": edges,
            "communities": self.communities,
        }

    def save(self, path: Optional[str] = None, *, compact: bool = False) -> str:
        """原子写入 JSON（先写临时文件再替换，避免写入中断导致图损坏）。"""

        target = os.path.abspath(path or GRAPH_PATH)
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        if not self.built_at:
            import time

            self.built_at = time.time()

        payload = self.to_dict()
        # compact is retained for compatibility; provenance is never discarded.
        fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(target) or ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
            os.replace(tmp_path, target)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        return target

    @classmethod
    def load(cls, path: Optional[str] = None) -> Optional["KnowledgeGraph"]:
        """加载图；文件不存在或损坏时返回 ``None``（调用方据此降级为无图检索）。"""

        target = os.path.abspath(path or GRAPH_PATH)
        if not os.path.exists(target):
            return None

        try:
            with open(target, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            return None

        graph = cls()
        graph.built_at = float(payload.get("built_at") or 0.0)
        graph.revision = str(payload.get("revision") or "")
        graph.communities = payload.get("communities") or {}

        for name, raw in (payload.get("nodes") or {}).items():
            if not isinstance(raw, dict):
                continue
            graph.nodes[name] = GraphNode(
                name=name,
                type=str(raw.get("t") or "concept"),
                count=int(raw.get("n") or 0),
                desc=str(raw.get("s") or ""),
                chunk_ids=list(raw.get("c") or []),
                doc_ids=list(raw.get("d") or []),
                aliases=list(raw.get("a") or []),
            )

        for edge in payload.get("edges") or []:
            if not isinstance(edge, dict):
                continue
            head, relation, tail = edge.get("h"), edge.get("r"), edge.get("t")
            if not head or not relation or not tail:
                continue
            key = (head, relation, tail)
            graph.edges[key] = {
                "h": head,
                "r": relation,
                "t": tail,
                "w": float(edge.get("w") or 1.0),
                "c": list(edge.get("c") or []),
                "d": list(edge.get("d") or []),
                "e": str(edge.get("e") or ""),
                "s": str(edge.get("s") or ""),
                "rd": str(edge.get("rd") or ""),
                "contributions": cls._contributions(edge),
                "order": edge.get("order", len(graph.edges)),
            }
            graph._refresh_edge(graph.edges[key])
            graph.adjacency[head].add(tail)
            graph.adjacency[tail].add(head)
            graph.incident[head].add(key)
            graph.incident[tail].add(key)

        return graph

    # ---- 统计 ----
    def stats(self) -> Dict[str, Any]:
        type_counter = Counter(node.type for node in self.nodes.values())
        return {
            "entities": len(self.nodes),
            "relations": len(self.edges),
            "communities": len(self.communities),
            "entity_types": dict(type_counter.most_common()),
            "built_at": self.built_at,
        }


# ---------------------------------------------------------------------------
# 4. 三元组抽取
# ---------------------------------------------------------------------------
def _sentences(text: str) -> List[str]:
    if not text:
        return []
    result: List[str] = []
    for chunk in _SENTENCE_SPLIT.split(text):
        chunk = chunk.strip()
        if chunk:
            result.append(chunk)
    return result


def _collect_entities(sentence: str) -> List[str]:
    """从句子中收集所有 PCB 领域实体（去重、规范）。"""

    found: List[str] = []
    seen: Set[str] = set()
    for _label, pattern in _COMPILED_PATTERNS:
        for match in pattern.finditer(sentence):
            name = normalize_entity(match.group(0))
            if name and name not in seen:
                seen.add(name)
                found.append(name)
    return found


def extract_triples_rule(
    text: str,
    *,
    chunk_id: str = "",
    doc_id: str = "",
    source_path: str = "",
    max_triples: int = GRAPH_MAX_TRIPLES_PER_CHUNK,
) -> List[Triple]:
    """规则抽取：句式规则为主，实体同句共现为辅。

    共现关系（``mentioned_with``）虽然语义弱，但能把标准号、材料、工艺
    串到同一张图里，让「镀金层厚度」这类查询有机会连到具体标准。
    """

    triples: List[Triple] = []
    if not text:
        return triples

    for sentence in _sentences(text):
        evidence = sentence[:GRAPH_EVIDENCE_CHARS]

        # 1) 句式规则
        for pattern, relation in _RULE_PATTERNS:
            for match in pattern.finditer(sentence):
                # 双次规范化：先剥包裹标点，再剥尾部助动词，最后清掉可能残留的标点
                head = normalize_entity(_TRAILING_AUX_RE.sub("", normalize_entity(match.group("h"))))
                tail = normalize_entity(match.group("t"))
                if not head or not tail or head == tail:
                    continue
                if len(head) < 2 or len(tail) < 2:
                    continue
                triples.append(
                    Triple(
                        head=head,
                        relation=relation,
                        tail=tail,
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        source_path=source_path,
                        evidence=evidence,
                        weight=1.2,
                    )
                )
                if len(triples) >= max_triples:
                    return triples

        # 2) 实体共现：把标准号 / 工艺 / 材料 / 参数互相连接
        entities = _collect_entities(sentence)
        if 2 <= len(entities) <= 6:
            for i, head in enumerate(entities):
                for tail in entities[i + 1 :]:
                    triples.append(
                        Triple(
                            head=head,
                            relation="mentioned_with",
                            tail=tail,
                            chunk_id=chunk_id,
                            doc_id=doc_id,
                            source_path=source_path,
                            evidence=evidence,
                            weight=0.4,
                        )
                    )
                    if len(triples) >= max_triples:
                        return triples

    return triples


# 实体类型表：给模型一个封闭的类型集合，避免同一类实体在图里碎成多种标签。
# 类型参与图谱统计与检索时的类型偏置，也决定 _normalize_entity_type 的映射目标。
_TRIPLE_TYPE_TABLE = """- Standard：标准、规范、条款（GB/T 4677-2002、IPC-2581C）
- Material：材料与基材（FR-4、PI、覆铜板、铜箔）
- Process：工艺步骤（沉金、镀金、蚀刻、层压、阻焊）
- Structure：结构层次（四层板、盲埋孔、阻抗线、焊盘）
- Parameter：可量化参数（镀层厚度、线宽、阻抗值、剥离强度）
- Equipment：设备与工装（飞针测试机、曝光机）
- Software：EDA 工具（Altium Designer、Cadence Allegro）
- Method：测试与检验方法（AOI、热冲击试验、可焊性试验）
- Defect：缺陷与失效（分层、起泡、开路、短路）
- Component：元件与器件（BGA、连接器、过孔）
- Concept：术语与抽象概念（信号完整性、EMI）
"""

_TRIPLE_JSON_SCHEMA = (
    '{"head": "实体名", "head_type": "Standard", "head_desc": "该实体是什么，40字内", '
    '"relation": "关系", "relation_desc": "这条关系说明了什么，40字内", '
    '"tail": "实体名", "tail_type": "Material", "tail_desc": "该实体是什么，40字内", '
    '"evidence": "逐字原文", "aliases": [], "qualifiers": {"standard": "标准号", '
    '"edition": "版本", "clause": "条款", "scope": "适用对象", "conditions": "适用条件", '
    '"parameter": "参数", "comparator": "比较符", "value": "原始数值", "unit": "原始单位"}}'
)


def _build_triple_prompt(text: str) -> str:
    """构造三元组抽取提示词。

    直接拼接而不使用 ``str.format``：提示词里的 JSON 示例含有花括号，
    用 format 会与之冲突（需要成对转义，极易出错）。

    相比只抽 ``head/relation/tail`` 的旧版，这里额外要求实体类型与一句描述 ——
    这是 LightRAG 图谱质量的主要来源：命中一个实体时，除了邻接关系还能拿到
    「它是什么」的说明，可直接进上下文而不必回读原文。
    """

    return (
        "你是 PCB 制造业知识工程师。请从下面的技术文档片段中抽取实体关系三元组。\n\n"
        "要求：\n"
        "1. 只抽取原文明确表述的内容，绝不臆造或补充常识；\n"
        "2. head / tail 必须是原文出现的实体（标准号、材料、工艺、结构、参数、"
        "测试方法、缺陷、EDA 工具等）；relation 用简短动词短语，例如：应符合、包含、"
        "采用、要求、限值、适用于、测试方法为；\n"
        "3. 每条三元组必须能在原文中找到依据；一句话涉及多个实体时拆成多条二元关系；\n"
        "4. head_desc / tail_desc 用一句第三人称说明该实体「是什么」，只依据原文，"
        f"不超过 40 字；relation_desc 用一句说明这条关系「说了什么」，不超过 40 字；\n"
        f"5. 实体类型只能从下列取值中选一个，无法归类用 Concept：\n{_TRIPLE_TYPE_TABLE}"
        f"6. 最多输出 {GRAPH_MAX_TRIPLES_PER_CHUNK} 条，没有则输出空数组 []；\n"
        "数值事实保留版本、对象、条件、比较符、原始值和单位；不同条件分别输出，不合并。"
        "缺失字段留空，evidence 必须逐字引用原文，aliases 只填原文明确给出的别名。\n"
        "7. 只输出 JSON 数组，不要输出解释文字或 Markdown 代码块，格式：\n"
        f"[{_TRIPLE_JSON_SCHEMA}]\n\n"
        f"片段：\n{text}"
    )


def _build_gleaning_prompt(text: str, existing: List[Dict[str, str]]) -> str:
    """gleaning 提示词：回灌已抽取结果，要求补全遗漏。

    参照 LightRAG 的 gleaning 做法 —— 首轮抽取常漏掉「次要但有用」的关系
    （例如参数与测试方法之间的约束），把结果回灌再问一轮，模型更容易意识到
    「这些类型的边还没有被抽到」，召回提升明显，代价只是多一次调用。
    """

    lines = []
    for item in existing[:40]:
        head = item.get("head", "")
        relation = item.get("relation", "")
        tail = item.get("tail", "")
        if head and relation and tail:
            lines.append(f"- {head} --[{relation}]--> {tail}")
    rendered = "\n".join(lines) if lines else "（无）"

    return (
        "你正在为 PCB 技术文档构建知识图谱。上一轮已从片段中抽取了下列三元组：\n\n"
        f"{rendered}\n\n"
        "请重新检查原文片段，**只补充上一轮遗漏的**实体关系。"
        "注意容易漏掉的类型：参数限值、测试方法、适用条件、材料与工艺的对应关系。\n"
        "不要重复上面已列出的三元组。\n\n"
        f"若确实没有遗漏，输出空数组 []。\n"
        "只输出 JSON 数组，不要输出解释文字或 Markdown 代码块。\n\n"
        f"格式：[{_TRIPLE_JSON_SCHEMA}]\n"
        f"片段：\n{text}"
    )


_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")


def _parse_triple_json(raw: str, *, strict: bool = False) -> List[Dict[str, Any]]:
    """从模型输出中稳健地解析三元组数组。

    兼容中英文键名与缺失的可选字段：旧提示词只产出 head/relation/tail，
    新提示词多出类型与描述。缺描述时留空，不阻断抽取。
    """

    if not raw:
        if strict:
            raise ValueError("empty graph extraction response")
        return []
    text = raw.strip()
    # 去掉 Markdown 围栏
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()

    match = _JSON_ARRAY_RE.search(text)
    if not match:
        if strict:
            raise ValueError("graph extraction response has no JSON array")
        return []
    try:
        payload = json.loads(match.group(0))
    except ValueError:
        if strict:
            raise ValueError("invalid graph extraction JSON") from None
        return []
    if not isinstance(payload, list):
        return []

    def _pick(item: Dict[str, Any], *keys: str) -> str:
        for key in keys:
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return ""

    triples: List[Dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            if strict:
                raise ValueError("graph extraction item must be an object")
            continue
        head = _pick(item, "head", "头实体", "source_entity")
        relation = _pick(item, "relation", "关系", "relationship")
        tail = _pick(item, "tail", "尾实体", "target_entity")
        if not (head and relation and tail):
            if strict:
                raise ValueError("graph extraction item has missing entities or relation")
            continue
        triples.append(
            {
                "head": head,
                "relation": relation,
                "tail": tail,
                "head_type": _pick(item, "head_type", "source_type", "头实体类型"),
                "head_desc": _pick(item, "head_desc", "source_desc", "头实体描述")[:GRAPH_LLM_DESC_CHARS],
                "relation_desc": _pick(item, "relation_desc", "relationship_desc", "关系描述")[:GRAPH_LLM_DESC_CHARS],
                "tail_type": _pick(item, "tail_type", "target_type", "尾实体类型"),
                "tail_desc": _pick(item, "tail_desc", "target_desc", "尾实体描述")[:GRAPH_LLM_DESC_CHARS],
                "evidence": _pick(item, "evidence", "原文"),
                "qualifiers": {key: str(value) for key, value in (item.get("qualifiers") or {}).items()
                               if key in {"standard", "edition", "clause", "scope", "conditions", "parameter", "comparator", "value", "unit"}
                               and isinstance(value, (str, int, float))} if isinstance(item.get("qualifiers"), dict) else {},
                "aliases": [value for value in item.get("aliases", []) if isinstance(value, str)][:10]
                           if isinstance(item.get("aliases"), list) else [],
            }
        )
    return triples


def _is_source_quote(evidence: str, snippet: str) -> bool:
    """判断 evidence 是否可追溯到原文。

    模型常把跨行引用压成单行，或把原文的换行/缩进换成空格；而换行在提示词里
    已被压成单个空格。两侧都删掉全部空白再比较，既容忍排版差异，也不放过
    真正的杜撰（非空白字符必须逐字出现）。
    """

    if evidence in snippet:
        return True
    return _WS_RE.sub("", evidence) in _WS_RE.sub("", snippet)


def _is_entity_mention(entity: str, snippet: str) -> bool:
    normalized = _WS_RE.sub("", normalize_entity(entity)).casefold()
    return bool(normalized) and normalized in _WS_RE.sub("", snippet).casefold()


def extract_triples_llm(
    text: str,
    llm: Any,
    *,
    chunk_id: str = "",
    doc_id: str = "",
    source_path: str = "",
    strict: bool = False,
) -> List[Triple]:
    """LLM 抽取：调用模型输出三元组 JSON（失败时返回空列表，由上层决定是否回退规则）。

    ``GRAPH_LLM_GLEANING > 0`` 时在首轮之后追加一轮补全：把已抽取的关系回灌，
    要求模型只补遗漏项。两轮结果按键三元组去重后合并。
    """

    if llm is None or not text or not text.strip():
        return []

    snippet = text[:GRAPH_LLM_MAX_CHARS]
    try:
        from pcb_rag.usage import model_complete
        response = model_complete(llm, _build_triple_prompt(snippet))
    except Exception:
        if strict:
            raise
        return []

    raw = str(getattr(response, "text", response))
    records = _parse_triple_json(raw, strict=strict)

    # gleaning：补一轮遗漏
    if records and GRAPH_LLM_GLEANING > 0:
        seen = {(r["head"], r["relation"], r["tail"], json.dumps(r.get("qualifiers", {}), sort_keys=True)) for r in records}
        for _ in range(GRAPH_LLM_GLEANING):
            try:
                more_raw = str(
                    getattr(model_complete(llm, _build_gleaning_prompt(snippet, records)), "text", "")
                )
            except Exception:
                if strict:
                    raise
                break
            added = 0
            for item in _parse_triple_json(more_raw, strict=strict):
                key = (item["head"], item["relation"], item["tail"], json.dumps(item.get("qualifiers", {}), sort_keys=True))
                if key in seen:
                    continue
                seen.add(key)
                records.append(item)
                added += 1
            if added == 0:
                break

    evidence = _WS_RE.sub(" ", snippet)[:GRAPH_EVIDENCE_CHARS]
    if strict:
        for item in records:
            quote = item.get("evidence", "")
            if not quote or not _is_source_quote(quote, snippet):
                raise ValueError("graph extraction evidence is not an exact source quote")
            if any(not _is_entity_mention(item[field], snippet) for field in ("head", "tail")):
                raise ValueError("graph extraction entities must appear in source")
    triples = [
        Triple(
            head=item["head"],
            relation=item["relation"],
            tail=item["tail"],
            chunk_id=chunk_id,
            doc_id=doc_id,
            source_path=source_path,
            evidence=(item.get("evidence") or evidence)[:GRAPH_EVIDENCE_CHARS],
            weight=1.5,
            head_type=item.get("head_type", ""),
            tail_type=item.get("tail_type", ""),
            head_desc=item.get("head_desc", ""),
            tail_desc=item.get("tail_desc", ""),
            relation_desc=item.get("relation_desc", ""),
            qualifiers=item.get("qualifiers", {}),
            aliases=item.get("aliases", []),
        )
        for item in records
    ]
    # gleaning 的补充量不占用首轮配额 —— 否则补出来的边会被上限直接截掉，
    # 白付一次调用。上限放宽到两倍，仍受总条数约束避免单块产出失控。
    cap = GRAPH_MAX_TRIPLES_PER_CHUNK * (2 if GRAPH_LLM_GLEANING > 0 else 1)
    return triples[:cap]


def extract_triples(
    text: str,
    llm: Any = None,
    *,
    backend: Optional[str] = None,
    chunk_id: str = "",
    doc_id: str = "",
    source_path: str = "",
) -> List[Triple]:
    """统一抽取入口：按配置选择规则 / LLM / 混合模式。"""

    mode = (backend or GRAPH_EXTRACT_BACKEND or "rule").strip().lower()

    if mode == "llm":
        llm_triples = extract_triples_llm(text, llm, chunk_id=chunk_id, doc_id=doc_id, source_path=source_path)
        if llm_triples:
            return llm_triples
        # 模型不可用或无输出时回退规则，保证图不会空转
        return extract_triples_rule(text, chunk_id=chunk_id, doc_id=doc_id, source_path=source_path)

    if mode == "hybrid":
        rule_triples = extract_triples_rule(text, chunk_id=chunk_id, doc_id=doc_id, source_path=source_path)
        llm_triples = extract_triples_llm(text, llm, chunk_id=chunk_id, doc_id=doc_id, source_path=source_path)
        return rule_triples + llm_triples

    return extract_triples_rule(text, chunk_id=chunk_id, doc_id=doc_id, source_path=source_path)


# ---------------------------------------------------------------------------
# 5. 图构建入口
# ---------------------------------------------------------------------------
def _node_text(node: Any) -> str:
    getter = getattr(node, "get_content", None)
    if callable(getter):
        try:
            return str(getter())
        except Exception:
            pass
    return str(getattr(node, "text", "") or "")


def _node_metadata(node: Any) -> Dict[str, Any]:
    metadata = getattr(node, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def build_graph_from_nodes(
    nodes: Optional[Sequence[Any]],
    llm: Any = None,
    *,
    graph: Optional[KnowledgeGraph] = None,
    backend: Optional[str] = None,
    progress: bool = False,
    strict: bool = False,
) -> KnowledgeGraph:
    """从切分好的 chunk 构建（或增量合并）知识图谱。

    - ``llm`` 仅在 ``backend`` 为 ``llm`` / ``hybrid`` 时使用
    - ``graph`` 传入已有图时执行增量合并（配合增量入库使用）
    """

    graph = graph if graph is not None else KnowledgeGraph()
    if not nodes:
        return graph

    mode = (backend or GRAPH_EXTRACT_BACKEND or "rule").strip().lower()
    triples: List[Triple] = []

    if strict and mode in {"llm", "hybrid"} and llm is None:
        raise ValueError("configured graph extraction requires an LLM")
    if mode in {"llm", "hybrid"} and llm is not None:
        triples = _extract_triples_concurrent(nodes, llm, mode=mode, progress=progress, strict=strict)
    else:
        for node in nodes:
            metadata = _node_metadata(node)
            triples.extend(
                extract_triples(
                    _node_text(node),
                    backend="rule",
                    chunk_id=str(getattr(node, "node_id", "") or getattr(node, "id_", "") or metadata.get("chunk_id", "") or ""),
                    doc_id=str(metadata.get("doc_node_id", "") or ""),
                    source_path=str(metadata.get("source_path") or metadata.get("file_path") or ""),
                )
            )

    from pcb_rag.pcb_facts import extract_qualified_facts
    metadata_by_id = {str(getattr(node, "node_id", "") or getattr(node, "id_", "") or ""): _node_metadata(node) for node in nodes}
    for triple in triples:
        metadata = metadata_by_id.get(triple.chunk_id, {})
        triple.revision = str(metadata.get("revision") or "")
        triple.qualifiers = {**{key: metadata.get(key, "") for key in ("standard", "edition", "scope", "conditions", "section", "page_label")},
                             **{key: value for key, value in triple.qualifiers.items() if value != ""}}
    for node in nodes:
        metadata = _node_metadata(node)
        chunk_id = str(getattr(node, "node_id", "") or getattr(node, "id_", "") or "")
        for fact in metadata.get("pcb_facts") or extract_qualified_facts(_node_text(node), metadata):
            parameter = fact.get("parameter", "")
            if parameter and len(parameter) <= 40:
                triples.append(Triple(head=parameter, relation=fact.get("comparator") or "限值",
                    tail=fact["value"] + " " + fact["unit"], chunk_id=chunk_id,
                    doc_id=str(metadata.get("doc_node_id") or ""), source_path=str(metadata.get("source_path") or ""),
                    evidence=fact["evidence"], head_type="parameter", revision=str(metadata.get("revision") or ""),
                    qualifiers=fact))
    added = graph.add_triples(triples)
    if progress:
        print(f"[GraphRAG] 抽取三元组 {len(triples)} 条，去重后新增/合并 {added} 条")

    try:
        from pcb_rag.observability import counter, record_event

        counter("graph.triples_extracted", len(triples))
        counter("graph.triples_added", added)
        record_event("graph.built", entities=len(graph.nodes), relations=len(graph.edges), backend=mode)
    except Exception:
        pass

    return graph


def _extract_triples_concurrent(
    nodes: Sequence[Any],
    llm: Any,
    *,
    mode: str,
    progress: bool = False,
    strict: bool = False,
) -> List[Triple]:
    """并发调用 LLM 抽取三元组（受 GRAPH_LLM_WORKERS 限制）。"""

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _work(node: Any) -> List[Triple]:
        metadata = _node_metadata(node)
        text = _node_text(node)
        kwargs = {
            "chunk_id": str(getattr(node, "node_id", "") or getattr(node, "id_", "") or metadata.get("chunk_id", "") or ""),
            "doc_id": str(metadata.get("doc_node_id", "") or ""),
            "source_path": str(metadata.get("source_path") or metadata.get("file_path") or ""),
        }
        if mode == "hybrid":
            return extract_triples_rule(text, **kwargs) + extract_triples_llm(text, llm, strict=strict, **kwargs)
        return extract_triples_llm(text, llm, strict=strict, **kwargs)

    results: List[Triple] = []
    workers = max(1, GRAPH_LLM_WORKERS)
    # 容忍模式下单块失败只记账、不中断整篇：模型偶发会返回坏 JSON 或改写引用，
    # 5,700 块规模下"任一块失败即作废整篇"等于永远跑不完。被接受的三元组仍经过
    # 严格证据校验，杜撰引用不会进入图谱；失败块重试一次，仍失败则跳过并计数。
    tolerate = env_bool("GRAPH_TOLERATE_CHUNK_FAILURES", False)

    def _collect(subset):
        got: List[Triple] = []
        failed: List[Any] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_work, node): node for node in subset}
            done = 0
            for future in as_completed(futures):
                done += 1
                try:
                    got.extend(future.result())
                except Exception:
                    if strict and not tolerate:
                        for pending in futures:
                            pending.cancel()
                        raise
                    failed.append(futures[future])
                if progress and done % 50 == 0:
                    print(f"[GraphRAG] 抽取进度 {done}/{len(subset)}")
        return got, failed

    results, failed = _collect(nodes)
    if failed and tolerate:
        print(f"[GraphRAG] {len(failed)} 块抽取失败，重试一次")
        retried, failed = _collect(failed)
        results.extend(retried)
    if failed:
        try:
            from pcb_rag.observability import counter

            counter("graph.chunk_failures", len(failed))
        except Exception:
            pass
        print(f"[GraphRAG] {len(failed)}/{len(nodes)} 块未产出三元组，已跳过并计数")

    return results


def load_graph(path: Optional[str] = None) -> Optional[KnowledgeGraph]:
    """便捷加载入口（等价于 ``KnowledgeGraph.load``）。"""

    return KnowledgeGraph.load(path)


# ---------------------------------------------------------------------------
# 6. 实体链接
# ---------------------------------------------------------------------------
def link_entities(
    query: str,
    graph: KnowledgeGraph,
    *,
    max_entities: int = GRAPH_MAX_ENTITIES_PER_QUERY,
    min_score: float = 0.34,
) -> List[Tuple[str, float]]:
    """把查询链接到图中的实体，返回 ``[(实体名, 分数), ...]`` 降序。

    打分考虑：实体名在查询中完整出现 > 查询中的片段命中实体 > 实体包含查询词。
    另外用「实体被提及次数」做轻微加权，避免罕见噪声实体抢占名额。
    """

    if not query or not graph or not graph.nodes:
        return []

    lowered = _WS_RE.sub(" ", query).strip().lower()
    if not lowered:
        return []

    # 先用查询词元缩小候选：直接枚举 n-gram 会退化成 O(实体数 × 查询长度)，
    # 在万级实体 + 长文档（chunk 扩展场景）下会明显拖慢检索
    terms = [t for t in extract_terms(query, max_terms=48) if len(t) >= 2]

    candidates: List[Tuple[float, str]] = []
    for name, node in graph.nodes.items():
        name_lower = name.lower()
        score = 0.0
        if any(alias.lower() in lowered for alias in node.aliases if len(alias) >= 2):
            score = 0.8

        if len(name_lower) >= 2 and name_lower in lowered:
            # 完整命中：名字越长越可信
            score = min(1.0, 0.55 + 0.03 * len(name_lower))
        else:
            # 查询词元被实体包含（如查询「镀金」命中实体「镀金层」）
            for term in terms:
                if term in name_lower:
                    score = max(score, min(0.5, 0.2 + 0.06 * len(term)))
                    break

        if score <= 0:
            continue

        # 提及次数加权（log 抑制长尾），上限 +0.15
        score += min(0.15, 0.05 * math.log1p(node.count))
        candidates.append((min(1.0, score), name))

    candidates.sort(key=lambda x: (-x[0], x[1]))
    return [(name, score) for score, name in candidates[:max_entities] if score >= min_score]


# ---------------------------------------------------------------------------
# 7. 社区检测与摘要
# ---------------------------------------------------------------------------
def detect_communities(graph: KnowledgeGraph, *, max_iter: int = 20) -> Dict[str, Dict[str, Any]]:
    """标签传播社区检测。

    返回 ``{社区标签: {"entities": [...], "size": n}}``。
    实现刻意保持确定性（邻居标签以 ``(票数, 标签)`` 排序），便于增量重建时结果稳定。
    """

    if not graph or not graph.nodes:
        return {}

    labels: Dict[str, str] = {name: name for name in graph.nodes}
    neighbors = {name: sorted(graph.adjacency.get(name, set())) for name in graph.nodes}

    for _ in range(max(1, max_iter)):
        changed = False
        for name in sorted(graph.nodes):
            adjacent = neighbors.get(name) or []
            if not adjacent:
                continue
            votes = Counter(labels[n] for n in adjacent if n in labels)
            if not votes:
                continue
            # tie-break：票数相同时取字典序最小的标签，保证确定性
            best_label = min(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            if labels[name] != best_label:
                labels[name] = best_label
                changed = True
        if not changed:
            break

    grouped: Dict[str, List[str]] = defaultdict(list)
    for name, label in labels.items():
        grouped[label].append(name)

    communities: Dict[str, Dict[str, Any]] = {}
    for label, members in grouped.items():
        if len(members) < GRAPH_COMMUNITY_MIN_SIZE:
            continue
        members.sort()
        communities[label] = {"entities": members, "size": len(members), "summary": ""}

    return communities


def summarize_communities(graph: KnowledgeGraph, llm: Any = None) -> Dict[str, Dict[str, Any]]:
    """为每个社区生成摘要（规则版拼接实体与关系；有 LLM 时可生成自然语言摘要）。"""

    if not graph:
        return {}
    if not graph.communities:
        graph.communities = detect_communities(graph)

    for label, community in graph.communities.items():
        if community.get("summary"):
            continue
        entities = community.get("entities", [])
        relation_lines: List[str] = []
        entity_set = set(entities)
        for edge in graph.edges.values():
            if edge["h"] in entity_set and edge["t"] in entity_set:
                relation_lines.append(f"{edge['h']} {edge['r']} {edge['t']}")
                if len(relation_lines) >= 20:
                    break

        fallback = "；".join(relation_lines) or "、".join(entities[:20])
        summary = fallback

        if llm is not None:
            prompt = (
                "以下是 PCB 知识图谱中一个主题簇的实体与关系，请用 2~3 句话概括该主题覆盖的内容，"
                "保留标准号与关键参数，不要编造。\n\n"
                f"实体：{'、'.join(entities[:30])}\n关系：{fallback}"
            )
            try:
                from pcb_rag.usage import model_complete
                response = model_complete(llm, prompt)
                summary = str(getattr(response, "text", response)).strip() or fallback
            except Exception:
                summary = fallback

        community["summary"] = summary[:GRAPH_COMMUNITY_MAX_SUMMARY_CHARS]

    return graph.communities


# ---------------------------------------------------------------------------
# 8. 图检索
# ---------------------------------------------------------------------------
class GraphRetriever:
    """基于知识图谱的检索器。

    ``retrieve`` 返回 ``llama_index`` 的 ``NodeWithScore`` 列表（延迟导入），
    其中节点文本由「实体邻域关系 + 证据片段」构成，可直接参与后续融合与生成。
    """

    def __init__(self, graph: KnowledgeGraph, *, hops: int = GRAPH_HOP, weight: float = GRAPH_WEIGHT) -> None:
        self.graph = graph
        self.hops = max(1, hops)
        self.weight = weight

    # ---- 纯逻辑：生成候选片段（便于单测，不依赖 llama-index） ----
    def build_passages(self, query: str, *, top_k: int = GRAPH_TOP_K) -> List[Dict[str, Any]]:
        """返回 ``[{text, score, metadata, entities}, ...]``，按分数降序。"""

        if not self.graph or not self.graph.nodes or not query:
            return []

        linked = link_entities(query, self.graph, max_entities=GRAPH_MAX_ENTITIES_PER_QUERY)
        if not linked:
            return []

        entity_names = [name for name, _score in linked]
        link_scores = {name: score for name, score in linked}
        edges = self.graph.related_edges(entity_names, hops=self.hops)

        # Preserve the nearest query-linked seed for each BFS layer. Filtering
        # only by seed endpoints silently reduces every multi-hop result to 1 hop.
        distances = {name: 0 for name in entity_names}
        queue = deque(entity_names)
        while queue:
            current = queue.popleft()
            depth = distances[current]
            if depth >= self.hops:
                continue
            for neighbor in self.graph.neighbors(current, hops=1):
                if neighbor not in distances:
                    distances[neighbor] = depth + 1
                    queue.append(neighbor)
        nearest = {}
        for name, depth in distances.items():
            seeds = [seed for seed in entity_names if name == seed]
            if not seeds:
                seeds = [seed for seed in entity_names if seed in self.graph.neighbors(name, hops=max(1, depth))]
            if seeds:
                nearest[name] = min(seeds, key=lambda seed: (-link_scores[seed], seed))
        # 按「种子实体」分组：同一实体的关系聚成一段，读起来更连贯
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for edge in edges:
            candidates = [nearest[name] for name in (edge["h"], edge["t"]) if name in nearest]
            if candidates:
                grouped[min(candidates, key=lambda name: (-link_scores[name], name))].append(edge)

        passages: List[Dict[str, Any]] = []
        for anchor, group_edges in grouped.items():
            group_edges.sort(key=lambda e: (-float(e.get("w", 0.0)), e["h"], e["t"]))
            lines: List[str] = []
            evidence_parts: List[str] = []
            chunk_ids: List[str] = []
            doc_ids: List[str] = []
            source_path = ""

            for edge in group_edges[:GRAPH_MAX_NEIGHBORS]:
                line = f"{edge['h']} --{edge['r']}--> {edge['t']}"
                if edge["t"] == anchor and edge["h"] != anchor:
                    line += "（反向）"
                lines.append(line)
                if edge.get("e"):
                    evidence_parts.append(str(edge["e"]))
                for cid in edge.get("c", []):
                    if cid not in chunk_ids:
                        chunk_ids.append(cid)
                for did in edge.get("d", []):
                    if did not in doc_ids:
                        doc_ids.append(did)
                if not source_path and edge.get("s"):
                    source_path = str(edge["s"])

            if not lines:
                continue

            # 权重：链接分数为主 + 关系强度 + 组规模
            strength = sum(float(e.get("w", 0.0)) for e in group_edges[:GRAPH_MAX_NEIGHBORS])
            score = (0.6 * link_scores.get(anchor, 0.3)) + 0.2 * min(1.0, math.log1p(strength) / 3.0) + 0.2 * min(
                1.0, len(group_edges) / 8.0
            )

            text = f"[知识图谱] 与「{anchor}」相关的实体关系：\n" + "\n".join(lines[:GRAPH_MAX_NEIGHBORS])
            if evidence_parts:
                unique_evidence: List[str] = []
                for piece in evidence_parts:
                    if piece not in unique_evidence:
                        unique_evidence.append(piece)
                    if len(unique_evidence) >= 3:
                        break
                text += "\n原文依据：" + " / ".join(unique_evidence)

            passages.append(
                {
                    "text": text,
                    "score": round(score * self.weight, 6),
                    "entities": [anchor] + [n for n in self.graph.neighbors(anchor, hops=self.hops, limit=10)],
                    "metadata": {
                        "source_type": "knowledge_graph",
                        "graph_anchor": anchor,
                        "chunk_ids": chunk_ids[:12],
                        "doc_ids": doc_ids[:8],
                        "source_path": source_path,
                        "relation_count": len(group_edges),
                    },
                }
            )

        passages.sort(key=lambda p: (-p["score"], p["metadata"]["graph_anchor"]))
        return passages[: max(1, top_k)]

    # ---- 对外：NodeWithScore 列表 ----
    def retrieve(self, query: str, top_k: int = GRAPH_TOP_K) -> List[Any]:
        """检索并转换为 llama-index ``NodeWithScore``（无命中时返回空列表）。"""

        passages = self.build_passages(query, top_k=top_k)
        if not passages:
            try:
                from pcb_rag.observability import counter, record_event

                counter("graph.retrieve_empty")
                record_event("graph.retrieve_empty", query=query[:120])
            except Exception:
                pass
            return []

        try:
            from llama_index.core.schema import NodeWithScore, TextNode
        except Exception:
            return []

        results: List[Any] = []
        for passage in passages:
            node = TextNode(
                text=passage["text"],
                metadata={
                    **passage["metadata"],
                    "entities": passage["entities"][:10],
                },
            )
            results.append(NodeWithScore(node=node, score=float(passage["score"])))

        try:
            from pcb_rag.observability import counter

            counter("graph.retrieve_hits", len(results))
        except Exception:
            pass

        return results

    def retrieve_community_summaries(self, query: str, top_k: int = 3) -> List[Any]:
        """全局检索：返回与查询实体最相关的社区摘要（用于「综述类」问题）。"""

        if not self.graph or not self.graph.communities:
            return []

        linked = {name for name, _ in link_entities(query, self.graph, max_entities=GRAPH_MAX_ENTITIES_PER_QUERY)}
        scored: List[Tuple[float, str, Dict[str, Any]]] = []
        for label, community in self.graph.communities.items():
            entities = set(community.get("entities", []))
            overlap = len(entities & linked)
            if overlap == 0:
                continue
            score = overlap / max(1, len(entities)) + 0.1 * math.log1p(len(entities))
            scored.append((score, label, community))

        if not scored:
            return []

        scored.sort(key=lambda x: (-x[0], x[1]))

        try:
            from llama_index.core.schema import NodeWithScore, TextNode
        except Exception:
            return []

        results: List[Any] = []
        for score, label, community in scored[:top_k]:
            text = f"[知识图谱·主题] {label}\n{community.get('summary', '')}"
            node = TextNode(
                text=text,
                metadata={
                    "source_type": "knowledge_graph_community",
                    "community": label,
                    "entities": community.get("entities", [])[:20],
                },
            )
            results.append(NodeWithScore(node=node, score=float(score) * self.weight))
        return results


def build_neighborhood_text(
    graph: KnowledgeGraph,
    anchor: str,
    *,
    max_edges: int = 8,
    max_lines: int = 12,
) -> str:
    """构造某实体的邻域证据文本（纯字符串，不依赖 llama-index）。

    文本构成：
      ``[知识图谱·邻域] 「锚点」：锚点描述``
      ``相关事实：`` 每条边一行，关系描述以括号附在句末；邻接实体首次出现时
      追加一行缩进的"是什么"说明。
      ``原文依据：`` 最多两条原文句子。

    关系描述与实体描述的来源见 :meth:`KnowledgeGraph.add_triple` 与
    :func:`_merge_node_desc`。没有描述的图（规则抽取）产出的文本与旧版一致，
    即多出来的只是"有则附加"的信息，不会因为缺描述而丢结构。
    """

    edges = graph.edges_of(anchor) if graph is not None else []
    if not edges:
        return ""

    lines: List[str] = []
    evidence: List[str] = []
    described: set = set()

    for edge in edges[:max_edges]:
        rel_desc = str(edge.get("rd") or "").strip()
        if rel_desc:
            lines.append(f"{edge['h']} --{edge['r']}--> {edge['t']}（{rel_desc}）")
        else:
            lines.append(f"{edge['h']} --{edge['r']}--> {edge['t']}")

        if edge.get("e"):
            evidence.append(str(edge["e"]))

        other = edge["t"] if edge["h"] == anchor else edge["h"]
        other_node = graph.nodes.get(other)
        desc = str(getattr(other_node, "desc", "") or "").strip()
        if desc and other not in described and len(lines) < max_lines:
            described.add(other)
            lines.append(f"  · {other}：{desc[:80]}")

    if not lines:
        return ""

    anchor_node = graph.nodes.get(anchor)
    anchor_desc = str(getattr(anchor_node, "desc", "") or "").strip()
    head = f"[知识图谱·邻域] 「{anchor}」"
    if anchor_desc:
        head += f"：{anchor_desc[:100]}"

    text = head + "\n相关事实：\n" + "\n".join(lines)
    if evidence:
        text += "\n原文依据：" + " / ".join(evidence[:2])
    return text


def expand_nodes_with_graph(
    nodes: Optional[Sequence[Any]],
    graph: Optional[KnowledgeGraph],
    *,
    max_extra: int = 3,
    hops: int = GRAPH_HOP,
) -> List[Any]:
    """用图扩展已有检索结果：从结果文本中识别实体，补充其邻域事实。

    与 ``GraphRetriever.retrieve`` 的区别：这里不依赖查询词命中实体，
    而是「顺着已召回内容里的实体」找补，适合补全多跳问题的缺失环节。
    """

    if not nodes or graph is None or not graph.nodes or max_extra <= 0:
        return list(nodes or [])

    anchors: List[str] = []
    for item in nodes:
        node = getattr(item, "node", item)
        text = _node_text(node)
        if not text:
            continue
        for name, _score in link_entities(text, graph, max_entities=6):
            if name not in anchors:
                anchors.append(name)
        if len(anchors) >= 6:
            break

    if not anchors:
        return list(nodes)

    additions: List[Any] = []
    seen_edges = set()
    for anchor in anchors[:4]:
        frontier = {anchor}
        visited = {anchor}
        edges = []
        for _ in range(max(1, hops)):
            next_frontier = set()
            for entity in sorted(frontier):
                for edge in graph.edges_of(entity):
                    key = (edge["h"], edge["r"], edge["t"])
                    if key in seen_edges:
                        continue
                    seen_edges.add(key)
                    edges.append(edge)
                    other = edge["t"] if edge["h"] == entity else edge["h"]
                    if other not in visited:
                        visited.add(other)
                        next_frontier.add(other)
            frontier = next_frontier
        if not edges:
            continue
        lines = [f"{edge['h']} --{edge['r']}--> {edge['t']}" for edge in edges[:12]]
        evidence = [str(edge["e"]) for edge in edges if edge.get("e")][:2]
        text = f"[知识图谱·邻域] 「{anchor}」\n相关事实：\n" + "\n".join(lines)
        if evidence:
            text += "\n原文依据：" + " / ".join(evidence)

        try:
            from llama_index.core.schema import NodeWithScore, TextNode
        except Exception:
            break

        node = TextNode(
            text=text,
            metadata={
                "source_type": "knowledge_graph_expand",
                "graph_anchor": anchor,
                "entities": [anchor],
                # 标记为「图证据」：调用方据此给它单独配额，避免它凭相对分数
                # 挤占真实检索结果的 top-k 名额（见 dify_external_api Step 7）
                "graph_evidence": True,
            },
        )
        # 分数取已召回结果的最高分略作衰减，保证排在原结果之后但仍可能进入上下文
        base = max((float(getattr(n, "score", 0.0) or 0.0) for n in nodes), default=0.5)
        additions.append(NodeWithScore(node=node, score=base * 0.75))

        if len(additions) >= max_extra:
            break

    if additions:
        try:
            from pcb_rag.observability import counter

            counter("graph.expanded_nodes", len(additions))
        except Exception:
            pass

    return list(nodes) + additions


# ---------------------------------------------------------------------------
# 9. 融合与诊断
# ---------------------------------------------------------------------------
def fuse_with_graph(
    base_nodes: Optional[Sequence[Any]],
    graph_nodes: Optional[Sequence[Any]],
    *,
    graph_weight: float = GRAPH_WEIGHT,
    rrf_k: int = GRAPH_RRF_K,
    overwrite_scores: bool = True,
    max_total: Optional[int] = None,
) -> List[Any]:
    """把图检索结果作为**独立一路**，用 RRF 与主结果融合。

    早先的实现是「图侧分数 × graph_weight 后与主结果同池按 score 排序」，
    问题在于两路分数量纲根本不可比：

    - 图侧是规则分（实体名长度 / 链接分），量级约 0.2 ~ 1.0；
    - 主结果是加权 RRF 分 ``w/(k+rank)``（k=40, w≈1）≈ 0.0125 ~ 0.024，
      或 rerank 概率 0 ~ 1。

    差两个数量级，于是 ``0.35 × 0.55 = 0.19 ≫ 0.02``：图节点无条件霸榜，
    把真正的检索结果挤下去（CLI 里融合发生在 rerank 之前，后果最严重）。

    RRF 只依赖排名、与绝对分数量纲无关，因此这里改为标准的
    ``score(d) = Σ w_route / (k + rank_route(d))``：
    - 主结果整体作为第 1 路（权重 1.0）
    - 图结果作为第 2 路（权重 ``graph_weight``）

    ``overwrite_scores``：
    - ``True``（默认，CLI 融合前置场景）：把融合分写回 ``item.score``；
    - ``False``（API 在 rerank 之后调用）：只决定顺序、保留原始分数，
      避免把已经量纲良好的 rerank 概率覆盖成 RRF 分。
    """

    base = list(base_nodes or [])
    graph = list(graph_nodes or [])
    if not base and not graph:
        return []
    if not base:
        base, graph = graph, []

    def _key(item: Any) -> str:
        node = getattr(item, "node", item)
        node_id = str(getattr(node, "node_id", "") or "")
        if node_id:
            return node_id
        text = _node_text(node)
        return str(hash(text[:200]))

    merged: List[Any] = []
    seen: Dict[str, int] = {}
    # 融合分单独放一张表，避免中途用 item.score 互相覆盖
    fused: List[float] = []

    def _mark_graph_evidence(item: Any) -> None:
        """标记"这条节点只有图这一路贡献"。

        调用方据此给它单独配额。否则它虽然排名被 RRF 正确压制，却仍会凭自带的
        规则分（0.2~1.0）在"按 score 排序截断 top_k"那一步挤掉真实检索结果。
        """
        try:
            node = getattr(item, "node", item)
            md = getattr(node, "metadata", None)
            if isinstance(md, dict):
                md["graph_evidence"] = True
        except Exception:
            pass

    def _add(item: Any, score: float, *, prefer_score: bool, graph_only: bool = False) -> None:
        key = _key(item)
        if key in seen:
            pos = seen[key]
            if prefer_score:
                old = float(getattr(merged[pos], "score", 0.0) or 0.0)
                new = float(getattr(item, "score", 0.0) or 0.0)
                if new > old:
                    merged[pos] = item
            fused[pos] += score
            return
        if graph_only:
            _mark_graph_evidence(item)
        seen[key] = len(merged)
        merged.append(item)
        fused.append(score)

    for rank, item in enumerate(base, 1):
        _add(item, 1.0 / (rrf_k + rank), prefer_score=True)
    for rank, item in enumerate(graph, 1):
        _add(item, float(graph_weight) / (rrf_k + rank), prefer_score=False, graph_only=True)

    order = sorted(range(len(merged)), key=lambda i: -fused[i])
    result = [merged[i] for i in order]

    if overwrite_scores:
        for i in order:
            try:
                merged[i].score = float(fused[i])
            except Exception:
                pass

    if max_total and len(result) > max_total:
        result = result[:max_total]
    return result


def describe_graph(graph: Optional[KnowledgeGraph] = None) -> Dict[str, Any]:
    """返回图配置与规模摘要，供启动日志与 ``/health`` 展示。"""

    info: Dict[str, Any] = {
        "enabled": GRAPH_RAG_ENABLED,
        "extract_backend": GRAPH_EXTRACT_BACKEND,
        "path": GRAPH_PATH,
        "hop": GRAPH_HOP,
        "top_k": GRAPH_TOP_K,
        "weight": GRAPH_WEIGHT,
        "community_enabled": GRAPH_COMMUNITY_ENABLED,
    }
    if graph is not None:
        info.update(graph.stats())
    return info
