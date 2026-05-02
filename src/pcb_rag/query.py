import os
# 设置 HuggingFace 镜像端点（国内加速）- 必须在任何 HF 相关 import 之前
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import asyncio
import json
import warnings
import re
import site
import sys
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from math import log
from pathlib import Path
from typing import Any, Iterable, Optional

def _bootstrap_local_venv() -> None:
    """允许模块运行时也能找到本项目 `.venv` 里的依赖。"""

    repo_dir = str(Path(__file__).resolve().parents[2])
    venv_site_packages = os.path.join(
        repo_dir,
        ".venv",
        "lib",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
        "site-packages",
    )

    if os.path.isdir(venv_site_packages) and venv_site_packages not in sys.path:
        sys.path.insert(0, venv_site_packages)
        site.addsitedir(venv_site_packages)


_bootstrap_local_venv()


def _sanitize_proxy_env_for_httpx() -> None:
    """修正服务器环境中不被 httpx 识别的代理 scheme。

    常见情况：某些环境会设置 ALL_PROXY=socks://127.0.0.1:10808/
    但 httpx 期望 socks5:// 或 socks5h://，否则会在 import 时直接 ValueError。

    这里做最小改动：
    - socks:// -> socks5://
    - 追加 NO_PROXY，避免本地服务（Ollama/Milvus）走代理
    """

    proxy_keys = [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ]

    for key in proxy_keys:
        val = os.environ.get(key)
        if not val:
            continue
        v = val.strip()
        if v.lower().startswith("socks://"):
            os.environ[key] = "socks5://" + v[len("socks://") :]

    # 避免本地服务走代理（即使设置了代理，也应直连 localhost/127.0.0.1）
    no_proxy_key = "NO_PROXY" if "NO_PROXY" in os.environ else "no_proxy" if "no_proxy" in os.environ else "NO_PROXY"
    existing = os.environ.get(no_proxy_key, "")
    entries = [e.strip() for e in existing.split(",") if e.strip()]
    for host in ("127.0.0.1", "localhost"):
        if host not in entries:
            entries.append(host)
    os.environ[no_proxy_key] = ",".join(entries)


_sanitize_proxy_env_for_httpx()

# Suppress known tokenizer warnings from some reranker implementations
warnings.filterwarnings(
    "ignore",
    message=r"You're using a Qwen2TokenizerFast tokenizer\..*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"`max_length` is ignored when `padding`=`True` and there is no truncation strategy\..*",
    category=UserWarning,
)

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle, TextNode
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

try:
    import nest_asyncio

    nest_asyncio.apply()
except Exception:
    pass

MILVUS_URI = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")
COLLECTION = os.getenv("COLLECTION", "pcb_kb")
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://127.0.0.1:11434")

HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com") 
# Huggingface国内代理

DEFAULT_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))       # LLM 上下文窗口
EMBED_NUM_CTX   = int(os.getenv("EMBED_NUM_CTX",   "2048"))       # Embedding 上下文窗口
DEFAULT_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "300"))

LEXICAL_CACHE_PATH = os.getenv("LEXICAL_CACHE_PATH", "./data/lexical_corpus.jsonl")
LEXICAL_EXPORT_BATCH = int(os.getenv("LEXICAL_EXPORT_BATCH", "2000"))
LEXICAL_EXPORT_LIMIT = int(os.getenv("LEXICAL_EXPORT_LIMIT", "-1"))
LEXICAL_ENABLED = os.getenv("LEXICAL_ENABLED", "1") not in {"0", "false", "False"}
LEXICAL_MAX_BYTES = int(os.getenv("LEXICAL_MAX_BYTES", "8000"))

# 层次化 Chunk 检索参数（group + expand 策略）
CHUNK_EXPAND_ENABLED = os.getenv("CHUNK_EXPAND_ENABLED", "1") not in {"0", "false", "False"}
CHUNK_EXPAND_NEIGHBORS = int(os.getenv("CHUNK_EXPAND_NEIGHBORS", "1"))  # 扩展的邻居数量（前后各N个）
CHUNK_EXPAND_PARENT = os.getenv("CHUNK_EXPAND_PARENT", "1") not in {"0", "false", "False"}  # 是否扩展父chunk（默认启用）
CHUNK_GROUP_BY_DOC = os.getenv("CHUNK_GROUP_BY_DOC", "1") not in {"0", "false", "False"}  # 按文档分组

# Milvus 中，LlamaIndex 会把 Node 的 metadata 展平为“动态字段”（不是一个名为 `metadata` 的 JSON 列）。
# 为了让本地 BM25/评测能够使用到这些元信息，这里显式列出我们关心的字段并在导出时打包为 dict。
LEXICAL_METADATA_KEYS: list[str] = [
    "source_type",
    "source_path",
    "vendor",
    "eda",
    "layer_count",
    "copper_oz",
    "effective_date",
]

# Rerank: 先高召回，再用 cross-encoder 精排，减少“看起来相关但其实不对”的段落
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "1") not in {"0", "false", "False"}
RERANK_MODEL = os.getenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-4B")
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "200"))  # Top200 精排
# rerank 后端：
# - sbert: SentenceTransformerRerank (cross-encoder)
# - hf: 直接用 HuggingFace Transformers 加载 rerank 模型（适合 Ollama 不支持的 rerank 模型）
# - qwen3reranker: Qwen3-Reranker 专用（生成式 reranker，使用 yes/no 分类 logits）
RERANK_BACKEND = os.getenv("RERANK_BACKEND", "qwen3reranker").strip().lower()

# HF rerank 相关（RERANK_BACKEND=hf 时生效）
HF_RERANK_MODEL = os.getenv("HF_RERANK_MODEL", "Qwen/Qwen3-Reranker-4B").strip()
HF_RERANK_MAX_LENGTH = int(os.getenv("HF_RERANK_MAX_LENGTH", "1024"))
HF_RERANK_BATCH_SIZE = int(os.getenv("HF_RERANK_BATCH_SIZE", "16"))
HF_RERANK_DEVICE = os.getenv("HF_RERANK_DEVICE", "auto").strip().lower()  # auto|cuda|cpu
HF_RERANK_DTYPE = os.getenv("HF_RERANK_DTYPE", "auto").strip().lower()  # auto|fp16|bf16|fp32

# 控制 HuggingFace 下载进度条刷新频率（秒）
HF_DOWNLOAD_MIN_INTERVAL = float(os.getenv("HF_DOWNLOAD_MIN_INTERVAL", "1"))
# 对大模型（如 Qwen3-Reranker）是否先 snapshot 预下载（进度可控、可续传）
HF_PREFETCH_SNAPSHOT = os.getenv("HF_PREFETCH_SNAPSHOT", "1") not in {"0", "false", "False"}

# Qwen3 reranker（Qwen/Qwen3-Reranker-*）参数
QWEN3_RERANK_INSTRUCTION = os.getenv(
    "QWEN3_RERANK_INSTRUCTION",
    "Given a search query, retrieve relevant candidates that answer the query.",
).strip()
# Qwen3-Reranker 模型最大上下文长度（官方支持 32k，这里默认用 HF_RERANK_MAX_LENGTH 可单独覆盖）
QWEN3_MAX_LENGTH = int(os.getenv("QWEN3_MAX_LENGTH", str(HF_RERANK_MAX_LENGTH)))
QWEN3_ATTN_IMPL = os.getenv("QWEN3_ATTN_IMPL", "").strip()  # e.g. flash_attention_2

# Reranker GPU 指定（多卡时避免与 Ollama/Embedding 争夺 GPU 0）
# auto = 多卡时自动选 GPU 1，单卡用 GPU 0；也可指定 0/1/cpu
RERANK_GPU_ID = os.getenv("RERANK_GPU_ID", "auto").strip().lower()

# 召回条数（rerank 之前）；候选越多 reranker 越不容易漏掉正确答案
RECALL_TOP_K = int(os.getenv("RECALL_TOP_K", "200"))

# 查询扩展：PCB专业术语同义词和缩写（按优先级排序，越靠前权重越高）
QUERY_EXPANSION_DICT = {
    # 核心术语
    "pcb": ["印制板", "印制电路板", "线路板", "printed circuit board"],
    "印制板": ["PCB", "印制电路板", "线路板"],
    "fpc": ["柔性电路板", "挠性板", "flexible printed circuit"],
    "hdi": ["高密度互连", "high density interconnect"],

    # 工艺相关
    "测试": ["试验", "检验", "检测", "test", "测量"],
    "试验": ["测试", "检验", "检测", "测量"],
    "检测": ["测试", "试验", "检验"],
    "阻燃": ["防火", "耐燃", "抗燃", "flame retardant", "阻燃型"],
    "镀金": ["沉金", "ENIG", "化金"],
    "喷锡": ["HASL", "热风整平"],
    "沉银": ["immersion silver"],
    "osp": ["有机保焊膜", "有机涂覆"],
    "蚀刻": ["etching", "腐蚀", "化学蚀刻"],
    "电镀": ["plating", "镀覆", "镀层"],

    # 材料相关
    "覆铜板": ["CCL", "copper clad laminate", "基材", "铜板"],
    "层压板": ["laminate", "叠层板"],
    "介质": ["dielectric", "绝缘", "介电"],
    "半固化片": ["prepreg", "PP", "粘结片"],
    "铜箔": ["copper foil", "铜皮"],
    "聚酰亚胺": ["PI", "polyimide", "亚胺薄膜"],
    "绝缘": ["insulation", "介质", "绝缘材料"],

    # 结构相关
    "焊盘": ["pad", "land", "盘"],
    "孔环": ["annular ring", "环宽"],
    "过孔": ["via", "导通孔", "通孔"],
    "盲孔": ["blind via", "盲埋孔"],
    "埋孔": ["buried via"],
    "导线": ["trace", "走线", "线路", "导体"],
    "线宽": ["trace width", "走线宽度"],

    # 电气相关
    "阻抗": ["impedance", "特性阻抗"],
    "差分": ["differential", "diff", "差分对"],
    "单端": ["single ended", "single-ended"],
    "带状线": ["stripline"],
    "微带线": ["microstrip"],
    "电阻": ["resistance", "阻值"],
    "击穿": ["breakdown", "击穿电压"],
    "耐压": ["withstand voltage", "电压测试"],

    # 测试测量相关
    "温度": ["temperature", "热", "高温", "低温"],
    "湿度": ["humidity", "潮湿", "湿热"],
    "尺寸": ["dimension", "规格", "size"],
    "厚度": ["thickness", "板厚"],
    "剥离": ["peel", "剥离强度"],
    "翘曲": ["warpage", "弯曲", "变形"],
    
    # 标准格式相关
    "语句": ["statement", "指令", "命令"],
    "属性": ["attribute", "参数", "property"],
    "格式": ["format", "规格", "file format"],
    "定义": ["definition", "规定", "描述"],
    "要求": ["requirement", "规范", "标准"],
    "程序": ["procedure", "步骤", "流程", "方法"],
    
    # 标准相关
    "国标": ["GB", "GB/T", "国家标准"],
    "行标": ["行业标准", "SJ", "SJ/T"],
    "标志": ["mark", "标记", "标识", "marking"],
    "颜色": ["color", "色标"],
}
QUERY_EXPANSION_ENABLED = os.getenv("QUERY_EXPANSION_ENABLED", "1") not in {"0", "false", "False"}
QUERY_EXPANSION_MAX_TERMS = int(os.getenv("QUERY_EXPANSION_MAX_TERMS", "6"))  # 最大扩展词数（增加覆盖面）

# =============================================================================
# Fusion + 多扩展查询 + Rerank 检索配置
# =============================================================================
# mode: RECIPROCAL_RANK (RRF) | DIST_BASED_SCORE (加权融合) | RELATIVE_SCORE
FUSION_MODE = os.getenv("FUSION_MODE", "RECIPROCAL_RANK").strip().upper()
# 权重：[向量检索权重, BM25权重]，平衡语义和精确匹配
FUSION_WEIGHTS = [float(x) for x in os.getenv("FUSION_WEIGHTS", "0.45,0.55").split(",")]
# 查询改写/扩展数量：多扩展查询核心参数，默认 4（平衡速度和召回覆盖）
FUSION_NUM_QUERIES = int(os.getenv("FUSION_NUM_QUERIES", "4"))
# 多扩展查询启用开关（默认启用，增加召回多样性）
MULTI_EXPAND_ENABLED = os.getenv("MULTI_EXPAND_ENABLED", "1") not in {"0", "false", "False"}
# LLM 查询改写启用开关（使用 LLM 生成高质量查询变体）
LLM_QUERY_REWRITE_ENABLED = os.getenv("LLM_QUERY_REWRITE_ENABLED", "0") not in {"0", "false", "False"}

# =============================================================================
# MultiPath Retriever + Late Interaction 配置 (Phase 3.1)
# =============================================================================
# 是否启用多路召回 + Late Interaction 模式（默认关闭，使用 Fusion+Expand+Rerank）
MULTIPATH_ENABLED = os.getenv("MULTIPATH_ENABLED", "0") not in {"0", "false", "False"}
# ColBERT 风格 Late Interaction reranker（注意：不适用于生成式模型如 Qwen3-Reranker）
COLBERT_RERANK_ENABLED = os.getenv("COLBERT_RERANK_ENABLED", "0") not in {"0", "false", "False"}
COLBERT_MODEL = os.getenv("COLBERT_MODEL", "BAAI/bge-reranker-v2-m3").strip()
COLBERT_MAX_LENGTH = int(os.getenv("COLBERT_MAX_LENGTH", "512"))
COLBERT_BATCH_SIZE = int(os.getenv("COLBERT_BATCH_SIZE", "16"))
# 多路召回参数
MULTIPATH_DENSE_TOP_K = int(os.getenv("MULTIPATH_DENSE_TOP_K", "100"))  # 向量召回数量
MULTIPATH_SPARSE_TOP_K = int(os.getenv("MULTIPATH_SPARSE_TOP_K", "100"))  # BM25召回数量
MULTIPATH_RERANK_CANDIDATES = int(os.getenv("MULTIPATH_RERANK_CANDIDATES", "50"))  # 送入 rerank 的候选数
MULTIPATH_RRF_K = int(os.getenv("MULTIPATH_RRF_K", "60"))  # RRF 融合参数 k

# =============================================================================
# HyDE + Query2Doc 查询增强配置 (Phase 3.3)
# =============================================================================
# 是否启用 HyDE (Hypothetical Document Embedding) 查询增强
HYDE_ENABLED = os.getenv("HYDE_ENABLED", "1") not in {"0", "false", "False"}  # 默认启用
# HyDE 假设文档最大长度（字符数）
HYDE_MAX_LENGTH = int(os.getenv("HYDE_MAX_LENGTH", "300"))
# HyDE 路降权系数（用于三路 RRF 融合：原始向量=1.0，HyDE向量=HYDE_ROUTE_WEIGHT）
# 提升 HyDE 路权重，因为 HyDE 生成的假设文档能显著提升语义匹配
HYDE_ROUTE_WEIGHT = float(os.getenv("HYDE_ROUTE_WEIGHT", "0.85"))
# 三路 RRF 融合参数（k 越小越倾向 top 排名，k=40 对 PCB 领域更佳）
FUSION_RRF_K = int(os.getenv("FUSION_RRF_K", "40"))
# HyDE few-shot 开关
HYDE_FEWSHOT_ENABLED = os.getenv("HYDE_FEWSHOT_ENABLED", "1") not in {"0", "false", "False"}
# 是否启用 Query2Doc 查询增强
QUERY2DOC_ENABLED = os.getenv("QUERY2DOC_ENABLED", "0") not in {"0", "false", "False"}  # 默认关闭（按当前策略禁用）
# Query2Doc 关键信息点最大数量
QUERY2DOC_MAX_POINTS = int(os.getenv("QUERY2DOC_MAX_POINTS", "5"))
# 短查询阈值（低于此长度才使用 HyDE/Query2Doc）
# 设置较高值使大多数查询都能受益于 HyDE 增强
SHORT_QUERY_THRESHOLD = int(os.getenv("SHORT_QUERY_THRESHOLD", "120"))
# 查询增强是否对所有查询生效（默认开启；关闭后仅短查询启用）
QUERY_ENHANCE_ALL = os.getenv("QUERY_ENHANCE_ALL", "1") not in {"0", "false", "False"}


def _gpu_free_mib() -> dict[int, int]:
    """返回各 GPU 空闲显存 (MiB)。失败时返回空 dict。"""
    try:
        import torch
        free: dict[int, int] = {}
        for i in range(torch.cuda.device_count()):
            f, _ = torch.cuda.mem_get_info(i)
            free[i] = int(f / (1024 * 1024))
        return free
    except Exception:
        return {}


def _try_build_reranker():
    if not RERANK_ENABLED:
        return None

    if RERANK_BACKEND in {"hf", "qwen3reranker"} or "qwen3-reranker" in HF_RERANK_MODEL.lower():
        model_id = HF_RERANK_MODEL
        # Qwen3-Reranker 是生成式 CausalLM 模型，使用 yes/no logits 评分；需走专用 reranker 逻辑
        if RERANK_BACKEND == "qwen3reranker" or "qwen3-reranker" in model_id.lower():
            # 预检 GPU 显存
            free = _gpu_free_mib()
            if free:
                best_gpu = max(free, key=free.get)  # type: ignore[arg-type]
                best_free = free[best_gpu]
                print(f"[RERANK] GPU 显存: " + ", ".join(f"GPU{k}={v}MiB free" for k, v in sorted(free.items())))
                if best_free < 4000:  # <4GB 连 4B 模型都放不下
                    print(f"[RERANK] ⚠️ 所有 GPU 均无足够显存 (最大空闲 {best_free}MiB < 4000MiB)")
                    print(f"[RERANK] 提示: 运行 `ollama ps` 检查，`ollama stop <model>` 释放显存")
                    return None
            try:
                return Qwen3Reranker(
                    top_n=RERANK_TOP_N,
                    model=model_id,
                    max_length=QWEN3_MAX_LENGTH,
                    instruction=QWEN3_RERANK_INSTRUCTION,
                    attn_implementation=QWEN3_ATTN_IMPL or None,
                )
            except Exception as e:
                import traceback
                print(f"[RERANK] Error loading Qwen3Reranker: {e}")
                traceback.print_exc()
                # 显示 GPU 状态帮助诊断 OOM
                free_after = _gpu_free_mib()
                if free_after:
                    print(f"[RERANK] 当前 GPU 显存: " + ", ".join(f"GPU{k}={v}MiB" for k, v in sorted(free_after.items())))
                    print(f"[RERANK] 提示: 运行 `ollama stop <model>` 释放显存后重试")
                return None

    if RERANK_BACKEND == "hf":
        try:
            return TransformersRerank(
                top_n=RERANK_TOP_N,
                model=HF_RERANK_MODEL,
                max_length=HF_RERANK_MAX_LENGTH,
                batch_size=HF_RERANK_BATCH_SIZE,
                device=HF_RERANK_DEVICE,
                dtype=HF_RERANK_DTYPE,
            )
        except Exception:
            return None

    # 默认：SentenceTransformer cross-encoder rerank
    try:
        from llama_index.core.postprocessor import SentenceTransformerRerank

        return SentenceTransformerRerank(top_n=RERANK_TOP_N, model=RERANK_MODEL)
    except Exception:
        return None


class TransformersRerank(BaseNodePostprocessor):
    model: str = Field(description="HuggingFace model id")
    top_n: int = Field(description="Number of nodes to return sorted by score.")
    max_length: int = Field(default=512, description="Tokenizer max_length")
    batch_size: int = Field(default=16, description="Batch size for rerank scoring")
    device: str = Field(default="auto", description="auto|cuda|cpu")
    dtype: str = Field(default="auto", description="auto|fp16|bf16|fp32")
    trust_remote_code: bool = Field(default=True, description="Whether to trust remote code.")

    _tokenizer: Any = PrivateAttr()
    _model: Any = PrivateAttr()
    _torch_device: Any = PrivateAttr()

    def __init__(
        self,
        top_n: int,
        model: str,
        max_length: int = 512,
        batch_size: int = 16,
        device: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = True,
    ):
        super().__init__(
            top_n=top_n,
            model=model,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except Exception as e:
            raise ImportError(
                "Cannot import transformers/torch for HF rerank. "
                "Please `pip install transformers torch accelerate`"
            ) from e

        # pick device
        if device == "cuda":
            torch_device = torch.device("cuda")
        elif device == "cpu":
            torch_device = torch.device("cpu")
        else:
            torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # pick dtype
        if torch_device.type == "cuda":
            if dtype == "bf16":
                torch_dtype = torch.bfloat16
            elif dtype in {"fp32", "float32"}:
                torch_dtype = torch.float32
            else:
                torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        self._torch_device = torch_device
        self._tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)

        # NOTE: 对于某些 reranker/VL reranker，可能需要 trust_remote_code 才能 load。
        # 这里优先按 sequence-classification 的 cross-encoder 方式加载；失败由外层回退。
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self._model.eval()
        self._model.to(torch_device)

    @classmethod
    def class_name(cls) -> str:
        return "TransformersRerank"

    def _score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        import torch

        scores: list[float] = []
        bs = max(int(self.batch_size), 1)
        for i in range(0, len(pairs), bs):
            batch = pairs[i : i + bs]
            q_list = [q for q, _ in batch]
            p_list = [p for _, p in batch]
            enc = self._tokenizer(
                q_list,
                p_list,
                padding=True,
                truncation=True,
                max_length=int(self.max_length),
                return_tensors="pt",
            )
            enc = {k: v.to(self._torch_device) for k, v in enc.items()}
            with torch.no_grad():
                out = self._model(**enc)
                logits = getattr(out, "logits", None)
                if logits is None:
                    raise RuntimeError("HF rerank model output missing logits")
                # logits: [B, 1] or [B, 2] typically
                if logits.dim() == 2 and logits.size(-1) == 1:
                    batch_scores = logits.squeeze(-1)
                elif logits.dim() == 2 and logits.size(-1) >= 2:
                    batch_scores = logits[:, 1]
                else:
                    batch_scores = logits.reshape(logits.size(0), -1).max(dim=-1).values
                scores.extend(batch_scores.detach().float().cpu().tolist())
        return [float(s) for s in scores]

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> list[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if not nodes:
            return []

        query_and_nodes = [
            (
                query_bundle.query_str,
                node.node.get_content(metadata_mode=MetadataMode.EMBED),
            )
            for node in nodes
        ]
        scores = self._score_pairs(query_and_nodes)
        if len(scores) != len(nodes):
            raise RuntimeError("HF rerank scores length mismatch")

        for node, score in zip(nodes, scores):
            node.score = score
        return sorted(nodes, key=lambda x: -x.score if x.score else 0)[: self.top_n]


def _snapshot_prefetch(repo_id: str) -> str:
    """用 snapshot_download 预下载整个 repo，并把进度刷新频率固定为每秒一次。"""

    from huggingface_hub import snapshot_download
    from tqdm.auto import tqdm as _tqdm

    class _Tqdm1s(_tqdm):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("mininterval", HF_DOWNLOAD_MIN_INTERVAL)
            super().__init__(*args, **kwargs)

    return snapshot_download(
        repo_id=repo_id,
        tqdm_class=_Tqdm1s,
    )


class Qwen3Reranker(BaseNodePostprocessor):
    """Qwen3-Reranker 生成式 reranker。

    使用 AutoModelForCausalLM 加载模型，通过对最后一个 token 的 "yes"/"no" logits
    计算相关性分数（官方 Transformers 推理方式，需 transformers>=4.51.0）。
    """

    model: str = Field(description="Qwen3 reranker model id")
    top_n: int = Field(description="Number of nodes to return sorted by score.")
    max_length: int = Field(default=8192, description="Max token length for reranker")
    instruction: str = Field(default=QWEN3_RERANK_INSTRUCTION, description="Rerank instruction")
    attn_implementation: Optional[str] = Field(default=None, description="Attention implementation")

    _tokenizer: Any = PrivateAttr()
    _model: Any = PrivateAttr()
    _device: Any = PrivateAttr()
    _input_device: Any = PrivateAttr()  # embedding 层所在设备，用于推理时移动输入 tensor
    _prefix_tokens: Any = PrivateAttr()
    _suffix_tokens: Any = PrivateAttr()
    _token_true_id: int = PrivateAttr()
    _token_false_id: int = PrivateAttr()

    def __init__(
        self,
        top_n: int,
        model: str,
        max_length: int = 8192,
        instruction: str = QWEN3_RERANK_INSTRUCTION,
        attn_implementation: Optional[str] = None,
    ):
        super().__init__(
            top_n=top_n,
            model=model,
            max_length=max_length,
            instruction=instruction,
            attn_implementation=attn_implementation,
        )

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Cannot import transformers/torch for Qwen3Reranker. "
                "Please `pip install 'transformers>=4.51.0' torch accelerate`"
            ) from e

        # ---------- GPU 选择 ----------
        # device_map 取值：
        #   "auto"          → 自动选择空闲显存最多的 GPU 单卡加载
        #   "cuda:<id>"     → 强制单卡
        #   "cpu"           → CPU
        target_gpu_id: int | None = None
        if torch.cuda.is_available():
            gpu_spec = RERANK_GPU_ID
            n_gpus = torch.cuda.device_count()
            if gpu_spec == "cpu":
                pass
            elif gpu_spec == "auto":
                free = _gpu_free_mib()
                if free:
                    target_gpu_id = max(free, key=free.get)
                    print(f"[RERANK] GPU 显存: " + ", ".join(f"GPU{k}={v}MiB free" for k, v in sorted(free.items())))
                    print(f"[RERANK] 选择空闲显存最多的 GPU {target_gpu_id} ({free[target_gpu_id]}MiB free)")
                else:
                    target_gpu_id = 0
                    print(f"[RERANK] 无法获取显存信息，默认加载到 GPU 0")
            elif gpu_spec.isdigit():
                target_gpu_id = int(gpu_spec)
                print(f"[RERANK] 将 Reranker 加载到 GPU {target_gpu_id}")
            elif ":" in gpu_spec:  # cuda:1
                target_gpu_id = int(gpu_spec.split(":")[-1])
                print(f"[RERANK] 将 Reranker 加载到 GPU {target_gpu_id}")
            else:
                target_gpu_id = 0
                print(f"[RERANK] 将 Reranker 加载到 GPU 0")

        # ---------- 预下载 ----------
        model_path = model
        if HF_PREFETCH_SNAPSHOT and not os.path.isdir(model):
            model_path = _snapshot_prefetch(model)

        # ---------- 构建加载参数 ----------
        load_kwargs: dict[str, Any] = {}
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation

        if not torch.cuda.is_available():
            torch_dtype = torch.float32
            device_map = "cpu"
        elif target_gpu_id is not None:
            torch_dtype = torch.bfloat16
            device_map = f"cuda:{target_gpu_id}"
        else:
            torch_dtype = torch.float32
            device_map = "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            **load_kwargs,
        ).eval()

        self._device = self._model.device
        self._input_device = next(self._model.parameters()).device

        # ---------- 构建 prefix/suffix tokens（官方模板） ----------
        # prefix: system prompt + user turn 开头
        # suffix: assistant 回复引导（含 <think></think> 短路思维链占位）
        prefix = (
            "<|im_start|>system\nJudge whether the Document meets the requirements based on the "
            'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
            "<|im_end|>\n<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self._prefix_tokens = self._tokenizer.encode(prefix, add_special_tokens=False)
        self._suffix_tokens = self._tokenizer.encode(suffix, add_special_tokens=False)
        self._token_true_id = self._tokenizer.convert_tokens_to_ids("yes")
        self._token_false_id = self._tokenizer.convert_tokens_to_ids("no")

    @classmethod
    def class_name(cls) -> str:
        return "Qwen3Reranker"

    def _format_pair(self, query: str, doc: str) -> str:
        """将 instruction/query/doc 格式化为模型输入文本（body 部分）。"""
        return (
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {doc}"
        )

    def _score_pairs(self, pairs: list[tuple[str, str]], batch_size: int = 8) -> list[float]:
        """批量计算 (query, doc) 对的相关性分数。"""
        import torch

        scores: list[float] = []
        max_len = int(self.max_length)
        body_max = max_len - len(self._prefix_tokens) - len(self._suffix_tokens)

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i: i + batch_size]
            texts = [self._format_pair(q, d) for q, d in batch]

            # tokenize body（留出 prefix/suffix 的 token 空间）
            encoded = self._tokenizer(
                texts,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=body_max,
            )
            # 手动拼接 prefix_tokens + body_tokens + suffix_tokens
            for j, ids in enumerate(encoded["input_ids"]):
                encoded["input_ids"][j] = self._prefix_tokens + ids + self._suffix_tokens

            # pad 成等长 batch tensor
            inputs = self._tokenizer.pad(
                encoded,
                padding=True,
                return_tensors="pt",
                max_length=max_len,
            )
            inputs = {k: v.to(self._input_device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self._model(**inputs).logits[:, -1, :]  # [B, vocab]
                # device_map 多卡时 logits 在最后层所在卡，统一移回 input_device 做后续运算
                logits = logits.to(self._input_device)

            true_vec = logits[:, self._token_true_id]
            false_vec = logits[:, self._token_false_id]
            stacked = torch.stack([false_vec, true_vec], dim=1)
            log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
            batch_scores = log_probs[:, 1].exp().tolist()
            scores.extend(batch_scores)

        return [float(s) for s in scores]

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> list[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if not nodes:
            return []

        pairs = [
            (
                query_bundle.query_str,
                node.node.get_content(metadata_mode=MetadataMode.EMBED),
            )
            for node in nodes
        ]
        scores = self._score_pairs(pairs)
        if len(scores) != len(nodes):
            raise RuntimeError("Qwen3 rerank scores length mismatch")
        for node, score in zip(nodes, scores):
            node.score = float(score)
        return sorted(nodes, key=lambda x: -x.score if x.score else 0)[: self.top_n]


def _ollama_list_models(base_url: str) -> list[dict]:
    url = base_url.rstrip("/") + "/api/tags"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = data.get("models")
            return models if isinstance(models, list) else []
    except Exception:
        return []


def _pick_llm_candidates(base_url: str) -> list[str]:
    env_model = os.getenv("OLLAMA_LLM_MODEL", "").strip()
    tags = _ollama_list_models(base_url)

    candidates: list[tuple[int, str]] = []
    for m in tags:
        name = (m.get("name") or m.get("model") or "").strip()
        if not name:
            continue
        if "embedding" in name.lower():
            continue
        size = m.get("size")
        try:
            size_int = int(size)
        except Exception:
            size_int = 1 << 62
        candidates.append((size_int, name))

    candidates.sort(key=lambda x: x[0])
    names = [n for _, n in candidates]

    # 若用户指定模型，强制置顶（即使不在 tags 列表里）
    if env_model:
        names = [env_model] + [n for n in names if n != env_model]
    return names


def _configure_llm(model_name: str) -> None:
    Settings.llm = Ollama(
        model=model_name,
        base_url=OLLAMA_BASE,
        request_timeout=DEFAULT_TIMEOUT,
        context_window=DEFAULT_NUM_CTX,
        additional_kwargs={
            "num_ctx": DEFAULT_NUM_CTX,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "repeat_penalty": 1.0,
            "presence_penalty": 1.5,
        },
        thinking=False,
    )


def _preprocess_query(query: str) -> str:
    """预处理查询：清理特殊字符，提取核心信息。
    
    处理以下情况：
    1. XML/HTML 标签：<tag> -> tag
    2. 移除多余空白
    3. 统一标点符号
    """
    if not query:
        return query
    
    import re
    
    # 保留 XML 标签内的内容用于检索（如 <design_rule_area> -> design_rule_area）
    # 同时保留原始形式用于精确匹配
    processed = query
    
    # 提取 XML 标签名并添加为扩展（不替换原查询）
    xml_tags = re.findall(r'<([a-zA-Z_][a-zA-Z0-9_:.-]*)>', query)
    if xml_tags:
        # 将标签名添加到查询中（用于 BM25 匹配）
        tag_terms = ' '.join(set(xml_tags))
        processed = f"{query} {tag_terms}"
    
    # 清理多余空白
    processed = re.sub(r'\s+', ' ', processed).strip()
    
    return processed


def _expand_query(query: str) -> str:
    """智能查询扩展：添加同义词和缩写以提升召回。

    改进点：
    1. 按匹配位置排序：越靠前的匹配词扩展优先级越高
    2. 避免重复扩展：已在查询中的词不再添加
    3. 智能截断：根据查询长度动态调整扩展数量
    4. 保护专业术语：不对已有的专业术语进行替换

    例: "PCB测试" -> "PCB测试 印制板 试验 检验"
    """
    if not QUERY_EXPANSION_ENABLED or not query:
        return query

    expanded_terms: list[tuple[int, str]] = []  # (position, term)
    query_lower = query.lower()

    for term, expansions in QUERY_EXPANSION_DICT.items():
        # 查找术语在查询中的位置
        term_lower = term.lower()
        pos = query_lower.find(term_lower)
        if pos == -1 and term in query:
            pos = query.find(term)
        if pos == -1:
            continue

        # 为每个扩展词计算优先级（位置越靠前、扩展列表越靠前优先级越高）
        for idx, exp in enumerate(expansions):
            exp_lower = exp.lower()
            # 避免重复添加
            if exp_lower in query_lower or exp in query:
                continue
            # 避免添加已有扩展
            if any(exp_lower == e[1].lower() for e in expanded_terms):
                continue
            # 优先级 = 位置 * 100 + 扩展索引
            priority = pos * 100 + idx
            expanded_terms.append((priority, exp))

    if not expanded_terms:
        return query

    # 按优先级排序
    expanded_terms.sort(key=lambda x: x[0])

    # 根据原始查询长度动态调整扩展数量
    max_terms = QUERY_EXPANSION_MAX_TERMS
    if len(query) > 30:
        max_terms = min(max_terms, 3)  # 长查询少扩展
    elif len(query) < 10:
        max_terms = min(max_terms + 2, 7)  # 短查询多扩展

    selected = [t[1] for t in expanded_terms[:max_terms]]
    return query + " " + " ".join(selected)


# =============================================================================
# LLM 查询改写优化 (Phase 1.4)
# =============================================================================

# LLM 查询改写 Prompt 模板
QUERY_REWRITE_PROMPT = """你是 PCB 电路板领域的专家。用户提出了一个问题，请生成 {num_queries} 个语义相似但表达不同的查询，用于检索相关文档。

要求：
1. 保持原始查询的核心意图
2. 使用不同的表达方式（同义词、专业术语、口语化表达）
3. 考虑可能的上下文和背景知识
4. 每个查询独立成行
5. 不要添加编号或其他前缀

原始查询：{query}

生成的查询（每行一个）："""


def llm_rewrite_query(query: str, num_queries: int = 4, llm=None) -> list[str]:
    """使用 LLM 生成高质量的查询变体。
    
    通过 LLM 理解查询意图，生成语义相似但表达不同的查询变体，
    提升检索的召回多样性。
    
    Args:
        query: 原始查询
        num_queries: 生成的查询变体数量（不含原始查询）
        llm: LLM 实例（如果为 None，使用 Settings.llm）
        
    Returns:
        查询变体列表（包含原始查询）
    """
    if llm is None:
        llm = Settings.llm
    
    if not query or not query.strip():
        return [query]
    
    try:
        prompt = QUERY_REWRITE_PROMPT.format(num_queries=num_queries, query=query)
        response = llm.complete(prompt)
        response_text = response.text if hasattr(response, 'text') else str(response)
        
        # 解析生成的查询变体
        variants = [query]  # 原始查询放在第一位
        
        for line in response_text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            # 去除可能的编号前缀（如 "1. ", "- ", "• " 等）
            cleaned = re.sub(r'^[\d\.\-\*\•\·\s]+', '', line).strip()
            if cleaned and cleaned != query and cleaned.lower() != query.lower():
                # 确保变体与原查询不完全相同
                if cleaned not in variants and cleaned.lower() not in [v.lower() for v in variants]:
                    variants.append(cleaned)
        
        # 如果 LLM 生成的变体不足，补充规则生成的变体
        if len(variants) < num_queries + 1:
            rule_variants = _build_rule_based_variants(query, num_queries + 1 - len(variants))
            for rv in rule_variants:
                if rv not in variants and rv.lower() not in [v.lower() for v in variants]:
                    variants.append(rv)
        
        return variants[:num_queries + 1]  # 返回原始查询 + num_queries 个变体
        
    except Exception as e:
        print(f"[LLM Rewrite] 查询改写失败: {e}")
        # 降级到规则生成
        return _build_rule_based_variants(query, num_queries + 1)


def _build_rule_based_variants(query: str, num_variants: int) -> list[str]:
    """基于规则生成查询变体（作为 LLM 改写的降级方案）。
    
    Args:
        query: 原始查询
        num_variants: 需要的变体数量
        
    Returns:
        查询变体列表
    """
    variants: list[str] = [query]
    query_lower = query.lower()
    
    # 1. 基于扩展词典生成变体
    for term, expansions in QUERY_EXPANSION_DICT.items():
        if len(variants) >= num_variants:
            break
        term_lower = term.lower()
        if term_lower in query_lower or term in query:
            for exp in expansions[:2]:
                if len(variants) >= num_variants:
                    break
                variant = query.replace(term, exp) if term in query else query_lower.replace(term_lower, exp.lower())
                if variant not in variants and variant.lower() not in [v.lower() for v in variants]:
                    variants.append(variant)
    
    # 2. 简化查询
    if len(variants) < num_variants:
        simplified = re.sub(r'(请问|请|帮我|我想|想要|想知道|如何|怎么|怎样)', '', query).strip()
        if simplified and simplified != query and simplified not in variants:
            variants.append(simplified)
    
    # 3. 关键词提取
    if len(variants) < num_variants:
        keywords = re.findall(r'[\u4e00-\u9fff]{2,}|[A-Za-z0-9]+', query)
        if keywords and len(keywords) >= 2:
            keyword_query = ' '.join(keywords[:5])
            if keyword_query not in variants:
                variants.append(keyword_query)
    
    return variants[:num_variants]


def _build_multi_expand_queries(query: str, num_queries: int = 7) -> list[str]:
    """构建多扩展查询列表，用于Fusion检索。
    
    生成多种查询变体以提升召回多样性：
    - 如果启用 LLM 改写（LLM_QUERY_REWRITE_ENABLED=1），使用 LLM 生成高质量变体
    - 否则使用规则生成变体：
      1. 原始查询
      2. 预处理后的查询（处理 XML 标签等）
      3. 扩展后的查询（添加同义词）
      4. 基于扩展词典的变体查询
      5. 简化查询（去除修饰词）
      6. 关键词查询
      7. 核心实体查询（提取关键名词短语）
    
    Args:
        query: 原始查询
        num_queries: 生成的查询数量
        
    Returns:
        查询变体列表
    """
    if not query:
        return [query]
    
    # 如果启用 LLM 查询改写，使用 LLM 生成高质量变体
    if LLM_QUERY_REWRITE_ENABLED:
        try:
            variants = llm_rewrite_query(query, num_queries=num_queries - 1)
            if len(variants) >= 2:  # 至少有原始查询 + 1个变体
                print(f"[LLM Rewrite] 生成 {len(variants)} 个查询变体")
                return variants[:num_queries]
        except Exception as e:
            print(f"[LLM Rewrite] 降级到规则生成: {e}")
    
    # 规则生成模式
    variants: list[str] = [query]  # 原始查询
    query_lower = query.lower()
    
    # 0. 预处理后的查询（处理 XML 标签等特殊情况）
    preprocessed = _preprocess_query(query)
    if preprocessed != query and preprocessed not in variants:
        variants.append(preprocessed)
    
    # 1. 添加扩展后的查询
    expanded = _expand_query(query)
    if expanded != query and expanded not in variants:
        variants.append(expanded)
    
    # 2. 基于扩展词典生成变体查询
    for term, expansions in QUERY_EXPANSION_DICT.items():
        if len(variants) >= num_queries:
            break
        term_lower = term.lower()
        if term_lower in query_lower or term in query:
            for exp in expansions[:2]:  # 每个术语最多2个扩展变体
                if len(variants) >= num_queries:
                    break
                # 替换术语生成新变体
                variant = query.replace(term, exp) if term in query else query_lower.replace(term_lower, exp.lower())
                if variant not in variants and variant.lower() not in [v.lower() for v in variants]:
                    variants.append(variant)
    
    # 3. 添加简化版查询（去除常见修饰词）
    if len(variants) < num_queries:
        simplified = re.sub(r'(请问|请|帮我|我想|想要|想知道|如何|怎么|怎样|在|中|的|时)', '', query).strip()
        if simplified and simplified != query and len(simplified) >= 4 and simplified not in variants:
            variants.append(simplified)
    
    # 4. 添加关键词提取查询（提取核心名词）
    if len(variants) < num_queries:
        # 提取中文词组和英文单词
        keywords = re.findall(r'[\u4e00-\u9fff]{2,}|[A-Za-z0-9]+', query)
        if keywords and len(keywords) >= 2:
            keyword_query = ' '.join(keywords[:5])
            if keyword_query not in variants:
                variants.append(keyword_query)
    
    # 5. 核心实体聚焦查询：提取疑似专业名词和实体
    if len(variants) < num_queries:
        # 提取带引号、书名号或大写的实体
        entities = re.findall(r'[【】《》""'']+([^【】《》""'']+)[【】《》""'']+|([A-Z][A-Za-z0-9_\-]+)', query)
        entity_list = [e[0] or e[1] for e in entities if e[0] or e[1]]
        if entity_list:
            entity_query = ' '.join(entity_list)
            if entity_query not in variants and len(entity_query) >= 3:
                variants.append(entity_query)
    
    return variants[:num_queries]


# =============================================================================
# HyDE + Query2Doc 查询增强实现 (Phase 3.3)
# =============================================================================

# HyDE Few-shot 示例（用于稳定生成质量，减少幻觉/发散）
HYDE_FEWSHOT_EXAMPLES = """示例1
问题：印制板导线电阻测试时，试样应满足哪些前处理条件？
输出：关键事实：试样应在(23±2)℃、相对湿度(50±10)%环境下预处理不少于24h；测试前检查导线无明显氧化、污染和机械损伤；按标准规定长度与截面积记录试样参数。

示例2
问题：挠性覆铜板选型时需要重点核对哪些参数？
输出：文档摘要：挠性覆铜板选型应核对基材类型（PI/PET）、铜箔厚度、胶层体系、剥离强度、耐热等级和尺寸稳定性；对动态弯折场景需优先验证弯折寿命与热冲击后电性能保持率。

示例3
问题：多层板层压后如何判定是否需要返工？
输出：关键事实：应检查层间对位偏差、孔位偏移、分层/空洞、板厚公差及翘曲度；若超出规范限值，需按工艺文件执行返工或报废判定，并保留过程记录。"""

# HyDE Prompt 模板（强约束：必须以“文档摘要：”或“关键事实：”开头）
HYDE_PROMPT_TEMPLATE = """你是 PCB 电路板领域的技术专家。请根据问题生成“用于检索”的高质量技术片段。

请严格遵守：
1) 输出只能是一段文本，且必须以“文档摘要：”或“关键事实：”开头。
2) 内容必须使用 PCB 领域专业术语，聚焦可验证的工艺/标准/参数事实。
3) 严禁编造具体标准号、具体数值、具体结论；若不确定，用“应核对相关标准条款/工艺文件”表达。
4) 长度控制在 100-220 字。

{fewshot_examples}

问题：{query}

输出："""


def _normalize_hyde_output(text: str) -> str:
    content = (text or "").strip()
    # 移除 Qwen3.5 可能残留的 <think>...</think> 标签
    content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
    if not content:
        return ""

    content = re.sub(r"^\s*(输出|回答)\s*[:：]\s*", "", content)
    content = re.sub(r"^\s*(文档摘要|关键事实)\s*[:：]\s*", "", content)
    content = content.strip()
    if not content:
        return ""
    return f"关键事实：{content}"

# Query2Doc Prompt 模板
QUERY2DOC_PROMPT_TEMPLATE = """你是 PCB 电路板领域的技术专家。请分析以下问题，列出回答这个问题可能需要查找的关键信息点。

要求：
1. 每个信息点一行
2. 最多列出 {max_points} 个关键信息点
3. 使用专业术语
4. 只输出信息点，不要有编号或解释

问题：{query}

关键信息点："""


def _is_short_query(query: str) -> bool:
    """判断是否为短查询。
    
    短查询的特征：
    1. 字符长度小于阈值
    2. 或者词数少于 3 个
    """
    if len(query) <= SHORT_QUERY_THRESHOLD:
        return True
    # 统计有意义的词（中文词组 + 英文单词）
    words = re.findall(r'[\u4e00-\u9fff]{2,}|[A-Za-z0-9]+', query)
    return len(words) < 4


def hyde_expand_query(query: str, llm=None) -> str:
    """HyDE (Hypothetical Document Embedding) 查询增强。
    
    通过 LLM 生成一个假设的文档片段，该片段可能包含问题的答案。
    然后将原始查询与假设文档结合，用于向量检索。
    
    原理：假设文档的 embedding 会更接近实际相关文档的 embedding，
    从而提升向量检索的召回效果。
    
    Args:
        query: 原始查询
        llm: LLM 实例（如果为 None，使用 Settings.llm）
        
    Returns:
        增强后的查询字符串
    """
    if not HYDE_ENABLED:
        return query
    
    # 取消短查询限制：对所有查询都应用 HyDE（提升语义召回）
    # 但仍检查 QUERY_ENHANCE_ALL 配置
    if not QUERY_ENHANCE_ALL and not _is_short_query(query):
        return query
    
    try:
        if llm is None:
            llm = Settings.llm

        fewshot_examples = HYDE_FEWSHOT_EXAMPLES if HYDE_FEWSHOT_ENABLED else ""
        prompt = HYDE_PROMPT_TEMPLATE.format(query=query, fewshot_examples=fewshot_examples)
        response = llm.complete(prompt)
        hypothetical_doc = _normalize_hyde_output(response.text)
        if not hypothetical_doc:
            return query
        
        # 截断过长的假设文档
        if len(hypothetical_doc) > HYDE_MAX_LENGTH:
            hypothetical_doc = hypothetical_doc[:HYDE_MAX_LENGTH] + "..."
        
        # 结合原始查询和假设文档
        enhanced_query = f"{query}\n\n{hypothetical_doc}"
        return enhanced_query
        
    except Exception as e:
        print(f"[HyDE] 生成假设文档失败: {e}")
        return query


def query2doc_expand(query: str, llm=None) -> str:
    """Query2Doc 查询增强。
    
    通过 LLM 提取回答问题所需的关键信息点，
    将这些信息点作为查询的补充，提升检索的精确度。
    
    与 HyDE 的区别：
    - HyDE 生成完整的假设文档，更适合语义检索
    - Query2Doc 生成关键词列表，更适合 BM25 检索
    
    Args:
        query: 原始查询
        llm: LLM 实例（如果为 None，使用 Settings.llm）
        
    Returns:
        增强后的查询字符串
    """
    if not QUERY2DOC_ENABLED:
        return query
    
    if not QUERY_ENHANCE_ALL and not _is_short_query(query):
        return query  # 仅在增强所有查询或短查询时启用
    
    try:
        if llm is None:
            llm = Settings.llm
        
        prompt = QUERY2DOC_PROMPT_TEMPLATE.format(
            query=query, 
            max_points=QUERY2DOC_MAX_POINTS
        )
        response = llm.complete(prompt)
        key_points = response.text.strip()
        
        # 清理关键信息点（去除编号、空行等）
        lines = [line.strip() for line in key_points.split('\n') if line.strip()]
        # 去除可能的编号
        cleaned_lines = []
        for line in lines[:QUERY2DOC_MAX_POINTS]:
            # 去除开头的数字、点、横线等
            cleaned = re.sub(r'^[\d\.\-\*\•\·\s]+', '', line).strip()
            if cleaned:
                cleaned_lines.append(cleaned)
        
        if cleaned_lines:
            key_points_str = ' '.join(cleaned_lines)
            enhanced_query = f"{query} {key_points_str}"
            return enhanced_query
        
        return query
        
    except Exception as e:
        print(f"[Query2Doc] 提取关键信息点失败: {e}")
        return query


def hybrid_query_enhance(query: str, llm=None, prefer_hyde: bool = True) -> tuple[str, str]:
    """混合查询增强：同时生成 HyDE 和 Query2Doc 增强版本。
    
    返回两个增强查询：
    1. 用于向量检索的增强查询（HyDE）
    2. 用于 BM25 检索的增强查询（Query2Doc）
    
    Args:
        query: 原始查询
        llm: LLM 实例
        prefer_hyde: 对于向量检索，是否优先使用 HyDE（否则使用原始查询）
        
    Returns:
        (向量检索查询, BM25检索查询)
    """
    should_enhance = QUERY_ENHANCE_ALL or _is_short_query(query)
    if not should_enhance:
        return query, query
    
    # 向量检索用 HyDE
    vector_query = hyde_expand_query(query, llm) if prefer_hyde and HYDE_ENABLED else query
    
    # BM25 检索用 Query2Doc
    bm25_query = query2doc_expand(query, llm) if QUERY2DOC_ENABLED else query
    
    return vector_query, bm25_query


def _node_id_key(node: NodeWithScore) -> str:
    try:
        nid = getattr(node.node, "node_id", None)
        if nid:
            return str(nid)
    except Exception:
        pass
    return str(id(node))


def _weighted_rrf_fuse_three_routes(
    routes: list[tuple[list[NodeWithScore], float]],
    top_n: int,
    rrf_k: int = 60,
) -> list[NodeWithScore]:
    scores: dict[str, float] = {}
    nodes_map: dict[str, NodeWithScore] = {}

    for nodes, route_weight in routes:
        weight = float(route_weight)
        if weight <= 0:
            continue
        for rank, n in enumerate(nodes, 1):
            key = _node_id_key(n)
            nodes_map.setdefault(key, n)
            scores[key] = scores.get(key, 0.0) + weight / float(rrf_k + rank)

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    out: list[NodeWithScore] = []
    for key, score in ranked[: max(0, int(top_n))]:
        node = nodes_map.get(key)
        if node is None:
            continue
        node.score = float(score)
        out.append(node)
    return out


class ThreeWayHyDEFusionRetriever(BaseRetriever):
    """三路召回：原始向量 + HyDE向量 + BM25+，使用加权 RRF 融合。"""

    def __init__(
        self,
        vector_retriever: BaseRetriever,
        bm25_retriever: BaseRetriever,
        similarity_top_k: int,
        vector_weight: float = 1.0,
        hyde_weight: float = 0.7,
        bm25_weight: float = 1.0,
        rrf_k: int = FUSION_RRF_K,
        llm: Any = None,
        use_hyde: bool = True,
        use_query2doc: bool = False,
        primed_queries: Optional[dict[str, tuple[str, str]]] = None,
    ):
        super().__init__()
        self._vector = vector_retriever
        self._bm25 = bm25_retriever
        self._k = int(similarity_top_k)
        self._vector_weight = float(vector_weight)
        self._hyde_weight = float(hyde_weight)
        self._bm25_weight = float(bm25_weight)
        self._rrf_k = int(rrf_k)
        self._llm = llm
        self._use_hyde = bool(use_hyde)
        self._use_query2doc = bool(use_query2doc)
        self._primed_queries = primed_queries or {}

    def _enhance_queries(self, query_str: str) -> tuple[str, str]:
        if query_str in self._primed_queries:
            return self._primed_queries[query_str]

        should_enhance = QUERY_ENHANCE_ALL or _is_short_query(query_str)
        hyde_query = query_str
        bm25_query = query_str
        if should_enhance:
            if self._use_hyde:
                hyde_query = hyde_expand_query(query_str, llm=self._llm)
            if self._use_query2doc:
                bm25_query = query2doc_expand(query_str, llm=self._llm)
        return hyde_query, bm25_query

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query_str = getattr(query_bundle, "query_str", str(query_bundle)).strip()
        if not query_str:
            return []

        hyde_query, bm25_query = self._enhance_queries(query_str)

        # ===== 多查询扩展：生成多个查询变体并聚合结果 =====
        if MULTI_EXPAND_ENABLED:
            # 生成查询变体
            raw_queries = _build_multi_expand_queries(query_str, num_queries=FUSION_NUM_QUERIES)
            bm25_queries = _build_multi_expand_queries(bm25_query, num_queries=FUSION_NUM_QUERIES)
            
            # 多查询向量检索（raw vec）
            all_raw_vec: list[NodeWithScore] = []
            seen_raw: set[str] = set()
            for q in raw_queries:
                for n in self._vector.retrieve(q)[: self._k]:
                    nid = _node_id_key(n)
                    if nid not in seen_raw:
                        seen_raw.add(nid)
                        all_raw_vec.append(n)
            raw_vec_nodes = all_raw_vec[: self._k * 2]  # 保留更多候选
            
            # 多查询 BM25 检索
            all_bm25: list[NodeWithScore] = []
            seen_bm25: set[str] = set()
            for q in bm25_queries:
                for n in self._bm25.retrieve(q)[: self._k]:
                    nid = _node_id_key(n)
                    if nid not in seen_bm25:
                        seen_bm25.add(nid)
                        all_bm25.append(n)
            bm25_nodes = all_bm25[: self._k * 2]
        else:
            raw_vec_nodes = self._vector.retrieve(query_str)[: self._k]
            bm25_nodes = self._bm25.retrieve(bm25_query)[: self._k]

        # HyDE 向量检索
        hyde_vec_nodes: list[NodeWithScore] = []
        if self._use_hyde and hyde_query and hyde_query != query_str:
            hyde_vec_nodes = self._vector.retrieve(hyde_query)[: self._k]

        routes: list[tuple[list[NodeWithScore], float]] = [
            (raw_vec_nodes, self._vector_weight),
            (bm25_nodes, self._bm25_weight),
        ]
        if hyde_vec_nodes:
            routes.insert(1, (hyde_vec_nodes, self._hyde_weight))

        return _weighted_rrf_fuse_three_routes(
            routes,
            top_n=self._k,
            rrf_k=self._rrf_k,
        )


# =============================================================================
# 答案生成质量优化：RAG Prompt + 引用生成 (Phase 3.4)
# =============================================================================

# 优化后的 RAG Prompt，增强答案质量和引用规范
RAG_PROMPT_TEMPLATE = """你是 PCB 电路板领域的技术专家。请根据以下参考文档回答用户问题。

要求：
1. 只基于参考文档中的信息回答，不要编造
2. 如果文档中没有相关信息，明确说明"根据提供的文档无法回答"
3. 回答要准确、专业、简洁
4. 在回答末尾标注信息来源（如：[来源：GBT4588.4-2017]）

参考文档：
{context}

用户问题：{query}

回答："""

# 环境变量控制引用功能
CITATION_ENABLED = os.getenv("CITATION_ENABLED", "1") not in {"0", "false", "False"}
# 用于生成答案的最大文档数
RAG_TOP_DOCS = int(os.getenv("RAG_TOP_DOCS", "5"))
# 每个文档内容的最大字符数
RAG_DOC_MAX_CHARS = int(os.getenv("RAG_DOC_MAX_CHARS", "1000"))

# =============================================================================
# Self-RAG (自反思 RAG) 配置
# =============================================================================
# 是否启用 Self-RAG 自反思机制
SELF_RAG_ENABLED = os.getenv("SELF_RAG_ENABLED", "0") not in {"0", "false", "False"}
# 最大迭代次数（防止无限循环）
SELF_RAG_MAX_ITERATIONS = int(os.getenv("SELF_RAG_MAX_ITERATIONS", "3"))
# 自评分阈值（低于此分数触发重新检索，1-5 分）
SELF_RAG_THRESHOLD = float(os.getenv("SELF_RAG_THRESHOLD", "3.0"))
# 是否在详细模式下显示自反思过程
SELF_RAG_VERBOSE = os.getenv("SELF_RAG_VERBOSE", "1") not in {"0", "false", "False"}

# Self-RAG 自评估 Prompt
SELF_RAG_EVALUATE_PROMPT = """你是一个严格的答案质量评估专家。请评估以下 RAG 系统的回答质量。

用户问题：{query}

检索到的参考文档：
{context}

系统生成的答案：
{answer}

请从以下维度评分（每项 1-5 分）：
1. 【相关性】答案是否回答了用户的问题？
2. 【完整性】答案是否完整，信息是否充分？
3. 【准确性】答案是否基于参考文档，没有编造？
4. 【可用性】检索到的文档是否足够支持回答？

请按以下 JSON 格式输出（只输出 JSON，不要其他内容）：
{{
    "relevance": <1-5>,
    "completeness": <1-5>,
    "accuracy": <1-5>,
    "sufficiency": <1-5>,
    "overall": <1-5>,
    "needs_more_retrieval": <true/false>,
    "retrieval_suggestion": "<如果需要重新检索，建议的新查询关键词>",
    "reason": "<简短的评估理由>"
}}"""

# Self-RAG 查询改写 Prompt（用于生成补充检索的查询）
SELF_RAG_REWRITE_PROMPT = """基于以下信息，生成一个新的检索查询来补充获取缺失的信息。

原始问题：{query}
已有答案：{answer}
缺失信息：{missing_info}

要求：
1. 新查询应该针对缺失的信息点
2. 使用 PCB 领域的专业术语
3. 只输出新的查询语句，不要其他解释

新检索查询："""


def extract_citations(answer: str, retrieved_docs: list) -> list[dict]:
    """从答案中提取引用信息。
    
    检测答案中出现的标准编号、文档名称等，并与检索到的文档进行匹配。
    
    Args:
        answer: LLM 生成的答案文本
        retrieved_docs: 检索到的文档列表（NodeWithScore 对象）
        
    Returns:
        引用列表，每个引用包含：
        - source: 来源文档路径/名称
        - matched_text: 匹配到的文本片段
        - confidence: 置信度（high/medium/low）
    """
    citations = []
    
    # 1. 提取答案中提到的标准编号（如 GBT4588.4-2017, IPC-A-600 等）
    standard_patterns = [
        r'GB[/T]*\s*[\d\.\-]+(?:\-\d{4})?',  # 国标：GBT4588.4-2017
        r'IPC-[A-Z]*-?\d+[A-Z]*',  # IPC 标准：IPC-A-600G
        r'SJ[/T]*\s*[\d\.\-]+(?:\-\d{4})?',  # 行业标准：SJT11363-2006
        r'ISO\s*\d+(?:\-\d+)?(?:\:\d{4})?',  # ISO 标准
    ]
    
    mentioned_standards = set()
    for pattern in standard_patterns:
        matches = re.findall(pattern, answer, re.IGNORECASE)
        mentioned_standards.update(m.upper().replace(' ', '') for m in matches)
    
    # 2. 与检索文档匹配
    for doc in retrieved_docs:
        node = doc.node if hasattr(doc, 'node') else doc
        source_path = node.metadata.get('source_path', '') if hasattr(node, 'metadata') else ''
        doc_text = node.get_content() if hasattr(node, 'get_content') else str(node)
        
        # 标准化来源路径
        source_name = Path(source_path).stem if source_path else '未知来源'
        
        # 检查是否在答案中被提及
        confidence = 'low'
        matched_text = ''
        
        # 检查标准编号匹配
        for std in mentioned_standards:
            std_normalized = std.replace('/', '').replace('-', '').replace(' ', '')
            source_normalized = source_name.upper().replace('/', '').replace('-', '').replace(' ', '')
            if std_normalized in source_normalized or source_normalized in std_normalized:
                confidence = 'high'
                matched_text = std
                break
        
        # 如果没有标准编号匹配，检查关键词匹配
        if confidence == 'low':
            # 提取来源文档的关键词
            source_keywords = re.findall(r'[\u4e00-\u9fff]{2,}|[A-Za-z0-9]+', source_name)
            for kw in source_keywords:
                if len(kw) >= 2 and kw.lower() in answer.lower():
                    confidence = 'medium'
                    matched_text = kw
                    break
        
        citations.append({
            'source': source_path or source_name,
            'source_name': source_name,
            'matched_text': matched_text,
            'confidence': confidence,
            'score': doc.score if hasattr(doc, 'score') else 0.0,
        })
    
    # 按置信度和分数排序
    confidence_order = {'high': 0, 'medium': 1, 'low': 2}
    citations.sort(key=lambda x: (confidence_order.get(x['confidence'], 3), -x['score']))
    
    return citations


def build_context_with_sources(retrieved_docs: list, max_docs: int = None, max_chars: int = None) -> tuple[str, list[dict]]:
    """构建带来源标注的上下文。
    
    Args:
        retrieved_docs: 检索到的文档列表
        max_docs: 最大文档数（默认使用 RAG_TOP_DOCS）
        max_chars: 每个文档最大字符数（默认使用 RAG_DOC_MAX_CHARS）
        
    Returns:
        (格式化的上下文文本, 来源信息列表)
    """
    max_docs = max_docs or RAG_TOP_DOCS
    max_chars = max_chars or RAG_DOC_MAX_CHARS
    
    context_texts = []
    sources_info = []
    
    for i, doc in enumerate(retrieved_docs[:max_docs]):
        node = doc.node if hasattr(doc, 'node') else doc
        
        # 获取文档内容
        if hasattr(node, 'get_content'):
            text = node.get_content(metadata_mode=MetadataMode.EMBED)
        else:
            text = str(node)
        
        # 获取来源信息
        metadata = node.metadata if hasattr(node, 'metadata') else {}
        source_path = metadata.get('source_path', '未知来源')
        source_name = Path(source_path).stem if source_path and source_path != '未知来源' else source_path
        
        # 截断过长内容
        if len(text) > max_chars:
            text = text[:max_chars] + '...'
        
        # 构建带标注的上下文
        context_texts.append(f"[文档{i+1}: {source_name}]\n{text}")
        
        sources_info.append({
            'index': i + 1,
            'source_path': source_path,
            'source_name': source_name,
            'score': doc.score if hasattr(doc, 'score') else 0.0,
            'char_count': len(text),
        })
    
    context = "\n\n---\n\n".join(context_texts)
    return context, sources_info


def generate_answer_with_citation(query: str, retrieved_docs: list, llm=None) -> dict:
    """生成带引用的答案。
    
    使用优化后的 RAG Prompt 生成答案，并自动提取和标注引用来源。
    
    Args:
        query: 用户查询
        retrieved_docs: 检索到的文档列表（NodeWithScore 对象）
        llm: LLM 实例（如果为 None，使用 Settings.llm）
        
    Returns:
        包含以下字段的字典：
        - answer: 生成的答案文本
        - citations: 引用列表（带置信度）
        - sources: 来源文档路径列表
        - context_info: 上下文构建信息
    """
    if llm is None:
        llm = Settings.llm
    
    if not retrieved_docs:
        return {
            'answer': '根据提供的文档无法回答该问题，未找到相关文档。',
            'citations': [],
            'sources': [],
            'context_info': {'doc_count': 0},
        }
    
    # 构建上下文
    context, sources_info = build_context_with_sources(retrieved_docs)
    
    # 生成答案
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, query=query)
    
    try:
        response = llm.complete(prompt)
        answer = response.text if hasattr(response, 'text') else str(response)
    except Exception as e:
        print(f"[RAG] 答案生成失败: {e}")
        answer = f"答案生成过程中出现错误：{e}"
    
    # 提取引用
    citations = []
    if CITATION_ENABLED:
        citations = extract_citations(answer, retrieved_docs)
    
    # 构建来源列表
    sources = [info['source_path'] for info in sources_info]
    
    return {
        'answer': answer,
        'citations': citations,
        'sources': sources,
        'context_info': {
            'doc_count': len(sources_info),
            'total_chars': sum(info['char_count'] for info in sources_info),
            'sources_detail': sources_info,
        },
    }


def format_answer_with_citations(result: dict, verbose: bool = False) -> str:
    """格式化带引用的答案输出。
    
    Args:
        result: generate_answer_with_citation 的返回结果
        verbose: 是否显示详细引用信息
        
    Returns:
        格式化后的答案字符串
    """
    output_parts = [result['answer']]
    
    if CITATION_ENABLED and result.get('citations'):
        # 筛选高置信度和中置信度的引用
        valid_citations = [c for c in result['citations'] if c['confidence'] in ('high', 'medium')]
        
        if valid_citations:
            output_parts.append('\n\n📚 参考来源：')
            for i, cite in enumerate(valid_citations[:5], 1):  # 最多显示5个
                confidence_marker = '✓' if cite['confidence'] == 'high' else '○'
                output_parts.append(f"  {confidence_marker} [{i}] {cite['source_name']}")
                if verbose and cite.get('matched_text'):
                    output_parts.append(f"      匹配: {cite['matched_text']}")
    
    return '\n'.join(output_parts)


# =============================================================================
# Self-RAG (自反思 RAG) 核心实现
# =============================================================================

def self_rag_evaluate(
    query: str,
    answer: str,
    retrieved_docs: list,
    llm=None,
) -> dict:
    """Self-RAG 自评估：让 LLM 评估自己的答案质量。
    
    Args:
        query: 用户原始查询
        answer: LLM 生成的答案
        retrieved_docs: 检索到的文档列表
        llm: LLM 实例
        
    Returns:
        评估结果字典，包含各维度评分和是否需要重新检索
    """
    if llm is None:
        llm = Settings.llm
    
    # 构建上下文摘要（用于评估）
    context_summary = []
    for i, doc in enumerate(retrieved_docs[:RAG_TOP_DOCS], 1):
        node = doc.node if hasattr(doc, 'node') else doc
        text = node.get_content() if hasattr(node, 'get_content') else str(node)
        # 截取前 500 字符用于评估
        context_summary.append(f"[文档{i}] {text[:500]}...")
    
    context_text = "\n\n".join(context_summary)
    
    # 构建评估 prompt
    prompt = SELF_RAG_EVALUATE_PROMPT.format(
        query=query,
        context=context_text,
        answer=answer
    )
    
    try:
        response = llm.complete(prompt)
        response_text = response.text if hasattr(response, 'text') else str(response)
        
        # 解析 JSON 响应
        # 尝试提取 JSON 部分
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            eval_result = json.loads(json_match.group())
        else:
            # 如果无法解析 JSON，返回默认评估
            eval_result = {
                "relevance": 3,
                "completeness": 3,
                "accuracy": 3,
                "sufficiency": 3,
                "overall": 3,
                "needs_more_retrieval": False,
                "retrieval_suggestion": "",
                "reason": "无法解析评估结果"
            }
        
        # 确保所有必要字段存在
        eval_result.setdefault("overall", 3)
        eval_result.setdefault("needs_more_retrieval", False)
        eval_result.setdefault("retrieval_suggestion", "")
        eval_result.setdefault("reason", "")
        
        return eval_result
        
    except Exception as e:
        if SELF_RAG_VERBOSE:
            print(f"[Self-RAG] 评估失败: {e}")
        return {
            "overall": 3,
            "needs_more_retrieval": False,
            "retrieval_suggestion": "",
            "reason": f"评估过程出错: {e}"
        }


def self_rag_generate_followup_query(
    query: str,
    answer: str,
    eval_result: dict,
    llm=None,
) -> str:
    """根据自评估结果生成补充检索的查询。
    
    Args:
        query: 原始查询
        answer: 当前答案
        eval_result: 自评估结果
        llm: LLM 实例
        
    Returns:
        新的检索查询字符串
    """
    if llm is None:
        llm = Settings.llm
    
    # 如果评估结果中有建议的查询，直接使用
    if eval_result.get("retrieval_suggestion"):
        suggestion = eval_result["retrieval_suggestion"].strip()
        if suggestion and len(suggestion) > 2:
            return suggestion
    
    # 否则使用 LLM 生成新查询
    missing_info = eval_result.get("reason", "信息不完整")
    
    prompt = SELF_RAG_REWRITE_PROMPT.format(
        query=query,
        answer=answer[:500],  # 截取部分答案
        missing_info=missing_info
    )
    
    try:
        response = llm.complete(prompt)
        new_query = response.text if hasattr(response, 'text') else str(response)
        new_query = new_query.strip()
        
        # 清理响应（去除可能的解释文本）
        lines = new_query.split('\n')
        new_query = lines[0].strip()
        
        if new_query and len(new_query) > 2:
            return new_query
        else:
            # 降级：使用原查询 + 补充关键词
            return f"{query} 详细 规范 标准"
            
    except Exception as e:
        if SELF_RAG_VERBOSE:
            print(f"[Self-RAG] 生成补充查询失败: {e}")
        return f"{query} 详细信息"


def self_rag_query(
    query: str,
    retriever,
    llm=None,
    max_iterations: int = None,
    threshold: float = None,
    verbose: bool = None,
) -> dict:
    """Self-RAG 主流程：评分器模式（不干预检索决策）。

    工作流程：
    1. 执行首轮检索和答案生成
    2. LLM 自我评估答案质量（仅打分）
    3. 不进行补充查询、不进行迭代重检索
    
    Args:
        query: 用户查询
        retriever: 检索器实例
        llm: LLM 实例
        max_iterations: 最大迭代次数（默认使用配置）
        threshold: 质量阈值（默认使用配置）
        verbose: 是否显示详细过程（默认使用配置）
        
    Returns:
        包含最终答案、引用和迭代历史的结果字典
    """
    if not SELF_RAG_ENABLED:
        # Self-RAG 未启用，使用普通流程
        docs = retriever.retrieve(query)
        return generate_answer_with_citation(query, docs, llm)
    
    if llm is None:
        llm = Settings.llm
    if max_iterations is None:
        max_iterations = 1
    if threshold is None:
        threshold = SELF_RAG_THRESHOLD
    if verbose is None:
        verbose = SELF_RAG_VERBOSE
    
    # 评分器模式：仅记录一轮
    iteration_history = []
    all_retrieved_docs = []
    seen_doc_ids = set()
    if verbose:
        print(f"\n[Self-RAG] === 评分模式（单轮）===")
        print(f"[Self-RAG] 当前查询: {query[:80]}...")

    # Step 1: 首轮检索
    new_docs = retriever.retrieve(query)

    # 去重并合并文档（仅一轮）
    for doc in new_docs:
        node = doc.node if hasattr(doc, 'node') else doc
        doc_id = node.node_id if hasattr(node, 'node_id') else str(id(node))
        if doc_id not in seen_doc_ids:
            seen_doc_ids.add(doc_id)
            all_retrieved_docs.append(doc)

    if verbose:
        print(f"[Self-RAG] 首轮检索文档: {len(new_docs)}, 去重后: {len(all_retrieved_docs)}")

    # Step 2: 生成答案（首轮文档不截断，保障召回覆盖）
    final_result = generate_answer_with_citation(
        query,
        all_retrieved_docs,
        llm
    )

    if verbose:
        print(f"[Self-RAG] 答案预览: {final_result['answer'][:100]}...")

    # Step 3: 自评估（仅评分，不驱动重检索）
    eval_result = self_rag_evaluate(
        query,
        final_result['answer'],
        all_retrieved_docs[:RAG_TOP_DOCS],
        llm
    )
    iteration_history.append({
        "iteration": 1,
        "query_used": query,
        "docs_retrieved": len(new_docs),
        "docs_total": len(all_retrieved_docs),
        "evaluation": eval_result,
    })

    if verbose:
        print(f"[Self-RAG] 自评分: {float(eval_result.get('overall', 3))}/5")
        print(f"[Self-RAG] 评估理由: {eval_result.get('reason', 'N/A')[:80]}")
    
    # 添加 Self-RAG 元信息
    final_result['self_rag_info'] = {
        'enabled': True,
        'mode': 'scorer_only',
        'iterations': len(iteration_history),
        'final_score': iteration_history[-1]['evaluation'].get('overall', 0) if iteration_history else 0,
        'history': iteration_history,
    }

    # 为评估流程提供检索结果（不影响线上展示）
    final_result['retrieved_docs'] = all_retrieved_docs
    retrieved_ids: list[str] = []
    for doc in all_retrieved_docs:
        try:
            node = doc.node if hasattr(doc, 'node') else doc
            nid = getattr(node, 'node_id', None) or getattr(node, 'id_', None) or getattr(node, 'id', None)
            if isinstance(nid, str) and nid:
                retrieved_ids.append(nid)
        except Exception:
            pass
    final_result['retrieved_node_ids'] = retrieved_ids
    
    return final_result


def format_self_rag_result(result: dict, show_history: bool = False) -> str:
    """格式化 Self-RAG 结果输出。
    
    Args:
        result: self_rag_query 的返回结果
        show_history: 是否显示迭代历史
        
    Returns:
        格式化后的字符串
    """
    output_parts = []
    
    # 主答案
    output_parts.append(format_answer_with_citations(result))
    
    # Self-RAG 信息
    self_rag_info = result.get('self_rag_info', {})
    if self_rag_info.get('enabled'):
        output_parts.append(f"\n\n🔄 Self-RAG 自反思：")
        output_parts.append(f"   迭代次数: {self_rag_info.get('iterations', 1)}")
        output_parts.append(f"   最终评分: {self_rag_info.get('final_score', 'N/A')}/5")
        
        if show_history and self_rag_info.get('history'):
            output_parts.append("\n   迭代历史:")
            for record in self_rag_info['history']:
                eval_info = record.get('evaluation', {})
                output_parts.append(
                    f"   [{record['iteration']}] 查询: {record['query_used'][:40]}... "
                    f"| 文档: {record['docs_retrieved']} | 评分: {eval_info.get('overall', 'N/A')}"
                )
    
    return '\n'.join(output_parts)


# =============================================================================
# 查询意图理解与分类系统
# =============================================================================
QUERY_ROUTING_ENABLED = os.getenv("QUERY_ROUTING_ENABLED", "1") not in {"0", "false", "False"}

# 查询类型配置：不同类型查询使用不同检索策略
QUERY_TYPES = {
    "definition": {  # 定义类：什么是XXX
        "patterns": ["什么是", "定义", "含义", "概念", "是什么", "指的是", "meaning", "definition"],
        "keywords": [],
        "strategy": {"vector_weight": 0.6, "bm25_weight": 0.4, "num_queries": 4},
        "description": "定义类查询，需要语义理解"
    },
    "procedure": {  # 流程类：如何做XXX
        "patterns": ["如何", "怎么", "怎样", "步骤", "方法", "流程", "过程", "工艺", "how to", "process"],
        "keywords": ["制造", "生产", "加工", "组装", "焊接", "蚀刻", "电镀", "钻孔"],
        "strategy": {"vector_weight": 0.5, "bm25_weight": 0.5, "num_queries": 5},
        "description": "流程类查询，平衡语义和关键词"
    },
    "specification": {  # 规格类：XXX的参数/标准
        "patterns": ["参数", "规格", "标准", "要求", "规定", "指标", "公差", "尺寸", "厚度", "宽度"],
        "keywords": ["mm", "mil", "oz", "μm", "ohm", "Ω", "层", "孔径"],
        "strategy": {"vector_weight": 0.3, "bm25_weight": 0.7, "num_queries": 4},
        "description": "规格类查询，精确匹配优先"
    },
    "comparison": {  # 对比类：A和B的区别
        "patterns": ["区别", "对比", "差异", "不同", "比较", "优缺点", "选择", "vs", "versus", "difference"],
        "keywords": [],
        "strategy": {"vector_weight": 0.55, "bm25_weight": 0.45, "num_queries": 5},
        "description": "对比类查询，需要理解多个概念"
    },
    "troubleshoot": {  # 故障类：为什么会XXX
        "patterns": ["为什么", "原因", "故障", "问题", "失败", "缺陷", "不良", "异常", "why", "cause", "failure"],
        "keywords": ["分层", "起泡", "短路", "断路", "翘曲", "偏移", "漏铜"],
        "strategy": {"vector_weight": 0.5, "bm25_weight": 0.5, "num_queries": 5},
        "description": "故障类查询，需要理解因果关系"
    },
    "standard": {  # 标准类：查询特定标准
        "patterns": [],
        "keywords": ["GB", "GB/T", "IPC", "JEDEC", "MIL", "ISO", "标准", "规范"],
        "regex": [r"GB[/T]*[-\s]*\d+", r"IPC[-\s]*\d+", r"JEDEC", r"MIL[-\s]*STD", r"ISO[-\s]*\d+"],
        "strategy": {"vector_weight": 0.25, "bm25_weight": 0.75, "num_queries": 3},
        "description": "标准类查询，精确匹配标准号"
    },
    "material": {  # 材料类：查询材料特性
        "patterns": ["材料", "基材", "介质", "铜箔", "树脂", "玻纤", "覆铜板"],
        "keywords": ["FR-4", "FR4", "CEM", "高频", "PTFE", "PI", "聚酰亚胺", "Tg", "Dk", "Df", "CTI"],
        "strategy": {"vector_weight": 0.4, "bm25_weight": 0.6, "num_queries": 4},
        "description": "材料类查询，需要匹配专业术语"
    },
    "test": {  # 测试类：测试方法和标准
        "patterns": ["测试", "试验", "检验", "检测", "验证", "test", "inspection"],
        "keywords": ["AOI", "X-ray", "飞针", "ICT", "FCT", "可靠性", "耐热", "剥离强度"],
        "strategy": {"vector_weight": 0.4, "bm25_weight": 0.6, "num_queries": 4},
        "description": "测试类查询，需要匹配测试方法"
    },
    "eda_operation": {  # EDA操作类：软件操作步骤
        "patterns": ["命令", "菜单", "操作", "设置", "配置", "面板", "对话框", "按钮", "选项"],
        "keywords": ["Altium", "Cadence", "Allegro", "OrCAD", "KiCad", "PADS", "AD", "PCB Editor",
                     "原理图", "封装", "网表", "布线", "布局", "DRC", "BOM"],
        "strategy": {"vector_weight": 0.35, "bm25_weight": 0.65, "num_queries": 4},
        "description": "EDA操作类查询，精确匹配软件命令和术语"
    }
}


def _classify_query(query: str) -> str:
    """智能分类查询意图，返回最匹配的查询类型。

    分类优先级：
    1. 正则匹配（标准号等精确模式）
    2. 关键词匹配（专业术语）
    3. 模式匹配（疑问词等）
    4. 默认为 general

    Returns:
        查询类型字符串
    """
    if not query:
        return "general"

    q = query.strip()
    q_lower = q.lower()

    # 记录各类型的匹配分数
    type_scores: dict[str, float] = {}

    for qtype, config in QUERY_TYPES.items():
        score = 0.0

        # 1. 正则匹配（最高优先级）
        if "regex" in config:
            for pattern in config["regex"]:
                if re.search(pattern, q, re.I):
                    score += 10.0
                    break

        # 2. 关键词匹配
        for kw in config.get("keywords", []):
            kw_lower = kw.lower()
            if kw_lower in q_lower or kw in q:
                score += 3.0
                # 完全匹配额外加分
                if re.search(rf"\b{re.escape(kw)}\b", q, re.I):
                    score += 1.0

        # 3. 模式匹配
        for pattern in config.get("patterns", []):
            if pattern.lower() in q_lower or pattern in q:
                score += 2.0
                # 开头匹配额外加分
                if q_lower.startswith(pattern.lower()) or q.startswith(pattern):
                    score += 1.0

        if score > 0:
            type_scores[qtype] = score

    # 选择得分最高的类型
    if type_scores:
        best_type = max(type_scores.items(), key=lambda x: x[1])
        if best_type[1] >= 2.0:  # 阈值：至少匹配一个模式
            return best_type[0]

    # 补充规则：数值规格查询
    if re.search(r"\b\d+(?:\.\d+)?\s*(mm|mil|oz|ohm|μm|um|Ω|层)\b", q, re.I):
        return "specification"

    # 补充规则：长句倾向语义理解
    if len(q) > 35 or q.count("，") > 2:
        return "definition"

    # 补充规则：短查询且包含专业术语
    if len(q) < 15 and any(term in q_lower for term in _PCB_TERMS):
        return "specification"

    return "general"


def _get_query_strategy(query_type: str) -> dict:
    """根据查询类型返回完整的检索策略配置。

    Returns:
        包含 vector_weight, bm25_weight, num_queries, description 的字典
    """
    if query_type in QUERY_TYPES:
        config = QUERY_TYPES[query_type]
        return {
            "type": query_type,
            "vector_weight": config["strategy"]["vector_weight"],
            "bm25_weight": config["strategy"]["bm25_weight"],
            "num_queries": config["strategy"]["num_queries"],
            "description": config.get("description", "")
        }

    # 默认 general 策略
    return {
        "type": "general",
        "vector_weight": float(FUSION_WEIGHTS[0]),
        "bm25_weight": float(FUSION_WEIGHTS[1]),
        "num_queries": FUSION_NUM_QUERIES,
        "description": "通用查询，平衡检索"
    }


def _get_retrieval_weights(query_type: str) -> tuple[float, float]:
    """根据查询类型返回检索权重 (向量权重, BM25权重)。"""
    strategy = _get_query_strategy(query_type)
    return (strategy["vector_weight"], strategy["bm25_weight"])


# =============================================================================
# ColBERT-style Late Interaction Reranker (Phase 3.1)
# =============================================================================


class ColBERTReranker:
    """ColBERT 风格的细粒度交互重排器。

    实现 Late Interaction 机制：
    1. 将 query 和 document 分别编码为 token-level embeddings
    2. 计算 MaxSim：每个 query token 与所有 doc tokens 的最大相似度
    3. 汇总所有 query tokens 的 MaxSim 得到最终得分

    这比传统 cross-encoder 更高效，同时保留了细粒度语义交互能力。
    """

    def __init__(
        self,
        model_name: str = COLBERT_MODEL,
        max_length: int = COLBERT_MAX_LENGTH,
        batch_size: int = COLBERT_BATCH_SIZE,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

        import torch

        # 选择设备
        if device == "cuda":
            self._device = torch.device("cuda")
        elif device == "cpu":
            self._device = torch.device("cpu")
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model = None
        self._tokenizer = None
        self._initialized = False

    def _lazy_init(self):
        """延迟加载模型，避免启动时内存占用过高。"""
        if self._initialized:
            return

        import torch
        from transformers import AutoModel, AutoTokenizer
        from huggingface_hub import model_info
        from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

        print(f"[ColBERT] 加载模型: {self.model_name}")
        
        # 尝试验证模型是否存在
        try:
            model_info(self.model_name)
        except (HfHubHTTPError, RepositoryNotFoundError) as e:
            error_msg = f"模型 '{self.model_name}' 不存在或无法访问。\n"
            error_msg += "\n推荐的 ColBERT/Reranker 模型：\n"
            error_msg += "  - BAAI/bge-reranker-v2-m3 (推荐，中文效果好)\n"
            error_msg += "  - BAAI/bge-reranker-base\n"
            error_msg += "  - BAAI/bge-reranker-large\n"
            error_msg += "\n请设置环境变量: export COLBERT_MODEL=BAAI/bge-reranker-v2-m3"
            raise ValueError(error_msg) from e
        
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self._model = AutoModel.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self._model.eval()
            self._model.to(self._device)
            self._initialized = True
            print(f"[ColBERT] 模型加载完成，device={self._device}")
        except Exception as e:
            error_msg = f"加载模型 '{self.model_name}' 失败: {e}\n"
            error_msg += "请确认模型名称正确，或尝试使用: BAAI/bge-reranker-v2-m3"
            raise RuntimeError(error_msg) from e

    def _encode(self, texts: list[str]) -> "torch.Tensor":
        """编码文本为 token-level embeddings。

        Returns:
            Tensor of shape [batch, seq_len, hidden_dim]
        """
        import torch

        self._lazy_init()

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            inputs = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                # 使用 last_hidden_state 作为 token embeddings
                # shape: [batch, seq_len, hidden_dim]
                token_embeddings = outputs.last_hidden_state
                # L2 归一化，便于计算余弦相似度
                token_embeddings = torch.nn.functional.normalize(
                    token_embeddings, p=2, dim=-1
                )
                all_embeddings.append(token_embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    def _maxsim_score(
        self, query_emb: "torch.Tensor", doc_emb: "torch.Tensor"
    ) -> float:
        """计算 ColBERT MaxSim 得分。

        MaxSim 算法：
        - 对于每个 query token，找到与所有 doc tokens 的最大相似度
        - 将所有 query tokens 的最大相似度求和

        Args:
            query_emb: [query_len, hidden_dim]
            doc_emb: [doc_len, hidden_dim]

        Returns:
            MaxSim 得分
        """
        import torch

        # 计算相似度矩阵: [query_len, doc_len]
        sim_matrix = torch.mm(query_emb, doc_emb.T)

        # 对每个 query token 取最大相似度，然后求和
        max_sim_per_query = sim_matrix.max(dim=1).values
        score = max_sim_per_query.sum().item()

        return score

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, Any]],
    ) -> list[tuple[str, Any, float]]:
        """对候选文档进行 ColBERT 重排。

        Args:
            query: 查询字符串
            candidates: [(doc_id, doc_content/node), ...] 候选列表

        Returns:
            [(doc_id, doc_content/node, score), ...] 按得分降序排列
        """
        if not candidates:
            return []

        # 提取文档文本
        doc_texts = []
        for _, content in candidates:
            if isinstance(content, str):
                doc_texts.append(content)
            elif hasattr(content, "text"):
                doc_texts.append(content.text)
            elif hasattr(content, "get_content"):
                doc_texts.append(content.get_content())
            else:
                doc_texts.append(str(content))

        # 编码 query 和 documents
        query_emb = self._encode([query])[0]  # [query_len, hidden_dim]
        doc_embs = self._encode(doc_texts)  # [num_docs, seq_len, hidden_dim]

        # 计算每个文档的 MaxSim 得分
        results = []
        for i, (doc_id, content) in enumerate(candidates):
            doc_emb = doc_embs[i]  # [seq_len, hidden_dim]
            score = self._maxsim_score(query_emb, doc_emb)
            results.append((doc_id, content, score))

        # 按得分降序排列
        results.sort(key=lambda x: x[2], reverse=True)
        return results


class MultiPathRetriever:
    """多路召回 + 细粒度交互 检索器 (Phase 3.1)

    实现思路：
    1. 多路召回：同时使用向量检索（Dense）和词法检索（Sparse/BM25）
    2. RRF 融合：使用 Reciprocal Rank Fusion 合并多路结果
    3. ColBERT 细粒度重排：使用 Late Interaction 对候选进行精排

    优点：
    - Dense 检索：语义理解强，处理同义词/近义词
    - Sparse 检索：精确匹配强，处理专业术语/标准号
    - ColBERT 重排：细粒度语义交互，提升排序质量
    """

    def __init__(
        self,
        dense_retriever: BaseRetriever,
        sparse_retriever: Optional["LocalBM25Retriever"] = None,
        colbert_reranker: Optional[ColBERTReranker] = None,
        dense_top_k: int = MULTIPATH_DENSE_TOP_K,
        sparse_top_k: int = MULTIPATH_SPARSE_TOP_K,
        rerank_candidates: int = MULTIPATH_RERANK_CANDIDATES,
        rrf_k: int = MULTIPATH_RRF_K,
        enable_colbert: bool = COLBERT_RERANK_ENABLED,
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.colbert_reranker = colbert_reranker
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.rerank_candidates = rerank_candidates
        self.rrf_k = rrf_k
        self.enable_colbert = enable_colbert and colbert_reranker is not None

    def _rrf_fuse(
        self,
        dense_results: list[NodeWithScore],
        sparse_results: list[NodeWithScore],
    ) -> list[tuple[str, NodeWithScore, float]]:
        """Reciprocal Rank Fusion 融合多路召回结果。

        RRF 公式：score(d) = Σ 1 / (k + rank(d))
        其中 k 是平滑参数（默认 60），rank 是文档在各路召回中的排名。

        Args:
            dense_results: 向量检索结果
            sparse_results: BM25 检索结果

        Returns:
            [(node_id, node, rrf_score), ...] 按 RRF 得分降序排列
        """
        k = self.rrf_k
        scores: dict[str, float] = {}
        node_map: dict[str, NodeWithScore] = {}

        # Dense 结果
        for rank, node in enumerate(dense_results):
            node_id = node.node.id_ or str(rank)
            scores[node_id] = scores.get(node_id, 0) + 1.0 / (k + rank + 1)
            if node_id not in node_map:
                node_map[node_id] = node

        # Sparse 结果
        for rank, node in enumerate(sparse_results):
            node_id = node.node.id_ or f"sparse_{rank}"
            scores[node_id] = scores.get(node_id, 0) + 1.0 / (k + rank + 1)
            if node_id not in node_map:
                node_map[node_id] = node

        # 按 RRF 得分排序
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        results = [(nid, node_map[nid], scores[nid]) for nid in sorted_ids]

        return results

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        expand_queries: Optional[list[str]] = None,
    ) -> list[NodeWithScore]:
        """执行多路召回 + 融合 + 重排。

        Args:
            query: 原始查询
            top_k: 最终返回的文档数
            expand_queries: 扩展查询列表（可选，用于增加召回多样性）

        Returns:
            排序后的 NodeWithScore 列表
        """
        queries = expand_queries if expand_queries else [query]

        # 1. 多路召回
        all_dense: list[NodeWithScore] = []
        all_sparse: list[NodeWithScore] = []

        for q in queries:
            # Dense 检索
            try:
                dense_results = self.dense_retriever._retrieve(QueryBundle(query_str=q))
                all_dense.extend(dense_results[:self.dense_top_k])
            except Exception as e:
                print(f"[MultiPath] Dense 检索失败: {e}")

            # Sparse 检索
            if self.sparse_retriever is not None:
                try:
                    sparse_results = self.sparse_retriever._retrieve(QueryBundle(query_str=q))
                    all_sparse.extend(sparse_results[:self.sparse_top_k])
                except Exception as e:
                    print(f"[MultiPath] Sparse 检索失败: {e}")

        # 去重（按 node_id）
        seen_dense: set[str] = set()
        unique_dense: list[NodeWithScore] = []
        for node in all_dense:
            nid = node.node.id_ or ""
            if nid and nid not in seen_dense:
                seen_dense.add(nid)
                unique_dense.append(node)

        seen_sparse: set[str] = set()
        unique_sparse: list[NodeWithScore] = []
        for node in all_sparse:
            nid = node.node.id_ or ""
            if nid and nid not in seen_sparse:
                seen_sparse.add(nid)
                unique_sparse.append(node)

        # 2. RRF 融合
        if unique_sparse:
            fused = self._rrf_fuse(unique_dense, unique_sparse)
        else:
            # 没有 sparse 结果时，仅使用 dense
            fused = [(n.node.id_ or str(i), n, n.score or 0.0) for i, n in enumerate(unique_dense)]

        # 3. ColBERT 细粒度重排
        candidates = fused[:self.rerank_candidates]

        if self.enable_colbert and self.colbert_reranker is not None and candidates:
            print(f"[MultiPath] ColBERT 重排 {len(candidates)} 个候选...")
            try:
                # 准备 rerank 输入
                rerank_input = [
                    (nid, node.node.get_content(metadata_mode=MetadataMode.EMBED))
                    for nid, node, _ in candidates
                ]
                reranked = self.colbert_reranker.rerank(query, rerank_input)

                # 构建最终结果
                nid_to_node = {nid: node for nid, node, _ in candidates}
                results: list[NodeWithScore] = []
                for nid, _, score in reranked[:top_k]:
                    if nid in nid_to_node:
                        node = nid_to_node[nid]
                        node.score = score
                        results.append(node)
                return results
            except Exception as e:
                print(f"[MultiPath] ColBERT 重排失败: {e}, 回退到 RRF 结果")

        # 不使用 ColBERT 时，直接返回 RRF 融合结果
        results: list[NodeWithScore] = []
        for nid, node, score in candidates[:top_k]:
            node.score = score
            results.append(node)

        return results


# ============================================================
# 层次化 Chunk 扩展策略 (Group + Expand)
# ============================================================

def expand_chunks_with_context(
    nodes: list[NodeWithScore],
    index: "VectorStoreIndex",
    expand_neighbors: int = CHUNK_EXPAND_NEIGHBORS,
    expand_parent: bool = CHUNK_EXPAND_PARENT,
    group_by_doc: bool = CHUNK_GROUP_BY_DOC,
) -> list[NodeWithScore]:
    """扩展检索结果，带上相关的上下文 chunks。
    
    实现 Group + Expand 策略：
    1. Group: 按文档/章节分组检索结果
    2. Expand: 递归扩展相邻 chunks（前后各 N 层）和父 chunks
    
    Args:
        nodes: 原始检索结果
        index: 向量索引（用于查询相关 chunks）
        expand_neighbors: 扩展的邻居层数（前后各 N 层，递归扩展）
        expand_parent: 是否扩展父 chunk
        group_by_doc: 是否按文档分组
    
    Returns:
        扩展后的节点列表（保持原有顺序，新增节点插入到相关位置）
    """
    if not nodes or not CHUNK_EXPAND_ENABLED:
        return nodes
    
    from pymilvus import connections, Collection
    from llama_index.core.schema import TextNode, NodeWithScore as NWS
    import json as _json
    
    # 确保 Milvus 连接
    conn_alias = "default"
    milvus_host = "127.0.0.1"
    milvus_port = "19530"
    
    try:
        if not connections.has_connection(conn_alias):
            connections.connect(alias=conn_alias, host=milvus_host, port=milvus_port)
        collection = Collection(COLLECTION, using=conn_alias)
        collection.load()
    except Exception as e:
        return nodes
    
    # 收集原始 IDs
    original_ids = set()
    node_by_id = {}
    parent_ids_to_expand = set()
    
    for node in nodes:
        nid = node.node.id_
        original_ids.add(nid)
        node_by_id[nid] = node
        
        metadata = node.node.metadata or {}
        
        # 收集父节点 ID
        if expand_parent:
            parent_id = metadata.get('parent_id', '')
            doc_node_id = metadata.get('doc_node_id', '')
            if parent_id and parent_id != doc_node_id and parent_id not in original_ids:
                parent_ids_to_expand.add(parent_id)
    
    # ========== 递归扩展邻居（真正的 N 层扩展） ==========
    all_expand_ids = set()
    
    if expand_neighbors > 0:
        # 第一层：从原始节点开始
        current_layer_ids = set(original_ids)
        current_layer_metadata = {nid: node_by_id[nid].node.metadata or {} for nid in original_ids}
        
        for layer in range(expand_neighbors):
            next_layer_ids = set()
            ids_to_fetch = set()
            
            # 从当前层收集需要扩展的邻居 ID
            for nid in current_layer_ids:
                metadata = current_layer_metadata.get(nid, {})
                prev_id = metadata.get('prev_id', '')
                next_id = metadata.get('next_id', '')
                
                if prev_id and prev_id not in original_ids and prev_id not in all_expand_ids:
                    ids_to_fetch.add(prev_id)
                    next_layer_ids.add(prev_id)
                if next_id and next_id not in original_ids and next_id not in all_expand_ids:
                    ids_to_fetch.add(next_id)
                    next_layer_ids.add(next_id)
            
            if not ids_to_fetch:
                break
            
            # 从 Milvus 获取这些节点的 metadata（用于下一层扩展）
            try:
                id_list = list(ids_to_fetch)
                expr = f'id in {_json.dumps(id_list)}'
                results = collection.query(
                    expr=expr, 
                    output_fields=['id', 'text', 'doc_id', 'prev_id', 'next_id', 'parent_id']
                )
                
                # 更新下一层的 metadata
                new_metadata = {}
                for r in results:
                    rid = r.get('id', '')
                    if rid:
                        new_metadata[rid] = {
                            'prev_id': r.get('prev_id', ''),
                            'next_id': r.get('next_id', ''),
                            'parent_id': r.get('parent_id', ''),
                            'doc_id': r.get('doc_id', ''),
                            'text': r.get('text', ''),
                        }
                
                all_expand_ids.update(ids_to_fetch)
                current_layer_ids = next_layer_ids
                current_layer_metadata = new_metadata
                
            except Exception as e:
                break
    
    # 添加父节点
    all_expand_ids.update(parent_ids_to_expand)
    
    if not all_expand_ids:
        return nodes
    
    # ========== 批量获取所有扩展节点 ==========
    expanded_nodes = []
    try:
        id_list = list(all_expand_ids)
        expr = f'id in {_json.dumps(id_list)}'
        results = collection.query(expr=expr, output_fields=['id', 'text', 'doc_id'])
        
        for r in results:
            node_id = r.get('id', '')
            text = r.get('text', '')
            doc_id = r.get('doc_id', '')
            
            if not text:
                continue
            
            # 构建 metadata
            metadata = {'is_context_expansion': True, 'doc_id': doc_id}
            
            # 从已有节点中找到同文档的节点，复制部分 metadata
            for orig_node in nodes:
                orig_doc = orig_node.node.metadata.get('doc_node_id') or orig_node.node.metadata.get('doc_id', '')
                if orig_doc == doc_id:
                    for key in ['source_path', 'file_name', 'doc_node_id', 'doc_type']:
                        if key in orig_node.node.metadata:
                            metadata[key] = orig_node.node.metadata[key]
                    break
            
            node = TextNode(text=text, id_=node_id, metadata=metadata)
            # 父节点给稍高分数，邻居节点给较低分数
            score = 0.15 if node_id in parent_ids_to_expand else 0.05
            expanded_nodes.append(NWS(node=node, score=score))
            
    except Exception as e:
        return nodes
    
    if not expanded_nodes:
        return nodes
    
    # 合并结果：简单追加到原始结果后面
    result = list(nodes)
    result.extend(expanded_nodes)
    
    return result


def group_chunks_by_document(
    nodes: list[NodeWithScore],
) -> dict[str, list[NodeWithScore]]:
    """按文档分组 chunks。
    
    Args:
        nodes: 检索结果
    
    Returns:
        {doc_id: [nodes...]} 映射
    """
    groups = {}
    for node in nodes:
        doc_id = node.node.metadata.get('doc_node_id') or node.node.metadata.get('source_path', 'unknown')
        if doc_id not in groups:
            groups[doc_id] = []
        groups[doc_id].append(node)
    
    # 每组内按 chunk_index 排序
    for doc_id in groups:
        groups[doc_id].sort(key=lambda n: n.node.metadata.get('chunk_index', 0))
    
    return groups


def format_grouped_context(
    groups: dict[str, list[NodeWithScore]],
    max_chunks_per_doc: int = 5,
) -> str:
    """格式化分组后的上下文，便于 LLM 理解。
    
    Args:
        groups: 按文档分组的 chunks
        max_chunks_per_doc: 每个文档最多取多少个 chunks
    
    Returns:
        格式化的上下文字符串
    """
    parts = []
    
    for doc_id, nodes in groups.items():
        # 获取文档标题
        if nodes:
            source_path = nodes[0].node.metadata.get('source_path', doc_id)
            doc_title = source_path.rsplit('/', 1)[-1].rsplit('.', 1)[0] if source_path else doc_id
        else:
            doc_title = doc_id
        
        parts.append(f"【文档：{doc_title}】")
        
        # 取前 N 个 chunks
        for i, node in enumerate(nodes[:max_chunks_per_doc]):
            chunk_idx = node.node.metadata.get('chunk_index', i)
            section = node.node.metadata.get('section_num', '')
            section_title = node.node.metadata.get('section_title', '')
            
            if section and section_title:
                parts.append(f"[{section} {section_title}]")
            elif chunk_idx is not None:
                parts.append(f"[段落 {chunk_idx + 1}]")
            
            # 标记是否为扩展的上下文
            if node.node.metadata.get('is_context_expansion'):
                parts.append(f"（上下文）{node.node.get_content()}")
            else:
                parts.append(node.node.get_content())
            parts.append("")  # 空行分隔
        
        if len(nodes) > max_chunks_per_doc:
            parts.append(f"... (还有 {len(nodes) - max_chunks_per_doc} 个相关段落)")
        
        parts.append("")  # 文档间分隔
    
    return "\n".join(parts)


def _try_build_colbert_reranker() -> Optional[ColBERTReranker]:
    """尝试构建 ColBERT reranker，失败返回 None。
    
    注意：ColBERTReranker 使用延迟加载，构建时不会实际加载模型。
    模型会在第一次调用 rerank() 时加载。
    """
    if not COLBERT_RERANK_ENABLED:
        return None

    # 验证模型名称格式
    model_name = COLBERT_MODEL.strip()
    if not model_name:
        print(f"[ColBERT] 错误: COLBERT_MODEL 为空")
        return None
    
    # 检查是否是常见的错误模型名称
    if model_name.lower() in {"qwen3-reranker", "qwen3-vl-reranker", "qwen-vl-reranker"}:
        print(f"[ColBERT] 警告: 模型名称 '{model_name}' 不正确")
        print(f"[ColBERT] 建议使用完整路径，如: Qwen/Qwen3-Reranker-4B")
        print(f"[ColBERT] 或使用推荐模型: BAAI/bge-reranker-v2-m3")
        return None

    try:
        # 只创建实例，不加载模型（延迟加载）
        reranker = ColBERTReranker(
            model_name=model_name,
            max_length=COLBERT_MAX_LENGTH,
            batch_size=COLBERT_BATCH_SIZE,
        )
        print(f"[ColBERT] Reranker 实例创建成功（模型将延迟加载）")
        return reranker
    except Exception as e:
        print(f"[ColBERT] 构建失败: {e}")
        return None


# BM25 停用词表（中英文常见虚词，避免稀释关键词权重）
_STOPWORDS: set[str] = {
    # 中文虚词
    "的", "了", "和", "与", "或", "在", "是", "为", "以", "及", "等", "之", "将", "被", "由", "对",
    "其", "该", "这", "那", "有", "无", "中", "上", "下", "内", "外", "可", "不", "应", "需", "到",
    "从", "于", "并", "而", "但", "若", "如", "则", "使", "让", "把", "给", "因", "故", "所", "者",
    # 英文虚词
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for", "by", "with", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "can", "this", "that", "these", "those", "it", "its",
}

# PCB领域专业术语（保护不被切分）
_PCB_TERMS: set[str] = {
    "pcb", "fpc", "hdi", "ccl", "emi", "emc", "via", "bga", "qfp", "smd", "dip", "sot", "mlcc",
    "ipc", "jedec", "mil", "oz", "tg", "dk", "df", "cte", "caf", "ecm", "pth", "npth", "hasl",
    "enig", "osp", "immersion", "gold", "silver", "tin", "annular", "ring", "aspect", "ratio",
    "impedance", "differential", "single", "ended", "microstrip", "stripline", "coplanar",
    "prepreg", "laminate", "copper", "foil", "solder", "mask", "silkscreen", "legend",
    "gerber", "drill", "route", "panelize", "netlist", "drc", "dfm", "cam",
    #新添加
    "fr-4", "g-11", "fr-5", "polyimide", "aluminum", "substrate", "metal", "base", 
    "core", "bonding", "sheet", "dielectric", "layer", "coverlay", "adhesive",
    "etching", "lamination", "plating", "desmear", "debur", "concave", "etch", "negative", 
    "etch", "blister", "delamination", "void", "dross", "solder", "wicking",
    "ctp", "ctb", "ppb", "test", "pattern", "land", "pad", "pristine", "area", "clearance", 
    "annular", "ring", "thermal", "relief", "test", "point", "fixture",
    "peel", "strength", "pull", "strength", "thermal", "conductivity", "thermal", "impedance", "cti", "arc", 
    "resistance", "voltage", "breakdown", "insulation", "resistance", "volume", "resistivity", "surface", "resistivity",
    "lsi", "package", "ball", "grid", "array", "bonding", "wire", "finger",
    "anti", "pad", "lpb", "spdr", "vna", "dsc", "tga", "tma", "dma",
    "gb/t", "iec", "sj/t", "qualification", "approval", "inspection", "lot", 
    "acceptance", "sampling", "aql", "reliability", "environmental", "stress", "temperature", "cycle"
    "schematic", "schdoc", "pcbdoc", "schlib", "pcblib", "prjpcb", "intlib", "netlabel", 
    "powerport", "offsheet", "connector", "sheet", "symbol", "sheetentry", "room", "polygon", "pour", "teardrop", "fanout", "interactive", "routing",
    "signal", "integrity", "si", "crosstalk", "reflection", "transmission", "line", "differential", 
    "pair", "matched", "length", "stub", "via", "stub", "impedance", "controlled",
    "component", "footprint", "model", "simulation", "library", "integrated", "database", 
    "svndblib", "dblib", "pcb3dlib", "vhdllib",
    "short", "circuit", "unrouted", "net", "unconnected", "pin", "width", "constraint", "routing", 
    "topology", "priority", "layer", "direction", "corner", "style", "via", "style",
    "power", "integrity", "pi", "simulation", "probe", "waveform", "oscillation", "overshoot", 
    "undershoot", "slope", "flight", "time", "stimulus",
    "smart", "pdf", "bill", "of", "materials", "bom", "cross", "reference", "report",  
    "drawing", "nc", "drill", "file", "gerber", "x2", "odb++", "step", "file",
    "timing", "constraint", "setup", "time", "hold", "time", "jitter", "skew", "propagation", "delay",
    "switch", "delay", "settle", "delay", "daisy", "chain", "star", "termination", "series", "resistor", "parallel", "resistor",
    "ibis", "model", "dml", "spice", "eslice", "sigrnoise", "sigexplorer", "sigwave", "constraint", "manager", "layout", 
    "cross", "section", "layer", "stack", "manager", "pdnsim", "thermal", "simulation",
    "characteristic", "impedance", "reflection", "coefficient", "standing", "wave", "ratio", "swr", 
    "nearend", "farend", "eye", "diagram", "eye", "height", "eye", "width", "noise", "margin", "monotonic",
    "prelayout", "postlayout", "topology", "extraction", "parameter", "sweep", "pulse", "duty", 
    "cycle", "overshoot", "high", "overshoot", "low",
    "power", "distribution", "system", "pds", "decoupling", "capacitor", "bypass", "capacitor", 
    "esr", "esl", "simultaneous", "switch", "noise", "ssn", "ground", "bounce", "power", "bounce",
    "offsheetconnector", "sheetentry", "engineeringchangeorder", "eco", "annotate", "crossprobe", 
    "crossselect", "mode", "polygonrepour", "maskexpansion", "soldermask", "pastemask", "powerplane", 
    "splitplane", "netclass", "componentclass", "designchannelclass", "testpoint", "solderpaste", 
    "reflowsoldering", "wavesoldering", "plasmacleaning",
    "componentcrossreferencereport", "designrulecheckreport", "boardinformationreport", "netstatusreport", 
    "singlepinnetsreport", "portcrossreferencereport", "simplebom", "csvfile", "gerberx2",
    "layerstackmanager", "splitplaneeditor", "polygonconnectstyle", "powerplaneconnectstyle", "fanoutcontrol", 
    "differentialpaireditor", "routingconflictresolution", "pushobstacles", "walkaroundobstacles", "autorouteall", 
    "autoroutenet", "autoroutearea", "autoroutecomponent", "unrouteall", "unroutenet", "unrouteconnection", "roomdefinition", "componentsclearancepermitted", "layerheightconstraint",
    "schematiclibraryeditor", "pcblibraryeditor", "libraryextractsources", "componentrulecheck", "libraryreport", 
    "componentgenerator", "multipartcomponent", "pinpropertyeditor", "symboleditor", "footprintwizard", "3dmodeeditor",
    "boardinsight", "display", "boardinsightlens", "mousewheelconfiguration", "graphicalediting", "preferences", 
    "snapgrid", "electricalgrid", "visiblegrid", "layercolors", "showhideobjects", "boardoptions", "measurementunit", "imperial", "metric", "routingwidthconstraints", "viastyleconstraints",
    "netlabel", "powerport", "offsheetconnector", "sheetentry", "bus", "busentry", "junction", "manualjunction", 
    "differentialpairnet", "netliststatus", "singlepinnets",
    "vsrc", "isrc", "vsin", "isin", "vpulse", "ipulse", "vpwl", "ipwl", "vexp", "iexp", "vsffm", 
    "isffm", "initialvoltage", "nodevoltage", "ns",
    "operatingpointanalysis", "transientanalysis", "dcsweepanalysis", "acsmallsignalanalysis", "noiseanalysis", "polezeroanalysis", "transferfunctionanalysis", "temperaturesweep", "parametersweep", "montecarloanalysis", "fourieranalysis", "simmodel", "simdata",
    "toplevelschematic", "subschematic", "sheetsymbol", "hierarchicaldesign", "updownhierarchy", "crossreference",
    "thermalreliefpad", "heatsink", "thermalvia", "exposedpad", "epad", "powerpad", "heatsinking", 
    "thermalmanagement", "copperpour", "gridpour", "thermalpad",
    "schlib", "pcblib", "intlib", "dblib", "svndblib", "prjpcb", "prjfpg", "schdoc", "pcbdoc", "offsheetconnector", 
    "sheetentry", "room", "polygonpour", "teardrop", "fanout", "interactiverouting", "differentialpairrouting", "lengthmatching", "testpoint", "clearanceconstraint", "layerstackmanager", "designrulecheck", "engineeringchangeorder", "eco", "annotate", "crossprobe", "crossselectmode", "polygonrepour", "teardropadd", "maskexpansion", "soldermask", "pastemask", "powerplane", "splitplane", "netclass", "componentclass", "designchannelclass",
    "componentcrossreferencereport", "designrulecheckreport", "boardinformationreport", "netstatusreport", 
    "singlepinnetsreport", "portcrossreferencereport", "billofmaterials", "bom", "simplebom", "csvfile", 
    "smartpdf", "export", "stepfile", "odb++", "gerberx2",
    "layerstackmanager", "splitplaneeditor", "polygonconnectstyle", "powerplaneconnectstyle", "fanoutcontrol", 
    "differentialpaireditor", "interactiveroutingconflictresolution", "pushobstacles", "walkaroundobstacles", "autorouteall", "autoroutenet", "autoroutearea", "autoroutecomponent", "unrouteall", "unroutenet", "unrouteconnection", "roomdefinition", "componentsclearancepermitted", "layerheightconstraint",
    "schematiclibraryeditor", "pcblibraryeditor", "libraryextractsources", "componentrulecheck", "libraryreport", 
    "componentgenerator", "multipartcomponent", "pinpropertyeditor", "symboleditor", "footprintwizard", "3dmodeleditor",
    "boardinsight", "display", "boardinsightlens", "mousewheelconfiguration", "graphicalediting", "preferences", 
    "snapgrid", "electricalgrid", "visiblegrid", "layercolors", "showhideobjects", "boardoptions", "measurementunit", "imperial", "metric", "routingwidthconstraints", "viastyleconstraints",
    "tms320f2812", "pgf", "dip-8", "soic", "qfn", "lga", "csp", "padshape", "annularring", "solder maske expansion", 
    "pastemaske expansion", "thermalreliefpad", "holesize", "viadiameter", "throughhole", "blindvia", "buriedvia", "microvia",
    "clearanceconstraint", "widthconstraint", "viastyleconstraint", "routingpriority", "layerdirection", 
    "cornerstyle", "fanoutcontrol", "differentialpairconstraint", "minimumannularring", "holesizeconstraint",
    "signalintegritysimulation", "powerintegritysimulation", "crosstalkanalysis", "reflectionanalysis", 
    "eyediagram", "jitteranalysis", "propagationdelay", "skewmatching", "impedancecalculation",
    "lib", "pcb3dlib", "vhdllib", "dxpprf", "dft", "rep", "html", "xml", "drc", "net",
    "crossprobe", "crossselectmode", "boardinsightlens", "autopan", "zoom", "maskhighlight", "interactiveediting", 
    "inplaceediting", "drag", "orthogonal", "snaptogrid", "electricalgrid", "visiblegrid", "layerstackmanager", 
    "boardoptions", "documentoptions", "preferences", "mousewheelconfiguration",
    "bus", "busentry", "junction", "manualjunction", "netclass", "differentialpair", "netlist", "netstatus", "singlepinnets",
    "etchresist", "develop", "strip", "platingcopper", "nickel", "gold", "tin", "lead-free", "hasl", "enig", 
    "osp", "immersionsilver", "solderpaste", "reflowsoldering", "wavesoldering", "drillbit", "routerbit", "debur", "desmear", "plasmacleaning",
    "visualinspection", "electricaltest", "continuitytest", "insulationresistancetest", "peelstrengthtest", 
    "thermalstresstest", "humiditytest", "drcviolationreport", "boardinformationreport", "componentcrossreferencereport",
    "gu555", "lf356n", "ne555p", "opa129", "opa695", "ads5500", "sl55016", "tlv1543", "adc774", "2n3904",
    "teardrop", "polygonpour", "splitplane", "powerplane", "groundplane", "solderpaste", "reflowsoldering", "wavesoldering", "plasmacleaning", "thermalvia", "exposedpad", "epad", "heatsink", "thermalmanagement",
    "signalintegrity", "crosstalk", "reflection", "overshoot", "undershoot", "jitter", "skew", 
    "propagationdelay", "eyediagram", "ibismodel", "simulation",
    "pcblibrary", "footprintwizard", "componentwizard", "schematiclibrary", "symboleditor", "pineditor", "3dmodel",
    "designrulecheck", "drc", "engineeringchangeorder", "eco", "billofmaterials", "bom", "netlist", "gerberfile", "drillfile", "odb++", "stepfile",
    "interactiverouting", "autorouting", "fanoutrouting", "differentialpairrouting", 
    "lengthmatching", "topologyrouting",
    "layerstack", "signal layer", "powerlayer", "groundlayer", "mechanical layer", "masklayer", 
    "silkscreenlayer", "keepoutlayer",
    "pad", "via", "trace", "copperpour", "thermalrelief", "solderbridge", "clearance", "annularring",
    "pcbeditor", "schematiceditor", "cameditor", "simulationeditor", "libraryeditor",
    "project", "prjpcb", "schdoc", "pcbdoc", "schlib", "pcblib", "intlib",
    "compile", "annotate", "backannotate", "update", "import", "changes", "validate", "execute",
    "no-erc", "marker", "compilemask", "layoutdirective", "testpoint",
    "netlabel", "powerport", "offsheetconnector", "sheetentry", "bus", "junction",
    "pcbboard", "boardoutline", "mechanicalboundary", "electricalboundary", "routingarea",
    "componentplacement", "autoplacement", "manualplacement", "alignment", "distribution",
    "wire", "bus", "busentry", "netlabel", "port", "junction", "node",
    "powerintegrity", "pdnsim", "decouplingcapacitor", "bypasscapacitor", "esr", "esl", "ssn", "groundbounce",
    "thermalanalysis", "heatdissipation", "thermalconductivity", "thermalresistance", "heatsink", "thermalvia",
    "manufacturingoutput", "gerber", "drilldrawing", "ncdrill", "odb++", "step", "pdf",
    "designformanufacturability", "dfm", "designforserviceability", "dfs", "designfortestability", "dft"
    "signalpropagation", "transmissionline", "characteristicimpedance", "propagationdelay", "skew", 
    "jitter", "eyeopening", "noisemargin", "crosstalkcoupling", "nearendcrosstalk", 
    "farendcrosstalk", "emccompliance", "shielding", "groundplane", "powerplane", 
    "decoupling", "placement", "stub", "length", "via", "stub", "reduction",
    "schematiclibrary", "pcblibrary", "integratedlibrary", "schematic symbol", 
    "pcbfootprint", "3dmodel", "simulationmodel", "signalintegritymodel", "pinmapping", "pinout", 
    "packageoutline", "thermalpad", "exposedpad", "landpattern", "padpitch", "paddiameter", "hole diameter",
    "electricalrule", "routingrule", "manufacturingrule", "placementrule", 
    "signalintegrityrule", "powerintegrityrule", "clearancerule", "widthrule", "spacingrule", 
    "topologyrule", "terminationrule", "differentialpairrule", "matchedlengthrule", "stub lengthrule", "viacountrule",
    "transientanalysis", "frequencyanalysis", "acanalysis", "dcanalysis", "noisesimulation", 
    "jitteranalysis", "eyediagramanalysis", "crosstalksimulation", "reflectionsimulation", 
    "impedancecalculation", "transmissiondelayanalysis",
    "leadfree", "hasl", "enig", "osp", "immersiongold", "immersiontin", "immersion silver", 
    "electroplating", "electrolessplating", "etchingprocess", "laminationprocess", "drilling", 
    "routing", "deburring", "desmearing", "plasmacleaning", "solderpasteprinting", "reflowsoldering", "wavesoldering",
    "backannotation", "netlistcomparison", "designcomparison", "engineeringchangeorder", 
    "ecomanagement", "parametereditor", "ruleeditor", "constrainteditor", "layerstackeditor", 
    "boardoutlineeditor", "shapeeditor", "polygoneditor",
    "continuitytest", "insulationtest", "hipottest", "thermalcyclingtest", "humiditytest", 
    "vibrationtest", "drcverification", "siverification", "piverification", "emcverification"
        "glassstransitiontemperature", "tma", "dsc", "tga", "thermaldecompositiontemperature", "td", "zaxiscte", 
        "thermalstratificationtime", "halogencontent", "ionchromatography", "oxygenbombcombustion", "oxygenbombcombustion",
    "verticalburning", "horizontalburning", "glowwiretest", "needleflametest", "flameretardantgrade", "fv-0", "fv-1", "fv-2", "fhb",
    "electricalstrength", "dielectricconstant", "dissipationfactor", "volumeresistivity", "surfaceresistivity", 
    "insulationresistance", "wetinsulationresistance", "voltagebreakdown",
    "ioncontamination", "chloridecontent", "sodiumchlorideequivalent", "solventresistance", "fluxresistance",
    "peelstrength", "pulloutstrength", "warpage", "twist", "bendingfatigue", "flexuralstrength",
    "microsection", "microscopiccutting", "etchback", "negativeetchback", "desmearing", "glassfiberprotrusion","resindrillingcontamination",
    "solderability", "wetting", "nonwetting", "dewetting", "acceleratedaging", "steamoxygenaging",
    "copperfoil", "electrodepositedcopperfoil", "rolledcopperfoil", "annealedcopperfoil", "shinyside", 
    "matteside", "treatedside", "thincopperfoil", "adhesivecoatedfoil",
    "e-glassfiber", "d-glassfiber", "s-glassfiber", "glassfabric", "non-wovenfabric", "warp-wise", "weft-wise", "threadcount", "plainweave", "size", "couplingagent",
    "epoxyresin", "phenolicresin", "polyesterresin", "unsaturatedpolyester", "acrylicresin", 
    "melamineformaldehyderesin", "polytetrafluoroethylene", "ptfe", "polyimideresin", "bismaleimidetriazineresin", "btresin", "perfluorinatedethylenepropylenecopolymer", "fep",
    "epoxyequivalent", "epoxyvalue", "dicyandiamide", "binder", "adhesive", "curingagent", "flameretardant", "bondenhancingtreatment", "compositemetallicmaterial", "carrierfoil", "curingtime",
    "prepreg", "bondingsheet", "adhesivecoateddielectricfilm", "unsupportedadhesivefilm", "laminateforadditiveprocess", 
    "masslaminationpanel",
    "subtractiveprocess", "additiveprocess", "semi-additiveprocess", "tenting", "smobc", "blank", "panel", 
    "multipleprintedpanel", "multiplepattern", "metallization", "touchingup", "reworking", "repairing",
    "artworkmaster", "productionmaster", "single-imageproductionmaster", "multiple-imageproductionmaster", "photographicreductiondimension", "photographicfilm", "silverfilm", "diazofilm", "positive", "positivepattern", "negative", "negativepattern", "photoplotting", "step-and-repeat", "flipflop", "emulsionside", "rightreading", "definition", "resolution", "density",
    "photoresist", "dryfilmphotoresist", "liquidphotoresist", "positive-actingresist", "negative-actingresist", 
    "platingresist", "platedresist", "permanentresist", "solderresist", "soldermaskink", "dryfilmsoldermask", 
    "liquidphotosensitivesolderresist", "stepscale", "grayscale", "exposure", "imaging", "screenprinting", "stencil", "thixotopic", "etchant", "etchingindicator", "brightdip", "undercut",
    "flex-rigidprintedboard", "rigidprintedboard", "flexibleprintedboard", "single-sidedprintedboard", 
    "double-sidedprintedboard", "multilayerprintedboard", "metalcoreprintedboard", "motherboard", "backplane", "multi-wiringprintedboard", "ceramic substrateprintedboard", "printedcomponent", "grid", "componentside", "solderside", "printing", "conductor", "conductorside", "flushconductor", "pattern", "conductivepattern", "non-conductivepattern", "legend", "mark",
    "base material", "metal-cladbase material", "copper-cladlaminate", "single-sidedcopper-cladlaminate", "double-sidedcopper-cladlaminate", "compositelaminate", "thinlaminate", "metalcorecopper-cladlaminate",
    "land", "offsetland", "nonfunctionalland", "landpattern", "anchoringspur", "annularring", "conductorlayer", "circuitlayer", "conductorlayerNo.1", "internallayer", "externallayer", "layer-to-layerspacing", "signalplane", "ground", "groundplane", "groundplaneclearance", "voltageplane", "voltageplaneclearance", "heatsinkplane", "heatshield", "primaryside", "secondaryside", "supportingplane", "basicdimension", "centertocenterspacing", "designspacingofconductor", "designwidthofconductor", "conductorspacing", "edgespacing", "pitch", "span", "edge-boardconnector", "right-angleedgeconnector", "connectorarea", "edgeboardcontact", "printedcontact", "contactarea", "componentlead", "componentpin", "nonfunctionalinterfacialconnection", "jumperwire", "haywire", "busbar", "cross-hatching", "datumreference", "referencedimension", "referenceedge", "cornermark", "trimline", "probepoint", "polarizingslot", "keyingslot", "registermark", "transmissionline", "capacitivecoupling", "crosstalk", "continuity", "current-carryingcapacity", "minimumelectricalspacing", "electromagneticshielding", "digitizing", "from-to-list", "netlist", "computer-aideddrawing", "printedboardcomputer-aideddesign", "placement", "routing", "designrulechecking",
    "paste misalignment", "excessivepaste", "insufficientpaste", "nopaste", "pastesmearing", "pastebridging", 
    "pastedepositshape",
    "adhesivemisalignment", "excessiveadhesive", "insufficientadhesive", "noadhesive", "adhesivestringing", "adhesivedotshape",
    "componentmisalignment", "missingcomponent", "reversedcomponent", "wrongcomponent", "componentonedge", "damagedcomponentbeforesoldering", "damagedcomponentaftersoldering", "damagedprintedboard",
    "solderjointmisalignment", "solderjointbridging", "insufficientsolder", "nosolderjoint", 
    "componenttombstoning", "solderwicking", "disturbedsolderjoint", "solderballs", "soldersplashes", "solderwebs", "badwetting",
    "microvia", "blindvia", "buriedvia", "throughholevia", "landlesshole", "componenthole", "mountinghole", "supportedhole", "unsupportedhole", "clearancehole", "accesshole", "dimensionedhole", "holelocation", "holepattern",
    "halogen-free", "chlorinecontent", "brominecontent", "ionchromatograph", "oxygenbomb", "combustion", "absorptionliquid", "sodiumhydroxide",
    "thermalreliefpad", "exposedpad", "epad", "powerpad", "heatsink", "thermalvia", "thermalmanagement", "copperpour", "gridpour",
    "dielectricmaterial", "prepreg", "bondingsheet", "adhesive", "coverlay", "basefilm", "polyimidfilm", "polyesterfilm", "fepfilm",
    "solderingflux", "rosinflux", "no-cleanflux", "corrosiveflux", "non-corrosiveflux",
    "barcode", "twodimensionalsymbol", "electroniccomponentpackaginglabel",
    "ballgridarray", "bga", "landgridarray", "lga", "solderjointvoid", "solderjointdurability", "temperaturecycling",
    "highdensityinterconnect", "hdi", "metalsubstratecopper-cladlaminate", "microwavefrequency", "relativedielectricconstant", "losstangent", "separatedielectricresonatormethod",
    "large-scaleintegratedcircuit", "lsi", "packaging", "common designstructure",
    "electronicmodule", "arraytypepackagesurfacemountdevice", "solderjointdurabilitytest"
}


# jieba 分词支持（延迟加载）
_JIEBA_INITIALIZED = False
_JIEBA_AVAILABLE = False


def _init_jieba():
    """延迟初始化 jieba 分词器，添加 PCB 领域自定义词典。"""
    global _JIEBA_INITIALIZED, _JIEBA_AVAILABLE
    if _JIEBA_INITIALIZED:
        return _JIEBA_AVAILABLE
    _JIEBA_INITIALIZED = True
    try:
        import jieba
        # 添加 PCB 领域自定义词典（高频专业词汇）
        _pcb_custom_words = [
            # 材料
            "覆铜板", "半固化片", "铜箔", "聚酰亚胺", "层压板", "基材", "粘结片",
            "玻璃布", "环氧树脂", "酚醛树脂", "聚酯薄膜", "聚四氟乙烯",
            # 结构
            "焊盘", "孔环", "过孔", "盲孔", "埋孔", "微孔", "导通孔",
            "阻焊膜", "丝印层", "阻焊层", "助焊层", "敷铜", "覆铜",
            "线路板", "印制板", "印制电路板", "挠性板", "柔性板",
            "多层板", "双面板", "单面板", "金属基板",
            # 工艺
            "蚀刻", "电镀", "沉金", "喷锡", "沉银", "沉锡",
            "热风整平", "化学镀", "有机保焊膜", "去钻污", "去毛刺",
            "层压", "压合", "钻孔", "铣板", "开窗",
            # 参数
            "阻抗", "介电常数", "损耗因子", "剥离强度", "翘曲度",
            "特性阻抗", "差分阻抗", "线宽", "线距", "板厚",
            "耐电压", "绝缘电阻", "体积电阻率", "表面电阻率",
            "击穿电压", "耐热性", "吸水率", "热膨胀系数",
            # 测试
            "飞针测试", "热冲击", "温度循环", "耐湿热", "可靠性测试",
            "显微切片", "金相分析", "离子污染",
            # 标准
            "国家标准", "行业标准", "国际标准", "分规范", "总规范",
            "试验方法", "技术条件", "验收规范", "质量保证",
            # EDA
            "原理图", "封装库", "网表", "设计规则", "约束规则",
            "自动布线", "手动布线", "差分对", "等长走线",
            "信号完整性", "电源完整性", "串扰", "反射",
            "元器件", "元件库", "管脚", "引脚", "焊盘库",
        ]
        for w in _pcb_custom_words:
            jieba.add_word(w, freq=50000)
        
        # 预加载词典
        jieba.initialize()
        _JIEBA_AVAILABLE = True
        return True
    except ImportError:
        _JIEBA_AVAILABLE = False
        return False


def _tokenize(text: str) -> list[str]:
    """面向 PCB 领域的改进分词：

    - 英文/数字/符号：保留连续串（如 `annular ring`, `IPC-6012`, `50ohm`, `diff_pair`）
    - 中文：优先使用 jieba 分词（比纯 bigram 准确度大幅提升）
    - 回退到 bigram + 单字（无 jieba 时）
    - 过滤停用词，保护专业术语
    - 添加 n-gram 以提升短语匹配
    """

    if not text:
        return []

    use_jieba = _init_jieba()

    tokens: list[str] = []
    raw_tokens: list[str] = []  # 保存原始 token 用于生成 n-gram

    for part in re.findall(r"[A-Za-z0-9_+\-./]+|[\u4e00-\u9fff]+", text):
        if not part:
            continue
        if re.fullmatch(r"[A-Za-z0-9_+\-./]+", part):
            lower = part.lower()
            # 过滤停用词（但保留专业术语）
            if lower in _STOPWORDS and lower not in _PCB_TERMS:
                continue
            tokens.append(lower)
            raw_tokens.append(lower)
            continue

        # CJK 文本处理
        part = part.strip()
        if not part:
            continue
        # 过滤单字停用词
        if len(part) == 1 and part in _STOPWORDS:
            continue

        if use_jieba:
            # ===== 使用 jieba 精确模式分词 =====
            import jieba
            # 精确模式 + 搜索引擎模式双重分词，最大化召回
            cut_words = list(jieba.cut(part, cut_all=False))
            search_words = list(jieba.cut_for_search(part))
            
            # 合并去重（保持顺序）
            seen_words: set[str] = set()
            jieba_tokens: list[str] = []
            for w in cut_words + search_words:
                w = w.strip()
                if not w or w in _STOPWORDS:
                    continue
                if w not in seen_words:
                    seen_words.add(w)
                    jieba_tokens.append(w)
            
            if jieba_tokens:
                tokens.extend(jieba_tokens)
                raw_tokens.extend(jieba_tokens)
            else:
                # jieba 输出为空时回退
                tokens.append(part)
                raw_tokens.append(part)
            
            # 补充添加完整词组（用于精确匹配长术语）
            if len(part) >= 2:
                tokens.append(part)
        else:
            # ===== 回退：bigram + 单字 =====
            tokens.append(part)
            raw_tokens.append(part)
            if len(part) >= 2:
                tokens.extend([part[i : i + 2] for i in range(len(part) - 1)])
            tokens.extend([c for c in part if c not in _STOPWORDS])

    # 添加英文 bigram（提升短语匹配，如 "annular ring"）
    en_tokens = [t for t in raw_tokens if re.fullmatch(r"[a-z0-9_+\-./]+", t)]
    if len(en_tokens) >= 2:
        for i in range(len(en_tokens) - 1):
            bigram = f"{en_tokens[i]}_{en_tokens[i+1]}"
            tokens.append(bigram)

    return tokens




def _extract_query_filters(user_query: str) -> tuple[str, Optional[MetadataFilters]]:
    """从用户输入里提取过滤条件，并返回 (clean_query, filters)。

    支持：
    - key=value / key:value（vendor/eda/layers/oz/source_type/effective_date）
    - 口语："JLCPCB + 4层 + 1oz" / "4-layer" / "kicad"
    """

    raw = (user_query or "").strip()
    if not raw:
        return user_query, None

    extracted: dict[str, object] = {}

    # 显式键值优先
    for m in re.finditer(
        r"\b(vendor|eda|layers|layer|oz|source_type|effective_date)\s*[:=]\s*([^\s;，,]+)",
        raw,
        re.I,
    ):
        k = m.group(1).lower()
        v = m.group(2)
        if k in {"layers", "layer"}:
            try:
                extracted["layer_count"] = int(re.sub(r"\D+", "", v) or "0")
            except Exception:
                pass
        elif k == "oz":
            try:
                extracted["copper_oz"] = float(re.sub(r"[^0-9.]+", "", v) or "0")
            except Exception:
                pass
        else:
            extracted[k] = v

    # vendor/eda 的口语匹配
    low = raw.lower()
    if "jlcpcb" in low or "嘉立创" in raw:
        extracted.setdefault("vendor", "JLCPCB")
    if "pcbway" in low:
        extracted.setdefault("vendor", "PCBWay")
    if "kicad" in low:
        extracted.setdefault("eda", "kicad")
    if "altium" in low or re.search(r"\bad\b", low):
        extracted.setdefault("eda", "altium")
    if "allegro" in low:
        extracted.setdefault("eda", "allegro")

    # layers: 4层 / 4-layer / 4 layers / 4L
    m = re.search(r"(?<!\d)(\d{1,2})\s*(?:层|layer|layers)\b", low)
    if not m:
        m = re.search(r"(?<!\d)(\d{1,2})\s*l\b", low)
    if m:
        try:
            extracted.setdefault("layer_count", int(m.group(1)))
        except Exception:
            pass

    # copper oz
    m = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*oz\b", low)
    if m:
        try:
            extracted.setdefault("copper_oz", float(m.group(1)))
        except Exception:
            pass

    if not extracted:
        return user_query, None

    filters: list[MetadataFilter] = []
    for k, v in extracted.items():
        if k in {"layers", "layer"}:
            k = "layer_count"
        if k == "oz":
            k = "copper_oz"
        filters.append(MetadataFilter(key=str(k), value=v, operator=FilterOperator.EQ))

    clean = re.sub(
        r"\b(vendor|eda|layers|layer|oz|source_type|effective_date)\s*[:=]\s*([^\s;，,]+)",
        " ",
        raw,
        flags=re.I,
    )
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean or user_query, MetadataFilters(filters=filters, condition=FilterCondition.AND)


@dataclass
class _Bm25Index:
    """改进的 BM25+ 索引实现。

    改进点：
    1. BM25+ 变体：添加 delta=1 防止长文档过度惩罚
    2. 改进的 IDF 计算：使用 log((N+1)/(n+0.5)) 避免负值
    3. 支持词项权重：专业术语额外加权
    4. 文档频率过滤：忽略过于常见的词（DF > 80% 文档）
    """

    docs: list[TextNode]
    tfs: list[Counter]
    df: Counter
    avgdl: float
    k1: float = 1.5  # 调高 k1 增强词频影响（PCB文档专业术语重复度高）
    b: float = 0.75
    delta: float = 1.0  # BM25+ 的 delta 参数，防止长文档过度惩罚

    def score(self, query_tokens: list[str], doc_i: int) -> float:
        """计算 BM25+ 评分。

        BM25+ 公式：
        score = Σ IDF(t) * (tf(t,d) * (k1+1)) / (tf(t,d) + k1*(1-b+b*|d|/avgdl)) + delta
        """
        if not query_tokens:
            return 0.0
        tf = self.tfs[doc_i]
        dl = sum(tf.values())
        if dl <= 0:
            return 0.0

        score = 0.0
        N = max(len(self.docs), 1)
        df_threshold = N * 0.8  # 文档频率阈值：超过80%文档包含的词不计入

        # 去重查询词并计算词频（同一查询词出现多次应增加权重）
        query_tf = Counter(query_tokens)

        for term, qtf in query_tf.items():
            f = tf.get(term)
            if not f:
                continue

            n_qi = self.df.get(term, 0)

            # 过滤过于常见的词（可能是停用词漏网之鱼）
            if n_qi > df_threshold:
                continue

            # 改进的 IDF：使用 log((N+1)/(n+0.5)) 确保非负
            idf = log((N + 1.0) / (n_qi + 0.5))

            # 专业术语加权：如果是 PCB 专业术语，IDF 额外乘以 1.2
            if term in _PCB_TERMS:
                idf *= 1.2

            # BM25+ 核心计算
            denom = f + self.k1 * (1.0 - self.b + self.b * dl / (self.avgdl or 1.0))
            tf_component = (f * (self.k1 + 1.0)) / (denom or 1.0)

            # BM25+ 添加 delta 防止长文档过度惩罚
            term_score = idf * (tf_component + self.delta)

            # 查询词频加权（同一词出现多次在查询中，增加权重）
            score += term_score * min(qtf, 3)  # 最多3倍

        return score


class LocalBM25Retriever(BaseRetriever):
    def __init__(
        self,
        bm25: _Bm25Index,
        similarity_top_k: int = 20,
        filters: Optional[MetadataFilters] = None,
    ):
        super().__init__()
        self._bm25 = bm25
        self._top_k = similarity_top_k
        self._filters = filters

    def _match_filter(self, metadata: dict, f: MetadataFilter) -> bool:
        val = metadata.get(f.key)
        op = f.operator
        tgt = f.value

        if op == FilterOperator.EQ:
            return val == tgt
        if op == FilterOperator.NE:
            return val != tgt
        if op == FilterOperator.IN:
            return val in (tgt or [])
        if op == FilterOperator.NIN:
            return val not in (tgt or [])
        if op == FilterOperator.GT:
            try:
                return float(val) > float(tgt)  # type: ignore[arg-type]
            except Exception:
                return False
        if op == FilterOperator.GTE:
            try:
                return float(val) >= float(tgt)  # type: ignore[arg-type]
            except Exception:
                return False
        if op == FilterOperator.LT:
            try:
                return float(val) < float(tgt)  # type: ignore[arg-type]
            except Exception:
                return False
        if op == FilterOperator.LTE:
            try:
                return float(val) <= float(tgt)  # type: ignore[arg-type]
            except Exception:
                return False
        if op == FilterOperator.CONTAINS:
            if isinstance(val, str) and isinstance(tgt, str):
                return tgt in val
            if isinstance(val, list):
                return tgt in val
            return False

        # 未实现的操作符：默认放行（避免误杀）
        return True

    def _match_filters(self, metadata: dict, filters: MetadataFilters) -> bool:
        parts: list[bool] = []
        for item in filters.filters:
            if isinstance(item, MetadataFilters):
                parts.append(self._match_filters(metadata, item))
            else:
                parts.append(self._match_filter(metadata, item))
        if filters.condition == FilterCondition.OR:
            return any(parts)
        return all(parts)

    def _retrieve(self, query_bundle) -> list[NodeWithScore]:
        query_str = getattr(query_bundle, "query_str", str(query_bundle))
        q_tokens = _tokenize(query_str)
        scored: list[tuple[float, int]] = []
        for i in range(len(self._bm25.docs)):
            if self._filters is not None:
                md = self._bm25.docs[i].metadata or {}
                if not self._match_filters(md, self._filters):
                    continue
            s = self._bm25.score(q_tokens, i)
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[NodeWithScore] = []
        for s, i in scored[: self._top_k]:
            results.append(NodeWithScore(node=self._bm25.docs[i], score=float(s)))
        return results


def _iter_milvus_rows(vector_store: MilvusVectorStore) -> Iterable[dict]:
    output_fields = ["id", "doc_id", "text", *LEXICAL_METADATA_KEYS]
    it = vector_store.client.query_iterator(
        collection_name=COLLECTION,
        batch_size=LEXICAL_EXPORT_BATCH,
        limit=LEXICAL_EXPORT_LIMIT,
        filter="id != ''",
        output_fields=output_fields,
        timeout=60,
    )
    while True:
        try:
            batch = it.next()
        except StopIteration:
            break
        except Exception:
            break

        if not batch:
            break

        if isinstance(batch, dict):
            yield batch
            continue

        for row in batch:
            if isinstance(row, dict):
                yield row


def _truncate_utf8_text(text: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    b = text.encode("utf-8")
    if len(b) <= max_bytes:
        return text
    return b[:max_bytes].decode("utf-8", errors="ignore")


def _load_or_build_bm25(vector_store: MilvusVectorStore) -> Optional[_Bm25Index]:
    if not LEXICAL_ENABLED:
        return None

    cache_path = Path(LEXICAL_CACHE_PATH)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    docs: list[TextNode] = []
    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    text = obj.get("text")
                    if not isinstance(text, str) or not text.strip():
                        continue
                    text = _truncate_utf8_text(text, LEXICAL_MAX_BYTES)
                    node = TextNode(text=text, id_=obj.get("id") or None, metadata=obj.get("metadata") or {})
                    docs.append(node)
        except Exception:
            docs = []

    # 如果没有缓存，则从 Milvus 导出一次并落盘缓存
    if not docs:
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as out:
                for row in _iter_milvus_rows(vector_store):
                    text = row.get("text")
                    if not isinstance(text, str) or not text.strip():
                        continue
                    text = _truncate_utf8_text(text, LEXICAL_MAX_BYTES)

                    md: dict[str, Any] = {}
                    for k in LEXICAL_METADATA_KEYS:
                        if k in row and row.get(k) is not None:
                            md[k] = row.get(k)
                    rec = {
                        "id": row.get("id"),
                        "doc_id": row.get("doc_id"),
                        "text": text,
                        "metadata": md,
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            tmp_path.replace(cache_path)
        except Exception:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

        # 重新读一次缓存
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    text = obj.get("text")
                    if not isinstance(text, str) or not text.strip():
                        continue
                    text = _truncate_utf8_text(text, LEXICAL_MAX_BYTES)
                    node = TextNode(text=text, id_=obj.get("id") or None, metadata=obj.get("metadata") or {})
                    docs.append(node)
        except Exception:
            docs = []

    if not docs:
        return None

    tfs: list[Counter] = []
    df: Counter = Counter()
    total_len = 0
    for node in docs:
        tokens = _tokenize(node.text)
        tf = Counter(tokens)
        tfs.append(tf)
        total_len += sum(tf.values())
        for term in tf.keys():
            df[term] += 1
    avgdl = total_len / max(len(docs), 1)
    return _Bm25Index(docs=docs, tfs=tfs, df=df, avgdl=avgdl)

def build_index(llm_model: str):
    _configure_llm(llm_model)
    Settings.embed_model = OllamaEmbedding(
        model_name="qwen3-embedding:8b-q8_0", 
        base_url=OLLAMA_BASE,
        ollama_additional_kwargs={"num_ctx": EMBED_NUM_CTX}
    )

    # Milvus 向量字段需要显式维度；优先从 embedding 模型自动推断，失败则允许用环境变量指定。
    embedding_field = os.getenv("MILVUS_EMBEDDING_FIELD", "embedding")
    embed_dim = None
    try:
        embed_dim = len(Settings.embed_model.get_query_embedding("test dimension"))
    except Exception:
        dim_env = os.getenv("EMBED_DIM") or os.getenv("MILVUS_DIM")
        if dim_env:
            try:
                embed_dim = int(dim_env)
            except Exception:
                embed_dim = None

    if not isinstance(embed_dim, int) or embed_dim <= 0:
        raise RuntimeError(
            "无法推断 embedding 维度（Milvus 需要 dim 才能创建/校验向量字段）。"
            "\n- 请确认 Ollama embedding 模型可用（当前：qwen3-embedding:8b-q8_0）"
            "\n- 或手动设置环境变量 EMBED_DIM（例如：EMBED_DIM=1024）"
        )

    async def _init_vector_store() -> MilvusVectorStore:
        return MilvusVectorStore(
            uri=MILVUS_URI,
            collection_name=COLLECTION,
            dim=embed_dim,
            embedding_field=embedding_field,
        )

    # `MilvusVectorStore` 内部会创建 `AsyncMilvusClient`，其初始化需要“正在运行”的事件循环。
    # 这里用一个短生命周期事件循环来完成初始化，避免脚本模式下报错。
    try:
        asyncio.get_running_loop()
        has_running_loop = True
    except RuntimeError:
        has_running_loop = False

    if has_running_loop:
        loop = asyncio.get_event_loop()
        vector_store = loop.run_until_complete(_init_vector_store())
    else:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            vector_store = loop.run_until_complete(_init_vector_store())
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)

def main():
    llm_candidates = _pick_llm_candidates(OLLAMA_BASE)
    if not llm_candidates:
        llm_candidates = ["qwen3:30b-a3b-instruct-2507-q4_K_M"]

    llm_idx = 0
    index = build_index(llm_candidates[llm_idx])

    rerank = _try_build_reranker()
    if rerank is not None:
        # 允许用户把 top_n 配小一点，但召回至少要 >= top_n
        recall_k = max(RECALL_TOP_K, RERANK_TOP_N)
        if RERANK_BACKEND == "hf":
            device = getattr(rerank, "_torch_device", None)
            print(
                f"[RERANK] HF Transformers: model={HF_RERANK_MODEL}, top_n={RERANK_TOP_N}, recall_k={recall_k}, device={device}"
            )
        elif RERANK_BACKEND == "qwen3reranker" or "qwen3-reranker" in HF_RERANK_MODEL.lower():
            device = getattr(rerank, "_device", None)
            print(
                f"[RERANK] Qwen3-Reranker: model={HF_RERANK_MODEL}, top_n={RERANK_TOP_N}, recall_k={recall_k}, device={device}"
            )
        else:
            print(f"[RERANK] SentenceTransformerRerank: model={RERANK_MODEL}, top_n={RERANK_TOP_N}, recall_k={recall_k}")
    else:
        recall_k = 20
        if RERANK_ENABLED:
            print("[RERANK] ⚠️ 未启动（GPU 显存不足或加载失败），回退为仅召回 (recall_k=20)")
            print("[RERANK] 提示: `ollama stop <model>` 释放显存，或设置 RERANK_GPU_ID=1")
        else:
            print("[RERANK] 关闭")

    # 词法（BM25）retriever：从 Milvus 导出 text 构建本地索引，失败则回退纯向量
    bm25_index = None
    try:
        # 这里复用 build_index() 中创建过的 vector store：
        # VectorStoreIndex.from_vector_store 仍持有底层 vector_store，可从 storage_context 取回。
        vector_store = index.storage_context.vector_store
        if isinstance(vector_store, MilvusVectorStore):
            bm25_index = _load_or_build_bm25(vector_store)
    except Exception:
        bm25_index = None

    # ColBERT reranker（多路召回模式使用）
    colbert_reranker = None
    if MULTIPATH_ENABLED and COLBERT_RERANK_ENABLED and bm25_index is not None:
        colbert_reranker = _try_build_colbert_reranker()

    # 显示检索模式信息
    if MULTIPATH_ENABLED and bm25_index is not None:
        print(f"[检索模式] MultiPath 多路召回 + Late Interaction")
        print(f"  - Dense召回: top_k={MULTIPATH_DENSE_TOP_K}")
        print(f"  - Sparse召回: top_k={MULTIPATH_SPARSE_TOP_K}")
        print(f"  - RRF融合: k={MULTIPATH_RRF_K}")
        if colbert_reranker is not None:
            model_name = getattr(colbert_reranker, 'model_name', COLBERT_MODEL)
            print(f"  - ColBERT重排: model={model_name}, candidates={MULTIPATH_RERANK_CANDIDATES}")
        else:
            if COLBERT_RERANK_ENABLED:
                print(f"  - ColBERT重排: 启用失败，使用标准Rerank")
            else:
                print(f"  - ColBERT重排: 未启用，使用标准Rerank")
    elif bm25_index is not None:
        fusion_mode_name = FUSION_MODE if hasattr(FUSION_MODES, FUSION_MODE) else "RECIPROCAL_RANK"
        print(f"[检索模式] 三路召回 Fusion + Rerank")
        print(f"  - 召回路由: 原始向量 + HyDE向量 + BM25+")
        print(f"  - Fusion: mode={fusion_mode_name}, weights={FUSION_WEIGHTS}")
        print(f"  - HyDE路降权: raw_vec=1.0, hyde_vec={HYDE_ROUTE_WEIGHT}")
        print(f"  - 多扩展查询: num_queries={FUSION_NUM_QUERIES}, enabled={MULTI_EXPAND_ENABLED}")
        print(f"  - Rerank: enabled={RERANK_ENABLED}, top_n={RERANK_TOP_N}")
    else:
        print(f"[检索模式] 仅向量检索（词法索引不可用）")

    print(f"[LLM] {llm_candidates[llm_idx]}  (num_ctx={DEFAULT_NUM_CTX})")
    if QUERY_ROUTING_ENABLED:
        print("[查询意图理解] 启用智能查询分类")
        print(f"  支持类型: {', '.join(QUERY_TYPES.keys())}, general")
    
    # 显示 Self-RAG 状态
    if SELF_RAG_ENABLED:
        print(f"[Self-RAG 自反思] 已启用")
        print(f"  最大迭代: {SELF_RAG_MAX_ITERATIONS}, 质量阈值: {SELF_RAG_THRESHOLD}/5")
    else:
        print(f"[Self-RAG 自反思] 默认关闭（当前问答默认模式为 Fusion+HyDE+Rerank）")
    
    # 显示 Chunk 扩展状态
    if CHUNK_EXPAND_ENABLED:
        print(f"[层次化Chunk扩展] 已启用")
        print(f"  邻居扩展: {CHUNK_EXPAND_NEIGHBORS}, 父节点扩展: {CHUNK_EXPAND_PARENT}, 按文档分组: {CHUNK_GROUP_BY_DOC}")
    
    # 显示 HyDE/Query2Doc 状态
    if HYDE_ENABLED or QUERY2DOC_ENABLED:
        print("[查询增强] ", end="")
        enhancements = []
        if HYDE_ENABLED:
            enhancements.append(f"HyDE(max_len={HYDE_MAX_LENGTH}, few_shot={HYDE_FEWSHOT_ENABLED})")
        if QUERY2DOC_ENABLED:
            enhancements.append(f"Query2Doc(max_points={QUERY2DOC_MAX_POINTS})")
        print(", ".join(enhancements))
        if HYDE_ENABLED:
            print("  HyDE格式约束: 关键事实:/文档摘要: 前缀")
        print(f"  短查询阈值: {SHORT_QUERY_THRESHOLD} 字符")
    if not MULTI_EXPAND_ENABLED:
        print(f"[多扩展查询] 已关闭（单查询 Fusion）")
    else:
        print(f"[多扩展查询] 已启用（num_queries={FUSION_NUM_QUERIES}）")

    while True:
        q = input("\nQ> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        clean_q, filters = _extract_query_filters(q)
        if filters is not None:
            fs = "; ".join([f"{f.key}{f.operator.value}{f.value}" for f in filters.filters if isinstance(f, MetadataFilter)])
            print(f"[FILTER] {fs}")

        # 智能查询意图理解
        query_type = "general"
        dynamic_weights = FUSION_WEIGHTS
        strategy = None
        if QUERY_ROUTING_ENABLED and bm25_index is not None:
            query_type = _classify_query(clean_q)
            strategy = _get_query_strategy(query_type)
            dynamic_weights = (strategy["vector_weight"], strategy["bm25_weight"])
            print(f"[查询意图] {query_type} ({strategy['description']})")
            print(f"  -> 权重(vec={dynamic_weights[0]:.2f}, bm25={dynamic_weights[1]:.2f}), 查询数={strategy['num_queries']}")

        # 查询扩展：添加同义词提升召回
        expanded_q = _expand_query(clean_q)
        if expanded_q != clean_q:
            print(f"[查询扩展] {expanded_q}")

        # HyDE / Query2Doc 查询增强（针对短查询）
        hyde_q = expanded_q  # 用于向量检索的增强查询
        q2doc_q = expanded_q  # 用于 BM25 检索的增强查询
        
        should_enhance_query = QUERY_ENHANCE_ALL or _is_short_query(clean_q)
        if should_enhance_query:
            if HYDE_ENABLED:
                if QUERY_ENHANCE_ALL:
                    print(f"[HyDE] 启用查询增强，生成假设文档...")
                else:
                    print(f"[HyDE] 检测到短查询，生成假设文档...")
                hyde_q = hyde_expand_query(expanded_q)
                if hyde_q != expanded_q:
                    # 只显示假设文档的前100字符
                    hyde_preview = hyde_q.replace(expanded_q, "").strip()[:100]
                    print(f"[HyDE] 假设文档: {hyde_preview}...")
            
            if QUERY2DOC_ENABLED:
                if QUERY_ENHANCE_ALL:
                    print(f"[Query2Doc] 启用查询增强，提取关键信息点...")
                else:
                    print(f"[Query2Doc] 检测到短查询，提取关键信息点...")
                q2doc_q = query2doc_expand(expanded_q)
                if q2doc_q != expanded_q:
                    # 显示增加的关键词
                    added_keywords = q2doc_q.replace(expanded_q, "").strip()[:80]
                    print(f"[Query2Doc] 关键信息: {added_keywords}...")

        # =================================================================
        # 检索流程：MultiPath 多路召回 或 Fusion 模式
        # =================================================================
        vector_retriever = index.as_retriever(similarity_top_k=recall_k, filters=filters)
        
        if MULTIPATH_ENABLED and bm25_index is not None:
            # ============================================================
            # MultiPath 多路召回 + Late Interaction 模式 (Phase 3.1)
            # ============================================================
            lexical_retriever = LocalBM25Retriever(bm25_index, similarity_top_k=MULTIPATH_SPARSE_TOP_K, filters=filters)

            # 构建多扩展查询变体（向量检索使用 HyDE 增强，BM25 使用 Query2Doc 增强）
            target_num_queries = strategy["num_queries"] if strategy else FUSION_NUM_QUERIES
            if MULTI_EXPAND_ENABLED and len(clean_q) <= SHORT_QUERY_THRESHOLD:
                # 使用 Query2Doc 增强的查询构建变体（用于 BM25）
                expand_queries = _build_multi_expand_queries(q2doc_q, num_queries=target_num_queries)
                if len(expand_queries) > 1:
                    print(f"[多扩展查询] 生成 {len(expand_queries)} 个查询变体")
                    for i, eq in enumerate(expand_queries[:3]):  # 只显示前3个
                        print(f"   {i+1}. {eq[:60]}{'...' if len(eq) > 60 else ''}")
            else:
                expand_queries = [q2doc_q]

            # 创建 MultiPathRetriever
            multipath_retriever = MultiPathRetriever(
                dense_retriever=vector_retriever,
                sparse_retriever=lexical_retriever,
                colbert_reranker=colbert_reranker,
                dense_top_k=MULTIPATH_DENSE_TOP_K,
                sparse_top_k=MULTIPATH_SPARSE_TOP_K,
                rerank_candidates=MULTIPATH_RERANK_CANDIDATES,
                rrf_k=MULTIPATH_RRF_K,
                enable_colbert=COLBERT_RERANK_ENABLED and colbert_reranker is not None,
            )

            # 执行多路召回检索（向量检索使用 HyDE 增强的查询）
            print(f"[MultiPath] 执行多路召回...")
            try:
                retrieved_nodes = multipath_retriever.retrieve(
                    hyde_q,  # 向量检索使用 HyDE 增强的查询
                    top_k=RERANK_TOP_N if rerank else 10,
                    expand_queries=expand_queries,  # BM25 使用 Query2Doc 增强的变体
                )
                
                # 如果 ColBERT 未启用但有标准 rerank，则进行后处理
                if rerank is not None and (colbert_reranker is None or not COLBERT_RERANK_ENABLED):
                    print(f"[Rerank] 标准精排，top_n={RERANK_TOP_N}")
                    retrieved_nodes = rerank._postprocess_nodes(
                        retrieved_nodes,
                        query_bundle=QueryBundle(query_str=expanded_q),
                    )
                
                # 层次化 Chunk 扩展：带上相关上下文
                if CHUNK_EXPAND_ENABLED and retrieved_nodes:
                    original_count = len(retrieved_nodes)
                    retrieved_nodes = expand_chunks_with_context(
                        retrieved_nodes,
                        index=index,
                        expand_neighbors=CHUNK_EXPAND_NEIGHBORS,
                        expand_parent=CHUNK_EXPAND_PARENT,
                        group_by_doc=CHUNK_GROUP_BY_DOC,
                    )
                    if len(retrieved_nodes) > original_count:
                        print(f"[ChunkExpand] 扩展上下文: {original_count} → {len(retrieved_nodes)} chunks")

                # 构建响应（需要 LLM 合成答案）- 使用优化后的带引用答案生成
                if retrieved_nodes:
                    # Self-RAG 自反思模式：如果启用，则进行迭代式检索
                    if SELF_RAG_ENABLED:
                        # 创建一个轻量级包装器来支持 Self-RAG 的迭代检索
                        class _MultiPathRetrieverWrapper:
                            def __init__(self, retriever, reranker, base_query, expand_queries):
                                self._retriever = retriever
                                self._reranker = reranker
                                self._base_query = base_query
                                self._expand_queries = expand_queries
                            
                            def retrieve(self, query_str):
                                # 检索（可能使用变体查询）
                                nodes = self._retriever.retrieve(
                                    query_str,
                                    top_k=RERANK_TOP_N if self._reranker else recall_k,
                                    expand_queries=self._expand_queries if query_str == self._base_query else [query_str],
                                )
                                # 重排
                                if self._reranker and (colbert_reranker is None or not COLBERT_RERANK_ENABLED):
                                    nodes = self._reranker._postprocess_nodes(
                                        nodes, query_bundle=QueryBundle(query_str=query_str)
                                    )
                                return nodes
                        
                        wrapper = _MultiPathRetrieverWrapper(multipath_retriever, rerank, hyde_q, expand_queries)
                        # 执行 Self-RAG 流程
                        result = self_rag_query(
                            query=expanded_q,  # 使用扩展后的原始查询
                            retriever=wrapper,
                            llm=Settings.llm,
                        )
                        # 格式化输出（包含 Self-RAG 信息）
                        formatted_answer = format_self_rag_result(result, show_history=False)
                    else:
                        # 普通模式：直接生成答案
                        result = generate_answer_with_citation(
                            query=expanded_q,
                            retrieved_docs=retrieved_nodes,
                            llm=Settings.llm,
                        )
                        formatted_answer = format_answer_with_citations(result, verbose=False)
                    
                    print(f"\nA> {formatted_answer}")
                    
                    # 显示检索统计（可选）
                    if result['context_info'].get('doc_count', 0) > 0:
                        high_conf = len([c for c in result.get('citations', []) if c['confidence'] == 'high'])
                        if high_conf > 0:
                            print(f"\n[引用] 高置信度匹配: {high_conf} 个来源")
                else:
                    print("\nA> 未找到相关文档")
                    
            except Exception as e:
                print(f"\n[错误] MultiPath 检索失败：{e}")
                import traceback
                traceback.print_exc()

        elif bm25_index is not None:
            # ============================================================
            # 三路召回 Fusion + Rerank 模式（原始向量 + HyDE向量 + BM25+）
            # ============================================================
            lexical_retriever = LocalBM25Retriever(bm25_index, similarity_top_k=recall_k, filters=filters)

            # 三路权重：原始向量 = vec_weight，HyDE向量 = vec_weight * HYDE_ROUTE_WEIGHT，BM25 = bm25_weight
            vec_weight = float(dynamic_weights[0]) if len(dynamic_weights) >= 1 else 1.0
            bm25_weight = float(dynamic_weights[1]) if len(dynamic_weights) >= 2 else 1.0
            hyde_weight = vec_weight * float(HYDE_ROUTE_WEIGHT)
            primed = {expanded_q: (hyde_q, q2doc_q)}

            fusion_retriever = ThreeWayHyDEFusionRetriever(
                vector_retriever=vector_retriever,
                bm25_retriever=lexical_retriever,
                similarity_top_k=recall_k,
                vector_weight=vec_weight,
                hyde_weight=hyde_weight,
                bm25_weight=bm25_weight,
                rrf_k=FUSION_RRF_K,
                llm=Settings.llm,
                use_hyde=HYDE_ENABLED,
                use_query2doc=QUERY2DOC_ENABLED,
                primed_queries=primed,
            )
            print(
                f"[三路融合] weights(raw_vec={vec_weight:.2f}, hyde_vec={hyde_weight:.2f}, bm25={bm25_weight:.2f}), "
                f"rrf_k={FUSION_RRF_K}"
            )
            
            # 构建查询引擎（带 Rerank 后处理）- 使用 HyDE 增强的查询
            postprocessors = []
            if rerank is not None:
                postprocessors.append(rerank)
                print(f"[Rerank] 启用精排，top_n={RERANK_TOP_N}")

            try:
                # Self-RAG 自反思模式
                if SELF_RAG_ENABLED:
                    # 创建融合检索器包装器
                    class _FusionRetrieverWrapper:
                        def __init__(self, retriever, postprocessors):
                            self._retriever = retriever
                            self._postprocessors = postprocessors or []
                        
                        def retrieve(self, query_str):
                            # 使用三路召回检索器
                            nodes = self._retriever._retrieve(QueryBundle(query_str=query_str))
                            # 应用后处理器（reranker）
                            for postprocessor in self._postprocessors:
                                nodes = postprocessor._postprocess_nodes(
                                    nodes, query_bundle=QueryBundle(query_str=query_str)
                                )
                            return nodes
                    
                    wrapper = _FusionRetrieverWrapper(fusion_retriever, postprocessors)
                    # 执行 Self-RAG 流程
                    result = self_rag_query(
                        query=expanded_q,
                        retriever=wrapper,
                        llm=Settings.llm,
                    )
                    # 格式化输出（包含 Self-RAG 信息）
                    formatted_answer = format_self_rag_result(result, show_history=False)
                    print(f"\nA> {formatted_answer}")
                else:
                    # 普通模式：使用 RetrieverQueryEngine
                    qe = RetrieverQueryEngine.from_args(
                        fusion_retriever,
                        node_postprocessors=postprocessors if postprocessors else None,
                    )
                    # 使用 HyDE 增强的查询进行检索（向量检索受益更大）
                    ans = qe.query(hyde_q)
                    print("\nA> ", ans)
            except Exception as e:
                # Ollama runner 可能因为模型过大/内存不足崩溃：500 + runner terminated。
                msg = str(e)
                can_fallback = llm_idx + 1 < len(llm_candidates)
                if ("runner process has terminated" in msg or "status code: 500" in msg) and can_fallback:
                    llm_idx += 1
                    _configure_llm(llm_candidates[llm_idx])
                    print(f"\n[提示] 当前模型运行失败，已自动切换到: {llm_candidates[llm_idx]}")
                    # 重试
                    try:
                        if SELF_RAG_ENABLED:
                            wrapper = _FusionRetrieverWrapper(fusion_retriever, postprocessors)
                            result = self_rag_query(query=expanded_q, retriever=wrapper, llm=Settings.llm)
                            print(f"\nA> {format_self_rag_result(result, show_history=False)}")
                        else:
                            qe = RetrieverQueryEngine.from_args(
                                fusion_retriever,
                                node_postprocessors=postprocessors if postprocessors else None,
                            )
                            ans = qe.query(hyde_q)
                            print("\nA> ", ans)
                    except Exception as e2:
                        print(f"\n[错误] 仍然失败：{e2}")
                else:
                    print(f"\n[错误] 查询失败：{e}")
        else:
            # ============================================================
            # 无 BM25 时使用纯向量检索 + Rerank
            # ============================================================
            postprocessors = []
            if rerank is not None:
                postprocessors.append(rerank)
                print(f"[Rerank] 启用精排，top_n={RERANK_TOP_N}")

            try:
                # Self-RAG 自反思模式
                if SELF_RAG_ENABLED:
                    class _VectorRetrieverWrapper:
                        def __init__(self, retriever, postprocessors):
                            self._retriever = retriever
                            self._postprocessors = postprocessors or []
                        
                        def retrieve(self, query_str):
                            nodes = self._retriever.retrieve(query_str)
                            for postprocessor in self._postprocessors:
                                nodes = postprocessor._postprocess_nodes(
                                    nodes, query_bundle=QueryBundle(query_str=query_str)
                                )
                            return nodes
                    
                    wrapper = _VectorRetrieverWrapper(vector_retriever, postprocessors)
                    result = self_rag_query(
                        query=expanded_q,
                        retriever=wrapper,
                        llm=Settings.llm,
                    )
                    formatted_answer = format_self_rag_result(result, show_history=False)
                    print(f"\nA> {formatted_answer}")
                else:
                    # 普通模式
                    qe = RetrieverQueryEngine.from_args(
                        vector_retriever,
                        node_postprocessors=postprocessors if postprocessors else None,
                    )
                    ans = qe.query(expanded_q)
                    print("\nA> ", ans)
            except Exception as e:
                msg = str(e)
                can_fallback = llm_idx + 1 < len(llm_candidates)
                if ("runner process has terminated" in msg or "status code: 500" in msg) and can_fallback:
                    llm_idx += 1
                    _configure_llm(llm_candidates[llm_idx])
                    print(f"\n[提示] 当前模型运行失败，已自动切换到: {llm_candidates[llm_idx]}")
                    try:
                        if SELF_RAG_ENABLED:
                            wrapper = _VectorRetrieverWrapper(vector_retriever, postprocessors)
                            result = self_rag_query(query=expanded_q, retriever=wrapper, llm=Settings.llm)
                            print(f"\nA> {format_self_rag_result(result, show_history=False)}")
                        else:
                            qe = RetrieverQueryEngine.from_args(
                                vector_retriever,
                                node_postprocessors=postprocessors if postprocessors else None,
                            )
                            ans = qe.query(expanded_q)
                            print("\nA> ", ans)
                    except Exception as e2:
                        print(f"\n[错误] 仍然失败：{e2}")
                else:
                    print(f"\n[错误] 查询失败：{e}")

if __name__ == "__main__":
    main()
