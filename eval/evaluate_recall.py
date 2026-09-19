import argparse
import json
import os
import re
import site
import sys
import hashlib
from collections import defaultdict
from math import log2
from pathlib import Path
from typing import Any, Iterable, Optional
import urllib.request


def _bootstrap_local_venv() -> None:
    """允许直接运行 `python eval/evaluate_recall.py` 时加载项目 `.venv` 依赖。

    兼容两种虚拟环境布局：

    - POSIX（部署在 Linux 服务器）：``.venv/lib/pythonX.Y/site-packages``
    - Windows（本地调试）：``.venv/Lib/site-packages``

    注意：本函数只是「用系统解释器跑、但依赖装在 .venv 里」时的便利措施。
    如果直接用 ``.venv/bin/python``（或 Windows 的 ``.venv\\Scripts\\python.exe``）
    运行，则不需要它。
    """

    # eval/ 的上一级是仓库根目录
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates: list[str] = []

    # Windows 布局：.venv/Lib/site-packages
    win_site_packages = os.path.join(repo_dir, ".venv", "Lib", "site-packages")
    if os.path.isdir(win_site_packages):
        candidates.append(win_site_packages)

    # POSIX 布局：.venv/lib/pythonX.Y/site-packages
    # 扫描（而非直接拼版本号）以便解释器小版本与建 venv 时不一致也能命中
    lib_dir = os.path.join(repo_dir, ".venv", "lib")
    try:
        if os.path.isdir(lib_dir):
            for name in os.listdir(lib_dir):
                if name.startswith("python"):
                    sp = os.path.join(lib_dir, name, "site-packages")
                    if os.path.isdir(sp):
                        candidates.append(sp)
    except Exception:
        pass

    if not candidates:
        return

    candidates.sort(reverse=True)
    for sp in candidates:
        if sp not in sys.path:
            sys.path.insert(0, sp)
            site.addsitedir(sp)
            return


_bootstrap_local_venv()

# 让 `python eval/evaluate_recall.py` 直接运行时能找到 src/pcb_rag（本项目为 src 布局）
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO_ROOT, "src"), _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 加载仓库根目录的 .env：直接 `python eval/evaluate_recall.py` 时也能读到
# LLM / Embedding / Rerank 的 API 配置（必须早于 import pcb_rag.query，
# 因为 query.py 在导入时就会读取环境变量）
from pcb_rag.env_loader import load_project_env  # noqa: E402

load_project_env()

from llama_index.core.schema import QueryBundle  # noqa: E402

try:
    import pcb_rag.query as rag_query  # noqa: E402
except ImportError as _exc:  # pragma: no cover - 部署环境缺依赖时给出可操作提示
    raise SystemExit(
        f"[ERROR] 无法导入 pcb_rag.query: {_exc}\n"
        "请先安装依赖后重试：\n"
        "  pip install -r requirements.txt && pip install -e .\n"
        f"（已尝试加入 sys.path: {os.path.join(_REPO_ROOT, 'src')}）"
    ) from _exc

# ═══════════════════════════════════════════════════════════════════════════════
#  新增检索模式支持 (Phase 3.1 - 3.3)
#  - multipath: MultiPath 多路召回 + RRF 融合
#  - multipath_colbert: MultiPath + ColBERT Late Interaction 重排
#  - hyde: HyDE 查询增强
#  - query2doc: Query2Doc 关键信息提取
# ═══════════════════════════════════════════════════════════════════════════════


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("eval dataset must be a JSON list")
    out: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        q = item.get("query")
        gt = item.get("ground_truth_ids")
        if not isinstance(q, str) or not q.strip():
            continue
        if not isinstance(gt, list) or not gt:
            continue
        out.append(item)
    return out


def _node_id(nws: Any) -> str | None:
    node = getattr(nws, "node", None)
    if node is None:
        return None

    # LlamaIndex nodes
    for attr in ("node_id", "id_", "id"):
        v = getattr(node, attr, None)
        if isinstance(v, str) and v:
            return v

    # sometimes wrapped
    inner = getattr(node, "node", None)
    if inner is not None:
        for attr in ("node_id", "id_", "id"):
            v = getattr(inner, attr, None)
            if isinstance(v, str) and v:
                return v
    return None


def _node_text(nws: Any) -> str:
    node = getattr(nws, "node", None)
    if node is None:
        return ""

    for attr in ("text", "content"):
        v = getattr(node, attr, None)
        if isinstance(v, str) and v:
            return v

    # llama_index nodes often support get_content
    get_content = getattr(node, "get_content", None)
    if callable(get_content):
        try:
            v = get_content()
            if isinstance(v, str) and v:
                return v
        except Exception:
            pass

    inner = getattr(node, "node", None)
    if inner is not None:
        for attr in ("text", "content"):
            v = getattr(inner, attr, None)
            if isinstance(v, str) and v:
                return v
        get_content = getattr(inner, "get_content", None)
        if callable(get_content):
            try:
                v = get_content()
                if isinstance(v, str) and v:
                    return v
            except Exception:
                pass

    return ""


def _node_score(nws: Any) -> Optional[float]:
    s = getattr(nws, "score", None)
    try:
        return float(s) if s is not None else None
    except Exception:
        return None


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _precision_at_k(got_ids: list[str], gt_ids: set[str], k: int) -> float:
    """Precision@K: 在 Top-K 结果中，相关文档的比例。

    P@K = |TopK ∩ GT| / K
    """
    if k <= 0:
        return 0.0
    top = got_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for nid in top if nid in gt_ids)
    return hits / k


def _average_precision(got_ids: list[str], gt_ids: set[str]) -> float:
    """Average Precision (AP): 用于计算 MAP。

    AP = (1/|GT|) * Σ_{k: rel_k=1} P@k
    """
    if not gt_ids:
        return 0.0
    if not got_ids:
        return 0.0

    hits = 0
    sum_precision = 0.0
    for i, nid in enumerate(got_ids, 1):
        if nid in gt_ids:
            hits += 1
            sum_precision += hits / i

    return sum_precision / len(gt_ids) if gt_ids else 0.0


def _r_precision(got_ids: list[str], gt_ids: set[str]) -> float:
    """R-Precision: 在 Top-R 结果中的精确率，R = |GT|。

    适合评估不同查询有不同数量相关文档的场景。
    """
    if not gt_ids:
        return 0.0
    r = len(gt_ids)
    return _precision_at_k(got_ids, gt_ids, r)


def _ndcg_at_k(got_ids: list[str], gt_ids: set[str], k: int) -> float:
    """NDCG@K with binary relevance (rel=1 if id in ground truth else 0).

    DCG@K = sum_{i=1..K} rel_i / log2(i+1)
    IDCG@K = sum_{i=1..min(|GT|,K)} 1 / log2(i+1)
    """

    if k <= 0:
        return 0.0
    if not gt_ids:
        return 0.0

    top = got_ids[:k]
    dcg = 0.0
    for i, nid in enumerate(top, 1):
        if nid in gt_ids:
            dcg += 1.0 / log2(i + 1)

    ideal = min(len(gt_ids), k)
    if ideal <= 0:
        return 0.0
    idcg = sum(1.0 / log2(i + 1) for i in range(1, ideal + 1))
    return (dcg / idcg) if idcg > 0 else 0.0


def _dcg_at_k_binary(rels: list[int], k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(rels[:k], 1):
        if rel:
            dcg += 1.0 / log2(i + 1)
    return dcg


def _ndcg_at_k_binary_rels(rels: list[int], ideal_relevant: int, k: int) -> float:
    if k <= 0:
        return 0.0
    if ideal_relevant <= 0:
        return 0.0
    dcg = _dcg_at_k_binary(rels, k)
    ideal = min(int(ideal_relevant), k)
    idcg = sum(1.0 / log2(i + 1) for i in range(1, ideal + 1))
    return (dcg / idcg) if idcg > 0 else 0.0


def _sha_key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        if not isinstance(p, str):
            p = str(p)
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\x1f")
    return h.hexdigest()


def _load_jsonl_kv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    out: dict[str, Any] = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                k = obj.get("k")
                v = obj.get("v")
                if isinstance(k, str):
                    out[k] = v
    except Exception:
        return {}
    return out


def _append_jsonl_kv(path: Path, k: str, v: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")


def _parse_bool_text(out_raw: str) -> bool:
    """把评测模型的输出解析成布尔值。

    兼容多种输出形态：``T/F``、``true/false``、``是/否``、``Yes/No``、
    JSON（``{"answer": true}``）以及纯 token 兜底。

    注意：解析失败一律返回 ``False``，因此当 judge 模型输出不规范时，软匹配会
    偏向"未命中"。如果换用不遵循指令的模型，应关注软匹配分数是否异常偏低。

    与原实现的差异（两处均为修正误判，会导致软匹配分数小幅上升，换判卷模型
    或重跑历史数据集时请留意）：

    1. 新增「去装饰后全串匹配」：``**T**`` / ``Yes.`` / ``T。`` / ``"是"`` 这类
       被 markdown 或标点包裹的输出，原来会在第 1 步因首字符不是 t/f 而落空、
       最终误判为 ``False``。
    2. 中文兜底改为先判否定：原实现用 ``"是" in out_raw`` 兜底，导致 ``不是``
       因包含 ``是`` 被误判为 ``True``。
    """

    out = (out_raw or "").strip().lower()

    # 0) 去掉 markdown 修饰与首尾标点后再做「全串匹配」
    out_clean = re.sub(r"[^\w\u4e00-\u9fff]", "", out)
    if out_clean in {"t", "true", "yes", "y", "是", "对", "對"}:
        return True
    if out_clean in {"f", "false", "no", "n", "否", "不"}:
        return False

    # 1) 期待的最严格输出：T / F
    if out.startswith("t"):
        return True
    if out.startswith("f"):
        return False

    # 2) 兼容 true/false（部分模型仍可能输出）
    if out.startswith("true"):
        return True
    if out.startswith("false"):
        return False

    # 3) 兼容中文/英文 Yes/No
    out_compact = re.sub(r"\s+", "", out)
    if out_compact in {"是", "对", "對", "yes", "y", "1"}:
        return True
    if out_compact in {"否", "不", "no", "n", "0"}:
        return False

    # 4) 兼容 JSON 输出：{"answer": true}
    try:
        if out.startswith("{") and out.endswith("}"):
            obj = json.loads(out_raw)
        else:
            m = re.search(r"\{[\s\S]*?\}", out_raw)
            obj = json.loads(m.group(0)) if m else None
        if isinstance(obj, dict):
            for k in ("answer", "result", "label", "relevant", "is_relevant"):
                v = obj.get(k)
                if isinstance(v, bool):
                    return bool(v)
                if isinstance(v, str):
                    vv = v.strip().lower()
                    if vv in {"t", "true", "yes", "y", "是", "对", "對", "1"}:
                        return True
                    if vv in {"f", "false", "no", "n", "否", "不", "0"}:
                        return False
    except Exception:
        pass

    # 5) 最后兜底：抓第一个独立 token
    m = re.search(r"\b(true|false)\b", out, re.I)
    if m:
        return m.group(1).lower() == "true"
    m = re.search(r"\b([tf])\b", out, re.I)
    if m:
        return m.group(1).lower() == "t"
    # 中文兜底：必须先判否定，否则 '不是' 会因为包含 '是' 被误判为 True
    if re.search(r"否|不|非", out_compact):
        return False
    if re.search(r"是|对|對", out_compact):
        return True
    return False


def _looks_like_ollama(base_url: str) -> bool:
    """按 URL 特征猜测后端类型（Ollama 默认端口 11434）。"""

    u = (base_url or "").strip().lower()
    return "11434" in u or u.rstrip("/").endswith("/api")


def _openai_chat_bool(
    base_url: str,
    model: str,
    prompt: str,
    api_key: str = "",
    timeout: float = 60.0,
) -> bool:
    """OpenAI 兼容 ``/chat/completions`` 判定。

    用于「无显卡、只有模型 API」的部署环境：不需要本地 Ollama 或任何
    本地推理依赖，只要有 OpenAI 兼容的 base_url + api_key + model 即可。
    """

    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是一个严格的二分类评估器。"
                    "你必须只输出单个字符：T 或 F（大写），不要输出其它任何内容。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 16,
        "stream": False,
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), method="POST"
    )
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    try:
        out_raw = str(payload["choices"][0]["message"]["content"] or "")
    except (KeyError, IndexError, TypeError):
        # 兜底：把整个响应体丢给解析器，避免因为非标准返回直接判 F
        out_raw = json.dumps(payload, ensure_ascii=False)
    return _parse_bool_text(out_raw)


def _judge_generate_bool(
    backend: str,
    base_url: str,
    model: str,
    prompt: str,
    api_key: str = "",
    timeout: float = 60.0,
) -> bool:
    """按 ``backend`` 分派软匹配判定请求。

    ``backend`` 取值：``auto``（按 base_url 猜测）/ ``ollama`` / ``openai``。
    """

    backend = (backend or "auto").strip().lower()
    if backend == "auto":
        backend = "ollama" if _looks_like_ollama(base_url) else "openai"
    if backend == "ollama":
        return _ollama_generate_bool(base_url, model, prompt, timeout=timeout)
    return _openai_chat_bool(base_url, model, prompt, api_key=api_key, timeout=timeout)


def _ollama_generate_bool(base_url: str, model: str, prompt: str, timeout: float = 60.0) -> bool:
    url = base_url.rstrip("/") + "/api/generate"
    body = {
        "model": model,
        "system": (
            "你是一个严格的二分类评估器。"
            "你必须只输出单个字符：T 或 F（大写），不要输出其它任何内容。"
        ),
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 8,
            "stop": ["\n", " ", "\t"],
        },
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    return _parse_bool_text(str(payload.get("response") or ""))


def _soft_match_llm(
    base_url: str,
    model: str,
    query: str,
    gt_texts: list[str],
    retrieved_text: str,
    cache: dict[str, Any],
    cache_path: Path,
    timeout: float = 60.0,
    backend: str = "auto",
    api_key: str = "",
) -> bool:
    # key 里不要放全部长文本，避免 cache 文件爆炸；存 hash 但计算时用原文参与 hash
    # 同时把 judge 的 backend/base_url/model 纳入 key：换判卷模型后旧缓存自动失效，
    # 避免沿用上一个模型（甚至另一个厂商）的判定结果
    gt_blob = "\n---\n".join([t[:800] for t in gt_texts if isinstance(t, str)])
    key = _sha_key("llm_v3", backend, base_url, model, query, gt_blob, retrieved_text[:1200])
    if key in cache:
        return bool(cache[key])

    prompt = (
        "任务：判断‘检索到的上下文’是否在语义上覆盖‘标准答案片段’，从而能回答该问题。\n"
        "判定标准：\n"
        "- 如果检索到的上下文包含/等价表达了标准答案片段中的关键事实/结论/条件（即便措辞不同），输出 T。\n"
        "- 如果只是主题相关但缺少关键事实，或无法支撑回答，输出 F。\n"
        "- 不要臆测，不要用常识补全。只基于给定文本判断。\n\n"
        f"问题：{query}\n\n"
        f"标准答案片段（Ground Truth）：\n{gt_blob}\n\n"
        f"检索到的上下文（Retrieved Context）：\n{retrieved_text[:2000]}\n\n"
        "输出只允许为单个字符：T 或 F。"
    )

    ok = _judge_generate_bool(
        backend, base_url, model, prompt, api_key=api_key, timeout=timeout
    )
    cache[key] = ok
    _append_jsonl_kv(cache_path, key, ok)
    return ok


_EMBED_CACHE: dict[str, list[float]] = {}


def _embed_cached(embed_model, text: str) -> list[float]:
    """带缓存的 embedding 调用。

    软匹配里同一个 Ground Truth 会和 Top-K 中每一条候选各比一次，不加缓存时
    调用量是 ``n_queries × top_k`` 量级（353 题 × 10 条 ≈ 3500 次）。在纯 API
    部署下这既是延迟也是费用，因此按文本哈希缓存：GT 只算一次，重复命中的
    候选也复用。
    """

    key = hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()
    hit = _EMBED_CACHE.get(key)
    if hit is not None:
        return hit
    vec = list(embed_model.get_text_embedding(text))
    _EMBED_CACHE[key] = vec
    return vec


def _soft_match_embed(
    embed_model,
    gt_texts: list[str],
    retrieved_text: str,
    threshold: float,
) -> bool:
    # 用最大相似度做判断（多 GT 时取 max）
    try:
        r_vec = _embed_cached(embed_model, retrieved_text[:2000])
    except Exception:
        return False

    def _cos(a, b) -> float:
        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b):
            dot += float(x) * float(y)
            na += float(x) * float(x)
            nb += float(y) * float(y)
        if na <= 0 or nb <= 0:
            return 0.0
        return dot / ((na ** 0.5) * (nb ** 0.5))

    best = 0.0
    for gt in gt_texts:
        if not isinstance(gt, str) or not gt.strip():
            continue
        try:
            g_vec = _embed_cached(embed_model, gt[:2000])
        except Exception:
            continue
        best = max(best, _cos(g_vec, r_vec))
        if best >= threshold:
            return True
    return best >= threshold


def _merge_union(vec_nodes: list[Any], bm25_nodes: list[Any], recall_k: int) -> list[Any]:
    out: list[Any] = []
    seen: set[str] = set()
    max_len = max(len(vec_nodes), len(bm25_nodes))
    for i in range(max_len):
        if i < len(bm25_nodes):
            n = bm25_nodes[i]
            key = _node_id(n) or f"bm25:{id(n)}"
            if key not in seen:
                seen.add(key)
                out.append(n)
        if i < len(vec_nodes):
            n = vec_nodes[i]
            key = _node_id(n) or f"vec:{id(n)}"
            if key not in seen:
                seen.add(key)
                out.append(n)
        if len(out) >= recall_k * 2:
            break
    return out


class _UnionRetriever:
    def __init__(self, vector_retriever, bm25_retriever, recall_k: int):
        self._vector = vector_retriever
        self._bm25 = bm25_retriever
        self._k = int(recall_k)

    def retrieve(self, query: str) -> list[Any]:
        vec_nodes = self._vector.retrieve(query)
        bm25_nodes = self._bm25.retrieve(query)
        return _merge_union(vec_nodes[: self._k], bm25_nodes[: self._k], self._k)


def _apply_query_enhancement(
    query: str,
    use_hyde: bool = False,
    use_query2doc: bool = False,
    llm=None,
) -> tuple[str, str]:
    """应用查询增强策略。

    Args:
        query: 原始查询
        use_hyde: 是否使用 HyDE 增强（生成假设文档）
        use_query2doc: 是否使用 Query2Doc 增强（提取关键信息）
        llm: LLM 实例

    Returns:
        (vector_query, bm25_query): 分别用于向量检索和 BM25 检索的查询
    """
    vector_query = query
    bm25_query = query

    # HyDE: 生成假设文档用于向量检索
    if use_hyde:
        try:
            hyde_func = getattr(rag_query, "hyde_expand_query", None)
            if hyde_func:
                enhanced = hyde_func(query, llm)
                if enhanced and enhanced != query:
                    vector_query = enhanced
                    print(f"[HyDE] 查询增强: {query[:30]}... -> {enhanced[:50]}...")
        except Exception as e:
            print(f"[HyDE] 增强失败: {e}")

    # Query2Doc: 提取关键信息用于 BM25 检索
    if use_query2doc:
        try:
            q2d_func = getattr(rag_query, "query2doc_expand", None)
            if q2d_func:
                enhanced = q2d_func(query, llm)
                if enhanced and enhanced != query:
                    bm25_query = enhanced
                    print(f"[Query2Doc] 查询增强: {query[:30]}... -> {enhanced[:50]}...")
        except Exception as e:
            print(f"[Query2Doc] 增强失败: {e}")

    return vector_query, bm25_query


def _build_query_variants(query: str, num_queries: int) -> list[str]:
    q = (query or "").strip()
    if not q:
        return []
    variants: list[str] = [q]
    if num_queries <= 1:
        return variants

    # 优先使用项目内的扩展函数
    try:
        expanded = rag_query._expand_query(q)
        if expanded and expanded not in variants:
            variants.append(expanded)
    except Exception:
        pass

    # 单词级追加扩展，补足到 num_queries
    try:
        if rag_query.QUERY_EXPANSION_ENABLED:
            q_low = q.lower()
            for term, exps in rag_query.QUERY_EXPANSION_DICT.items():
                if term.lower() in q_low:
                    for exp in exps:
                        v = (q + " " + str(exp)).strip()
                        if v not in variants:
                            variants.append(v)
                        if len(variants) >= num_queries:
                            return variants[:num_queries]
    except Exception:
        pass

    return variants[:num_queries]


def _aggregate_nodes_by_min_rank(list_of_nodes: list[list[Any]]) -> list[Any]:
    best_rank: dict[str, int] = {}
    best_node: dict[str, Any] = {}
    for nodes in list_of_nodes:
        for rank, n in enumerate(nodes, 1):
            key = _node_id(n)
            if not key:
                continue
            if key not in best_rank or rank < best_rank[key]:
                best_rank[key] = rank
                best_node[key] = n

    ranked = sorted(best_rank.items(), key=lambda x: x[1])
    out: list[Any] = []
    for nid, _ in ranked:
        n = best_node.get(nid)
        if n is not None:
            out.append(n)
    return out


def _rrf_fuse(
    vec_nodes: list[Any],
    bm25_nodes: list[Any],
    topn: int = 10,
    k: int = 60,
    allowed_ids: Optional[set[str]] = None,
    weights: tuple[float, float] = (0.4, 0.6),  # BM25 权重更高（精确匹配更重要）
    boost_overlap: bool = True,  # 双命中加成
    top_boost: int = 5,  # Top-N 位置额外加成
) -> list[Any]:
    """优化的 RRF 融合算法（**评测历史口径**，与线上实现不同）。

    ⚠️ 这里不是线上跑的融合。线上（CLI / API）用的是
    ``pcb_rag.fusion.weighted_rrf_fuse``，公式为 ``score = w/(k+rank)``，
    没有双命中加成、没有 Top-N 加成、没有动态 k。

    本函数保留了这些额外偏置，用于**复现历史报告**（旧版本评测脚本的行为）。
    把它和线上指标直接对比是错的 —— 两套融合的分数分布与排序都不同。

    改进点：
    1. 支持检索器权重（BM25 对精确术语更重要，权重更高）
    2. 双命中加成：同时被两个检索器召回的文档额外加分
    3. Top-N 位置加成：排名前 N 的文档获得额外权重
    4. 排名衰减：高排名位置权重更高
    5. BM25 优先打破平局：分数相同时优先 BM25 排名
    """
    scores: dict[str, float] = {}
    nodes_map: dict[str, Any] = {}
    vec_rank: dict[str, int] = {}
    bm25_rank: dict[str, int] = {}

    w_vec, w_bm25 = weights

    for rank, n in enumerate(vec_nodes, 1):
        key = _node_id(n)
        if not key:
            continue
        if allowed_ids is not None and key not in allowed_ids:
            continue
        nodes_map.setdefault(key, n)
        vec_rank.setdefault(key, rank)
        # 加权 RRF 分数
        base_score = w_vec / (k + rank)
        # Top-N 位置加成：排名前 top_boost 的文档获得额外权重
        if rank <= top_boost:
            base_score *= (1.0 + 0.15 * (top_boost - rank + 1) / top_boost)
        scores[key] = scores.get(key, 0.0) + base_score

    for rank, n in enumerate(bm25_nodes, 1):
        key = _node_id(n)
        if not key:
            continue
        if allowed_ids is not None and key not in allowed_ids:
            continue
        nodes_map.setdefault(key, n)
        bm25_rank.setdefault(key, rank)
        base_score = w_bm25 / (k + rank)
        # Top-N 位置加成
        if rank <= top_boost:
            base_score *= (1.0 + 0.2 * (top_boost - rank + 1) / top_boost)  # BM25 top 加成更大
        scores[key] = scores.get(key, 0.0) + base_score

    # 双命中加成：如果文档同时被两个检索器召回，额外奖励
    if boost_overlap:
        overlap_ids = set(vec_rank.keys()) & set(bm25_rank.keys())
        for nid in overlap_ids:
            # 加成幅度与两边排名的调和平均成反比（调和平均更强调较小排名）
            v_r = vec_rank.get(nid, 10**9)
            b_r = bm25_rank.get(nid, 10**9)
            harmonic_mean_rank = 2 * v_r * b_r / (v_r + b_r + 1e-9)
            # 排名越靠前加成越大，最高约 0.15
            boost = 0.15 / (1 + harmonic_mean_rank / 10)
            scores[nid] = scores.get(nid, 0.0) + boost

    def _sort_key(item):
        nid, score = item
        b = bm25_rank.get(nid, 10**9)
        v = vec_rank.get(nid, 10**9)
        # 优先分数，次优先 BM25 排名（精确匹配优先），再次优先向量排名
        return (-score, b, v)

    ranked = sorted(scores.items(), key=_sort_key)
    out: list[Any] = []
    for nid, _ in ranked[: max(0, int(topn))]:
        n = nodes_map.get(nid)
        if n is not None:
            out.append(n)
    return out


def _rrf_fuse_multi(
    node_lists: list[list[Any]],
    route_weights: list[float],
    topn: int = 10,
    k: int = 60,
    allowed_ids: Optional[set[str]] = None,
) -> list[Any]:
    """多路加权 RRF 融合（用于原始向量 + HyDE向量 + BM25 三路召回）。"""
    scores: dict[str, float] = {}
    nodes_map: dict[str, Any] = {}

    for idx, nodes in enumerate(node_lists):
        weight = float(route_weights[idx]) if idx < len(route_weights) else 1.0
        if weight <= 0:
            continue
        for rank, n in enumerate(nodes, 1):
            nid = _node_id(n)
            if not nid:
                continue
            if allowed_ids is not None and nid not in allowed_ids:
                continue
            nodes_map.setdefault(nid, n)
            scores[nid] = scores.get(nid, 0.0) + weight / float(k + rank)

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    out: list[Any] = []
    for nid, score in ranked[: max(0, int(topn))]:
        n = nodes_map.get(nid)
        if n is None:
            continue
        try:
            n.score = float(score)
        except Exception:
            pass
        out.append(n)
    return out


def _score_fuse(
    vec_nodes: list[Any],
    bm25_nodes: list[Any],
    topn: int = 10,
    weights: tuple[float, float] = (0.5, 0.5),
    allowed_ids: Optional[set[str]] = None,
    normalize_method: str = "minmax",  # 新增：归一化方法
) -> list[Any]:
    """改进的分数融合算法。

    改进点：
    1. 支持多种归一化方法：minmax / zscore / rank
    2. 处理分数缺失：未被某检索器召回的文档使用最低分而非0
    3. 排名兜底：分数相同时优先 BM25 排名（精确匹配优先）
    """

    def _normalize_minmax(scores_dict: dict[str, float]) -> dict[str, float]:
        if not scores_dict:
            return {}
        vals = list(scores_dict.values())
        mn, mx = min(vals), max(vals)
        if mx <= mn:
            return {k: 0.5 for k in scores_dict}
        return {k: (v - mn) / (mx - mn) for k, v in scores_dict.items()}

    def _normalize_rank(nodes: list[Any]) -> dict[str, float]:
        """基于排名的归一化：排名1得分1.0，排名n得分接近0"""
        out: dict[str, float] = {}
        n = len(nodes)
        for rank, node in enumerate(nodes, 1):
            nid = _node_id(node)
            if nid:
                out[nid] = 1.0 - (rank - 1) / max(n, 1)
        return out

    def _scores(nodes: list[Any]) -> dict[str, float]:
        out: dict[str, float] = {}
        for n in nodes:
            nid = _node_id(n)
            if not nid:
                continue
            sc = _node_score(n)
            if sc is None:
                continue
            out[nid] = float(sc)
        return out

    # 根据归一化方法处理分数
    if normalize_method == "rank":
        v_scores = _normalize_rank(vec_nodes)
        b_scores = _normalize_rank(bm25_nodes)
    else:  # minmax
        v_scores = _normalize_minmax(_scores(vec_nodes))
        b_scores = _normalize_minmax(_scores(bm25_nodes))

    nodes_map: dict[str, Any] = {}
    vec_rank: dict[str, int] = {}
    bm25_rank: dict[str, int] = {}

    for rank, n in enumerate(vec_nodes, 1):
        nid = _node_id(n)
        if not nid:
            continue
        nodes_map.setdefault(nid, n)
        vec_rank.setdefault(nid, rank)

    for rank, n in enumerate(bm25_nodes, 1):
        nid = _node_id(n)
        if not nid:
            continue
        nodes_map.setdefault(nid, n)
        bm25_rank.setdefault(nid, rank)

    fused: dict[str, float] = {}
    # 计算所有节点的最低分作为缺失值填充
    all_v = list(v_scores.values())
    all_b = list(b_scores.values())
    min_v = min(all_v) if all_v else 0.0
    min_b = min(all_b) if all_b else 0.0

    for nid in set(list(nodes_map.keys())):
        if allowed_ids is not None and nid not in allowed_ids:
            continue
        # 使用最低分而非0填充缺失（避免单检索器文档被过度惩罚）
        vs = v_scores.get(nid, min_v * 0.5)
        bs = b_scores.get(nid, min_b * 0.5)
        fused[nid] = weights[0] * vs + weights[1] * bs

    def _sort_key(item):
        nid, score = item
        b = bm25_rank.get(nid, 10**9)
        v = vec_rank.get(nid, 10**9)
        return (-score, b, v)

    ranked = sorted(fused.items(), key=_sort_key)
    out: list[Any] = []
    for nid, _ in ranked[: max(0, int(topn))]:
        n = nodes_map.get(nid)
        if n is not None:
            out.append(n)
    return out


class _RRFRetriever:
    def __init__(
        self,
        vector_retriever,
        bm25_retriever,
        recall_k: int,
        topn: int = 10,
        use_union: bool = False,
        num_queries: int = 1,
        fusion_mode: str = "rrf",
    ):
        self._vector = vector_retriever
        self._bm25 = bm25_retriever
        self._k = int(recall_k)
        self._topn = int(topn)
        self._use_union = bool(use_union)
        self._num_queries = max(1, int(num_queries))
        self._fusion_mode = str(fusion_mode)

    def retrieve(self, query: str) -> list[Any]:
        queries = _build_query_variants(query, self._num_queries)
        if not queries:
            return []

        vec_lists: list[list[Any]] = []
        bm25_lists: list[list[Any]] = []
        for q in queries:
            vec_lists.append(self._vector.retrieve(q)[: self._k])
            bm25_lists.append(self._bm25.retrieve(q)[: self._k])

        vec_nodes = _aggregate_nodes_by_min_rank(vec_lists)
        bm25_nodes = _aggregate_nodes_by_min_rank(bm25_lists)
        allowed_ids: Optional[set[str]] = None
        if self._use_union:
            union_nodes = _merge_union(vec_nodes, bm25_nodes, self._k)
            allowed_ids = set(_node_id(n) for n in union_nodes if _node_id(n))
        if self._fusion_mode == "score":
            try:
                w = getattr(rag_query, "FUSION_WEIGHTS", [0.5, 0.5])
                weights = (float(w[0]), float(w[1])) if len(w) >= 2 else (0.5, 0.5)
            except Exception:
                weights = (0.5, 0.5)
            return _score_fuse(vec_nodes, bm25_nodes, topn=self._topn, weights=weights, allowed_ids=allowed_ids)
        return _rrf_fuse(vec_nodes, bm25_nodes, topn=self._topn, allowed_ids=allowed_ids)


class _ScoreFusionRetriever:
    def __init__(self, vector_retriever, bm25_retriever, recall_k: int):
        self._vector = vector_retriever
        self._bm25 = bm25_retriever
        self._k = int(recall_k)

    def retrieve(self, query: str) -> list[Any]:
        vec_nodes = self._vector.retrieve(query)[: self._k]
        bm25_nodes = self._bm25.retrieve(query)[: self._k]
        try:
            w = getattr(rag_query, "FUSION_WEIGHTS", [0.5, 0.5])
            weights = (float(w[0]), float(w[1])) if len(w) >= 2 else (0.5, 0.5)
        except Exception:
            weights = (0.5, 0.5)
        return _score_fuse(vec_nodes, bm25_nodes, topn=self._k, weights=weights)


class _FusionEnhancedRetriever:
    """三路召回 Fusion 检索包装器（原始向量 + HyDE向量 + BM25）。"""

    def __init__(
        self,
        vector_retriever,
        bm25_retriever,
        recall_k: int,
        num_queries: int = 3,
        use_hyde: bool = True,
        use_query2doc: bool = False,
        llm=None,
    ):
        self._vector = vector_retriever
        self._bm25 = bm25_retriever
        self._k = int(recall_k)
        self._num_queries = max(1, int(num_queries))
        self._use_hyde = bool(use_hyde)
        self._use_query2doc = bool(use_query2doc)
        self._llm = llm

    def retrieve(self, query: str) -> list[Any]:
        # 向量侧：原始查询 + HyDE 增强查询（两路）
        # 稀疏侧：BM25（Query2Doc 默认关闭）
        vec_query, bm25_query = _apply_query_enhancement(
            query,
            use_hyde=self._use_hyde,
            use_query2doc=self._use_query2doc,
            llm=self._llm,
        )

        raw_vec_queries = _build_query_variants(query, self._num_queries)
        vec_queries = _build_query_variants(vec_query, self._num_queries)
        bm25_queries = _build_query_variants(bm25_query, self._num_queries)

        raw_vec_lists: list[list[Any]] = []
        vec_lists: list[list[Any]] = []
        bm25_lists: list[list[Any]] = []
        for q in raw_vec_queries:
            raw_vec_lists.append(self._vector.retrieve(q)[: self._k])
        for q in vec_queries:
            vec_lists.append(self._vector.retrieve(q)[: self._k])
        for q in bm25_queries:
            bm25_lists.append(self._bm25.retrieve(q)[: self._k])

        raw_vec_nodes = _aggregate_nodes_by_min_rank(raw_vec_lists)
        vec_nodes = _aggregate_nodes_by_min_rank(vec_lists)
        bm25_nodes = _aggregate_nodes_by_min_rank(bm25_lists)

        try:
            w = getattr(rag_query, "FUSION_WEIGHTS", [0.5, 0.5])
            vec_weight = float(w[0]) if len(w) >= 1 else 1.0
            bm25_weight = float(w[1]) if len(w) >= 2 else 1.0
        except Exception:
            vec_weight, bm25_weight = (1.0, 1.0)
        hyde_scale = float(getattr(rag_query, "HYDE_ROUTE_WEIGHT", 0.7))
        hyde_weight = vec_weight * hyde_scale
        rrf_k = int(getattr(rag_query, "FUSION_RRF_K", 60))

        use_hyde_route = bool(self._use_hyde and vec_query and vec_query != query)
        if use_hyde_route:
            return _rrf_fuse_multi(
                [raw_vec_nodes, vec_nodes, bm25_nodes],
                [vec_weight, hyde_weight, bm25_weight],
                topn=self._k,
                k=rrf_k,
            )
        return _rrf_fuse_multi(
            [raw_vec_nodes, bm25_nodes],
            [vec_weight, bm25_weight],
            topn=self._k,
            k=rrf_k,
        )


class _MultiPathRetrieverWrapper:
    """MultiPath 多路召回检索器包装器（用于评估）。

    支持功能：
    - Dense + Sparse 多路召回
    - 多查询变体扩展（关键优化！）
    - RRF 融合
    - 可选 ColBERT Late Interaction 重排
    - 可选 HyDE / Query2Doc 查询增强
    """

    def __init__(
        self,
        vector_retriever,
        bm25_retriever,
        recall_k: int,
        use_colbert: bool = False,
        use_hyde: bool = False,
        use_query2doc: bool = False,
        colbert_candidates: int = 50,
        rrf_k: int = 60,
        num_queries: int = 3,  # 新增：多查询扩展数量
        llm=None,
    ):
        self._vector = vector_retriever
        self._bm25 = bm25_retriever
        self._k = int(recall_k)
        self._use_colbert = use_colbert
        self._use_hyde = use_hyde
        self._use_query2doc = use_query2doc
        self._colbert_candidates = colbert_candidates
        self._rrf_k = rrf_k
        self._num_queries = max(1, int(num_queries))  # 多查询扩展
        self._llm = llm
        self._colbert_reranker = None

        # 延迟加载 ColBERT
        if use_colbert:
            self._init_colbert()

    def _init_colbert(self):
        """初始化 ColBERT reranker。"""
        try:
            build_colbert = getattr(rag_query, "_try_build_colbert_reranker", None)
            if build_colbert:
                self._colbert_reranker = build_colbert()
                if self._colbert_reranker:
                    print("[MultiPath-Eval] ColBERT reranker 初始化成功")
                else:
                    print("[MultiPath-Eval] ColBERT reranker 构建失败，将使用 RRF 融合")
        except Exception as e:
            print(f"[MultiPath-Eval] ColBERT 初始化异常: {e}")
            self._colbert_reranker = None

    def _rrf_fuse_simple(
        self,
        vec_nodes: list[Any],
        bm25_nodes: list[Any],
    ) -> list[tuple[str, Any, float]]:
        """简单 RRF 融合。"""
        k = self._rrf_k
        scores: dict[str, float] = {}
        node_map: dict[str, Any] = {}

        for rank, n in enumerate(vec_nodes):
            nid = _node_id(n)
            if not nid:
                continue
            scores[nid] = scores.get(nid, 0) + 1.0 / (k + rank + 1)
            if nid not in node_map:
                node_map[nid] = n

        for rank, n in enumerate(bm25_nodes):
            nid = _node_id(n)
            if not nid:
                continue
            scores[nid] = scores.get(nid, 0) + 1.0 / (k + rank + 1)
            if nid not in node_map:
                node_map[nid] = n

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [(nid, node_map[nid], scores[nid]) for nid in sorted_ids]

    def retrieve(self, query: str) -> list[Any]:
        # 1. 查询增强：HyDE 用于向量检索，Query2Doc 用于 BM25
        vec_query, bm25_query = _apply_query_enhancement(
            query,
            use_hyde=self._use_hyde,
            use_query2doc=self._use_query2doc,
            llm=self._llm,
        )

        # 2. 构建多查询变体（关键优化！与 fusion_expand 保持一致）
        # 向量检索使用 HyDE 增强后的查询作为基础
        vec_queries = _build_query_variants(vec_query, self._num_queries)
        # BM25 使用 Query2Doc 增强后的查询作为基础
        bm25_queries = _build_query_variants(bm25_query, self._num_queries)

        # 3. 多路并行召回
        vec_lists: list[list[Any]] = []
        bm25_lists: list[list[Any]] = []
        
        for vq in vec_queries:
            vec_lists.append(self._vector.retrieve(vq)[: self._k])
        
        if self._bm25:
            for bq in bm25_queries:
                bm25_lists.append(self._bm25.retrieve(bq)[: self._k])

        # 4. 聚合多查询结果（按最小排名）
        vec_nodes = _aggregate_nodes_by_min_rank(vec_lists)
        bm25_nodes = _aggregate_nodes_by_min_rank(bm25_lists) if bm25_lists else []

        # 5. RRF 融合
        if bm25_nodes:
            fused = self._rrf_fuse_simple(vec_nodes, bm25_nodes)
        else:
            fused = [(_node_id(n) or str(i), n, _node_score(n) or 0.0) for i, n in enumerate(vec_nodes)]

        # 6. ColBERT 重排（可选）
        candidates = fused[:self._colbert_candidates]

        if self._use_colbert and self._colbert_reranker and candidates:
            try:
                # 准备 rerank 输入
                rerank_input = [
                    (nid, _node_text(node))
                    for nid, node, _ in candidates
                ]
                reranked = self._colbert_reranker.rerank(query, rerank_input)

                # 构建结果
                nid_to_node = {nid: node for nid, node, _ in candidates}
                results: list[Any] = []
                for nid, _, score in reranked[:self._k]:
                    if nid in nid_to_node:
                        results.append(nid_to_node[nid])
                return results
            except Exception as e:
                print(f"[MultiPath-Eval] ColBERT 重排失败: {e}")

        # 返回 RRF 融合结果
        return [node for _, node, _ in candidates[:self._k]]


class _HyDERetriever:
    """HyDE 增强检索器包装器。"""

    def __init__(self, base_retriever, recall_k: int, llm=None):
        self._base = base_retriever
        self._k = int(recall_k)
        self._llm = llm

    def retrieve(self, query: str) -> list[Any]:
        # 使用 HyDE 增强查询
        enhanced_query, _ = _apply_query_enhancement(
            query, use_hyde=True, use_query2doc=False, llm=self._llm
        )
        return self._base.retrieve(enhanced_query)[: self._k]


class _Query2DocRetriever:
    """Query2Doc 增强检索器包装器（用于 BM25）。"""

    def __init__(self, base_retriever, recall_k: int, llm=None):
        self._base = base_retriever
        self._k = int(recall_k)
        self._llm = llm

    def retrieve(self, query: str) -> list[Any]:
        # 使用 Query2Doc 增强查询
        _, enhanced_query = _apply_query_enhancement(
            query, use_hyde=False, use_query2doc=True, llm=self._llm
        )
        return self._base.retrieve(enhanced_query)[: self._k]


def _build_retriever(
    index,
    recall_k: int,
    mode: str = "fusion",
    use_colbert: bool = False,
    use_hyde: bool = False,
    use_query2doc: bool = False,
    colbert_candidates: int = 50,
    rrf_k: int = 60,
    llm=None,
):
    """构建检索器。

    支持的模式：
    - vector: 纯向量检索
    - bm25: 纯 BM25 检索
    - fusion: QueryFusion 混合检索
    - fusion_rerank: QueryFusion + Rerank
    - fusion_expand: QueryFusion + 查询扩展
    - fusion_score: 分数加权融合
    - union: 交替合并
    - rrf: RRF 融合
    - union_rrf: Union + RRF
    - multipath: MultiPath 多路召回 + RRF（Phase 3.1）
    - multipath_colbert: MultiPath + ColBERT 重排（Phase 3.1）
    - hyde: HyDE 查询增强（Phase 3.3）
    - query2doc: Query2Doc 查询增强（Phase 3.3）
    - multipath_hyde: MultiPath + HyDE
    - multipath_q2d: MultiPath + Query2Doc
    - multipath_full: MultiPath + HyDE + Query2Doc + ColBERT（完整优化）
    """
    vector_store = index.storage_context.vector_store
    bm25 = None
    try:
        if isinstance(vector_store, rag_query.MilvusVectorStore):
            bm25 = rag_query._load_or_build_bm25(vector_store)
    except Exception:
        bm25 = None

    vector_retriever = index.as_retriever(similarity_top_k=recall_k)
    
    # 获取多查询扩展数量（与 fusion_expand 保持一致）
    num_queries = max(3, int(getattr(rag_query, "FUSION_NUM_QUERIES", 5)))

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 3.1 - MultiPath 多路召回
    # ─────────────────────────────────────────────────────────────────────────
    if mode == "multipath":
        lexical_retriever = rag_query.LocalBM25Retriever(bm25, similarity_top_k=recall_k) if bm25 else None
        return _MultiPathRetrieverWrapper(
            vector_retriever,
            lexical_retriever,
            recall_k,
            use_colbert=False,
            use_hyde=use_hyde,
            use_query2doc=use_query2doc,
            rrf_k=rrf_k,
            num_queries=num_queries,
            llm=llm,
        )

    if mode == "multipath_colbert":
        lexical_retriever = rag_query.LocalBM25Retriever(bm25, similarity_top_k=recall_k) if bm25 else None
        return _MultiPathRetrieverWrapper(
            vector_retriever,
            lexical_retriever,
            recall_k,
            use_colbert=True,
            use_hyde=use_hyde,
            use_query2doc=use_query2doc,
            colbert_candidates=colbert_candidates,
            rrf_k=rrf_k,
            num_queries=num_queries,
            llm=llm,
        )

    if mode == "multipath_hyde":
        lexical_retriever = rag_query.LocalBM25Retriever(bm25, similarity_top_k=recall_k) if bm25 else None
        return _MultiPathRetrieverWrapper(
            vector_retriever,
            lexical_retriever,
            recall_k,
            use_colbert=False,
            use_hyde=True,
            use_query2doc=False,
            rrf_k=rrf_k,
            num_queries=num_queries,
            llm=llm,
        )

    if mode == "multipath_q2d":
        lexical_retriever = rag_query.LocalBM25Retriever(bm25, similarity_top_k=recall_k) if bm25 else None
        return _MultiPathRetrieverWrapper(
            vector_retriever,
            lexical_retriever,
            recall_k,
            use_colbert=False,
            use_hyde=False,
            use_query2doc=True,
            rrf_k=rrf_k,
            num_queries=num_queries,
            llm=llm,
        )

    if mode == "multipath_enhanced":
        # MultiPath + HyDE + Query2Doc（不含 ColBERT 重排）
        lexical_retriever = rag_query.LocalBM25Retriever(bm25, similarity_top_k=recall_k) if bm25 else None
        return _MultiPathRetrieverWrapper(
            vector_retriever,
            lexical_retriever,
            recall_k,
            use_colbert=False,
            use_hyde=True,
            use_query2doc=True,
            rrf_k=rrf_k,
            num_queries=num_queries,
            llm=llm,
        )

    if mode == "multipath_full":
        # 完整优化：MultiPath + HyDE + Query2Doc + ColBERT
        lexical_retriever = rag_query.LocalBM25Retriever(bm25, similarity_top_k=recall_k) if bm25 else None
        return _MultiPathRetrieverWrapper(
            vector_retriever,
            lexical_retriever,
            recall_k,
            use_colbert=True,
            use_hyde=True,
            use_query2doc=True,
            colbert_candidates=colbert_candidates,
            rrf_k=rrf_k,
            num_queries=num_queries,
            llm=llm,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 3.3 - HyDE / Query2Doc 单独使用
    # ─────────────────────────────────────────────────────────────────────────
    if mode == "hyde":
        return _HyDERetriever(vector_retriever, recall_k, llm=llm)

    if mode == "query2doc":
        if bm25 is None:
            print("[WARN] query2doc 模式需要 BM25，回退到 vector 模式")
            return vector_retriever
        lexical_retriever = rag_query.LocalBM25Retriever(bm25, similarity_top_k=recall_k)
        return _Query2DocRetriever(lexical_retriever, recall_k, llm=llm)

    # ─────────────────────────────────────────────────────────────────────────
    # 原有模式
    # ─────────────────────────────────────────────────────────────────────────
    if mode == "vector" or bm25 is None:
        return vector_retriever

    if mode == "bm25":
        return rag_query.LocalBM25Retriever(bm25, similarity_top_k=recall_k)

    if mode == "union":
        lexical_retriever = rag_query.LocalBM25Retriever(bm25, similarity_top_k=recall_k)
        return _UnionRetriever(vector_retriever, lexical_retriever, recall_k)

    if mode == "rrf":
        lexical_retriever = rag_query.LocalBM25Retriever(bm25, similarity_top_k=recall_k)
        return _RRFRetriever(
            vector_retriever,
            lexical_retriever,
            recall_k,
            topn=recall_k,
            use_union=False,
            num_queries=1,
            fusion_mode="rrf",
        )

    if mode == "union_rrf":
        lexical_retriever = rag_query.LocalBM25Retriever(bm25, similarity_top_k=recall_k)
        num_q = max(3, int(getattr(rag_query, "FUSION_NUM_QUERIES", 3)))
        return _RRFRetriever(
            vector_retriever,
            lexical_retriever,
            recall_k,
            topn=recall_k,
            use_union=True,
            num_queries=num_q,
            fusion_mode="score",
        )

    if mode == "fusion_score":
        lexical_retriever = rag_query.LocalBM25Retriever(bm25, similarity_top_k=recall_k)
        return _ScoreFusionRetriever(vector_retriever, lexical_retriever, recall_k)

    if mode == "fusion_hyde_rerank":
        lexical_retriever = rag_query.LocalBM25Retriever(bm25, similarity_top_k=recall_k)
        # Fusion + HyDE + 多查询扩展，充分利用查询变体提升召回
        num_q = max(3, int(getattr(rag_query, "FUSION_NUM_QUERIES", 4)))
        return _FusionEnhancedRetriever(
            vector_retriever,
            lexical_retriever,
            recall_k,
            num_queries=num_q,
            use_hyde=True,
            use_query2doc=False,
            llm=llm,
        )

    # 向后兼容旧模式名
    if mode == "fusion_hyde_q2d":
        lexical_retriever = rag_query.LocalBM25Retriever(bm25, similarity_top_k=recall_k)
        num_q = 1
        return _FusionEnhancedRetriever(
            vector_retriever,
            lexical_retriever,
            recall_k,
            num_queries=num_q,
            use_hyde=True,
            use_query2doc=False,
            llm=llm,
        )

    # fusion / fusion_rerank / fusion_expand
    lexical_retriever = rag_query.LocalBM25Retriever(bm25, similarity_top_k=recall_k)
    if mode == "fusion_expand":
        num_q = max(3, int(getattr(rag_query, "FUSION_NUM_QUERIES", 3)))
    else:
        num_q = 3
    return rag_query.QueryFusionRetriever(
        retrievers=[vector_retriever, lexical_retriever],
        similarity_top_k=recall_k,
        num_queries=num_q,
        mode=rag_query.FUSION_MODES.RECIPROCAL_RANK,
        use_async=False,
    )


def _maybe_rerank(reranker, nodes: list[Any], query: str) -> list[Any]:
    if reranker is None:
        return nodes
    if not nodes:
        return nodes
    try:
        return reranker.postprocess_nodes(nodes, query_bundle=QueryBundle(query_str=query))
    except Exception as e:
        print(f"[WARN] rerank failed: {e}", file=sys.stderr)
        return nodes


def _maybe_rerank_range(
    reranker,
    nodes: list[Any],
    query: str,
    start: int,
    end: int,
) -> list[Any]:
    if reranker is None:
        return nodes
    if not nodes:
        return nodes
    if start < 1:
        start = 1
    if end < start:
        return nodes
    n = len(nodes)
    if n <= 1:
        return nodes
    s = min(start - 1, n)
    e = min(end, n)
    if s >= e:
        return nodes
    head = nodes[:s]
    mid = nodes[s:e]
    tail = nodes[e:]
    try:
        mid = reranker.postprocess_nodes(mid, query_bundle=QueryBundle(query_str=query))
    except Exception as e:
        print(f"[WARN] rerank failed: {e}", file=sys.stderr)
        return nodes
    return head + mid + tail


def _iter_ks(ks: str) -> list[int]:
    out: list[int] = []
    for part in ks.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    out = sorted(set([k for k in out if k > 0]))
    if not out:
        out = [1, 3, 5, 10, 20]
    return out


def _parse_range(spec: str) -> tuple[int, int] | None:
    if not spec:
        return None
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if len(parts) != 2:
        return None
    try:
        start = int(parts[0])
        end = int(parts[1])
        return (start, end)
    except Exception:
        return None


def _evaluate(
    dataset: list[dict[str, Any]],
    retriever,
    reranker,
    ks: list[int],
    score_threshold: Optional[float] = None,
    rerank_range: Optional[tuple[int, int]] = None,
    soft_match: str = "none",
    soft_llm_base_url: str = "http://127.0.0.1:11434",
    soft_llm_model: str = "",
    soft_llm_timeout: float = 60.0,
    soft_llm_backend: str = "auto",
    soft_llm_api_key: str = "",
    soft_cache_path: Optional[Path] = None,
    embed_threshold: float = 0.78,
    chunk_expand: bool = False,  # 新增：是否启用 chunk 扩展
    self_rag_enabled: bool = False,
    self_rag_llm=None,
    self_rag_max_iterations: Optional[int] = None,
    self_rag_threshold: Optional[float] = None,
    self_rag_verbose: bool = False,
) -> dict[str, Any]:
    hits_at_k = {k: 0 for k in ks}
    recall_sum_at_k = {k: 0.0 for k in ks}
    precision_sum_at_k = {k: 0.0 for k in ks}  # 新增：Precision@K
    mrr_sum = 0.0
    ndcg_sum_at_k = {k: 0.0 for k in ks}
    ap_sum = 0.0  # 新增：用于计算 MAP
    r_precision_sum = 0.0  # 新增：R-Precision

    soft_hits_at_k = {k: 0 for k in ks}
    soft_mrr_sum = 0.0
    soft_ndcg_sum_at_k = {k: 0.0 for k in ks}

    total = 0

    # group analysis
    group_hits: dict[str, dict[str, Any]] = defaultdict(lambda: {"n": 0, "hits@10": 0})

    soft_cache: dict[str, Any] = {}
    if soft_match != "none":
        if soft_cache_path is None:
            soft_cache_path = Path("./eval/softmatch_cache.jsonl")
        soft_cache = _load_jsonl_kv(soft_cache_path)
    embed_model = getattr(rag_query.Settings, "embed_model", None)

    for item in dataset:
        q = str(item["query"]).strip()
        gt_ids = set([str(x) for x in item.get("ground_truth_ids") or [] if x])
        gt_texts = item.get("ground_truth_text") or []
        if not isinstance(gt_texts, list):
            gt_texts = [str(gt_texts)]
        gt_texts = [str(t) for t in gt_texts if isinstance(t, str) and t.strip()]
        if not gt_ids:
            continue

        total += 1
        def _retrieve_with_postprocess(query_str: str) -> list[Any]:
            nodes_local = retriever.retrieve(query_str)

            # Chunk 扩展（在 rerank 之前，增加上下文召回）
            if chunk_expand and hasattr(rag_query, 'expand_chunks_with_context'):
                try:
                    expand_enabled = getattr(rag_query, 'CHUNK_EXPAND_ENABLED', True)
                    expand_neighbors = int(getattr(rag_query, 'CHUNK_EXPAND_NEIGHBORS', 1))
                    expand_parent = getattr(rag_query, 'CHUNK_EXPAND_PARENT', True)
                    if expand_enabled:
                        nodes_local = rag_query.expand_chunks_with_context(
                            nodes_local,
                            index=None,  # 直接使用 Milvus
                            expand_neighbors=expand_neighbors,
                            expand_parent=expand_parent,
                        )
                except Exception:
                    pass  # 扩展失败时静默继续

            if rerank_range is not None:
                nodes_local = _maybe_rerank_range(
                    reranker,
                    nodes_local,
                    query_str,
                    rerank_range[0],
                    rerank_range[1],
                )
            else:
                nodes_local = _maybe_rerank(reranker, nodes_local, query_str)
            return nodes_local

        if self_rag_enabled and hasattr(rag_query, 'self_rag_query'):
            class _SelfRAGEvalRetriever:
                def __init__(self, retrieve_func):
                    self._retrieve_func = retrieve_func

                def retrieve(self, query_str: str):
                    return self._retrieve_func(query_str)

            try:
                wrapped = _SelfRAGEvalRetriever(_retrieve_with_postprocess)
                sr_result = rag_query.self_rag_query(
                    query=q,
                    retriever=wrapped,
                    llm=self_rag_llm,
                    max_iterations=self_rag_max_iterations,
                    threshold=self_rag_threshold,
                    verbose=self_rag_verbose,
                )
                nodes = sr_result.get('retrieved_docs') or []
                if not nodes:
                    nodes = _retrieve_with_postprocess(q)
            except Exception as e:
                print(f"[WARN] self-rag eval failed: {e}", file=sys.stderr)
                nodes = _retrieve_with_postprocess(q)
        else:
            nodes = _retrieve_with_postprocess(q)
        got_ids: list[str] = []
        got_texts: list[str] = []
        got_scores: list[Optional[float]] = []
        for nws in nodes:
            nid = _node_id(nws)
            if nid:
                got_ids.append(nid)
                got_texts.append(_node_text(nws))
                got_scores.append(_node_score(nws))

        # fusion / multi-retriever 可能返回重复节点；评估时按“第一次出现”计分
        if got_ids:
            # 同步去重 ids/text/score
            seen: set[str] = set()
            new_ids: list[str] = []
            new_texts: list[str] = []
            new_scores: list[Optional[float]] = []
            for nid, txt, sc in zip(got_ids, got_texts, got_scores):
                if nid in seen:
                    continue
                seen.add(nid)
                new_ids.append(nid)
                new_texts.append(txt)
                new_scores.append(sc)
            got_ids, got_texts, got_scores = new_ids, new_texts, new_scores

        # 按 score_threshold 过滤（保持与线上一致的“低分不入上下文”逻辑）
        if score_threshold is not None:
            f_ids: list[str] = []
            f_texts: list[str] = []
            f_scores: list[Optional[float]] = []
            for nid, txt, sc in zip(got_ids, got_texts, got_scores):
                if sc is None:
                    continue
                if sc >= float(score_threshold):
                    f_ids.append(nid)
                    f_texts.append(txt)
                    f_scores.append(sc)
            got_ids, got_texts, got_scores = f_ids, f_texts, f_scores

        # hit@k / recall@k / precision@k
        for k in ks:
            top = set(got_ids[:k])
            inter = top & gt_ids
            if inter:
                hits_at_k[k] += 1
            if gt_ids:
                recall_sum_at_k[k] += len(inter) / float(len(gt_ids))
            # Precision@K
            precision_sum_at_k[k] += _precision_at_k(got_ids, gt_ids, k)

        # NDCG@K
        for k in ks:
            ndcg_sum_at_k[k] += _ndcg_at_k(got_ids, gt_ids, k)

        # MRR (first correct rank)
        rr = 0.0
        for i, nid in enumerate(got_ids, 1):
            if nid in gt_ids:
                rr = 1.0 / float(i)
                break
        mrr_sum += rr

        # MAP (Mean Average Precision)
        ap_sum += _average_precision(got_ids, gt_ids)

        # R-Precision
        r_precision_sum += _r_precision(got_ids, gt_ids)

        # Soft Match（语义命中）：用 LLM 或 embedding 判断 retrieved_text 是否覆盖 GT 片段
        soft_first_rank = 0
        soft_rels: list[int] = [0 for _ in got_ids]
        if soft_match != "none" and gt_texts:
            soft_top_k = max(ks) if ks else 10
            for i, txt in enumerate(got_texts[:soft_top_k], 1):
                if not isinstance(txt, str) or not txt.strip():
                    continue
                ok = False
                if soft_match == "llm":
                    if not soft_llm_model:
                        # 兜底顺序：显式 judge 模型 → 当前 LLM 后端模型 → 本地 Ollama 模型
                        if os.getenv("LLM_BACKEND", "local").strip().lower() == "api":
                            soft_llm_model = os.getenv("LLM_MODEL", "")
                        else:
                            soft_llm_model = os.getenv(
                                "OLLAMA_EVAL_JUDGE_MODEL", os.getenv("OLLAMA_LLM_MODEL", "")
                            )
                    if soft_llm_model:
                        ok = _soft_match_llm(
                            base_url=soft_llm_base_url,
                            model=soft_llm_model,
                            query=q,
                            gt_texts=gt_texts,
                            retrieved_text=txt,
                            cache=soft_cache,
                            cache_path=soft_cache_path,
                            timeout=soft_llm_timeout,
                            backend=soft_llm_backend,
                            api_key=soft_llm_api_key,
                        )
                elif soft_match == "embed":
                    if embed_model is not None:
                        ok = _soft_match_embed(embed_model, gt_texts, txt, threshold=embed_threshold)

                if ok:
                    soft_rels[i - 1] = 1
                    if soft_first_rank == 0:
                        soft_first_rank = i

            # 裁剪：最多允许命中 len(GT) 个相关文档，避免 soft NDCG > 1
            ideal_relevant = max(1, len(gt_ids))
            kept = 0
            for j in range(len(soft_rels)):
                if soft_rels[j]:
                    kept += 1
                    if kept > ideal_relevant:
                        soft_rels[j] = 0

        # soft hit@k / soft mrr / soft ndcg@k
        for k in ks:
            if any(soft_rels[:k]):
                soft_hits_at_k[k] += 1
            soft_ndcg_sum_at_k[k] += _ndcg_at_k_binary_rels(
                soft_rels,
                ideal_relevant=max(1, len(gt_ids)),
                k=k,
            )
        if soft_first_rank:
            soft_mrr_sum += 1.0 / float(soft_first_rank)

        md = item.get("ground_truth_metadata") or {}
        if isinstance(md, dict):
            source_type = str(md.get("source_type") or "unknown")
        else:
            source_type = "unknown"
        g = group_hits[source_type]
        g["n"] += 1
        if set(got_ids[:10]) & gt_ids:
            g["hits@10"] += 1

    metrics = {
        "n": total,
        "mrr": (mrr_sum / total) if total else 0.0,
        "map": (ap_sum / total) if total else 0.0,  # 新增：Mean Average Precision
        "r_precision": (r_precision_sum / total) if total else 0.0,  # 新增：R-Precision
        # 命中率（Hit Rate）/ Recall@K：Top-K 内是否至少命中一个 GT chunk
        "hit_rate": {str(k): (hits_at_k[k] / total) if total else 0.0 for k in ks},
        # Recall@K: 平均覆盖率 |TopK ∩ GT| / |GT|
        "recall": {str(k): (recall_sum_at_k[k] / total) if total else 0.0 for k in ks},
        # Precision@K: Top-K 结果中相关文档的比例
        "precision": {str(k): (precision_sum_at_k[k] / total) if total else 0.0 for k in ks},
        "ndcg": {str(k): (ndcg_sum_at_k[k] / total) if total else 0.0 for k in ks},
        # F1@K: Precision 和 Recall 的调和平均
        "f1": {
            str(k): (
                2 * (precision_sum_at_k[k] / total) * (recall_sum_at_k[k] / total)
                / ((precision_sum_at_k[k] / total) + (recall_sum_at_k[k] / total) + 1e-10)
            ) if total else 0.0
            for k in ks
        },
        "soft": {
            "mode": soft_match,
            "mrr": (soft_mrr_sum / total) if total else 0.0,
            "hit_rate": {str(k): (soft_hits_at_k[k] / total) if total else 0.0 for k in ks},
            "ndcg": {str(k): (soft_ndcg_sum_at_k[k] / total) if total else 0.0 for k in ks},
        },
        "by_source_type": {
            k: {
                "n": v["n"],
                "recall@10": (v["hits@10"] / v["n"]) if v["n"] else 0.0,
            }
            for k, v in sorted(group_hits.items(), key=lambda x: (-x[1]["n"], x[0]))
        },
    }
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate retrieval recall@K on eval_dataset.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
新增检索模式（Phase 3.1 - 3.3）：
    fusion_hyde_rerank Fusion + HyDE + Rerank（问答默认链路）
    fusion_hyde_q2d   (兼容别名) 当前等价于 fusion_hyde_rerank
  multipath          MultiPath 多路召回 + RRF 融合
  multipath_colbert  MultiPath + ColBERT Late Interaction 重排
  multipath_hyde     MultiPath + HyDE 查询增强
  multipath_q2d      MultiPath + Query2Doc 查询增强
  multipath_enhanced MultiPath + HyDE + Query2Doc（不含 ColBERT）
  multipath_full     完整优化：MultiPath + HyDE + Query2Doc + ColBERT
  hyde               单独使用 HyDE 增强向量检索
  query2doc          单独使用 Query2Doc 增强 BM25 检索

示例：
  # 评估 MultiPath 多路召回
  python evaluate_recall.py --mode multipath --recall-k 200

    # 评估 Fusion + HyDE + Rerank
    python evaluate_recall.py --mode fusion_hyde_rerank --rerank --rerank-top-n 30

  # 评估 MultiPath + ColBERT 重排
  python evaluate_recall.py --mode multipath_colbert --colbert-candidates 100

  # 评估完整优化方案
  python evaluate_recall.py --mode multipath_full --recall-k 200 --rerank
"""
    )
    ap.add_argument("--dataset", default="./eval/eval_dataset.json")
    ap.add_argument("--out", default="./eval/eval_report.json")
    # 默认留空 → 交给 build_llm 按 LLM_BACKEND 解析（api 用 LLM_MODEL，local 用 OLLAMA_LLM_MODEL）。
    # 注意：api_clients.build_llm 内部是 `model or LLM_MODEL`，一旦这里给了非空值就会覆盖
    # LLM_MODEL；在纯 API 部署下会把 Ollama 的模型名打到 API 上，触发 404 model not found。
    ap.add_argument(
        "--llm",
        default=os.getenv("EVAL_LLM_MODEL", ""),
        help="覆盖 LLM 模型名；留空则按 LLM_BACKEND 使用 .env 里的 LLM_MODEL / OLLAMA_LLM_MODEL",
    )
    ap.add_argument(
        "--mode",
        choices=[
            # 原有模式
            "vector",
            "bm25",
            "fusion",
            "fusion_rerank",
            "fusion_expand",
            "fusion_score",
            "fusion_hyde_rerank",
            "fusion_hyde_q2d",
            "union",
            "rrf",
            "union_rrf",
            # Phase 3.1 - MultiPath 多路召回
            "multipath",
            "multipath_colbert",
            "multipath_hyde",
            "multipath_q2d",
            "multipath_enhanced",
            "multipath_full",
            # Phase 3.3 - HyDE / Query2Doc
            "hyde",
            "query2doc",
        ],
        default="fusion_hyde_rerank",
    )
    ap.add_argument("--recall-k", type=int, default=int(os.getenv("RECALL_TOP_K", "40")))
    ap.add_argument("--ks", default="1,3,5,10,20")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--score-threshold", type=float, default=float(os.getenv("SCORE_THRESHOLD", "nan")))
    ap.add_argument("--soft-match", choices=["none", "llm", "embed"], default=os.getenv("SOFT_MATCH", "none"))
    # 软匹配 LLM 判定：默认复用当前 LLM 后端（纯 API 部署下即 LLM_BASE_URL / LLM_MODEL），
    # 未配置时回退到本地 Ollama，保持对历史运行的向后兼容。
    # 想用「另一个厂商的模型当判卷人」时，只需单独设置 SOFT_JUDGE_*。
    ap.add_argument(
        "--soft-llm-base-url",
        default=os.getenv("SOFT_JUDGE_BASE_URL")
        or os.getenv("LLM_BASE_URL")
        or os.getenv("OLLAMA_BASE", "http://127.0.0.1:11434"),
    )
    ap.add_argument(
        "--soft-llm-model",
        default=os.getenv("SOFT_JUDGE_MODEL")
        or os.getenv("LLM_MODEL")
        or os.getenv("OLLAMA_EVAL_JUDGE_MODEL", ""),
    )
    ap.add_argument(
        "--soft-llm-api-key",
        default=os.getenv("SOFT_JUDGE_API_KEY") or os.getenv("LLM_API_KEY", ""),
    )
    ap.add_argument(
        "--soft-llm-backend",
        choices=["auto", "ollama", "openai"],
        default=os.getenv("SOFT_JUDGE_BACKEND", "auto"),
        help="软匹配判定的传输方式：auto=按 base_url 猜测（含 11434 视为 Ollama）",
    )
    ap.add_argument(
        "--soft-llm-timeout",
        type=float,
        default=float(
            os.getenv("SOFT_JUDGE_TIMEOUT", os.getenv("OLLAMA_EVAL_JUDGE_TIMEOUT", "60"))
        ),
    )
    ap.add_argument("--soft-cache", default=os.getenv("SOFT_MATCH_CACHE", "./eval/softmatch_cache.jsonl"))
    ap.add_argument("--embed-threshold", type=float, default=float(os.getenv("EMBED_MATCH_THRESHOLD", "0.78")))
    ap.add_argument("--rerank", dest="rerank", action="store_true", default=True, help="Enable rerank postprocessing")
    ap.add_argument("--no-rerank", dest="rerank", action="store_false", help="Disable rerank postprocessing")
    ap.add_argument("--rerank-top-n", type=int, default=0, help="Override RERANK_TOP_N when rerank is enabled")
    ap.add_argument("--rerank-range", default="", help="Only rerank a range (1-based, inclusive), e.g. 100,200")
    # 新增：Chunk 扩展参数
    ap.add_argument("--chunk-expand", action="store_true", help="Enable chunk context expansion (neighbors + parent)")
    # Phase 3.1 新增参数
    ap.add_argument("--colbert-candidates", type=int, default=50, help="ColBERT rerank candidate count")
    ap.add_argument("--rrf-k", type=int, default=60, help="RRF fusion smoothing parameter k")
    # Phase 3.3 新增参数
    ap.add_argument("--use-hyde", action="store_true", help="Enable HyDE query enhancement (for multipath modes)")
    ap.add_argument("--use-query2doc", action="store_true", help="Enable Query2Doc enhancement (for multipath modes)")
    ap.add_argument(
        "--self-rag",
        dest="self_rag",
        action="store_true",
        default=(os.getenv("SELF_RAG_ENABLED", "0") not in {"0", "false", "False"}),
        help="Enable Self-RAG iterative retrieval during evaluation",
    )
    ap.add_argument(
        "--no-self-rag",
        dest="self_rag",
        action="store_false",
        help="Disable Self-RAG iterative retrieval during evaluation",
    )
    ap.add_argument("--self-rag-max-iterations", type=int, default=0, help="Override Self-RAG max iterations")
    ap.add_argument("--self-rag-threshold", type=float, default=float("nan"), help="Override Self-RAG score threshold")
    ap.add_argument("--self-rag-verbose", action="store_true", help="Show Self-RAG iteration logs in evaluation")
    args = ap.parse_args()

    ds_path = Path(args.dataset)
    if not ds_path.exists():
        print(f"[ERROR] dataset not found: {ds_path}", file=sys.stderr)
        return 2

    dataset = _load_dataset(ds_path)
    if args.limit and args.limit > 0:
        dataset = dataset[: args.limit]

    index = rag_query.build_index(args.llm)

    # 获取 LLM 实例（用于 HyDE / Query2Doc）
    # 【注意】HyDE/Query2Doc 在当前 PCB 数据集上可能导致性能下降
    # 生成的假设文档质量不高，反而干扰检索。建议：
    # - 使用 multipath 模式（不含 HyDE）而非 multipath_enhanced
    # - 或者使用 fusion_expand + rerank 作为基线
    llm_instance = None
    uses_hyde = args.use_hyde or args.mode in ("hyde", "fusion_hyde_rerank", "fusion_hyde_q2d", "multipath_hyde", "multipath_enhanced", "multipath_full")
    # 按当前策略禁用 Query2Doc（即使 mode 名称中含有 q2d）
    uses_query2doc = False
    
    if uses_hyde or uses_query2doc:
        # 强制设置环境变量，绕过 query.py 中的默认关闭检查
        if uses_hyde:
            rag_query.HYDE_ENABLED = True
            # 评测时放宽短查询限制，让更多查询受益于 HyDE
            rag_query.SHORT_QUERY_THRESHOLD = 100
            print("[Eval] 强制启用 HyDE，短查询阈值设为 100 字符")
            print("[Eval] ⚠️ 警告: HyDE 在当前数据集上可能导致性能下降，建议使用 --mode multipath 代替")
        rag_query.QUERY2DOC_ENABLED = False
        
        try:
            llm_instance = getattr(rag_query.Settings, "llm", None)
        except Exception:
            llm_instance = None

    # Self-RAG 评估需要 LLM，如果前面没有初始化则复用当前 Settings.llm
    if args.self_rag and llm_instance is None:
        try:
            llm_instance = getattr(rag_query.Settings, "llm", None)
        except Exception:
            llm_instance = None
    rag_query.SELF_RAG_ENABLED = bool(args.self_rag)

    # 构建检索器
    retriever = _build_retriever(
        index,
        recall_k=int(args.recall_k),
        mode=str(args.mode),
        use_colbert=("colbert" in args.mode),
        use_hyde=uses_hyde,
        use_query2doc=uses_query2doc,
        colbert_candidates=int(args.colbert_candidates),
        rrf_k=int(args.rrf_k),
        llm=llm_instance,
    )

    # 构建 reranker（用于后处理）
    reranker = None
    if args.rerank:
        rag_query.RERANK_ENABLED = True
        if args.rerank_top_n and args.rerank_top_n > 0:
            rag_query.RERANK_TOP_N = int(args.rerank_top_n)
        if str(rag_query.RERANK_BACKEND).strip().lower() == "api":
            # 纯 API 部署：没有本地 GPU / 模型，打印 url 与远程模型名才有诊断价值
            print("[Eval] 正在构建 Reranker: backend=api, "
                  f"url={getattr(rag_query, 'RERANK_API_URL', '') or '(未配置)'}, "
                  f"model={getattr(rag_query, 'RERANK_API_MODEL', '') or '(服务默认)'}, "
                  f"top_n={rag_query.RERANK_TOP_N}")
        else:
            print(f"[Eval] 正在构建 Reranker: backend={rag_query.RERANK_BACKEND}, "
                  f"model={rag_query.HF_RERANK_MODEL}, top_n={rag_query.RERANK_TOP_N}, "
                  f"gpu_id={rag_query.RERANK_GPU_ID}")
        reranker = rag_query._try_build_reranker()
        if reranker is None:
            print("[ERROR] rerank enabled but failed to initialize; "
                  "fallback to retrieval only (检查上方堆栈确认原因)", file=sys.stderr)
        else:
            dev = getattr(reranker, "_device", "unknown")
            print(f"[Eval] Reranker 初始化成功, device={dev}")

    ks = _iter_ks(args.ks)

    score_threshold = None
    if args.score_threshold == args.score_threshold:  # not NaN
        score_threshold = float(args.score_threshold)

    rerank_range = _parse_range(str(args.rerank_range))
    metrics = _evaluate(
        dataset,
        retriever,
        reranker,
        ks,
        score_threshold=score_threshold,
        rerank_range=rerank_range,
        soft_match=str(args.soft_match),
        soft_llm_base_url=str(args.soft_llm_base_url),
        soft_llm_model=str(args.soft_llm_model),
        soft_llm_timeout=float(args.soft_llm_timeout),
        soft_llm_backend=str(args.soft_llm_backend),
        soft_llm_api_key=str(args.soft_llm_api_key),
        soft_cache_path=Path(str(args.soft_cache)),
        embed_threshold=float(args.embed_threshold),
        chunk_expand=args.chunk_expand,  # 新增：chunk 扩展
        self_rag_enabled=bool(args.self_rag),
        self_rag_llm=llm_instance,
        self_rag_max_iterations=(int(args.self_rag_max_iterations) if int(args.self_rag_max_iterations) > 0 else None),
        self_rag_threshold=(float(args.self_rag_threshold) if args.self_rag_threshold == args.self_rag_threshold else None),
        self_rag_verbose=bool(args.self_rag_verbose),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # 输出评估结果
    print(f"\n{'='*60}")
    print(f"  评估报告 - 模式: {args.mode}")
    print(f"{'='*60}")
    print(f"[配置] recall_k={args.recall_k}, rerank={args.rerank}, chunk_expand={args.chunk_expand}, self_rag={args.self_rag}")
    if args.self_rag:
        iter_show = int(args.self_rag_max_iterations) if int(args.self_rag_max_iterations) > 0 else int(getattr(rag_query, "SELF_RAG_MAX_ITERATIONS", 3))
        thr_show = float(args.self_rag_threshold) if args.self_rag_threshold == args.self_rag_threshold else float(getattr(rag_query, "SELF_RAG_THRESHOLD", 3.0))
        print(f"[配置] self_rag_max_iterations={iter_show}, self_rag_threshold={thr_show}")
    if args.chunk_expand:
        expand_n = int(getattr(rag_query, 'CHUNK_EXPAND_NEIGHBORS', 1))
        expand_p = getattr(rag_query, 'CHUNK_EXPAND_PARENT', True)
        print(f"[配置] expand_neighbors={expand_n}, expand_parent={expand_p}")
    if args.mode.startswith("multipath"):
        print(f"[配置] rrf_k={args.rrf_k}, colbert_candidates={args.colbert_candidates}")
        print(f"[配置] use_hyde={uses_hyde}")
        print(f"[配置] use_query2doc={uses_query2doc}")
    print("-" * 60)
    print(
        f"[指标] n={metrics['n']} mrr={metrics['mrr']:.4f} map={metrics['map']:.4f} "
        f"r_precision={metrics['r_precision']:.4f}"
    )
    print(f"[指标] hit_rate={metrics['hit_rate']}")
    print(f"[指标] precision={metrics['precision']}")
    print(f"[指标] recall={metrics['recall']}")
    print(f"[指标] ndcg={metrics['ndcg']}")
    print(f"[指标] f1={metrics['f1']}")
    if isinstance(metrics.get("soft"), dict):
        s = metrics["soft"]
        print(
            f"[指标] soft(mode={s.get('mode')}) mrr={float(s.get('mrr') or 0.0):.4f} "
            f"hit_rate={s.get('hit_rate')} ndcg={s.get('ndcg')}"
        )
    print("-" * 60)
    print(f"[完成] 报告已写入: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
