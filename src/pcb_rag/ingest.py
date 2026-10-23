import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

import hashlib
import os
import re
import site
import sys
import json
from pathlib import Path
from typing import List, Tuple, Optional

# 尝试导入chardet用于编码检测
try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False

def _bootstrap_local_venv() -> None:
    """允许模块运行时也能找到本项目 `.venv` 里的依赖（兼容 Windows / POSIX）。"""

    repo_dir = Path(__file__).resolve().parents[2]
    venv_dir = repo_dir / ".venv"

    # 无论能否找到 .venv，都把 src 目录加入 sys.path，保证可直接运行脚本
    src_dir = repo_dir / "src"
    if src_dir.is_dir() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    # 坏链接或无权访问时 is_dir() 会抛 OSError（Windows 上 .venv 指向另一个
    # 平台的路径时，stat 返回 WinError 1920）。这里只需要判断"能否用本地
    # venv"，取不到就当作没有 —— 不应让整个模块导入失败。
    try:
        if not venv_dir.is_dir():
            return
    except OSError:
        return

    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        venv_dir / "Lib" / "site-packages",            # Windows
        venv_dir / "lib" / version / "site-packages",  # POSIX
    ]

    lib_dir = venv_dir / "lib"
    if lib_dir.is_dir():
        try:
            for name in sorted(os.listdir(lib_dir), reverse=True):
                if name.startswith("python"):
                    candidates.append(lib_dir / name / "site-packages")
        except OSError:
            pass

    for sp in candidates:
        sp_str = str(sp)
        if sp.is_dir() and sp_str not in sys.path:
            sys.path.insert(0, sp_str)
            site.addsitedir(sp_str)
            return


_bootstrap_local_venv()


def _sanitize_proxy_env_for_httpx() -> None:
    """修正服务器环境中不被 httpx 识别的代理 scheme。

    常见情况：ALL_PROXY=socks://127.0.0.1:10808/
    httpx 期望 socks5:// 或 socks5h://，否则会在 import 时 ValueError。
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

    no_proxy_key = "NO_PROXY" if "NO_PROXY" in os.environ else "no_proxy" if "no_proxy" in os.environ else "NO_PROXY"
    existing = os.environ.get(no_proxy_key, "")
    entries = [e.strip() for e in existing.split(",") if e.strip()]
    for host in ("127.0.0.1", "localhost"):
        if host not in entries:
            entries.append(host)
    os.environ[no_proxy_key] = ",".join(entries)


# 先加载仓库根目录的 .env（直接 `python -m pcb_rag.ingest` 时也能读到配置），
# 再执行代理规范化：这样 .env 里写的 HTTP_PROXY / ALL_PROXY 也会被一并处理
from pcb_rag.env_loader import load_project_env
load_project_env()

_sanitize_proxy_env_for_httpx()

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser, HierarchicalNodeParser
from llama_index.core.schema import TextNode
from llama_index.readers.file import PyMuPDFReader, DocxReader, RTFReader
from llama_index.vector_stores.milvus import MilvusVectorStore

# 统一模型客户端工厂：LLM / Embedding 支持 local（Ollama）与 api 双后端
from pcb_rag import incremental
from pcb_rag.chunk_quality import assess_chunk_quality
from pcb_rag.api_clients import (
    EMBED_BACKEND,
    LLM_BACKEND as LLM_BACKEND,
    OLLAMA_BASE,
    OLLAMA_EMBED_MODEL as OLLAMA_EMBED_MODEL,
    build_embed_model,
    build_llm,
    get_embedding_dim,
)

# P1-2：GraphRAG —— 入库阶段抽取实体关系并维护知识图谱
from pcb_rag.graph_rag import (
    GRAPH_COMMUNITY_ENABLED,
    GRAPH_EXTRACT_BACKEND,
    GRAPH_RAG_ENABLED,
    KnowledgeGraph,
    build_graph_from_nodes,
    summarize_communities,
)

# P2：多模态 —— PDF 图片描述与表格结构化
from pcb_rag.multimodal import augment_documents_with_multimodal

# P2：访问控制 —— 入库时写入租户 / 可视性字段
from pcb_rag.security import ACL_ENABLED, acl_metadata_for_ingest

try:
    import nest_asyncio

    nest_asyncio.apply()
except Exception:
    pass

DATA_DIR = os.getenv("DATA_DIR", "./data/clear_docs")
MILVUS_URI = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")
COLLECTION = os.getenv("COLLECTION", "pcb_kb")

# 切块策略配置：parent_child（两层结构）| structure（结构感知）| semantic（语义感知）| hierarchical（层次化）| sentence（固定大小）| attention（注意力语义感知）
NODE_PARSER_MODE = os.getenv("NODE_PARSER_MODE", "parent_child").strip().lower()

# ========== Parent-Child 两层分块参数（parent_child 模式）==========
# Parent 层：按结构划分（section/段落块）
PARENT_MAX_SIZE = int(os.getenv("PARENT_MAX_SIZE", "3000"))      # Parent 最大字符数
PARENT_MIN_SIZE = int(os.getenv("PARENT_MIN_SIZE", "200"))       # Parent 最小字符数（太小合并）
# 标题小节过短时仍保留的最小长度（避免标题被吞掉，影响命中）
PARENT_MIN_SECTION_KEEP = int(os.getenv("PARENT_MIN_SECTION_KEEP", "120"))

# Child 层：在 Parent 内部细分
CHILD_MAX_SIZE = int(os.getenv("CHILD_MAX_SIZE", "800"))         # Child 最大字符数
CHILD_MIN_SIZE = int(os.getenv("CHILD_MIN_SIZE", "150"))         # Child 最小字符数
CHILD_OVERLAP = int(os.getenv("CHILD_OVERLAP", "80"))            # Child 间重叠字符数

# 语义切分触发条件
SEMANTIC_SPLIT_THRESHOLD = int(os.getenv("SEMANTIC_SPLIT_THRESHOLD", "1600"))  # Parent 长度超过此值才考虑语义切分（CHILD_MAX_SIZE × 2）
SEMANTIC_VARIANCE_THRESHOLD = float(os.getenv("SEMANTIC_VARIANCE_THRESHOLD", "0.15"))  # 相似度波动超过此值触发语义切分

# ========== Chunk 质量评估 ==========
# 切块器只负责"按结构切开"，不判断切开后有没有检索价值。PCB 规范文档里
# 存在大量裸标签罗列块（<default> <port> ...），语义空洞但会占用向量库与
# BM25 的统计量。这里在入库时给低价值块打标记：
#   off  = 不评估
#   flag = 只在 metadata 写 low_value / chunk_quality（默认，可回滚）
#   drop = 直接丢弃（块数会减少，prev/next 链出现空洞，慎用）
CHUNK_QUALITY_MODE = os.getenv("CHUNK_QUALITY_MODE", "flag").strip().lower()


# 结构化切块参数（structure模式 - 旧版）
STRUCTURE_MAX_CHUNK_SIZE = int(os.getenv("STRUCTURE_MAX_CHUNK_SIZE", "2000"))  # 最大chunk大小（字符）
STRUCTURE_MIN_CHUNK_SIZE = int(os.getenv("STRUCTURE_MIN_CHUNK_SIZE", "200"))   # 最小chunk大小（字符）
STRUCTURE_OVERLAP = int(os.getenv("STRUCTURE_OVERLAP", "100"))                  # chunk重叠

# 语义切块参数（semantic模式）
SEMANTIC_BUFFER_SIZE = int(os.getenv("SEMANTIC_BUFFER_SIZE", "1"))  # 句子缓冲
SEMANTIC_BREAKPOINT_THRESHOLD = int(os.getenv("SEMANTIC_BREAKPOINT_THRESHOLD", "90"))  # 相似度阈值百分位

# Transformer 注意力语义切块参数（attention模式）
ATTENTION_MODEL = os.getenv("ATTENTION_MODEL", "qwen3-embedding:8b-q8_0")  # Ollama 嵌入模型
ATTENTION_USE_OLLAMA = os.getenv("ATTENTION_USE_OLLAMA", "1") not in {"0", "false", "False"}  # 是否使用 Ollama
ATTENTION_MIN_CHUNK_SIZE = int(os.getenv("ATTENTION_MIN_CHUNK_SIZE", "80"))  # 最小chunk大小(字符) - 捕获独立引脚/图号/参数行
ATTENTION_MAX_CHUNK_SIZE = int(os.getenv("ATTENTION_MAX_CHUNK_SIZE", "1000"))  # 最大chunk大小(字符) - 避免过载
ATTENTION_THRESHOLD = float(os.getenv("ATTENTION_THRESHOLD", "0.45"))        # 相似度阈值(低于此值切分) - 更积极切分
ATTENTION_BREAKPOINT_PERCENTILE = int(os.getenv("ATTENTION_BREAKPOINT_PERCENTILE", "22"))  # 断点百分位(越高越积极)
ATTENTION_OVERLAP = int(os.getenv("ATTENTION_OVERLAP", "220"))               # 字符重叠 - 增加补偿上下文割裂
ATTENTION_WINDOW_SIZE = int(os.getenv("ATTENTION_WINDOW_SIZE", "6"))         # 滑动窗口大小 - 轻微提升平滑度

# 层次化 Chunk 结构参数（父子/邻接关系）
CHUNK_HIERARCHY_ENABLED = os.getenv("CHUNK_HIERARCHY_ENABLED", "1") not in {"0", "false", "False"}
CHUNK_EXPAND_NEIGHBORS = int(os.getenv("CHUNK_EXPAND_NEIGHBORS", "1"))       # 检索时扩展的邻居数量
CHUNK_EXPAND_PARENT = os.getenv("CHUNK_EXPAND_PARENT", "1") not in {"0", "false", "False"}  # 是否扩展父chunk

# 层次化切块参数（hierarchical模式）
HIERARCHICAL_CHUNK_SIZES = [int(x) for x in os.getenv("HIERARCHICAL_CHUNK_SIZES", "1024,384,128").split(",")]

# 传统固定切块参数（sentence模式及其他作为fallback）
# 优化：从512减到384，overlap增加到150，提高召回精度
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "384"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# 编码修复开关
AUTO_FIX_ENCODING = os.getenv("AUTO_FIX_ENCODING", "1") not in {"0", "false", "False"}
# OCR乱码清理开关
CLEAN_OCR_GARBAGE = os.getenv("CLEAN_OCR_GARBAGE", "1") not in {"0", "false", "False"}


# ================== 编码检测与修复 ==================

def _detect_encoding_simple(data: bytes) -> str:
    """简化的编码检测（无chardet时使用）"""
    # 检查BOM
    if data.startswith(b'\xef\xbb\xbf'):
        return 'utf-8-sig'
    if data.startswith(b'\xff\xfe'):
        return 'utf-16-le'
    if data.startswith(b'\xfe\xff'):
        return 'utf-16-be'
    
    # 尝试UTF-8
    try:
        data.decode('utf-8')
        return 'utf-8'
    except UnicodeDecodeError:
        pass
    
    # 尝试GBK（中文Windows常见）
    try:
        decoded = data.decode('gbk')
        chinese_ratio = len(re.findall(r'[\u4e00-\u9fff]', decoded)) / max(len(decoded), 1)
        if chinese_ratio > 0.05:
            return 'gbk'
    except UnicodeDecodeError:
        pass
    
    # 尝试GB18030（最全的中文编码）
    try:
        data.decode('gb18030')
        return 'gb18030'
    except UnicodeDecodeError:
        pass
    
    return 'utf-8'


def _detect_encoding(data: bytes) -> Tuple[str, float]:
    """检测字节数据的编码"""
    # 首先尝试UTF-8（最常见且最安全）
    try:
        decoded = data.decode('utf-8')
        # 验证是否有合理的中文内容
        chinese_ratio = len(re.findall(r'[\u4e00-\u9fff]', decoded)) / max(len(decoded), 1)
        garbage_ratio = _calculate_garbage_ratio(decoded)
        
        # 如果UTF-8解码成功且质量好，直接返回
        if garbage_ratio < 0.05 and chinese_ratio > 0.01:
            return 'utf-8', 0.99
    except UnicodeDecodeError:
        pass
    
    if HAS_CHARDET:
        result = chardet.detect(data)
        encoding = result.get('encoding', 'utf-8') or 'utf-8'
        confidence = result.get('confidence', 0.0) or 0.0
        
        # chardet有时会误判UTF-8为GB2312
        if encoding.lower() in ('gb2312', 'gbk', 'gb18030'):
            # 再次验证UTF-8
            try:
                utf8_decoded = data.decode('utf-8')
                utf8_garbage = _calculate_garbage_ratio(utf8_decoded)
                
                # 如果UTF-8质量更好，使用UTF-8
                if utf8_garbage < 0.05:
                    return 'utf-8', 0.99
            except Exception:
                pass
        
        # chardet有时会误判中文为ISO-8859-1
        if encoding.lower() in ('iso-8859-1', 'ascii', 'windows-1252') and confidence < 0.9:
            try:
                decoded = data.decode('gbk')
                chinese_count = len(re.findall(r'[\u4e00-\u9fff]', decoded))
                if chinese_count > 10:
                    return 'gbk', 0.95
            except Exception:
                pass
        
        return encoding, confidence
    else:
        return _detect_encoding_simple(data), 0.8


def _decode_with_fallback(data: bytes) -> Tuple[str, str]:
    """尝试多种编码解码，返回(解码文本, 使用的编码)"""
    encoding, _ = _detect_encoding(data)
    
    # 如果检测为UTF-8，直接返回
    if encoding.lower() in ('utf-8', 'utf-8-sig'):
        try:
            return data.decode('utf-8'), 'utf-8'
        except Exception:
            pass
    
    encodings_to_try = [encoding]
    if encoding.lower() not in ('utf-8', 'utf-8-sig'):
        encodings_to_try.append('utf-8')
    if encoding.lower() not in ('gbk', 'gb2312', 'gb18030'):
        encodings_to_try.extend(['gbk', 'gb18030', 'gb2312'])
    
    best_result = None
    best_score = float('inf')
    
    for enc in encodings_to_try:
        try:
            decoded = data.decode(enc)
            # 验证解码质量
            score = _score_text_quality(decoded)
            
            # 选择质量最好的解码结果
            if score < best_score:
                best_score = score
                best_result = (decoded, enc)
            
            # 如果找到高质量结果，提前返回
            if score < 0.05:
                return decoded, enc
        except (UnicodeDecodeError, LookupError):
            continue
    
    if best_result:
        repaired = _try_mojibake_repair(best_result[0])
        if repaired and repaired[2] + 0.02 < best_score:
            return repaired[0], repaired[1]
    
    if best_result and best_score < 0.3:
        return best_result
    
    # 最后强制UTF-8
    return data.decode('utf-8', errors='replace'), 'utf-8(forced)'


# ================== OCR乱码检测与清理 ==================

# 合法的Unicode范围
VALID_UNICODE_RANGES = [
    (0x0000, 0x007F),   # ASCII
    (0x00A0, 0x00FF),   # 拉丁补充
    (0x0370, 0x03FF),   # 希腊字母：μm、Ω、Δ 等工程单位和公式符号
    (0x2000, 0x206F),   # 通用标点
    (0x2070, 0x209F),   # 上下标
    (0x20A0, 0x20CF),   # 货币符号
    (0x2100, 0x214F),   # 字母符号
    (0x2150, 0x218F),   # 数字形式
    (0x2190, 0x21FF),   # 箭头
    (0x2200, 0x22FF),   # 数学运算符
    (0x2300, 0x23FF),   # 杂项技术符号
    (0x2460, 0x24FF),   # 带圈字母数字
    (0x2500, 0x257F),   # 制表符
    (0x25A0, 0x25FF),   # 几何形状
    (0x2600, 0x26FF),   # 杂项符号
    (0x3000, 0x303F),   # CJK标点
    (0x3040, 0x309F),   # 平假名
    (0x30A0, 0x30FF),   # 片假名
    (0x3100, 0x312F),   # 注音符号
    (0x3200, 0x32FF),   # 带圈CJK
    (0x3400, 0x4DBF),   # CJK扩展A
    (0x4E00, 0x9FFF),   # CJK基本
    (0xF900, 0xFAFF),   # CJK兼容
    (0xFE30, 0xFE4F),   # CJK兼容形式
    (0xFF00, 0xFFEF),   # 半角全角形式
]


def _is_valid_char(char: str) -> bool:
    """检查字符是否在合法Unicode范围内"""
    code = ord(char)
    for start, end in VALID_UNICODE_RANGES:
        if start <= code <= end:
            return True
    return False


def _calculate_garbage_ratio(text: str) -> float:
    """计算文本中乱码字符的比例"""
    if not text:
        return 0.0
    
    total = 0
    garbage_count = 0
    
    for char in text:
        if char.isspace():
            continue
        total += 1
        if char == '\ufffd':  # Unicode replacement char
            garbage_count += 1
            continue
        if not _is_valid_char(char):
            garbage_count += 1
    
    return garbage_count / total if total > 0 else 0.0


def _calculate_chinese_ratio(text: str) -> float:
    if not text:
        return 0.0
    chinese_count = len(re.findall(r'[\u4e00-\u9fff]', text))
    return chinese_count / max(len(text), 1)


def _calculate_replacement_ratio(text: str) -> float:
    if not text:
        return 0.0
    return text.count('\ufffd') / max(len(text), 1)


def _calculate_mojibake_ratio(text: str) -> float:
    if not text:
        return 0.0
    mojibake_count = 0
    for char in text:
        code = ord(char)
        if 0x00C0 <= code <= 0x00FF:
            mojibake_count += 1
    return mojibake_count / max(len(text), 1)


def _score_text_quality(text: str) -> float:
    garbage_ratio = _calculate_garbage_ratio(text)
    replacement_ratio = _calculate_replacement_ratio(text)
    mojibake_ratio = _calculate_mojibake_ratio(text)
    chinese_ratio = _calculate_chinese_ratio(text)

    score = garbage_ratio + replacement_ratio * 2.0 + mojibake_ratio * 0.6

    if len(text) > 50 and chinese_ratio < 0.01:
        score += 0.05
    else:
        score -= min(chinese_ratio, 0.2) * 0.5

    return score


def _should_try_mojibake_repair(text: str) -> bool:
    if not text or len(text) < 50:
        return False
    replacement_ratio = _calculate_replacement_ratio(text)
    mojibake_ratio = _calculate_mojibake_ratio(text)
    chinese_ratio = _calculate_chinese_ratio(text)

    return replacement_ratio > 0.002 or (mojibake_ratio > 0.08 and chinese_ratio < 0.02)


def _try_mojibake_repair(text: str) -> Optional[Tuple[str, str, float]]:
    if not _should_try_mojibake_repair(text):
        return None

    candidates: List[Tuple[float, str, str]] = []
    source_encodings = ['latin1', 'cp1252']
    target_encodings = ['utf-8', 'gb18030', 'gbk']

    for src in source_encodings:
        try:
            raw = text.encode(src)
        except UnicodeEncodeError:
            continue
        for dst in target_encodings:
            try:
                decoded = raw.decode(dst)
            except UnicodeDecodeError:
                continue
            score = _score_text_quality(decoded)
            candidates.append((score, decoded, f"mojibake:{src}->{dst}"))

    if not candidates:
        return None

    best = min(candidates, key=lambda x: x[0])
    return best[1], best[2], best[0]


def _clean_ocr_garbage(text: str) -> str:
    """清理OCR乱码字符"""
    if not text:
        return ""
    
    # 1. 移除控制字符和替换字符
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    text = text.replace('\ufffd', '')
    
    # 2. 移除常见OCR乱码序列
    garbage_patterns = [
        r'[ӡưճƬ֧أ͵һڱ]+',
        r'[ͶĄ]+',
    ]
    for pattern in garbage_patterns:
        text = re.sub(pattern, '', text)
    
    # 3. 逐字符过滤非法Unicode
    cleaned_chars = []
    for char in text:
        if char.isspace() or _is_valid_char(char):
            cleaned_chars.append(char)
    
    text = ''.join(cleaned_chars)
    
    # 4. 清理多余空白
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


# ================== 结构化文本展平 & 图表锚点增强 ==================

def _is_aligned_param_row(line: str) -> bool:
    """判断一行是否像“表格/引脚参数/对齐字段”的参数行。

    目标：识别由“竖线/制表符/多空格”分隔的多列行，避免把整张表作为连续文本进入切块。
    """
    if not line:
        return False
    s = line.rstrip("\n")
    stripped = s.strip()
    if len(stripped) < 4:
        return False

    # 典型 markdown/纯文本表格
    if stripped.count("|") >= 2:
        return True
    # 制表符对齐
    if "\t" in s:
        return True
    # 多空格分列（至少 3 列）
    cols = [c for c in re.split(r" {2,}", stripped) if c]
    if len(cols) >= 3:
        return True

    return False


def _flatten_aligned_blocks_to_paragraphs(text: str) -> str:
    """将连续对齐的参数行展平：每行强制变成独立自然段（用空行分隔）。

    仅对“连续块”生效，避免误伤普通正文：默认要求连续 >= 3 行才视为结构块。
    可用环境变量关闭：STRUCTURED_FLATTEN=0
    """
    if not text or "\n" not in text:
        return text

    try:
        min_run = int(os.getenv("STRUCTURED_FLATTEN_MIN_RUN", "3"))
    except Exception:
        min_run = 3

    lines = text.splitlines()
    out: list[str] = []
    run: list[str] = []

    def _flush_run():
        nonlocal run
        if not run:
            return
        if len(run) >= min_run:
            # 每行作为独立段落：行 + 空行
            for r in run:
                out.append(r.rstrip())
                out.append("")
        else:
            out.extend([r.rstrip() for r in run])
        run = []

    for line in lines:
        if _is_aligned_param_row(line):
            run.append(line)
            continue

        # 遇到非表格行：先落盘结构块
        _flush_run()
        out.append(line.rstrip())

    _flush_run()
    return "\n".join(out)


_FIG_TABLE_REF_RE = re.compile(
    r"(?P<kind>图|表)\s*(?P<num>(?:\d+|[一二三四五六七八九十百]+)(?:\s*[-–—]\s*(?:\d+|[一二三四五六七八九十百]+))+)",
    re.IGNORECASE,
)


def _anchor_figure_table_refs(text: str) -> str:
    """图/表引用锚点增强：把引用行后面的若干行核心描述复制到引用附近。

    目的：避免“图X-X”在一个 chunk、描述在下一个 chunk，导致图号检索命中但内容缺失。
    可用环境变量控制：
      - ANCHOR_FIGURE_TABLE=0 关闭
      - ANCHOR_FIGURE_TABLE_LINES=4 (建议 3~5)
      - ANCHOR_FIGURE_TABLE_MAX_CHARS=240
    """
    if not text or "\n" not in text:
        return text

    try:
        n_lines = int(os.getenv("ANCHOR_FIGURE_TABLE_LINES", "4"))
    except Exception:
        n_lines = 4
    try:
        max_chars = int(os.getenv("ANCHOR_FIGURE_TABLE_MAX_CHARS", "240"))
    except Exception:
        max_chars = 240

    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)

        m = _FIG_TABLE_REF_RE.search(line)
        if not m:
            i += 1
            continue

        # 避免重复注入（如果下一行已经是锚点）
        if i + 1 < len(lines) and lines[i + 1].lstrip().startswith("锚点："):
            i += 1
            continue

        # 只对“较短引用行”或“包含‘如图/见图/如下图’”的行注入，避免正文中每次提及都复制
        short_line = len(line.strip()) <= 60
        has_ref_phrase = any(p in line for p in ("如图", "见图", "如下图", "如表", "见表"))
        if not (short_line or has_ref_phrase):
            i += 1
            continue

        # 抽取后续若干非空行作为锚点内容
        picked: list[str] = []
        j = i + 1
        while j < len(lines) and len(picked) < n_lines:
            cand = lines[j].strip()
            if not cand:
                j += 1
                continue
            # 遇到新章节/新图表标题时停止
            if _FIG_TABLE_REF_RE.search(cand):
                break
            if re.match(r"^(\d+(?:\.\d+){0,4})\s+\S", cand):
                break
            if re.match(r"^(第[一二三四五六七八九十百]+[章节条款])\b", cand):
                break
            picked.append(cand)
            j += 1

        if picked:
            anchor = " ".join(picked)
            if len(anchor) > max_chars:
                anchor = anchor[:max_chars] + "…"
            out.append(f"锚点：{anchor}")

        i += 1

    return "\n".join(out)


# ================== 结构化切块 ==================

# GB/T标准文档章节模式
SECTION_PATTERNS = [
    # 主章节：1 范围、2 术语
    (re.compile(r'^(\d+)\s+(\S.{0,50})$', re.MULTILINE), 1),
    # 二级：1.1 一般要求
    (re.compile(r'^(\d+\.\d+)\s+(\S.{0,50})$', re.MULTILINE), 2),
    # 三级：1.1.1 xxx
    (re.compile(r'^(\d+\.\d+\.\d+)\s+(\S.{0,50})$', re.MULTILINE), 3),
    # 四级及以上
    (re.compile(r'^(\d+(?:\.\d+){3,})\s+(\S.{0,40})$', re.MULTILINE), 4),
    # 附录：附录A、附录 B
    (re.compile(r'^(附录\s*[A-Z])\s*(.*)$', re.MULTILINE), 1),
]


class StructureAwareSplitter:
    """结构感知切块器 - 按GB/T标准文档章节结构切块"""
    
    def __init__(
        self, 
        max_chunk_size: int = 2000,
        min_chunk_size: int = 200,
        overlap: int = 100
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap
    
    def _find_section_boundaries(self, text: str) -> List[Tuple[int, int, str]]:
        """找到所有章节边界
        
        返回: [(position, level, title), ...]
        """
        boundaries = []
        
        for pattern, level in SECTION_PATTERNS:
            for match in pattern.finditer(text):
                num = match.group(1)
                title = match.group(2) if match.lastindex >= 2 else ""
                boundaries.append((match.start(), level, f"{num} {title}".strip()))
        
        # 按位置排序
        boundaries.sort(key=lambda x: x[0])
        return boundaries
    
    def _split_by_structure(self, text: str) -> List[Tuple[str, dict]]:
        """按文档结构切块
        
        返回: [(chunk_text, metadata), ...]
        """
        if not text.strip():
            return []
        
        boundaries = self._find_section_boundaries(text)
        
        # 如果没有检测到章节结构，回退到段落切块
        if not boundaries:
            return self._split_by_paragraphs(text)
        
        chunks = []
        
        # 添加文档开头到第一个章节的内容
        if boundaries[0][0] > 0:
            intro = text[:boundaries[0][0]].strip()
            if len(intro) >= self.min_chunk_size:
                chunks.append((intro, {"section": "前言", "level": 0}))
        
        # 处理每个章节
        for i, (pos, level, title) in enumerate(boundaries):
            # 确定本章节的结束位置
            if i + 1 < len(boundaries):
                end_pos = boundaries[i + 1][0]
            else:
                end_pos = len(text)
            
            section_text = text[pos:end_pos].strip()
            
            # 如果章节太长，进一步切分
            if len(section_text) > self.max_chunk_size:
                sub_chunks = self._split_long_section(section_text, title, level)
                chunks.extend(sub_chunks)
            elif len(section_text) >= self.min_chunk_size:
                chunks.append((section_text, {"section": title, "level": level}))
            # 太短的章节与下一章节合并（在后处理中处理）
        
        # 后处理：合并过短的chunk
        chunks = self._merge_short_chunks(chunks)
        
        return chunks
    
    def _split_long_section(self, text: str, section_title: str, level: int) -> List[Tuple[str, dict]]:
        """切分过长的章节"""
        chunks = []
        
        # 按段落切分
        paragraphs = re.split(r'\n{2,}', text)
        current_chunk = ""
        chunk_idx = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(current_chunk) + len(para) + 2 <= self.max_chunk_size:
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para
            else:
                if current_chunk:
                    chunks.append((current_chunk, {
                        "section": section_title,
                        "level": level,
                        "part": chunk_idx
                    }))
                    chunk_idx += 1
                    
                    # 添加重叠
                    if self.overlap > 0 and len(current_chunk) > self.overlap:
                        overlap_text = current_chunk[-self.overlap:]
                        current_chunk = overlap_text + "\n\n" + para
                    else:
                        current_chunk = para
                else:
                    # 单个段落就超过限制，强制切分
                    if len(para) > self.max_chunk_size:
                        for i in range(0, len(para), self.max_chunk_size - self.overlap):
                            chunk_part = para[i:i + self.max_chunk_size]
                            if chunk_part.strip():
                                chunks.append((chunk_part, {
                                    "section": section_title,
                                    "level": level,
                                    "part": chunk_idx
                                }))
                                chunk_idx += 1
                    else:
                        current_chunk = para
        
        if current_chunk:
            chunks.append((current_chunk, {
                "section": section_title,
                "level": level,
                "part": chunk_idx if chunk_idx > 0 else None
            }))
        
        return chunks
    
    def _split_by_paragraphs(self, text: str) -> List[Tuple[str, dict]]:
        """按段落切块（无结构时的回退方案）"""
        chunks = []
        paragraphs = re.split(r'\n{2,}', text)
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(current_chunk) + len(para) + 2 <= self.max_chunk_size:
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para
            else:
                if current_chunk and len(current_chunk) >= self.min_chunk_size:
                    chunks.append((current_chunk, {}))
                current_chunk = para
        
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append((current_chunk, {}))
        
        return chunks
    
    def _merge_short_chunks(self, chunks: List[Tuple[str, dict]]) -> List[Tuple[str, dict]]:
        """合并过短的chunk"""
        if len(chunks) <= 1:
            return chunks
        
        merged = []
        i = 0
        while i < len(chunks):
            text, meta = chunks[i]
            
            # 如果当前chunk太短，尝试与下一个合并
            while len(text) < self.min_chunk_size and i + 1 < len(chunks):
                i += 1
                next_text, next_meta = chunks[i]
                text = text + "\n\n" + next_text
                # 保留第一个的metadata
            
            merged.append((text, meta))
            i += 1
        
        return merged
    
    def split_text(self, text: str) -> List[Tuple[str, dict]]:
        """切分文本，返回chunks列表"""
        return self._split_by_structure(text)


# =============================================================================
# Parent-Child 两层结构分块器
# =============================================================================

class HierarchicalStructureSplitter:
    """两层结构分块器：Parent (结构) + Child (内容)
    
    设计理念：
    - Parent 层：按文档结构划分（标题section / 段落块）
    - Child 层：在 Parent 内部按需细分（语义或长度）
    
    Parent 构建规则：
    - 有标题：parent = 标题小节（section）
    - 无标题：parent = 段落块（空行分段/列表块/代码块）
    
    Child 构建规则：
    - Parent 太长（> child_max_size × 2）→ 判断是否需要语义切分
    - 段内相似度剧烈波动 → 启用语义边界切分
    - 否则直接长度切（带 overlap）
    """
    
    def __init__(
        self,
        parent_max_size: int = None,
        parent_min_size: int = None,
        child_max_size: int = None,
        child_min_size: int = None,
        child_overlap: int = None,
        semantic_split_threshold: int = None,
        semantic_variance_threshold: float = None,
        embed_model_name: str = None,
    ):
        self.parent_max_size = parent_max_size or PARENT_MAX_SIZE
        self.parent_min_size = parent_min_size or PARENT_MIN_SIZE
        self.child_max_size = child_max_size or CHILD_MAX_SIZE
        self.child_min_size = child_min_size or CHILD_MIN_SIZE
        self.child_overlap = child_overlap or CHILD_OVERLAP
        self.semantic_split_threshold = semantic_split_threshold or SEMANTIC_SPLIT_THRESHOLD
        self.semantic_variance_threshold = semantic_variance_threshold or SEMANTIC_VARIANCE_THRESHOLD
        self.embed_model_name = embed_model_name or ATTENTION_MODEL
        
        # 延迟加载的语义切分器
        self._semantic_splitter = None
        
        # 章节标题模式
        self._section_patterns = [
            # 主章节：1 范围、2 术语
            (re.compile(r'^(\d+)\s+(\S.{0,50})$', re.MULTILINE), 1),
            # 二级：1.1 一般要求
            (re.compile(r'^(\d+\.\d+)\s+(\S.{0,50})$', re.MULTILINE), 2),
            # 三级：1.1.1 xxx
            (re.compile(r'^(\d+\.\d+\.\d+)\s+(\S.{0,50})$', re.MULTILINE), 3),
            # 四级及以上
            (re.compile(r'^(\d+(?:\.\d+){3,})\s+(\S.{0,40})$', re.MULTILINE), 4),
            # 附录：附录A、附录 B
            (re.compile(r'^(附录\s*[A-Z])\s*(.*)$', re.MULTILINE), 1),
            # 中文章节：第一章、第二节
            (re.compile(r'^(第[一二三四五六七八九十百]+[章节条款])\s*(.*)$', re.MULTILINE), 1),
        ]
        
        # 段落块模式（列表、代码等）
        self._list_pattern = re.compile(r'^(?:[\d]+[.、)]|[a-z][.)]|[-•·※])\s+', re.MULTILINE)
        self._code_pattern = re.compile(r'^[\s]*(?:def |class |import |from |if |for |while |#)', re.MULTILINE)
    
    def _get_semantic_splitter(self):
        """延迟加载语义切分器"""
        if self._semantic_splitter is None:
            self._semantic_splitter = AttentionSemanticSplitter(
                model_name=self.embed_model_name,
                max_chunk_size=self.child_max_size,
                min_chunk_size=self.child_min_size,
                threshold=ATTENTION_THRESHOLD,
                overlap=self.child_overlap,
                window_size=ATTENTION_WINDOW_SIZE,
                breakpoint_percentile=ATTENTION_BREAKPOINT_PERCENTILE,
            )
        return self._semantic_splitter
    
    def _find_section_boundaries(self, text: str) -> List[Tuple[int, int, str, int]]:
        """找到所有章节边界
        
        返回: [(position, end_position, title, level), ...]
        """
        boundaries = []
        
        for pattern, level in self._section_patterns:
            for match in pattern.finditer(text):
                num = match.group(1)
                title = match.group(2) if match.lastindex >= 2 else ""
                boundaries.append((match.start(), match.end(), f"{num} {title}".strip(), level))
        
        # 按位置排序
        boundaries.sort(key=lambda x: x[0])
        return boundaries
    
    def _split_into_paragraphs(self, text: str) -> List[Tuple[str, int, int, str]]:
        """将无标题文本按段落块切分
        
        识别：空行分段、列表块、代码块
        返回: [(text, start, end, block_type), ...]
        """
        if not text.strip():
            return []
        
        blocks = []
        
        # 按双换行（或更多）分割
        parts = re.split(r'\n{2,}', text)
        pos = 0
        
        for part in parts:
            part_stripped = part.strip()
            if not part_stripped:
                pos = text.find('\n\n', pos) + 2 if '\n\n' in text[pos:] else len(text)
                continue
            
            # 确定块类型
            start = text.find(part_stripped, pos)
            end = start + len(part_stripped)
            
            if self._list_pattern.search(part_stripped[:100]):
                block_type = "list"
            elif self._code_pattern.search(part_stripped[:100]):
                block_type = "code"
            else:
                block_type = "paragraph"
            
            blocks.append((part_stripped, start, end, block_type))
            pos = end
        
        return blocks
    
    def _build_parents(self, text: str) -> List[dict]:
        """构建 Parent 层
        
        规则：
        - 有标题：parent = 标题小节（section）
        - 无标题：parent = 段落块组合
        
        返回: [{"text": str, "start": int, "end": int, "type": str, "title": str, "level": int}, ...]
        """
        if not text.strip():
            return []
        
        parents = []
        section_boundaries = self._find_section_boundaries(text)
        
        if section_boundaries:
            # 有章节结构：按章节划分 parent
            
            # 处理第一个章节之前的内容（前言）
            if section_boundaries[0][0] > 0:
                intro_text = text[:section_boundaries[0][0]].strip()
                if len(intro_text) >= self.parent_min_size:
                    parents.append({
                        "text": intro_text,
                        "start": 0,
                        "end": section_boundaries[0][0],
                        "type": "section",
                        "title": "前言",
                        "level": 0,
                    })
            
            # 处理每个章节
            for i, (pos, title_end, title, level) in enumerate(section_boundaries):
                # 确定章节结束位置
                if i + 1 < len(section_boundaries):
                    section_end = section_boundaries[i + 1][0]
                else:
                    section_end = len(text)
                
                section_text = text[pos:section_end].strip()
                
                if len(section_text) >= self.parent_min_size or len(section_text) >= PARENT_MIN_SECTION_KEEP:
                    parents.append({
                        "text": section_text,
                        "start": pos,
                        "end": section_end,
                        "type": "section",
                        "title": title,
                        "level": level,
                    })
                elif parents:
                    # 太短的章节合并到前一个（极短小节）
                    parents[-1]["text"] += "\n\n" + section_text
                    parents[-1]["end"] = section_end
        else:
            # 无章节结构：按段落块组合
            paragraph_blocks = self._split_into_paragraphs(text)
            
            if not paragraph_blocks:
                # 整个文本作为一个 parent
                parents.append({
                    "text": text.strip(),
                    "start": 0,
                    "end": len(text),
                    "type": "paragraph",
                    "title": "",
                    "level": 0,
                })
            else:
                # 合并相邻段落直到达到合理大小
                current_parent = None
                
                for block_text, start, end, block_type in paragraph_blocks:
                    if current_parent is None:
                        current_parent = {
                            "text": block_text,
                            "start": start,
                            "end": end,
                            "type": block_type,
                            "title": "",
                            "level": 0,
                        }
                    elif len(current_parent["text"]) + len(block_text) + 2 <= self.parent_max_size:
                        # 如果块类型变化且当前 parent 足够大，先收口，避免混合语义
                        if current_parent["type"] != block_type and len(current_parent["text"]) >= self.parent_min_size:
                            parents.append(current_parent)
                            current_parent = {
                                "text": block_text,
                                "start": start,
                                "end": end,
                                "type": block_type,
                                "title": "",
                                "level": 0,
                            }
                        else:
                            # 合并到当前 parent
                            current_parent["text"] += "\n\n" + block_text
                            current_parent["end"] = end
                            # 如果新块有不同类型，更新为混合类型
                            if current_parent["type"] != block_type:
                                current_parent["type"] = "mixed"
                    else:
                        # 当前 parent 已满，保存并开始新的
                        if len(current_parent["text"]) >= self.parent_min_size:
                            parents.append(current_parent)
                        current_parent = {
                            "text": block_text,
                            "start": start,
                            "end": end,
                            "type": block_type,
                            "title": "",
                            "level": 0,
                        }
                
                # 添加最后一个 parent
                if current_parent and len(current_parent["text"]) >= self.parent_min_size:
                    parents.append(current_parent)
                elif current_parent and parents:
                    # 太短则合并到前一个
                    parents[-1]["text"] += "\n\n" + current_parent["text"]
                    parents[-1]["end"] = current_parent["end"]
                elif current_parent:
                    # 没有前一个，直接添加
                    parents.append(current_parent)
        
        return parents
    
    def _should_use_semantic_split(self, parent_text: str) -> Tuple[bool, Optional[List[float]]]:
        """判断是否需要使用语义切分
        
        条件：
        1. Parent 长度超过阈值（child_max_size × 2）
        2. 内部相似度波动超过阈值
        
        返回: (should_split, similarity_scores)
        """
        # 条件1：长度检查
        if len(parent_text) <= self.semantic_split_threshold:
            return False, None
        
        # 条件2：计算相似度波动
        try:
            splitter = self._get_semantic_splitter()
            
            # 按句子切分
            sentences = splitter._split_into_sentences(parent_text)
            if len(sentences) < 3:
                return False, None
            
            # 获取嵌入
            sentence_texts = [s[0] for s in sentences]
            embeddings = splitter._get_embeddings(sentence_texts)
            
            if len(embeddings) < 3:
                return False, None
            
            # 计算相似度
            similarity_scores = splitter._compute_similarity_scores(embeddings)
            
            # 计算相似度波动（标准差）
            import numpy as np
            scores_array = np.array(similarity_scores[1:])  # 跳过第一个（总是1.0）
            variance = np.std(scores_array)
            
            # 如果波动大，使用语义切分
            if variance >= self.semantic_variance_threshold:
                return True, similarity_scores
            
            return False, similarity_scores
            
        except Exception as e:
            print(f"  [语义检测] 失败: {e}")
            return False, None
    
    @staticmethod
    def _hard_split(sentence: str, max_size: int) -> List[str]:
        """把超过 ``max_size`` 的"句子"强制切段。

        背景：``_split_by_length`` 按"句子"组装，遇到**没有任何句末标点的超长段**
        （如 PDF 抽取出的 XML 片段 / 连排图注，实测有单"句"1901 字）时，旧实现会把
        整句塞进一个 chunk，导致 chunk 远超配置的 child 上限（实测 max=1901 > 800）。
        这里在窗口内找最后一个**句内标点**（，、；：等）作为切点，尽量少破坏可读性；
        找不到或切点太靠前时才按长度硬切。
        """
        if len(sentence) <= max_size:
            return [sentence]
        parts: List[str] = []
        i = 0
        while i < len(sentence):
            if len(sentence) - i <= max_size:
                parts.append(sentence[i:])
                break
            window = sentence[i:i + max_size]
            cut = max(window.rfind(p) for p in "，、；：,;:")
            if cut < max_size // 2:  # 切点太靠前会留下过短块，直接硬切
                cut = max_size - 1
            parts.append(sentence[i:i + cut + 1])
            i += cut + 1
        return parts

    @staticmethod
    def _overlap_tail(chunk: str, overlap: int) -> str:
        """取块尾 ``overlap`` 字作为下一块的开头。

        旧实现取末 overlap 字后又 ``rfind('。')`` **向前截断**：末 80 字里几乎必含
        句号，实际残留往往只剩几个字 —— 配置写 80、实际近 0（实测相邻块命中率 1.1%）。
        现改为**向后补齐**：从"末 overlap 字"往前扩到上一个句首，保证
        实际重叠 ∈ [overlap, 2·overlap)，且新块从句子边界开始。
        """
        if overlap <= 0 or len(chunk) <= overlap:
            return chunk
        start = len(chunk) - overlap
        # 向前找句边界补齐（中英文句号都认，与句子切分正则一致）；
        # 补齐距离限制在一个 overlap 内，避免携带整个长句
        b = max(chunk.rfind("。", 0, start), chunk.rfind(".", 0, start))
        if b != -1 and (start - (b + 1)) <= overlap:
            start = b + 1
        return chunk[start:]

    def _split_by_length(self, text: str, max_size: int, min_size: int, overlap: int) -> List[str]:
        """按长度切分（句对齐 + 真实 overlap + 超长句兜底）。

        三个保证（对应 tests/test_chunk_split.py 的断言）：
        1. 相邻块 overlap 真实存在：下一块以前一块末 ``overlap`` 字开头
           （句边界对齐后 ∈ [overlap, 2·overlap)），不再被句号截没；
        2. 所有块 ≤ ``max_size``：单"句"超限时先在句内标点处硬切；
           组装后仍有超限块（如短尾并入前块）再兜底二切；
        3. 除兜底二切外，块按句子边界组装，不在句中间断开。
        """
        if len(text) <= max_size:
            return [text]

        # 超长"句"先在句内标点处切段，保证任何 piece ≤ max_size
        pieces: List[str] = []
        for sent in re.split(r'(?<=[。！？.!?])', text):
            sent = sent.strip()
            if not sent:
                continue
            if len(sent) > max_size:
                pieces.extend(self._hard_split(sent, max_size))
            else:
                pieces.append(sent)

        chunks: List[str] = []
        current = ""
        for piece in pieces:
            if current and len(current) + len(piece) <= max_size:
                current += piece
                continue
            if current and len(current) >= min_size:
                chunks.append(current)
                carry = self._overlap_tail(current, overlap) if overlap > 0 else ""
                # 只有 carry + piece 仍不超限时才带 overlap，否则放弃该次重叠
                if carry and len(carry) + len(piece) <= max_size:
                    current = carry + piece
                else:
                    current = piece
            else:
                # 首块，或当前块太短（< min_size）：继续攒（可能略超限，兜底二切会处理）
                current += piece

        if current:
            if chunks and len(current) < min_size:
                chunks[-1] += current  # 太短则并入前一个
            else:
                chunks.append(current)

        # 兜底：组装后仍超限的块（短尾并入 / carry 组合）二次硬切
        out: List[str] = []
        for c in chunks:
            c = c.strip()
            if not c:
                continue
            if len(c) > max_size:
                out.extend(self._hard_split(c, max_size))
            else:
                out.append(c)
        return out
    
    def _split_by_semantic(self, text: str, similarity_scores: List[float] = None) -> List[str]:
        """使用语义切分"""
        try:
            splitter = self._get_semantic_splitter()
            chunks_with_meta = splitter.split_text(text)
            return [chunk for chunk, _ in chunks_with_meta]
        except Exception as e:
            print(f"  [语义切分] 失败，回退到长度切分: {e}")
            return self._split_by_length(text, self.child_max_size, self.child_min_size, self.child_overlap)
    
    def _build_children(self, parent: dict) -> List[dict]:
        """在 Parent 内部构建 Child 层
        
        规则：
        - Parent 太长（> semantic_split_threshold）且相似度波动大 → 语义切分
        - 否则 → 长度切分（带 overlap）
        
        返回: [{"text": str, "parent_idx": int, "child_idx": int, "split_method": str}, ...]
        """
        parent_text = parent["text"]
        
        # 如果 parent 足够短，直接作为一个 child
        if len(parent_text) <= self.child_max_size:
            return [{
                "text": parent_text,
                "child_idx": 0,
                "split_method": "none",
            }]
        
        # 判断是否需要语义切分
        use_semantic, similarity_scores = self._should_use_semantic_split(parent_text)
        
        if use_semantic:
            # 使用语义切分
            child_texts = self._split_by_semantic(parent_text, similarity_scores)
            split_method = "semantic"
        else:
            # 使用长度切分
            child_texts = self._split_by_length(
                parent_text, 
                self.child_max_size, 
                self.child_min_size, 
                self.child_overlap
            )
            split_method = "length"
        
        children = []
        for idx, text in enumerate(child_texts):
            children.append({
                "text": text,
                "child_idx": idx,
                "split_method": split_method,
            })
        
        return children
    
    def split_document(self, text: str, doc_metadata: dict = None) -> List[Tuple[str, dict]]:
        """切分文档，返回 (chunk_text, metadata) 列表
        
        构建两层结构：
        1. 先构建 Parent（section / 段落块）
        2. 在每个 Parent 内部构建 Child（语义或长度切分）
        """
        if not text or not text.strip():
            return []
        
        doc_metadata = doc_metadata or {}
        results = []
        
        # 第一步：构建 Parent 层
        parents = self._build_parents(text)
        
        if not parents:
            # 回退：整个文档作为一个 chunk
            return [(text.strip(), {
                "parent_idx": 0,
                "child_idx": 0,
                "parent_type": "document",
                "split_method": "none",
                **doc_metadata,
            })]
        
        # 第二步：在每个 Parent 内部构建 Child
        chunk_idx = 0
        
        for parent_idx, parent in enumerate(parents):
            children = self._build_children(parent)
            
            for child in children:
                child_text = child["text"]
                parent_title = parent.get("title", "").strip()
                if parent.get("type") == "section" and parent_title:
                    # 将父标题前缀到每个 child，提升检索命中标题类问题
                    if parent_title not in child_text[:120]:
                        child_text = f"标题：{parent_title}\n{child_text}"
                metadata = {
                    # Parent 信息
                    "parent_idx": parent_idx,
                    "parent_type": parent["type"],
                    "parent_title": parent.get("title", ""),
                    "parent_level": parent.get("level", 0),
                    "parent_size": len(parent["text"]),
                    "_parent_text": parent["text"],
                    "parent_start": parent.get("start", 0),
                    "parent_end": parent.get("end", len(parent["text"])),
                    
                    # Child 信息
                    "child_idx": child["child_idx"],
                    "children_count": len(children),
                    "split_method": child["split_method"],
                    
                    # 全局索引
                    "chunk_idx": chunk_idx,
                    
                    # 继承文档元数据
                    **doc_metadata,
                }
                
                results.append((child_text, metadata))
                chunk_idx += 1
        
        return results


# =============================================================================
# Transformer 语义感知切块器
# =============================================================================

class AttentionSemanticSplitter:
    """基于 Transformer 嵌入的语义感知切块器
    
    工作原理：
    1. 将文本按句子分割
    2. 使用 Transformer 模型计算每个句子的嵌入向量
    3. 计算相邻句子的余弦相似度
    4. 在相似度低于阈值的位置（语义断点）切分
    5. 结合文档结构（章节标题）优化切分
    
    优化点：
    - 句子级别语义分析（比 token 级别更稳定）
    - 滑动窗口平滑（避免噪声）
    - 结构感知（保留章节完整性）
    - 动态阈值（适应不同文档）
    """
    
    def __init__(
        self,
        model_name: str = None,
        max_chunk_size: int = None,
        min_chunk_size: int = None,
        threshold: float = None,
        overlap: int = None,
        window_size: int = None,
        device: str = None,
        breakpoint_percentile: int = None,
    ):
        self.model_name = model_name or ATTENTION_MODEL
        self.max_chunk_size = max_chunk_size or ATTENTION_MAX_CHUNK_SIZE
        self.min_chunk_size = min_chunk_size or ATTENTION_MIN_CHUNK_SIZE
        self.threshold = threshold or ATTENTION_THRESHOLD
        self.overlap = overlap or ATTENTION_OVERLAP
        self.window_size = window_size or ATTENTION_WINDOW_SIZE
        self.breakpoint_percentile = breakpoint_percentile or ATTENTION_BREAKPOINT_PERCENTILE
        
        # Embedding 后端：EMBED_BACKEND=api 直接走 API；local 时按模型名判断 Ollama / HF
        self.use_api_embed = EMBED_BACKEND == "api"
        self.use_ollama = (
            not self.use_api_embed
            and ATTENTION_USE_OLLAMA
            and ':' in self.model_name
            and '/' not in self.model_name
        )
        
        self._model = None
        self._tokenizer = None
        self._device = device
        self._ollama_embed = None
        
        # 章节标题模式（与结构感知切块共用）
        self._section_patterns = [
            re.compile(r'^(\d+)\s+(\S.{0,50})$', re.MULTILINE),
            re.compile(r'^(\d+\.\d+)\s+(\S.{0,50})$', re.MULTILINE),
            re.compile(r'^(\d+\.\d+\.\d+)\s+(\S.{0,50})$', re.MULTILINE),
            re.compile(r'^(第[一二三四五六七八九十百]+[章节条款])\s*(.*)$', re.MULTILINE),
            re.compile(r'^(附录\s*[A-Z])\s*(.*)$', re.MULTILINE),
        ]
    
    def _load_model(self):
        """延迟加载模型"""
        if self._model is not None or self._ollama_embed is not None:
            return
        
        if self.use_api_embed:
            # 注意：走 API 时**不能**把 self.model_name 当模型 id —— 它是 ATTENTION_MODEL，
            # 默认值是 Ollama 的 "qwen3-embedding:8b-q8_0"。发给 OpenAI 兼容网关会被 400
            # 拒绝（"Model qwen3-embedding does not exist"），而且每个 batch 都会重试：
            # 既白烧网关配额，又静默退化成"按长度切分"（语义切块失效但日志只轻描淡写）。
            # API 后端一律使用配置好的 EMBED_MODEL。
            try:
                embed_model = build_embed_model()
                print(
                    f"[SemanticSplitter] 使用 API embedding 模型: "
                    f"{getattr(embed_model, 'model_name', '(未命名)')}"
                )
                self._ollama_embed = embed_model
                print("[SemanticSplitter] API embedding 初始化完成")
            except Exception as e:
                print(f"[SemanticSplitter] API embedding 初始化失败: {e}")
                self._ollama_embed = None
        elif self.use_ollama:
            # 使用 Ollama
            try:
                from llama_index.embeddings.ollama import OllamaEmbedding
                print(f"[SemanticSplitter] 使用 Ollama 模型: {self.model_name}")
                self._ollama_embed = OllamaEmbedding(
                    model_name=self.model_name,
                    base_url=OLLAMA_BASE,
                    ollama_additional_kwargs={"num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "2048"))},
                )
                print("[SemanticSplitter] Ollama 模型初始化完成")
            except Exception as e:
                print(f"[SemanticSplitter] Ollama 初始化失败: {e}")
                self._ollama_embed = None
        else:
            # 使用 Hugging Face Transformers
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
                
                if self._device is None or self._device == "auto":
                    self._device = "cuda" if torch.cuda.is_available() else "cpu"
                
                print(f"[SemanticSplitter] 加载 HF 模型: {self.model_name} (device={self._device})")
                
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, trust_remote_code=True
                )
                self._model = AutoModel.from_pretrained(
                    self.model_name, trust_remote_code=True
                ).to(self._device)
                self._model.eval()
                
                print("[SemanticSplitter] HF 模型加载完成")
            except Exception as e:
                print(f"[SemanticSplitter] HF 模型加载失败: {e}")
                self._model = None
    
    def _get_embeddings(self, texts: List[str]) -> 'np.ndarray':
        """批量获取文本嵌入"""
        import numpy as np
        
        if not texts:
            return np.array([])
        
        self._load_model()
        
        if (self.use_ollama or self.use_api_embed) and self._ollama_embed is not None:
            # 使用 Ollama / API 获取嵌入
            try:
                # 必须走批接口。早先是 for text in texts: get_text_embedding(text)，
                # 等于"每个句子一次 HTTP 请求"—— 一篇 3 万字的文档就是几百次请求，
                # 既打爆网关限流，也让语义切块成为整个入库流程的瓶颈。
                show_progress = len(texts) > 20
                if hasattr(self._ollama_embed, "get_text_embedding_batch"):
                    embeddings = self._ollama_embed.get_text_embedding_batch(
                        list(texts), show_progress=show_progress
                    )
                else:  # pragma: no cover - 兜底：实现不支持批接口时退回逐条
                    embeddings = []
                    for i, text in enumerate(texts):
                        if show_progress and i % 10 == 0:
                            print(f"  嵌入进度: {i}/{len(texts)}", end='\r')
                        embeddings.append(self._ollama_embed.get_text_embedding(text))
                    if show_progress:
                        print(f"  嵌入完成: {len(texts)}/{len(texts)}")
                arr = np.array(embeddings)
                # L2 归一化，避免余弦相似度数值不稳定
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms = np.where(norms < 1e-8, 1.0, norms)
                return arr / norms
            except Exception as e:
                print(f"[SemanticSplitter] 嵌入失败: {e}")
                return np.array([])
        elif self._model is not None:
            # 使用 Hugging Face Transformers
            import torch
            
            embeddings = []
            batch_size = 16
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                
                inputs = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self._device)
                
                with torch.no_grad():
                    outputs = self._model(**inputs)
                    # 使用 [CLS] token 或 mean pooling
                    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                        batch_emb = outputs.pooler_output
                    else:
                        # Mean pooling
                        attention_mask = inputs['attention_mask']
                        token_embeddings = outputs.last_hidden_state
                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        batch_emb = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    
                    embeddings.append(batch_emb.cpu().numpy())
            
            if not embeddings:
                return np.array([])
            arr = np.vstack(embeddings)
            # L2 归一化，避免余弦相似度数值不稳定
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.where(norms < 1e-8, 1.0, norms)
            return arr / norms
        else:
            return np.array([])
    
    def _split_into_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """将文本分割为段落/句子单元，返回 (文本, 起始位置, 结束位置)
        
        优化策略（v2 - 段落优先）：
        - 优先以段落为最小语义单元，保持段落完整性
        - 仅当段落超过 max_chunk_size 时才降级到句子级分割
        - 相邻短段落合并为一个单元，提升 chunk 的信息密度
        """
        units = []
        
        # 第一步：按双换行（或更多空行）分割成段落
        paragraphs = []
        
        for match in re.finditer(r'[^\n]+(?:\n[^\n]+)*', text):
            para = match.group().strip()
            if para and len(para) > 5:  # 忽略过短的段落
                paragraphs.append((para, match.start(), match.end()))
        
        # 第二步：合并相邻短段落，提升信息密度
        merged_paragraphs = []
        current_text = ""
        current_start = 0
        current_end = 0
        merge_threshold = self.max_chunk_size * 0.5  # 段落短于此值时尝试合并
        
        for para, start, end in paragraphs:
            if not current_text:
                current_text = para
                current_start = start
                current_end = end
            elif len(current_text) + len(para) + 2 <= merge_threshold:
                # 两个短段落合并
                current_text = current_text + "\n" + para
                current_end = end
            else:
                # 当前积累的已足够大，保存并开始新的
                merged_paragraphs.append((current_text, current_start, current_end))
                current_text = para
                current_start = start
                current_end = end
        if current_text:
            merged_paragraphs.append((current_text, current_start, current_end))
        
        # 第三步：处理每个合并后的段落
        for para, start, end in merged_paragraphs:
            # 如果段落 <= max_chunk_size，整体保留（段落优先策略的关键）
            if len(para) <= self.max_chunk_size:
                units.append((para, start, end))
            else:
                # 段落过长：按句子分割，但合并阈值提高到 max_chunk_size * 0.4
                sent_pattern = r'([^。！？\n]+[。！？]?)'
                sub_units = []
                current_text = ""
                current_start = start
                sentence_merge_threshold = max(400, self.max_chunk_size * 0.4)
                
                for sent_match in re.finditer(sent_pattern, para):
                    sent = sent_match.group().strip()
                    if not sent:
                        continue
                    
                    # 累计句子直到达到合理大小
                    if len(current_text) + len(sent) < sentence_merge_threshold:  # 合并短句（阈值提高）
                        if current_text:
                            current_text += sent
                        else:
                            current_text = sent
                            current_start = start + sent_match.start()
                    else:
                        if current_text:
                            sub_units.append((current_text, current_start, start + sent_match.start()))
                        current_text = sent
                        current_start = start + sent_match.start()
                
                if current_text:
                    sub_units.append((current_text, current_start, end))
                
                units.extend(sub_units if sub_units else [(para, start, end)])
        
        return units if units else [(text, 0, len(text))]
    
    def _detect_section_boundaries(self, text: str) -> set:
        """检测章节标题位置"""
        boundaries = set()
        for pattern in self._section_patterns:
            for match in pattern.finditer(text):
                boundaries.add(match.start())
        return boundaries
    
    def _compute_similarity_scores(self, embeddings: 'np.ndarray') -> List[float]:
        """计算相邻句子的相似度分数"""
        import numpy as np
        
        if len(embeddings) < 2:
            return [1.0] * len(embeddings)
        
        scores = [1.0]  # 第一个句子没有前一个
        
        for i in range(1, len(embeddings)):
            # 余弦相似度
            a = embeddings[i-1]
            b = embeddings[i]
            denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            similarity = np.dot(a, b) / denom
            if not np.isfinite(similarity):
                similarity = 1.0
            scores.append(float(similarity))
        
        return scores
    
    def _smooth_scores(self, scores: List[float]) -> List[float]:
        """滑动窗口平滑"""
        import numpy as np
        
        if len(scores) <= self.window_size:
            return scores
        
        smoothed = []
        half_window = self.window_size // 2
        
        for i in range(len(scores)):
            start = max(0, i - half_window)
            end = min(len(scores), i + half_window + 1)
            smoothed.append(np.mean(scores[start:end]))
        
        return smoothed
    
    def _find_breakpoints(
        self, 
        sentences: List[Tuple[str, int, int]], 
        similarity_scores: List[float],
        section_boundaries: set,
        text: str,
    ) -> List[int]:
        """找到语义断点位置 - v2 优化：极度保守的切分策略
        
        核心改进：
        - 降低百分位阈值（默认15%），只在最显著的语义断裂处切分
        - 要求更大的累积长度才允许切分，避免碎片化
        - 强语义断点需要更大的相邻对比差异
        - 低波动文档（技术标准等）几乎不做语义切分，靠长度 + 章节边界
        """
        import numpy as np
        
        if len(sentences) < 2:
            return []
        
        # 平滑相似度分数
        smoothed = self._smooth_scores(similarity_scores)
        
        # 动态阈值：使用更低的百分位数（只切最明显的语义断裂）
        scores_array = np.array(smoothed[1:])  # 跳过第一个
        if len(scores_array) == 0:
            return []
        
        percentile_threshold = np.percentile(scores_array, self.breakpoint_percentile)
        # 取百分位阈值和静态阈值的较小值（更保守）
        dynamic_threshold = min(percentile_threshold, self.threshold)
        
        # 如果相似度波动很小，说明文档语义连贯，几乎不做语义切分
        score_std = float(np.std(scores_array))
        low_variance = score_std < 0.08  # 放宽低波动判定
        
        breakpoints = []
        cumulative_len = 0
        
        for i in range(len(sentences)):
            sent_text, start, end = sentences[i]
            score = smoothed[i] if i < len(smoothed) else 1.0
            cumulative_len += len(sent_text)
            
            is_semantic_break = (score < dynamic_threshold) and not low_variance
            is_section_start = start in section_boundaries
            
            # 检测强语义断点：局部极小值且显著低于两侧邻居（提高差异阈值 0.1 -> 0.15）
            is_strong_break = False
            if i > 0 and i < len(smoothed) - 1:
                if (score < smoothed[i-1] - 0.15 and 
                    score < smoothed[i+1] - 0.15 and
                    score < dynamic_threshold):
                    is_strong_break = True
            
            # 决策逻辑（极度保守）：
            should_break = False
            
            if is_section_start and cumulative_len >= self.min_chunk_size * 0.5:
                # 章节边界：仅在已有一定长度时才切（避免章节标题被切得太碎）
                should_break = True
            elif cumulative_len >= self.max_chunk_size:
                # 超过最大长度：优先在语义断点处切
                if is_semantic_break or is_strong_break:
                    should_break = True
                elif cumulative_len >= self.max_chunk_size * 1.3:
                    # 超出 30% 才强制切（给段落更多容纳空间）
                    should_break = True
            elif is_strong_break and cumulative_len >= self.min_chunk_size * 2:
                # 强语义断点：需要至少 2 倍最小长度才允许切（避免小 chunk）
                should_break = True
            
            if should_break and cumulative_len >= self.min_chunk_size:
                breakpoints.append(start)
                cumulative_len = 0
        
        return sorted(set(breakpoints))
    
    def _create_chunks(
        self, 
        text: str, 
        breakpoints: List[int],
    ) -> List[Tuple[str, dict]]:
        """根据断点创建 chunks，确保所有内容都被包含"""
        chunks = []
        
        # 确保断点列表完整：起点 + 断点 + 终点
        all_points = sorted(set([0] + breakpoints + [len(text)]))
        
        chunk_idx = 0
        current_text = ""
        
        for i in range(len(all_points) - 1):
            start = all_points[i]
            end = all_points[i + 1]
            segment = text[start:end].strip()
            
            if not segment:
                continue
            
            # 尝试累积到当前 chunk
            if current_text:
                potential = current_text + "\n" + segment
            else:
                potential = segment
            
            if len(potential) <= self.max_chunk_size:
                current_text = potential
            else:
                # 当前 chunk 已满，先保存
                if current_text and len(current_text) >= self.min_chunk_size:
                    chunks.append((current_text.strip(), {
                        "split_method": "semantic",
                        "chunk_idx": chunk_idx,
                    }))
                    chunk_idx += 1
                    current_text = ""
                
                # 处理当前 segment
                if len(segment) > self.max_chunk_size:
                    # segment 本身过长，需要强制切分
                    sub_chunks = self._force_split_by_sentences(segment)
                    for sc in sub_chunks:
                        if len(sc) >= self.min_chunk_size:
                            chunks.append((sc.strip(), {
                                "split_method": "semantic",
                                "chunk_idx": chunk_idx,
                                "force_split": True,
                            }))
                            chunk_idx += 1
                else:
                    # segment 正常大小，作为新 chunk 的开始
                    current_text = segment
        
        # 处理最后的 chunk（不管大小都保存，避免丢内容）
        if current_text:
            if len(current_text) >= self.min_chunk_size:
                chunks.append((current_text.strip(), {
                    "split_method": "semantic",
                    "chunk_idx": chunk_idx,
                }))
            elif chunks:
                # 太短则合并到前一个
                prev_text, prev_meta = chunks[-1]
                chunks[-1] = (prev_text + "\n" + current_text.strip(), prev_meta)
            else:
                # 没有前一个，直接添加
                chunks.append((current_text.strip(), {
                    "split_method": "semantic",
                    "chunk_idx": chunk_idx,
                }))
        
        # 合并过小的 chunks
        chunks = self._merge_small_chunks(chunks)
        
        # 添加重叠
        chunks = self._add_overlap(chunks)
        
        return chunks
    
    def _force_split_by_sentences(self, text: str) -> List[str]:
        """按句子强制切分过长文本"""
        chunks = []
        sentences = re.split(r'([。！？\n])', text)
        current = ""
        
        for i in range(0, len(sentences), 2):
            sent = sentences[i]
            delim = sentences[i+1] if i+1 < len(sentences) else ""
            
            if len(current) + len(sent) + len(delim) <= self.max_chunk_size:
                current += sent + delim
            else:
                if current:
                    chunks.append(current.strip())
                current = sent + delim
        
        if current:
            chunks.append(current.strip())
        
        return chunks
    
    def _merge_small_chunks(self, chunks: List[Tuple[str, dict]]) -> List[Tuple[str, dict]]:
        """合并过小的 chunks - v2：更积极地合并以避免碎片化"""
        if len(chunks) <= 1:
            return chunks
        
        merged = []
        current_text = ""
        current_meta = None
        
        for text, meta in chunks:
            potential = (current_text + "\n" + text).strip() if current_text else text
            # 只要合并后不超过 max_chunk_size，且当前积累量 < min_chunk_size，就继续合并
            if len(current_text) < self.min_chunk_size and len(potential) <= self.max_chunk_size:
                current_text = potential
                if current_meta is None:
                    current_meta = meta
            else:
                if current_text and len(current_text) >= self.min_chunk_size:
                    merged.append((current_text.strip(), current_meta or {}))
                elif current_text and merged:
                    # 太短则并入前一个 chunk
                    prev_text, prev_meta = merged[-1]
                    if len(prev_text) + len(current_text) <= self.max_chunk_size:
                        merged[-1] = ((prev_text + "\n" + current_text).strip(), prev_meta)
                    else:
                        merged.append((current_text.strip(), current_meta or {}))
                elif current_text:
                    merged.append((current_text.strip(), current_meta or {}))
                current_text = text
                current_meta = meta
        
        # 处理末尾残留
        if current_text:
            if len(current_text) >= self.min_chunk_size:
                merged.append((current_text.strip(), current_meta or {}))
            elif merged:
                prev_text, prev_meta = merged[-1]
                if len(prev_text) + len(current_text) <= self.max_chunk_size:
                    merged[-1] = ((prev_text + "\n" + current_text).strip(), prev_meta)
                else:
                    merged.append((current_text.strip(), current_meta or {}))
            else:
                merged.append((current_text.strip(), current_meta or {}))
        
        return merged
    
    def _add_overlap(self, chunks: List[Tuple[str, dict]]) -> List[Tuple[str, dict]]:
        """添加 chunk 间重叠"""
        if len(chunks) <= 1 or self.overlap <= 0:
            return chunks
        
        result = []
        for i, (text, meta) in enumerate(chunks):
            if i > 0:
                # 取前一个 chunk 的尾部
                prev_text = chunks[i-1][0]
                overlap_text = prev_text[-self.overlap:] if len(prev_text) > self.overlap else prev_text
                # 找到句子边界
                match = re.search(r'[。！？\.\!\?]\s*', overlap_text)
                if match:
                    overlap_text = overlap_text[match.end():]
                if overlap_text:
                    text = overlap_text + " " + text
                    meta = {**meta, "has_overlap": True}
            result.append((text, meta))
        
        return result
    
    def split_text(self, text: str) -> List[Tuple[str, dict]]:
        """切分文本的主入口"""
        if not text or not text.strip():
            return []
        
        text = text.strip()
        
        # 短文本直接返回
        if len(text) <= self.max_chunk_size:
            return [(text, {"split_method": "semantic", "chunk_idx": 0})]
        
        try:
            # 1. 分割句子
            sentences = self._split_into_sentences(text)
            if len(sentences) < 2:
                return [(text, {"split_method": "semantic", "chunk_idx": 0})]
            
            # 2. 检测章节边界
            section_boundaries = self._detect_section_boundaries(text)
            
            # 3. 获取句子嵌入
            sentence_texts = [s[0] for s in sentences]
            embeddings = self._get_embeddings(sentence_texts)
            
            if len(embeddings) == 0:
                # 回退到简单切分
                return self._fallback_split(text)
            
            # 4. 计算相似度
            similarity_scores = self._compute_similarity_scores(embeddings)
            
            # 5. 找断点
            breakpoints = self._find_breakpoints(
                sentences, similarity_scores, section_boundaries, text
            )
            
            # 6. 创建 chunks
            chunks = self._create_chunks(text, breakpoints)
            
            return chunks if chunks else [(text, {"split_method": "semantic", "chunk_idx": 0})]
            
        except Exception as e:
            print(f"[SemanticSplitter] 语义切分失败: {e}, 回退到简单切分")
            return self._fallback_split(text)
    
    def _fallback_split(self, text: str) -> List[Tuple[str, dict]]:
        """回退：按段落和句子切分"""
        chunks = []
        paragraphs = re.split(r'\n{2,}', text)
        current = ""
        idx = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(current) + len(para) <= self.max_chunk_size:
                current = current + "\n\n" + para if current else para
            else:
                if current and len(current) >= self.min_chunk_size:
                    chunks.append((current, {"split_method": "fallback", "chunk_idx": idx}))
                    idx += 1
                
                if len(para) > self.max_chunk_size:
                    sub = self._force_split_by_sentences(para)
                    for s in sub:
                        if len(s) >= self.min_chunk_size:
                            chunks.append((s, {"split_method": "fallback", "chunk_idx": idx}))
                            idx += 1
                    current = ""
                else:
                    current = para
        
        if current and len(current) >= self.min_chunk_size:
            chunks.append((current, {"split_method": "fallback", "chunk_idx": idx}))
        
        return chunks


_VENDOR_ALIASES: list[tuple[str, str]] = [
    ("jlcpcb", "JLCPCB"),
    ("嘉立创", "JLCPCB"),
    ("pcbway", "PCBWay"),
    ("捷配", "JiePei"),
    ("nextpcb", "NextPCB"),
]

_EDA_ALIASES: list[tuple[str, str]] = [
    ("kicad", "kicad"),
    ("altium", "altium"),
    ("altiumdesigner", "altium"),
    ("ad", "altium"),
    ("allegro", "allegro"),
    ("orcad", "orcad"),
    ("pads", "pads"),
    ("eagle", "eagle"),
]


def _infer_source_type(rel_posix: str) -> str:
    """推断文档来源类型：优先目录层级，其次文件名。

    语料常以平铺方式放在 ``DATA_DIR`` 下（例如
    ``clear_docs/GBT2036-1994印制电路术语.txt``），此时路径里不存在
    「标准/国家标准」这类层级。原实现只认目录，会把全部文档归为
    ``unknown`` —— 后果不是少一个标签，而是 ``source_type`` 过滤条件
    全部落空、``by_source_type`` 分组统计失真。

    文件名里的标准号前缀（GB / IPC / SJ / ISO）与 EDA 工具名足以区分，
    因此补一层兜底；目录层级仍然优先，显式分类不被文件名覆盖。
    """

    path = (rel_posix or "").replace("\\", "/")

    # 0) 归一化目录名：语料目录常带排序前缀（如 ``01_国家标准``、``03_工程语料``），
    #    只对已知分类目录剥前缀，避免误伤 ``01_PCB基础概念与分类.txt`` 这类
    #    文件名（rel_norm 末尾带斜杠，宽松的 lookahead 会把文件名也当成目录）。
    path = re.sub(
        r"(?<=/)\d+_(?=(?:国家标准|国际标准|行业标准|标准|教材与专著|教材|"
        r"工程语料|文献|PCB设计|PCB资料|其他)/)",
        "",
        path,
    )

    # 1) 目录层级优先：有显式分类目录时以目录为准
    if "/标准/国际标准/" in path:
        return "standard_international"
    if "/标准/国家标准/" in path:
        return "standard_national"
    if "/国家标准/" in path:
        return "standard_national"
    if "/国际标准/" in path:
        return "standard_international"
    if "/行业标准/" in path:
        return "standard_industry"
    if "/标准/" in path:
        return "standard"
    if "/工程语料/" in path:
        return "engineering"
    if "/文献/" in path:
        return "literature"
    if "/PCB设计/" in path:
        return "pcb_design"
    if "/PCB资料/" in path:
        return "pcb_material"

    # 2) 文件名兜底
    #
    # 注意 rstrip("/")：调用方传入的 rel_norm 形如 "/dir/file.txt/"，末尾带斜杠，
    # 直接 rsplit("/")[-1] 会取到空串，兜底逻辑静默失效。
    #
    # 标准号在路径段首匹配而不是只看 basename：GB/T、SJ/T 这类写法本身含斜杠，
    # 一旦出现在路径里（如 "/GB/T 4677/x.txt"）或文件名被截断，只看 basename
    # 就会漏掉。用 (?:^|/) 锚定段首，避免 "GBK"/"IPCX" 之类的子串误命中。
    probe = path.rstrip("/")
    upper_path = probe.upper()
    basename = probe.rsplit("/", 1)[-1].upper()

    if re.search(r"(?:^|/)GB[/T\s]*\d", upper_path):
        return "standard_national"
    if re.search(r"(?:^|/)IPC[-\s]?[A-Z]*\d", upper_path):
        return "standard_international"
    if re.search(r"(?:^|/)SJ[/T\s]*\d", upper_path):
        return "standard_industry"
    if re.search(r"(?:^|/)ISO[-\s]?\d", upper_path):
        return "standard_international"

    if re.search(r"ALTIUM|CADENCE|ALLEGRO|PADS|MENTOR|PROTEL|POWERPCB", basename):
        return "pcb_design"
    if re.search(r"材料|层压|覆铜|基材|板材", basename):
        return "pcb_material"
    if re.search(r"教程|设计|LAYOUT|布线|仿真", basename, re.IGNORECASE):
        return "pcb_design"
    if re.search(r"测试|试验|检验|检测", basename):
        return "standard"

    return "unknown"


def _infer_effective_date(text: str) -> str | None:
    # YYYY-MM-DD / YYYY.MM.DD / YYYY_MM_DD
    m = re.search(r"(20\d{2})[-._](\d{1,2})[-._](\d{1,2})", text)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{mo:02d}-{d:02d}"
    # YYYYMMDD
    m = re.search(r"(20\d{2})(\d{2})(\d{2})", text)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{mo:02d}-{d:02d}"
    return None


def _infer_metadata(file_path: str) -> dict:
    p = Path(file_path)
    text = (str(p) + " " + p.name).lower()

    # 尽量用相对 DATA_DIR 的路径来做分类
    try:
        rel = p.resolve().relative_to(Path(DATA_DIR).resolve()).as_posix()
    except Exception:
        rel = p.as_posix()
    rel_norm = "/" + rel.replace("\\", "/").strip("/") + "/"

    md: dict[str, object] = {
        "source_type": _infer_source_type(rel_norm),
        "source_path": rel.strip("/"),
    }

    # vendor
    for needle, vendor in _VENDOR_ALIASES:
        if needle in text:
            md["vendor"] = vendor
            break

    # eda
    for needle, eda in _EDA_ALIASES:
        if needle in text:
            md["eda"] = eda
            break

    # layers (e.g. 4层 / 4-layer / 4 layers / 4L)
    # 注意：不能依赖 \b —— \w 含中文，"4层板"/"4层PCB" 中 "层" 之后不存在词边界
    m = re.search(r"(?<!\d)(\d{1,2})(?!\d)\s*(?:层|layer|layers)", text)
    if not m:
        m = re.search(r"(?<!\d)(\d{1,2})(?!\d)\s*l(?![a-z0-9])", text)
    if m:
        try:
            md["layer_count"] = int(m.group(1))
        except Exception:
            pass

    # copper oz (e.g. 1oz / 0.5oz)
    m = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*oz\b", text)
    if m:
        try:
            md["copper_oz"] = float(m.group(1))
        except Exception:
            pass

    eff = _infer_effective_date(str(p))
    if eff:
        md["effective_date"] = eff

    return md


# =============================================================================
# 文档结构化增强 (Phase 2.4) - 添加丰富的元数据以提升检索精度
# =============================================================================

# PCB 领域关键实体模式
PCB_ENTITY_PATTERNS = {
    # 标准编号
    "standard_gb": re.compile(r'GB[/T]*\s*[\d\.\-]+(?:\-\d{4})?', re.IGNORECASE),
    "standard_ipc": re.compile(r'IPC-[A-Z]*-?\d+[A-Z]*', re.IGNORECASE),
    "standard_sj": re.compile(r'SJ[/T]*\s*[\d\.\-]+(?:\-\d{4})?', re.IGNORECASE),
    "standard_iso": re.compile(r'ISO\s*\d+(?:\-\d+)?(?:\:\d{4})?', re.IGNORECASE),
    
    # 材料相关
    "material_fr4": re.compile(r'\bFR[-\s]?4\b', re.IGNORECASE),
    "material_cem": re.compile(r'\bCEM[-\s]?\d\b', re.IGNORECASE),
    "material_rogers": re.compile(r'\bRogers\s*\d+', re.IGNORECASE),
    "material_pi": re.compile(r'\b(聚酰亚胺|PI|Polyimide)\b', re.IGNORECASE),
    
    # 工艺相关
    "process_hasl": re.compile(r'\b(HASL|喷锡|热风整平)\b', re.IGNORECASE),
    "process_enig": re.compile(r'\b(ENIG|沉金|化金|镀金)\b', re.IGNORECASE),
    "process_osp": re.compile(r'\b(OSP|有机保焊|有机涂覆)\b', re.IGNORECASE),
    "process_immersion": re.compile(r'\b(沉银|沉锡|Immersion)\b', re.IGNORECASE),
    
    # 技术参数
    "param_impedance": re.compile(r'\b(阻抗|impedance|特性阻抗)\b', re.IGNORECASE),
    "param_thickness": re.compile(r'\b(厚度|thickness)\b', re.IGNORECASE),
    "param_copper_weight": re.compile(r'\b(\d+(?:\.\d+)?\s*oz)\b', re.IGNORECASE),
    "param_trace_width": re.compile(r'\b(线宽|trace\s*width)\b', re.IGNORECASE),
    
    # 测试相关
    "test_aoi": re.compile(r'\b(AOI|自动光学检测)\b', re.IGNORECASE),
    "test_flying_probe": re.compile(r'\b(飞针|flying\s*probe)\b', re.IGNORECASE),
    "test_ict": re.compile(r'\b(ICT|在线测试)\b', re.IGNORECASE),
}

# 文档类型关键词
DOC_TYPE_KEYWORDS = {
    "national_standard": ["GB", "GB/T", "国家标准", "中华人民共和国"],
    "industry_standard": ["SJ", "SJ/T", "行业标准", "电子行业"],
    "international_standard": ["IPC", "ISO", "IEC", "JEDEC", "国际标准"],
    "test_procedure": ["测试方法", "试验方法", "检验方法", "测试规范", "试验规范"],
    "design_guide": ["设计指南", "设计规范", "设计要求", "Layout", "布局"],
    "manufacturing_spec": ["制造规范", "工艺规范", "加工要求", "生产规范"],
    "material_spec": ["材料规格", "基材", "覆铜板", "半固化片", "铜箔"],
    "quality_standard": ["质量标准", "验收标准", "检验标准", "合格判定"],
}


def classify_document_type(text: str, file_path: str = "") -> str:
    """分类文档类型
    
    根据文档内容和文件路径判断文档类型。
    
    Args:
        text: 文档文本内容
        file_path: 文件路径
        
    Returns:
        文档类型标识符
    """
    text_lower = text.lower() if text else ""
    path_lower = file_path.lower() if file_path else ""
    combined = text_lower + " " + path_lower
    
    # 按优先级检查标准类型
    if re.search(r'GB[/T]*\s*\d+', text, re.IGNORECASE):
        return "national_standard"
    if re.search(r'IPC-[A-Z]*-?\d+', text, re.IGNORECASE):
        return "ipc_standard"
    if re.search(r'SJ[/T]*\s*\d+', text, re.IGNORECASE):
        return "industry_standard"
    if re.search(r'ISO\s*\d+', text, re.IGNORECASE):
        return "international_standard"
    
    # 检查内容关键词
    for doc_type, keywords in DOC_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in combined:
                return doc_type
    
    # 根据内容特征判断
    if any(kw in combined for kw in ["测试", "试验", "检验", "检测"]):
        return "test_procedure"
    if any(kw in combined for kw in ["设计", "layout", "布局", "走线"]):
        return "design_guide"
    if any(kw in combined for kw in ["制造", "工艺", "加工", "生产"]):
        return "manufacturing_spec"
    
    return "general"


def extract_pcb_entities(text: str) -> dict:
    """从文本中提取 PCB 领域关键实体
    
    Args:
        text: 文档文本内容
        
    Returns:
        提取到的实体字典，格式为 {entity_type: [entities...]}
    """
    if not text:
        return {}
    
    entities = {}
    
    for entity_type, pattern in PCB_ENTITY_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            # 去重并限制数量
            unique_matches = list(dict.fromkeys(matches))[:10]
            # 清理并规范化
            cleaned = [m.strip() for m in unique_matches if m.strip()]
            if cleaned:
                entities[entity_type] = cleaned
    
    return entities


def extract_section_info(text: str) -> dict:
    """从文本中提取章节信息
    
    Args:
        text: 文本内容（通常是 chunk）
        
    Returns:
        章节信息字典
    """
    section_info = {
        "section": "",
        "subsection": "",
        "has_table": False,
        "has_formula": False,
        "has_list": False,
    }
    
    if not text:
        return section_info
    
    # 检测章节标题（取第一个匹配）
    for pattern, level in SECTION_PATTERNS:
        match = pattern.search(text[:500])  # 只检查开头部分
        if match:
            title = f"{match.group(1)} {match.group(2) if match.lastindex >= 2 else ''}".strip()
            if level <= 2:
                section_info["section"] = title
            else:
                section_info["subsection"] = title
            break
    
    # 检测结构特征
    # 表格特征：多个制表符或竖线分隔
    if re.search(r'(?:\t.*){3,}|\|.*\|.*\|', text):
        section_info["has_table"] = True
    
    # 公式特征：数学符号密集
    if re.search(r'[=≤≥±×÷∑∏√∫]{2,}|[a-zA-Z]\s*[=]\s*\d', text):
        section_info["has_formula"] = True
    
    # 列表特征：多个编号或项目符号
    if re.search(r'(?:^|\n)\s*(?:[\d]+[.、)]|[a-z][.)]|[-•·])\s+\S', text, re.MULTILINE):
        section_info["has_list"] = True
    
    return section_info


def enhance_chunk_metadata(chunk_text: str, chunk_meta: dict, doc_text: str, file_path: str) -> dict:
    """增强 chunk 元数据
    
    为文档切块添加丰富的元数据，以提升检索精度。
    
    Args:
        chunk_text: 切块文本内容
        chunk_meta: 切块原有元数据（来自 StructureAwareSplitter）
        doc_text: 完整文档文本（用于提取文档级信息）
        file_path: 文件路径
        
    Returns:
        增强后的元数据字典
    """
    enhanced_meta = dict(chunk_meta)  # 复制原有元数据

    # 1. 来源类型 / 文档类型
    #
    # 这两个字段服务于不同维度，都要保证有值：
    #   source_type —— 粗粒度来源（standard_national / pcb_design / ...），
    #                  供元数据过滤与评测分组使用；
    #   doc_type    —— 细粒度文体（national_standard / design_guide / ...），
    #                  供语境前缀与检索意图判断使用。
    # 历史上只有 doc_type 被写入，且 _infer_source_type 依赖目录层级，
    # 平铺语料下会退化为 unknown，因此这里统一兜底。
    current_source_type = str(enhanced_meta.get("source_type") or "").strip().lower()
    if current_source_type in {"", "unknown", "none"}:
        enhanced_meta["source_type"] = _infer_source_type(file_path)

    if "doc_type" not in enhanced_meta:
        enhanced_meta["doc_type"] = classify_document_type(doc_text[:5000], file_path)

    # 2. 提取章节信息
    section_info = extract_section_info(chunk_text)
    if section_info["section"] and not enhanced_meta.get("section"):
        enhanced_meta["section"] = section_info["section"]
    if section_info["subsection"]:
        enhanced_meta.setdefault("subsection", section_info["subsection"])
    
    # 3. 添加结构特征标记
    enhanced_meta["has_table"] = section_info["has_table"]
    enhanced_meta["has_formula"] = section_info["has_formula"]
    enhanced_meta["has_list"] = section_info["has_list"]
    
    # 4. 提取关键实体（仅存储实体类型，不存储具体值以节省空间）
    entities = extract_pcb_entities(chunk_text)
    if entities:
        # 只保存实体类型列表，方便过滤
        enhanced_meta["entity_types"] = list(entities.keys())
        # 如果有标准编号，单独存储（常用于检索）
        standards = []
        for etype in ["standard_gb", "standard_ipc", "standard_sj", "standard_iso"]:
            if etype in entities:
                standards.extend(entities[etype])
        if standards:
            enhanced_meta["standards"] = standards[:5]  # 最多5个标准编号
    
    # 5. 添加内容摘要特征
    enhanced_meta["char_count"] = len(chunk_text)
    enhanced_meta["line_count"] = chunk_text.count('\n') + 1
    
    return enhanced_meta


def _apply_chunk_quality(meta: dict, text: str) -> bool:
    """评估 chunk 质量并标记；返回 False 表示该块应被丢弃。

    只对低价值块写字段（``chunk_quality`` / ``low_value`` / ``quality_flags``），
    正常块不加键 —— 这些键会一并进入 Milvus metadata 与 BM25 词法语料，
    给每个块都塞三个无意义字段会白白增加索引体积。

    ``drop`` 模式会减少块数，``prev_id`` / ``next_id`` 链随之出现空洞
    （邻居查不到就少扩展一条，不会报错）。默认 ``flag`` 只打标记，
    是否降权交给检索侧决定 —— 误删一条条款的代价远高于多留一条低价值候选。
    """
    if CHUNK_QUALITY_MODE not in {"flag", "drop"}:
        return True
    try:
        result = assess_chunk_quality(text)
    except Exception:
        return True

    if not result["is_low_value"]:
        return True

    if CHUNK_QUALITY_MODE == "drop":
        return False

    meta["chunk_quality"] = result["score"]
    meta["low_value"] = True
    meta["quality_flags"] = ",".join(result["flags"])
    return True



# =============================================================================
# Contextual Retrieval：为 chunk 补充语境前缀（Phase 4）
#
#   问题：切块会丢失上下文。"应不小于 1.0 N/mm。" 这类片段脱离文档后，
#         既看不出是哪篇文档、哪一章节，也看不出在说什么，很难被检索命中。
#   方案：入库时给每个 chunk 拼一句「属于哪篇文档 · 哪个章节 · 在说什么」的前缀，
#         该前缀同时参与 embedding 与 BM25 索引。
#
#   CONTEXTUAL_RETRIEVAL_MODE:
#     off  = 关闭
#     rule = 规则版：文档名 + 标题路径 + 标准号拼装（零 LLM 成本，推荐默认）
#     llm  = LLM 版：模型生成一句话语境（效果更好，入库时每个 chunk 需一次调用）
# =============================================================================

CONTEXTUAL_RETRIEVAL_MODE = os.getenv("CONTEXTUAL_RETRIEVAL_MODE", "rule").strip().lower()
CONTEXTUAL_RETRIEVAL_ENABLED = CONTEXTUAL_RETRIEVAL_MODE in {"rule", "llm"}
CONTEXTUAL_LLM_WORKERS = int(os.getenv("CONTEXTUAL_LLM_WORKERS", "4"))
CONTEXTUAL_LLM_MAX_CHARS = int(os.getenv("CONTEXTUAL_LLM_MAX_CHARS", "120"))
CONTEXTUAL_LLM_BATCH = int(os.getenv("CONTEXTUAL_LLM_BATCH", "1000"))


def _set_node_text(node, text: str) -> None:
    """兼容不同 LlamaIndex 版本，安全设置 TextNode 正文。"""
    if hasattr(node, "set_content"):
        node.set_content(text)
    else:
        node.text = text


def build_contextual_prefix(chunk_text: str, chunk_meta: dict, doc_meta: dict) -> str:
    """规则版语境前缀：文档名 · 章节路径 · 文档类型 · 标准号。"""

    doc_meta = doc_meta or {}
    chunk_meta = chunk_meta or {}
    chunk_text = chunk_text or ""

    parts: list[str] = []

    source = doc_meta.get("source_path") or doc_meta.get("file_name") or ""
    if source:
        parts.append(Path(str(source)).stem)

    section = (
        chunk_meta.get("parent_title")
        or chunk_meta.get("chunk_section")
        or chunk_meta.get("section")
        or ""
    )
    if section:
        section = str(section).strip()
        # 正文首行通常就是本节标题，若再放进前缀，等于同一句文本被嵌入两遍。
        # 实测语料中 52.8% 的块存在这种重复：前缀「· 8.2.5.3 语句组成」+
        # 正文首行「8.2.5.3 语句组成」。占用的虽只有十几个字符，但它同时进了
        # 向量与 BM25 统计量（同一个标题词被计两次频次），属于纯噪声。
        # 这里只丢掉与正文首行重合的**末级**标题，父级路径仍然保留。
        levels = [x.strip() for x in re.split(r"\s*(?:>|＞|·|/)\s*", section) if x.strip()]
        body_head = ""
        for line in (chunk_text or "").splitlines():
            if line.strip():
                body_head = re.sub(r"\s+", "", line)
                break
        if levels and body_head:
            last = re.sub(r"\s+", "", levels[-1])
            if last and (last in body_head or body_head in last):
                levels = levels[:-1]
        if levels:
            parts.append(" > ".join(levels))

    doc_type = chunk_meta.get("doc_type") or ""
    if doc_type and doc_type != "general":
        parts.append(str(doc_type))

    standards = chunk_meta.get("standards") or []
    if isinstance(standards, (list, tuple)) and standards:
        parts.append("/".join(str(s) for s in list(standards)[:3]))

    parts = [p for p in parts if p]
    if not parts:
        return ""

    prefix = "【上下文：" + " · ".join(parts) + "】"
    # 前缀过长、或已注入过，则跳过
    if len(prefix) > 200 or prefix in chunk_text:
        return ""
    return prefix


_CONTEXTUAL_LLM_PROMPT = """你在为 PCB 领域文档构建检索索引。

【文档全文】
{document}

【待处理片段】
{chunk}

请用一句话（{max_chars} 字以内）说明：这个片段出自文档的哪个部分、在讲什么、能回答什么问题。
只输出这句话，不要任何前缀、编号或解释。"""


def _generate_contextual_summary(chunk_text: str, doc_text: str, llm) -> str:
    """LLM 版：为单个 chunk 生成语境说明，失败返回空串。"""
    try:
        doc_excerpt = (doc_text or "")[:6000]
        prompt = _CONTEXTUAL_LLM_PROMPT.format(
            document=doc_excerpt,
            chunk=(chunk_text or "")[:1500],
            max_chars=CONTEXTUAL_LLM_MAX_CHARS,
        )
        from pcb_rag.usage import model_complete
        response = model_complete(llm, prompt)
        text = response.text if hasattr(response, "text") else str(response)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        text = text.splitlines()[0].strip() if text else ""
        return text[: CONTEXTUAL_LLM_MAX_CHARS * 2]
    except Exception:
        return ""


def _apply_contextual_with_llm(nodes: list, doc_text_by_id: dict, llm) -> list:
    """LLM 版：并发为每个 chunk 生成语境说明，失败时回退规则前缀。"""

    from concurrent.futures import ThreadPoolExecutor, as_completed

    total = len(nodes)
    targets = nodes
    if total > CONTEXTUAL_LLM_BATCH > 0:
        print(
            f"[Contextual] chunk 数 {total} 超过批量上限 {CONTEXTUAL_LLM_BATCH}，"
            f"仅处理前 {CONTEXTUAL_LLM_BATCH} 个（可调整 CONTEXTUAL_LLM_BATCH）"
        )
        targets = nodes[:CONTEXTUAL_LLM_BATCH]

    def _task(node):
        meta = getattr(node, "metadata", None) or {}
        doc_id = str(meta.get("doc_node_id") or "")
        summary = _generate_contextual_summary(
            node.text or "", doc_text_by_id.get(doc_id, ""), llm
        )
        if summary:
            return node, f"【上下文：{summary}】"
        # 回退到规则前缀
        return node, build_contextual_prefix(node.text or "", meta, {})

    applied = 0
    with ThreadPoolExecutor(max_workers=max(1, CONTEXTUAL_LLM_WORKERS)) as pool:
        futures = [pool.submit(_task, n) for n in targets]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                node, prefix = fut.result()
                if prefix:
                    _set_node_text(node, f"{prefix}\n{node.text}")
                    meta = getattr(node, "metadata", None)
                    if isinstance(meta, dict):
                        meta["contextual_prefix"] = True
                    applied += 1
            except Exception:
                continue
            if i % 50 == 0:
                print(f"[Contextual] 进度 {i}/{len(targets)}")

    print(f"[Contextual] LLM 版完成：{applied}/{len(targets)} 个 chunk 已注入语境前缀")
    return nodes


def apply_contextual_retrieval(nodes: list, documents: list) -> list:
    """为所有 chunk 注入语境前缀（就地修改节点正文与 metadata）。"""

    if not CONTEXTUAL_RETRIEVAL_ENABLED or not nodes:
        return nodes

    # 文档 id -> 原文，供 LLM 版使用
    doc_text_by_id: dict[str, str] = {}
    for d in documents or []:
        did = getattr(d, "id_", None)
        if did:
            doc_key = incremental.doc_node_id_of(d)
            doc_text_by_id[doc_key] = doc_text_by_id.get(doc_key, "") + "\n" + (getattr(d, "text", "") or "")

    if CONTEXTUAL_RETRIEVAL_MODE == "llm":
        try:
            llm = build_llm()
            print(f"[Contextual] LLM 版已启用，模型={getattr(llm, 'model', 'unknown')}")
            return _apply_contextual_with_llm(nodes, doc_text_by_id, llm)
        except Exception as e:
            print(f"[Contextual] LLM 初始化失败，回退规则版: {e}")

    applied = 0
    for node in nodes:
        meta = getattr(node, "metadata", None) or {}
        doc_meta = {
            "source_path": meta.get("source_path", ""),
            "file_name": meta.get("file_name", ""),
        }
        prefix = build_contextual_prefix(node.text or "", meta, doc_meta)
        if not prefix:
            continue
        _set_node_text(node, f"{prefix}\n{node.text}")
        meta["contextual_prefix"] = True
        applied += 1

    print(f"[Contextual] 规则版完成：{applied}/{len(nodes)} 个 chunk 已注入语境前缀")
    return nodes


def _normalize_text_for_embed(text: str) -> str:
    """文本清洗：修复编码、清理乱码、减少噪声

    功能：
    1. 修复编码问题（如果是bytes）
    2. 清理OCR乱码字符
    3. 规范化空白符
    4. 保留表格/公式结构
    """
    if not isinstance(text, str) or not text:
        return ""
    
    t = text
    
    # 1. 基本换行符规范化
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace("\x00", "")
    
    # 2. OCR乱码清理（如果启用）
    if CLEAN_OCR_GARBAGE:
        t = _clean_ocr_garbage(t)

    # 2.1 结构化文本展平（表格/参数行 -> 每行一个自然段）
    if os.getenv("STRUCTURED_FLATTEN", "1") not in {"0", "false", "False"}:
        try:
            t = _flatten_aligned_blocks_to_paragraphs(t)
        except Exception:
            pass

    # 2.2 图/表引用锚点增强（复制后续若干行描述到引用附近）
    if os.getenv("ANCHOR_FIGURE_TABLE", "1") not in {"0", "false", "False"}:
        try:
            t = _anchor_figure_table_refs(t)
        except Exception:
            pass
    
    # 3. 合并过多空白；保留单个换行
    t = re.sub(r"[ \t\f\v]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    
    # 4. 移除每行首尾空白
    lines = [line.strip() for line in t.split('\n')]
    t = '\n'.join(lines)
    
    return t.strip()

def _utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


def _truncate_utf8(text: str, max_bytes: int, suffix: str) -> str:
    if max_bytes <= 0:
        return ""
    b = text.encode("utf-8")
    if len(b) <= max_bytes:
        return text
    suffix_b = suffix.encode("utf-8")
    if max_bytes <= len(suffix_b):
        return suffix_b[:max_bytes].decode("utf-8", errors="ignore")
    keep = max_bytes - len(suffix_b)
    return b[:keep].decode("utf-8", errors="ignore") + suffix


def _chunk_utf8(text: str, max_bytes: int) -> list[str]:
    if max_bytes <= 0:
        return [""]
    b = text.encode("utf-8")
    if len(b) <= max_bytes:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(b):
        end = min(start + max_bytes, len(b))
        chunk = b[start:end].decode("utf-8", errors="ignore")
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks or [""]


def _split_text_to_chunks(text: str, max_bytes: int) -> list[str]:
    if not text:
        return [""]
    if _utf8_len(text) <= max_bytes:
        return [text]

    chunks: list[str] = []
    paragraphs = re.split(r"\n{2,}", text)
    current = ""

    def _flush_current():
        nonlocal current
        if current:
            chunks.append(current)
            current = ""

    for p in paragraphs:
        part = p.strip()
        if not part:
            continue
        candidate = part if not current else current + "\n\n" + part
        if _utf8_len(candidate) <= max_bytes:
            current = candidate
            continue

        _flush_current()
        if _utf8_len(part) <= max_bytes:
            chunks.append(part)
            continue

        # 句子级拆分
        sentences = re.split(r"(?<=[。！？.!?])\s+", part)
        buf = ""
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            cand = s if not buf else buf + " " + s
            if _utf8_len(cand) <= max_bytes:
                buf = cand
                continue
            if buf:
                chunks.append(buf)
                buf = ""
            if _utf8_len(s) <= max_bytes:
                chunks.append(s)
            else:
                chunks.extend(_chunk_utf8(s, max_bytes))
        if buf:
            chunks.append(buf)

    if current:
        chunks.append(current)
    return chunks


def _truncate_text_nodes(nodes, max_bytes=60000, target_bytes=12000):
    """Split oversized passages while preserving identity, ACL and adjacency.

    Milvus stores passage text separately from metadata. Metadata must never
    be erased to make text fit; source snapshots retain the full provenance.
    """
    import copy
    from pcb_rag.pcb_facts import extract_qualified_facts
    result, spans, internal_links = [], {}, set()
    for original in nodes:
        if not isinstance(getattr(original, "text", None), str):
            result.append(original)
            continue
        chunks = [original.text]
        if _utf8_len(original.text) > max_bytes:
            chunks = _split_text_to_chunks(original.text, min(max_bytes, target_bytes))
        group = []
        for index, text in enumerate(chunks):
            node = copy.deepcopy(original)
            node.text = text
            if index:
                node.id_ = hashlib.md5(f"{original.id_}-part-{index}".encode(), usedforsecurity=False).hexdigest()
            node.metadata["logical_chunk_id"] = node.id_
            if len(chunks) > 1:
                node.metadata["pcb_facts"] = extract_qualified_facts(text, node.metadata)
                node.metadata["split_part"] = index
            node.excluded_embed_metadata_keys = list(node.metadata)
            node.excluded_llm_metadata_keys = list(node.metadata)
            if _utf8_len(text) > max_bytes:
                raise ValueError("passage exceeds Milvus text field limit after splitting")
            group.append(node)
        for index, node in enumerate(group):
            if index:
                node.metadata["prev_id"] = group[index - 1].id_
                internal_links.add((node.id_, "prev_id"))
            if index + 1 < len(group):
                node.metadata["next_id"] = group[index + 1].id_
                internal_links.add((node.id_, "next_id"))
        if group:
            spans[original.id_] = (group[0].id_, group[-1].id_)
        result.extend(group)
    for node in result:
        for field, end in (("prev_id", 1), ("next_id", 0)):
            linked = node.metadata.get(field)
            if linked in spans and linked != node.id_ and (node.id_, field) not in internal_links:
                node.metadata[field] = spans[linked][end]
    return result


def _build_node_parser(embed_model):
    """根据配置构建node parser：parent_child > 结构感知 > 注意力语义 > 语义感知 > 层次化 > 传统固定切块"""
    
    if NODE_PARSER_MODE == "parent_child":
        print("[切块策略] Parent-Child 两层结构切块")
        print(f"  Parent: max={PARENT_MAX_SIZE}, min={PARENT_MIN_SIZE}")
        print(f"  Child: max={CHILD_MAX_SIZE}, min={CHILD_MIN_SIZE}, overlap={CHILD_OVERLAP}")
        print(f"  语义切分阈值: 长度>{SEMANTIC_SPLIT_THRESHOLD}, 波动>{SEMANTIC_VARIANCE_THRESHOLD}")
        return "parent_child"  # 标记使用 parent_child 模式
    
    if NODE_PARSER_MODE == "structure":
        print(f"[切块策略] 结构感知切块 (max={STRUCTURE_MAX_CHUNK_SIZE}, min={STRUCTURE_MIN_CHUNK_SIZE}, overlap={STRUCTURE_OVERLAP})")
        # 结构感知切块使用自定义处理，这里返回一个基本的SentenceSplitter作为后备
        # 实际切块在main()中通过StructureAwareSplitter处理
        return None  # 标记使用结构感知模式
    
    if NODE_PARSER_MODE == "attention":
        print("[切块策略] Transformer 注意力语义切块")
        print(f"  模型: {ATTENTION_MODEL}")
        print(f"  参数: max={ATTENTION_MAX_CHUNK_SIZE}, min={ATTENTION_MIN_CHUNK_SIZE}, threshold={ATTENTION_THRESHOLD}, overlap={ATTENTION_OVERLAP}")
        print(f"  断点百分位: {ATTENTION_BREAKPOINT_PERCENTILE}%, 窗口: {ATTENTION_WINDOW_SIZE}")
        # 注意力语义切块使用自定义处理
        return "attention"  # 标记使用注意力语义模式
    
    if NODE_PARSER_MODE == "semantic":
        try:
            # 添加chunk_size限制，避免产生超大chunk（Milvus限制65535字符）
            semantic_chunk_size = int(os.getenv("SEMANTIC_CHUNK_SIZE", "3000"))
            print(f"[切块策略] 语义感知切块 (buffer={SEMANTIC_BUFFER_SIZE}, threshold={SEMANTIC_BREAKPOINT_THRESHOLD}%, max_chunk_size={semantic_chunk_size})")
            return SemanticSplitterNodeParser(
                buffer_size=SEMANTIC_BUFFER_SIZE,
                breakpoint_percentile_threshold=SEMANTIC_BREAKPOINT_THRESHOLD,
                embed_model=embed_model,
                chunk_size=semantic_chunk_size,  # 限制最大chunk大小
            )
        except Exception as e:
            print(f"[警告] 语义切块初始化失败: {e}, 回退到固定切块")
    
    if NODE_PARSER_MODE == "hierarchical":
        try:
            safe_sizes = [max(256, s) for s in HIERARCHICAL_CHUNK_SIZES]
            if safe_sizes != HIERARCHICAL_CHUNK_SIZES:
                print(f"[切块策略] 层次化切块 size 过小，已自动调整为 {safe_sizes}")
            print(f"[切块策略] 层次化切块 (sizes={safe_sizes})")
            return HierarchicalNodeParser.from_defaults(
                chunk_sizes=safe_sizes,
            )
        except Exception as e:
            print(f"[警告] 层次化切块初始化失败: {e}, 回退到固定切块")
    
    # 默认：传统固定切块
    print(f"[切块策略] 固定大小切块 (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


def _structure_aware_parse(documents) -> list:
    """使用结构感知切块器处理文档"""
    
    splitter = StructureAwareSplitter(
        max_chunk_size=STRUCTURE_MAX_CHUNK_SIZE,
        min_chunk_size=STRUCTURE_MIN_CHUNK_SIZE,
        overlap=STRUCTURE_OVERLAP
    )
    
    nodes = []
    for doc in documents:
        text = doc.text if hasattr(doc, 'text') else str(doc)
        doc_id = getattr(doc, 'id_', '') or ''
        doc_metadata = dict(getattr(doc, 'metadata', {}) or {})
        file_path = doc_metadata.get('source_path', '') or doc_metadata.get('file_path', '')
        
        # 使用结构感知切块
        chunks = splitter.split_text(text)
        
        for idx, (chunk_text, chunk_meta) in enumerate(chunks):
            # 生成唯一ID
            node_id = hashlib.md5(f"{doc_id}-{idx}".encode('utf-8'), usedforsecurity=False).hexdigest()
            
            # 合并metadata（基础）
            merged_meta = {**doc_metadata}
            if chunk_meta:
                merged_meta['chunk_section'] = chunk_meta.get('section', '')
                merged_meta['chunk_level'] = chunk_meta.get('level', 0)
                if chunk_meta.get('part') is not None:
                    merged_meta['chunk_part'] = chunk_meta['part']
            
            # 增强元数据（Phase 2.4 文档结构化增强）
            enhanced_meta = enhance_chunk_metadata(
                chunk_text=chunk_text,
                chunk_meta=merged_meta,
                doc_text=text,
                file_path=file_path
            )
            # 低价值块（裸标签罗列等）：flag 模式打标记，drop 模式跳过
            if not _apply_chunk_quality(enhanced_meta, chunk_text):
                continue
            
            node = TextNode(
                text=chunk_text,
                id_=node_id,
                metadata=enhanced_meta
            )
            nodes.append(node)
    
    return nodes


def _parent_child_parse(documents) -> list:
    """使用 Parent-Child 两层结构切块器处理文档
    
    设计理念：
    1. Parent 层：按文档结构划分（标题section / 段落块）
    2. Child 层：在 Parent 内部按需细分（语义或长度）
    
    检索优势：
    - 稳定性：结构化的 Parent 保证上下文完整
    - 语义纯度：Child 层按需使用 Transformer 语义切分
    """
    
    print("[Parent-Child] 初始化两层结构切块器...")
    print(f"  Parent: max={PARENT_MAX_SIZE}, min={PARENT_MIN_SIZE}")
    print(f"  Child: max={CHILD_MAX_SIZE}, min={CHILD_MIN_SIZE}, overlap={CHILD_OVERLAP}")
    
    splitter = HierarchicalStructureSplitter(
        parent_max_size=PARENT_MAX_SIZE,
        parent_min_size=PARENT_MIN_SIZE,
        child_max_size=CHILD_MAX_SIZE,
        child_min_size=CHILD_MIN_SIZE,
        child_overlap=CHILD_OVERLAP,
        semantic_split_threshold=SEMANTIC_SPLIT_THRESHOLD,
        semantic_variance_threshold=SEMANTIC_VARIANCE_THRESHOLD,
    )
    
    nodes = []
    total_chunks = 0
    total_parents = 0
    semantic_split_count = 0
    length_split_count = 0
    
    for doc_idx, doc in enumerate(documents):
        text = doc.text if hasattr(doc, 'text') else str(doc)
        doc_id = getattr(doc, 'id_', '') or ''
        doc_metadata = dict(getattr(doc, 'metadata', {}) or {})
        file_path = doc_metadata.get('source_path', '') or doc_metadata.get('file_path', '')
        
        # 生成文档级别的 ID
        doc_node_id = incremental.doc_node_id_of(doc)
        page_identity = str(doc_metadata.get("page_label") or doc_metadata.get("page") or doc_metadata.get("page_number") or "")
        fragment_id = hashlib.sha256((doc_id + "|" + page_identity + "|" + text).encode()).hexdigest()[:24]
        
        print(f"[Parent-Child] [{doc_idx + 1}/{len(documents)}] 处理: {file_path or doc_id[:50]}... (长度: {len(text)} 字符)")
        
        # 使用 Parent-Child 切块
        try:
            chunks = splitter.split_document(text, doc_metadata)
            
            # 统计切分方法
            parent_indices = set()
            for _, meta in chunks:
                parent_indices.add(meta.get('parent_idx', 0))
                if meta.get('split_method') == 'semantic':
                    semantic_split_count += 1
                elif meta.get('split_method') == 'length':
                    length_split_count += 1
            
            doc_parent_count = len(parent_indices)
            total_parents += doc_parent_count
            
            print(f"  → {doc_parent_count} parents, {len(chunks)} children")
        except Exception as e:
            print(f"[Parent-Child] 文档 {file_path} 切块失败: {e}, 使用回退方案")
            chunks = [(text, {"parent_idx": 0, "child_idx": 0, "split_method": "fallback"})]
        
        # 第一遍：生成所有 node IDs 用于建立邻接关系
        chunk_node_ids = []
        for idx, (chunk_text, chunk_meta) in enumerate(chunks):
            node_id = hashlib.md5(f"{fragment_id}-pc-{idx}".encode('utf-8'), usedforsecurity=False).hexdigest()
            chunk_node_ids.append(node_id)
        
        # 预先聚合每个 parent 的子块区间：一次 O(n) 扫描。
        # 原实现是在下面这层循环里再遍历一遍 chunks 找同 parent 的首尾，
        # 整体是 O(n²)：1e4 chunk 约 30~60s，1e5 chunk 要跑一小时以上。
        parent_span: dict = {}
        for _idx, (_, _meta) in enumerate(chunks):
            _pidx = _meta.get('parent_idx', 0)
            _span = parent_span.get(_pidx)
            if _span is None:
                parent_span[_pidx] = [_idx, _idx]
            else:
                _span[1] = _idx

        # 第二遍：构建节点
        for idx, (chunk_text, chunk_meta) in enumerate(chunks):
            node_id = chunk_node_ids[idx]
            
            # 构建邻接关系
            prev_id = chunk_node_ids[idx - 1] if idx > 0 else None
            next_id = chunk_node_ids[idx + 1] if idx < len(chunks) - 1 else None
            
            # 同一 parent 内的位置信息（O(1) 查表）
            current_parent_idx = chunk_meta.get('parent_idx', 0)
            parent_first_child, parent_last_child = parent_span.get(current_parent_idx, [idx, idx])

            parent_id = hashlib.md5(f"{fragment_id}-parent-{current_parent_idx}".encode(), usedforsecurity=False).hexdigest()
            parent_text = chunk_meta.get("_parent_text") or text
            from pcb_rag.pcb_facts import enrich_table_child
            chunk_text, table_metadata = enrich_table_child(chunk_text, parent_text)
            chunk_meta = {key: value for key, value in chunk_meta.items() if key != "_parent_text"}
            if idx == parent_first_child:
                parent_meta = dict(doc_metadata, **chunk_meta)
                parent_meta.update(doc_node_id=doc_node_id, node_level="parent", parent_id="", logical_chunk_id=parent_id)
                nodes.append(TextNode(text=parent_text, id_=parent_id, metadata=parent_meta))

            # 构建完整元数据
            #
            # 必须以 doc 级元数据打底：source_type / source_path / vendor / eda
            # 这些字段在文档加载阶段（_infer_metadata）才产生，切块器产出的
            # chunk_meta 只带 parent_* / child_* 等位置信息。此前这里写成
            # `{**chunk_meta, ...}` 且未并入 doc_metadata，导致 parent_child
            # 模式下所有 chunk 的 source_type 落到 schema 默认值 unknown ——
            # 按类型过滤全部失效、引用无法定位章节、评测分组统计失真。
            merged_meta = {
                **doc_metadata,
                **chunk_meta,
                **table_metadata,
                'node_level': 'child',
                # 邻接关系
                'prev_id': prev_id or '',
                'next_id': next_id or '',
                'doc_node_id': doc_node_id,
                # 父块锚点（供 expand_chunks_with_context 做父级扩展）
                'parent_id': parent_id,
                
                # Parent 内的位置信息
                'is_first_in_parent': (idx == parent_first_child),
                'is_last_in_parent': (idx == parent_last_child),
                
                # 全局位置
                'total_chunks_in_doc': len(chunks),
            }

            # 章节字段归一：下游（build_context_with_sources / 引用溯源）
            # 依次取 parent_title → chunk_section → section，这里统一补一份，
            # 避免不同切块器的字段命名差异导致章节信息丢失。
            if not merged_meta.get('section'):
                merged_meta['section'] = (
                    merged_meta.get('parent_title')
                    or merged_meta.get('chunk_section')
                    or ''
                )
            
            # 增强元数据
            enhanced_meta = enhance_chunk_metadata(
                chunk_text=chunk_text,
                chunk_meta=merged_meta,
                doc_text=text,
                file_path=file_path
            )
            # 低价值块（裸标签罗列等）：flag 模式打标记，drop 模式跳过
            if not _apply_chunk_quality(enhanced_meta, chunk_text):
                continue
            
            node = TextNode(
                text=chunk_text,
                id_=node_id,
                metadata=enhanced_meta
            )
            nodes.append(node)
            total_chunks += 1
    
    print(f"[Parent-Child] 完成：{len(documents)} 文档 -> {total_parents} parents -> {total_chunks} children")
    print(f"  切分方法统计: 语义切分={semantic_split_count}, 长度切分={length_split_count}, 无需切分={total_chunks - semantic_split_count - length_split_count}")
    
    return nodes


def _attention_semantic_parse(documents) -> list:
    """使用 Transformer 语义切块器处理文档 (优化版)
    
    基于预训练 Transformer 的句子嵌入计算语义相似度，
    在相似度较低的位置（语义断点）切分文档。
    """
    
    print("[SemanticSplitter] 初始化 Transformer 语义切块器...")
    print(f"  模型: {ATTENTION_MODEL}")
    print(f"  参数: max={ATTENTION_MAX_CHUNK_SIZE}, min={ATTENTION_MIN_CHUNK_SIZE}, threshold={ATTENTION_THRESHOLD}, overlap={ATTENTION_OVERLAP}")
    print(f"  断点百分位: {ATTENTION_BREAKPOINT_PERCENTILE}%, 窗口: {ATTENTION_WINDOW_SIZE}")
    
    splitter = AttentionSemanticSplitter(
        model_name=ATTENTION_MODEL,
        max_chunk_size=ATTENTION_MAX_CHUNK_SIZE,
        min_chunk_size=ATTENTION_MIN_CHUNK_SIZE,
        threshold=ATTENTION_THRESHOLD,
        overlap=ATTENTION_OVERLAP,
        window_size=ATTENTION_WINDOW_SIZE,
        breakpoint_percentile=ATTENTION_BREAKPOINT_PERCENTILE,
    )
    
    nodes = []
    total_chunks = 0
    
    # 用于存储层次结构关系
    hierarchy_map = {}  # node_id -> {parent_id, prev_id, next_id, children_ids, doc_id, chunk_index}
    
    for doc_idx, doc in enumerate(documents):
        text = doc.text if hasattr(doc, 'text') else str(doc)
        doc_id = getattr(doc, 'id_', '') or ''
        doc_metadata = dict(getattr(doc, 'metadata', {}) or {})
        file_path = doc_metadata.get('source_path', '') or doc_metadata.get('file_path', '')
        
        # 生成文档级别的 ID（作为所有 chunk 的父节点）
        doc_node_id = hashlib.md5(f"doc-{doc_id}".encode('utf-8'), usedforsecurity=False).hexdigest()
        
        print(f"[SemanticSplitter] [{doc_idx + 1}/{len(documents)}] 处理: {file_path or doc_id[:50]}... (长度: {len(text)} 字符)")
        
        # 使用语义切块
        try:
            chunks = splitter.split_text(text)
            print(f"  → 生成 {len(chunks)} 个 chunks")
        except Exception as e:
            print(f"[SemanticSplitter] 文档 {file_path} 切块失败: {e}, 使用回退方案")
            chunks = [(text, {"split_method": "fallback", "chunk_idx": 0})]
        
        # 检测章节标题模式
        section_pattern = re.compile(r'^(\d+(?:\.\d+)*)\s+(.+?)$', re.MULTILINE)
        
        # 第一遍：生成所有 node IDs 用于建立关系
        chunk_node_ids = []
        chunk_sections = []  # 记录每个 chunk 的章节信息
        
        for idx, (chunk_text, chunk_meta) in enumerate(chunks):
            node_id = hashlib.md5(f"{doc_id}-sem-{idx}".encode('utf-8'), usedforsecurity=False).hexdigest()
            chunk_node_ids.append(node_id)
            
            # 检测 chunk 中的章节标题（用于父子关系）
            section_match = section_pattern.search(chunk_text[:200])  # 只检查开头
            if section_match:
                section_num = section_match.group(1)
                section_title = section_match.group(2)
                chunk_sections.append({
                    'section_num': section_num,
                    'section_title': section_title,
                    'level': len(section_num.split('.'))  # 1, 2, 3... 层级
                })
            else:
                chunk_sections.append(None)
        
        # 第二遍：构建层次关系
        section_stack = []  # 用于追踪父章节 [(section_num, node_id), ...]
        
        for idx, (chunk_text, chunk_meta) in enumerate(chunks):
            node_id = chunk_node_ids[idx]
            section_info = chunk_sections[idx]
            
            # 构建邻接关系
            prev_id = chunk_node_ids[idx - 1] if idx > 0 else None
            next_id = chunk_node_ids[idx + 1] if idx < len(chunks) - 1 else None
            
            # 构建父子关系
            parent_id = doc_node_id  # 默认父节点是文档
            
            if CHUNK_HIERARCHY_ENABLED and section_info:
                current_level = section_info['level']
                
                # 弹出同级或更深层级的章节
                while section_stack and section_stack[-1][1] >= current_level:
                    section_stack.pop()
                
                # 如果有上级章节，设置为父节点
                if section_stack:
                    parent_id = section_stack[-1][0]
                
                # 将当前章节压入栈
                section_stack.append((node_id, current_level))
            elif CHUNK_HIERARCHY_ENABLED and section_stack:
                # 没有章节标题的 chunk，使用最近的章节作为父节点
                parent_id = section_stack[-1][0]
            
            # 存储层次结构
            hierarchy_map[node_id] = {
                'parent_id': parent_id,
                'prev_id': prev_id,
                'next_id': next_id,
                'doc_id': doc_node_id,
                'chunk_index': idx,
                'total_chunks': len(chunks),
            }
            
            # 合并metadata
            merged_meta = {**doc_metadata}
            if chunk_meta:
                merged_meta['split_method'] = chunk_meta.get('split_method', 'semantic')
                merged_meta['chunk_idx'] = chunk_meta.get('chunk_idx', idx)
                if chunk_meta.get('has_overlap'):
                    merged_meta['has_overlap'] = True
                if chunk_meta.get('force_split'):
                    merged_meta['force_split'] = True
            
            # 添加层次结构元数据
            if CHUNK_HIERARCHY_ENABLED:
                merged_meta['parent_id'] = parent_id
                merged_meta['prev_id'] = prev_id or ''
                merged_meta['next_id'] = next_id or ''
                merged_meta['doc_node_id'] = doc_node_id
                merged_meta['chunk_index'] = idx
                merged_meta['total_chunks'] = len(chunks)
                if section_info:
                    merged_meta['section_num'] = section_info['section_num']
                    merged_meta['section_title'] = section_info['section_title']
                    merged_meta['section_level'] = section_info['level']
            
            # 增强元数据
            enhanced_meta = enhance_chunk_metadata(
                chunk_text=chunk_text,
                chunk_meta=merged_meta,
                doc_text=text,
                file_path=file_path
            )
            # 低价值块（裸标签罗列等）：flag 模式打标记，drop 模式跳过
            if not _apply_chunk_quality(enhanced_meta, chunk_text):
                continue
            
            node = TextNode(
                text=chunk_text,
                id_=node_id,
                metadata=enhanced_meta
            )
            nodes.append(node)
            total_chunks += 1
    
    if CHUNK_HIERARCHY_ENABLED:
        print(f"[AttentionSplitter] 完成：{len(documents)} 文档 -> {total_chunks} chunks (层次结构已启用)")
    else:
        print(f"[AttentionSplitter] 完成：{len(documents)} 文档 -> {total_chunks} chunks")
    return nodes


# =============================================================================
# 增量入库：文档变更检测（Phase 5）
#
#   - manifest 记录每个文档的指纹（路径 + 大小 + mtime）
#   - 入库时只处理「新增 / 变更」的文档，未变更的跳过（省去重复切块与向量化）
#   - 数据目录中已移除的文档，其 chunk 会从向量库删除
#   - 设 INGEST_INCREMENTAL=0 可退回「每次全量处理」的旧行为
# =============================================================================

INGEST_INCREMENTAL = os.getenv("INGEST_INCREMENTAL", "1") not in {"0", "false", "False"}
INGEST_MANIFEST_PATH = os.getenv("INGEST_MANIFEST_PATH", "./data/ingest_manifest.json")

# 指纹 / 比对 / 清单读写的实现已下沉到 pcb_rag/incremental.py（纯标准库，可单测）。
# 这里保留原名作为薄封装：ingest 内部与 tests 都在按旧名引用，
# 改名会造成无谓破坏；同时保证"仓库里只有一份增量逻辑"。
MANIFEST_VERSION = incremental.MANIFEST_VERSION


def _content_hash(path: Path) -> str:
    return incremental.content_hash(path)


def _file_fingerprint(path: Path) -> str:
    return incremental.file_fingerprint(path)


def _doc_fingerprint_of(doc) -> str:
    return incremental.doc_fingerprint_of(doc)


def _doc_source_of(doc) -> str:
    return incremental.doc_source_of(doc)


def _doc_node_id_of(doc) -> str:
    """文档级节点 id，与 _parent_child_parse 中保持一致。"""
    return incremental.doc_node_id_of(doc)


def _load_manifest() -> dict:
    path = Path(INGEST_MANIFEST_PATH)
    manifest = incremental.load_manifest(INGEST_MANIFEST_PATH)
    if path.exists() and incremental.needs_full_reprocess(manifest):
        print(
            f"ℹ️  入库清单为旧版本(v{manifest.get('version')})：指纹算法已从 mtime 改为内容哈希，"
            "本次将全量重嵌入一次，之后恢复正常增量。"
        )
    return manifest


def _save_manifest(docs_map: dict) -> None:
    try:
        incremental.save_manifest(INGEST_MANIFEST_PATH, docs_map)
    except Exception as e:
        print(f"⚠️  写入入库清单失败: {e}")


def _diff_documents(documents: list, manifest: dict) -> tuple:
    """比对文档与清单（实现见 :func:`pcb_rag.incremental.diff_documents`）。

    Returns:
        (待处理文档列表, 需要先清理旧 chunk 的 doc_node_id 列表, 新的清单条目)
    """
    return incremental.diff_documents(documents, manifest)


def _delete_chunks_by_doc_node_id(vector_store, doc_node_id: str) -> int:
    """按 doc_node_id 删除该文档的所有 chunk（用于变更覆盖与文档删除）。

    ``MilvusVectorStore.client`` 是**同步**的 ``MilvusClient``（异步版是 ``.aclient``）。
    早期实现对同步 ``client.delete()`` 的返回值做了 ``await``，触发
    ``TypeError: object dict can't be used in 'await' expression`` 并被兜底 ``except`` 吞掉，
    结果是“删除”从未真正执行过。这里直接走同步接口。
    """
    if not doc_node_id:
        return 0
    try:
        client = getattr(vector_store, "client", None)
        if client is None:
            return 0
        # 防御：doc_node_id 是 md5 hex，不应含引号；仍做一次转义避免过滤表达式注入
        safe_id = str(doc_node_id).replace('"', '\\"')
        result = client.delete(
            collection_name=COLLECTION,
            filter=f'doc_node_id == "{safe_id}"',
        )
        if isinstance(result, dict):
            return int(result.get("delete_count", 0) or 0)
        # 旧版 pymilvus 可能返回 list[pk]
        if isinstance(result, (list, tuple)):
            return len(result)
        return 0
    except Exception as e:
        raise RuntimeError(f"failed deleting chunks for {doc_node_id}") from e


def _ensure_doc_node_id(nodes: list, documents: list) -> list:
    """保证每个 chunk 的 metadata 都带 ``doc_node_id``。

    Parent-Child 切块会自己写入该字段；结构感知 / 默认 SentenceSplitter 等模式不会。
    增量入库依赖该字段做「按文档删除」，缺失时变更 / 删除文档的旧 chunk 永远清不掉。
    """
    by_doc_id = {}
    for d in documents:
        did = getattr(d, "id_", None)
        if did:
            by_doc_id[str(did)] = _doc_node_id_of(d)

    patched = 0
    for n in nodes:
        meta = getattr(n, "metadata", None)
        if meta is None:
            continue
        if meta.get("doc_node_id"):
            continue
        ref = getattr(n, "ref_doc_id", None)
        if not ref:
            src = getattr(n, "source_node", None)
            ref = getattr(src, "node_id", None) if src is not None else None
        target = by_doc_id.get(str(ref)) if ref else None
        if target:
            meta["doc_node_id"] = target
            patched += 1
    if patched:
        print(f"🔖 已为 {patched} 个 chunk 补齐 doc_node_id（供增量删除使用）")
    return nodes


def _update_knowledge_graph(nodes: list, stale_doc_ids=None) -> None:
    """维护知识图谱：合并本次抽取的三元组，并清理已变更 / 已删除文档的旧关系。

    图谱是**旁路产物**：无论抽取还是落盘失败，都只告警，绝不中断向量入库主流程。
    """

    if not GRAPH_RAG_ENABLED:
        return

    try:
        graph = KnowledgeGraph.load() or KnowledgeGraph()

        if stale_doc_ids:
            removed = graph.remove_docs(stale_doc_ids)
            if removed:
                print(f"[GraphRAG] 已清理 {removed} 条过期关系")

        if not nodes:
            graph.save()
            return

        llm = None
        if GRAPH_EXTRACT_BACKEND in {"llm", "hybrid"}:
            llm = getattr(Settings, "llm", None)

        graph = build_graph_from_nodes(nodes, llm, graph=graph, progress=True)

        if GRAPH_COMMUNITY_ENABLED:
            communities = summarize_communities(graph, llm)
            print(f"[GraphRAG] 社区摘要: {len(communities)} 个主题簇")

        path = graph.save()
        stats = graph.stats()
        print(f"[GraphRAG] 知识图谱已更新: {path} (实体={stats['entities']}, 关系={stats['relations']})")
    except Exception as exc:
        raise RuntimeError("knowledge graph update failed") from exc


def _load_ingest_documents(data_dir=None):
    directory = Path(data_dir or DATA_DIR)
    if not directory.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {directory}")
    if not any(path.is_file() for path in directory.rglob("*")):
        return []
    documents = SimpleDirectoryReader(data_dir or DATA_DIR, recursive=True,
        file_extractor={".pdf": PyMuPDFReader(), ".docx": DocxReader(), ".doc": RTFReader()}).load_data()
    page_counts = {}
    for document in documents:
        source = str(document.metadata.get("file_path") or document.metadata.get("filename") or document.hash)
        file_id = hashlib.md5(source.encode(), usedforsecurity=False).hexdigest()
        document.metadata["doc_node_id"] = hashlib.md5(f"doc-{file_id}".encode(), usedforsecurity=False).hexdigest()
        page_counts[source] = page_counts.get(source, 0) + 1
        page = str(document.metadata.get("page_label") or document.metadata.get("page")
                   or document.metadata.get("page_number") or page_counts[source])
        document.id_ = hashlib.md5(f"{file_id}|{page}|{page_counts[source]}".encode(), usedforsecurity=False).hexdigest()
        document.metadata.setdefault("page_label", page)
        document.metadata.setdefault("source_path", source)
        for key, value in _infer_metadata(source).items():
            document.metadata.setdefault(key, value)
        if ACL_ENABLED:
            for key, value in acl_metadata_for_ingest().items():
                document.metadata.setdefault(key, value)
        if os.getenv("NORMALIZE_TEXT", "1").lower() not in {"0", "false"}:
            _set_node_text(document, _normalize_text_for_embed(document.text))
    from pcb_rag.pcb_facts import link_continued_tables
    return link_continued_tables(documents)


def _parse_ingest_documents(documents):
    from pcb_rag.pcb_facts import extract_qualified_facts
    documents = augment_documents_with_multimodal(documents, progress=True)
    parser = _build_node_parser(Settings.embed_model)
    if parser == "parent_child":
        nodes = _parent_child_parse(documents)
    elif parser == "attention":
        nodes = _attention_semantic_parse(documents)
    elif parser is None:
        nodes = _structure_aware_parse(documents)
    else:
        nodes = parser.get_nodes_from_documents(documents, show_progress=True)
    nodes = _ensure_doc_node_id(nodes, documents)
    parents = [node for node in nodes if node.metadata.get("node_level") == "parent"]
    children = [node for node in nodes if node.metadata.get("node_level") != "parent"]
    source_metadata = {incremental.doc_node_id_of(doc): doc.metadata for doc in documents}
    for node in children:
        node.metadata = dict(source_metadata.get(node.metadata.get("doc_node_id"), {}), **node.metadata)
        node.metadata["node_level"] = "child"
        source = node.metadata.get("source_path") or node.metadata.get("file_path", "")
        node.metadata = enhance_chunk_metadata(node.text, node.metadata, "", source)
        node.metadata["pcb_facts"] = extract_qualified_facts(node.text, node.metadata)
    children = apply_contextual_retrieval(children, documents)
    children = _truncate_text_nodes(children, max_bytes=60000, target_bytes=12000)
    return parents + children


def main():
    from pcb_rag.ingestion import run_ingestion
    from pcb_rag.revisions import RevisionStore, pipeline_fingerprints, state_path
    from pcb_rag.graph_semantic import build_graph_vectors
    from pcb_rag.api_clients import get_embedding_profile
    from pcb_rag.embedding_contract import ensure_ingest_embedding_profile
    import inspect

    documents = _load_ingest_documents()
    config = dict(os.environ)
    # Include effective defaults and implementation hashes: a code upgrade can
    # require backfill even when environment and source bytes do not change.
    for key in list(config) + [key for key in globals() if key.isupper()]:
        value = globals().get(key)
        if isinstance(value, (str, int, float, bool)) and not any(word in key for word in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            config[key] = value
    config["PARSER_IMPLEMENTATION"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config["METADATA_IMPLEMENTATION"] = hashlib.sha256(inspect.getsource(enhance_chunk_metadata).encode()).hexdigest()
    config["CONTEXTUAL_IMPLEMENTATION"] = hashlib.sha256(inspect.getsource(apply_contextual_retrieval).encode()).hexdigest()
    from pcb_rag.graph_rag import _build_triple_prompt, _build_gleaning_prompt
    config["GRAPH_PROMPT_IMPLEMENTATION"] = hashlib.sha256((inspect.getsource(_build_triple_prompt) + inspect.getsource(_build_gleaning_prompt)).encode()).hexdigest()
    config["EMBEDDING_PROFILE"] = get_embedding_profile(document_text="passage_text_only_v2") if EMBED_BACKEND == "api" else {
        "backend": EMBED_BACKEND, "model": config.get("OLLAMA_EMBED_MODEL"),
        "dimension": config.get("EMBED_DIM"), "document_text": "passage_text_only_v2",
    }
    config["METADATA_IMPLEMENTATION"] += hashlib.sha256(Path(__file__).with_name("pcb_facts.py").read_bytes()).hexdigest()
    config["LEXICAL_IMPLEMENTATION"] = hashlib.sha256(Path(__file__).with_name("query.py").read_bytes()).hexdigest()
    config["GRAPH_IMPLEMENTATION"] = hashlib.sha256(
        Path(__file__).with_name("graph_rag.py").read_bytes() + Path(__file__).with_name("graph_semantic.py").read_bytes()
    ).hexdigest()
    fingerprints = pipeline_fingerprints(config)
    store = RevisionStore(state_path(COLLECTION))
    # Preview is read-only and does not initialize a model or connect to Milvus.
    if os.getenv("INGEST_PLAN_ONLY", "0") == "1":
        active = store.read().get("active", {})
        selected, stale, current = incremental.diff_documents(documents, {"docs": active.get("docs", {})}, pipeline=fingerprints)
        print(json.dumps({"collection": COLLECTION, "documents": len(current), "parse_fragments": len(selected),
                          "removed_or_changed": len(stale), "pipeline": fingerprints}, ensure_ascii=False, indent=2))
        return
    Settings.llm = build_llm()
    Settings.embed_model = build_embed_model()
    dimension = get_embedding_dim(Settings.embed_model)

    async def create_store():
        return MilvusVectorStore(uri=MILVUS_URI, collection_name=COLLECTION, overwrite=False,
                                 upsert_mode=True, dim=dimension, embedding_field="embedding", consistency_level="Strong")
    vector_store = asyncio.run(create_store())
    if os.getenv("INGEST_REQUIRE_EMPTY_COLLECTION", "0") == "1" and not (
        store.read().get("active") or store.read().get("pending")
    ):
        if int(vector_store.client.get_collection_stats(COLLECTION).get("row_count", 0)):
            raise ValueError("migration target must be an empty, dedicated collection")
    if os.getenv("INGEST_OVERWRITE", "0") == "1" and vector_store.client.get_collection_stats(COLLECTION).get("row_count", 0):
        raise ValueError("use a new COLLECTION for rebuilding; published rows are retained for rollback")
    if EMBED_BACKEND == "api":
        ensure_ingest_embedding_profile(vector_store.client, COLLECTION,
                                       get_embedding_profile(dimension, document_text="passage_text_only_v2"))
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    def verify(ids):
        for start in range(0, len(ids), 256):
            batch = ids[start:start + 256]
            rows = vector_store.client.get(collection_name=COLLECTION, ids=batch, output_fields=["id"], consistency_level="Strong", timeout=30)
            if {str(row["id"]) for row in rows} != set(batch):
                raise RuntimeError("vector write verification failed; published revision unchanged")

    def graph_build(nodes):
        graph = build_graph_from_nodes(nodes, Settings.llm, backend=GRAPH_EXTRACT_BACKEND, progress=True, strict=True)
        if GRAPH_COMMUNITY_ENABLED:
            summarize_communities(graph, Settings.llm)
        return graph

    def graph_vectors(graph, path):
        return build_graph_vectors(graph, Settings.embed_model, path, client=vector_store.client,
                                   collection_prefix=COLLECTION, backend=os.getenv("GRAPH_SEMANTIC_BACKEND", "milvus"))

    active = run_ingestion(documents, pipeline=fingerprints, collection=COLLECTION,
        parse=_parse_ingest_documents, insert=index.insert_nodes, verify=verify,
        graph_build=graph_build if GRAPH_RAG_ENABLED else None,
        graph_vectors=graph_vectors if GRAPH_RAG_ENABLED and os.getenv("GRAPH_SEMANTIC_ENABLED", "1") != "0" else None,
        store=store)
    incremental.save_manifest(INGEST_MANIFEST_PATH, active["docs"], pipeline=fingerprints, revision=active["revision"])
    print(f"Ingest published: collection={COLLECTION}, revision={active['revision']}, documents={len(active['docs'])}")


def _fix_file_encodings(data_dir: str) -> int:
    """修复目录下文本文件的编码问题"""
    fixed_count = 0
    skipped_count = 0
    
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            if not filename.endswith('.txt'):
                continue
            
            file_path = os.path.join(root, filename)
            
            try:
                # 读取原始字节
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                
                # 检测编码
                encoding, confidence = _detect_encoding(raw_data)
                
                # 如果已经是UTF-8且质量好，跳过
                if encoding.lower() in ('utf-8', 'utf-8-sig'):
                    try:
                        utf8_decoded = raw_data.decode('utf-8')
                        quality_score = _score_text_quality(utf8_decoded)
                        if quality_score < 0.05 and not _should_try_mojibake_repair(utf8_decoded):
                            skipped_count += 1
                            continue
                    except Exception:
                        pass
                
                # 如果不是UTF-8，尝试修复
                if encoding.lower() not in ('utf-8', 'utf-8-sig', 'ascii'):
                    decoded, used_encoding = _decode_with_fallback(raw_data)
                    
                    # 检查解码质量
                    garbage_ratio = _calculate_garbage_ratio(decoded)
                    
                    if garbage_ratio < 0.3:  # 乱码率低于30%认为成功
                        # 清理乱码
                        if CLEAN_OCR_GARBAGE:
                            decoded = _clean_ocr_garbage(decoded)
                        
                        # 写回UTF-8
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(decoded)
                        
                        fixed_count += 1
                        print(f"   ✅ 修复: {filename} ({used_encoding} → UTF-8)")
                    else:
                        print(f"   ⚠️  乱码率过高: {filename} ({garbage_ratio:.1%})")
                        
            except Exception as e:
                print(f"   ❌ 处理失败: {filename} - {e}")
    
    if fixed_count > 0:
        print(f"✅ 共修复 {fixed_count} 个文件的编码")
    if skipped_count > 0:
        print(f"ℹ️  跳过 {skipped_count} 个质量良好的UTF-8文件")
    
    return fixed_count


if __name__ == "__main__":
    main()
