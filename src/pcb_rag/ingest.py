import asyncio
import hashlib
import os
import re
import site
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# 尝试导入chardet用于编码检测
try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False

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

    # 首先尝试按当前解释器的版本添加
    if os.path.isdir(venv_site_packages) and venv_site_packages not in sys.path:
        sys.path.insert(0, venv_site_packages)
        site.addsitedir(venv_site_packages)
        return

    # 如果没有与当前 Python 完全匹配的 site-packages，尝试扫描 `.venv/lib` 下的 pythonX.Y 目录，
    # 将第一个可用的 site-packages 加入 sys.path（兼容在其他 Python 版本下运行脚本但依赖安装在特定 venv 的场景）。
    lib_dir = os.path.join(repo_dir, ".venv", "lib")
    try:
        candidates = []
        if os.path.isdir(lib_dir):
            for name in os.listdir(lib_dir):
                if name.startswith("python"):
                    sp = os.path.join(lib_dir, name, "site-packages")
                    if os.path.isdir(sp):
                        candidates.append(sp)
        # 按名称倒序（通常 python3.10 < python3.11），选择最高版本优先
        candidates.sort(reverse=True)
        for sp in candidates:
            if sp not in sys.path:
                sys.path.insert(0, sp)
                site.addsitedir(sp)
                return
    except Exception:
        pass


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


_sanitize_proxy_env_for_httpx()

from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser, HierarchicalNodeParser
from llama_index.core.schema import MetadataMode, TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.readers.file import PyMuPDFReader, DocxReader, RTFReader
from llama_index.vector_stores.milvus import MilvusVectorStore

try:
    import nest_asyncio

    nest_asyncio.apply()
except Exception:
    pass

DATA_DIR = os.getenv("DATA_DIR", "./data/clear_docs")
MILVUS_URI = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")
COLLECTION = os.getenv("COLLECTION", "pcb_kb")
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://127.0.0.1:11434")

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
            except:
                pass
        
        # chardet有时会误判中文为ISO-8859-1
        if encoding.lower() in ('iso-8859-1', 'ascii', 'windows-1252') and confidence < 0.9:
            try:
                decoded = data.decode('gbk')
                chinese_count = len(re.findall(r'[\u4e00-\u9fff]', decoded))
                if chinese_count > 10:
                    return 'gbk', 0.95
            except:
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
        except:
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
        r'[ͶͶͶĄIJͶ]+',
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
        current_section = "前言"
        
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
    
    def _split_by_length(self, text: str, max_size: int, min_size: int, overlap: int) -> List[str]:
        """按长度切分（带 overlap）"""
        if len(text) <= max_size:
            return [text]
        
        chunks = []
        
        # 按句子切分以保持完整性
        sentences = re.split(r'(?<=[。！？.!?])', text)
        
        current_chunk = ""
        
        for sent in sentences:
            if not sent.strip():
                continue
            
            if len(current_chunk) + len(sent) <= max_size:
                current_chunk += sent
            else:
                if current_chunk and len(current_chunk) >= min_size:
                    chunks.append(current_chunk.strip())
                    
                    # 添加 overlap
                    if overlap > 0:
                        overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                        # 找到句子边界
                        boundary = overlap_text.rfind('。')
                        if boundary == -1:
                            boundary = overlap_text.rfind('.')
                        if boundary > 0:
                            overlap_text = overlap_text[boundary+1:]
                        current_chunk = overlap_text + sent
                    else:
                        current_chunk = sent
                else:
                    current_chunk += sent
        
        # 添加最后一个 chunk
        if current_chunk:
            if len(current_chunk) >= min_size:
                chunks.append(current_chunk.strip())
            elif chunks:
                # 太短则合并到前一个
                chunks[-1] += current_chunk
            else:
                chunks.append(current_chunk.strip())
        
        return chunks
    
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
        
        # 判断是否使用 Ollama（模型名包含 : 且不包含 /）
        self.use_ollama = ATTENTION_USE_OLLAMA and ':' in self.model_name and '/' not in self.model_name
        
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
        
        if self.use_ollama:
            # 使用 Ollama
            try:
                from llama_index.embeddings.ollama import OllamaEmbedding
                print(f"[SemanticSplitter] 使用 Ollama 模型: {self.model_name}")
                self._ollama_embed = OllamaEmbedding(
                    model_name=self.model_name,
                    base_url=OLLAMA_BASE,
                    ollama_additional_kwargs={"num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "2048"))},
                )
                print(f"[SemanticSplitter] Ollama 模型初始化完成")
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
                
                print(f"[SemanticSplitter] HF 模型加载完成")
            except Exception as e:
                print(f"[SemanticSplitter] HF 模型加载失败: {e}")
                self._model = None
    
    def _get_embeddings(self, texts: List[str]) -> 'np.ndarray':
        """批量获取文本嵌入"""
        import numpy as np
        
        if not texts:
            return np.array([])
        
        self._load_model()
        
        if self.use_ollama and self._ollama_embed is not None:
            # 使用 Ollama 获取嵌入
            try:
                embeddings = []
                # 显示进度（每 10 个打印一次）
                show_progress = len(texts) > 20
                for i, text in enumerate(texts):
                    if show_progress and i % 10 == 0:
                        print(f"  嵌入进度: {i}/{len(texts)}", end='\r')
                    emb = self._ollama_embed.get_text_embedding(text)
                    embeddings.append(emb)
                if show_progress:
                    print(f"  嵌入完成: {len(texts)}/{len(texts)}")
                arr = np.array(embeddings)
                # L2 归一化，避免余弦相似度数值不稳定
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms = np.where(norms < 1e-8, 1.0, norms)
                return arr / norms
            except Exception as e:
                print(f"[SemanticSplitter] Ollama 嵌入失败: {e}")
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
    if "/标准/国际标准/" in rel_posix:
        return "standard_international"
    if "/标准/国家标准/" in rel_posix:
        return "standard_national"
    if "/标准/" in rel_posix:
        return "standard"
    if "/文献/" in rel_posix:
        return "literature"
    if "/PCB设计/" in rel_posix:
        return "pcb_design"
    if "/PCB资料/" in rel_posix:
        return "pcb_material"
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
    m = re.search(r"(?<!\d)(\d{1,2})\s*(?:层|layer|layers)\b", text)
    if not m:
        m = re.search(r"(?<!\d)(\d{1,2})\s*l\b", text)
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
    
    # 1. 添加文档类型
    if "doc_type" not in enhanced_meta:
        enhanced_meta["doc_type"] = classify_document_type(doc_text[:5000], file_path)
    
    # 2. 提取章节信息
    section_info = extract_section_info(chunk_text)
    if section_info["section"] and "section" not in enhanced_meta:
        enhanced_meta["section"] = section_info["section"]
    if section_info["subsection"]:
        enhanced_meta["subsection"] = section_info["subsection"]
    
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
    """切分/截断过长的文本节点，避免超过 Milvus 的 varchar 限制(65535 bytes)。

    - 先按段落/句子切分到 target_bytes
    - 仍超过 max_bytes 的再截断
    注意：Milvus写入的text字段会包含metadata，需按 UTF-8 字节长度计入。
    """
    truncated_count = 0
    split_count = 0
    dropped_meta_count = 0
    suffix = "\n\n[...文本过长已截断...]"

    new_nodes = []
    for node in nodes:
        if not (hasattr(node, "text") and isinstance(node.text, str)):
            new_nodes.append(node)
            continue

        try:
            content_none = node.get_content(metadata_mode=MetadataMode.NONE)
            content_all = node.get_content(metadata_mode=MetadataMode.ALL)
        except Exception:
            content_none = node.text
            content_all = node.text

        if _utf8_len(content_all) <= max_bytes:
            new_nodes.append(node)
            continue

        original_len = _utf8_len(content_all)
        meta_len = max(0, _utf8_len(content_all) - _utf8_len(content_none))

        # 如果metadata本身过大，直接剥离metadata
        if meta_len >= max_bytes:
            try:
                node.metadata = {}
                dropped_meta_count += 1
            except Exception:
                pass
            try:
                content_none = node.get_content(metadata_mode=MetadataMode.NONE)
                meta_len = 0
            except Exception:
                content_none = node.text
                meta_len = 0

        allow = max_bytes - meta_len
        if allow <= 0:
            node.text = ""
            truncated_count += 1
            new_nodes.append(node)
            continue

        base_text = content_none if isinstance(content_none, str) else node.text
        if not isinstance(base_text, str):
            base_text = ""

        split_max = max(512, min(target_bytes, allow))
        chunks = _split_text_to_chunks(base_text, split_max)
        if len(chunks) <= 1:
            node.text = _truncate_utf8(base_text, allow, suffix)
            truncated_count += 1
            new_nodes.append(node)
        else:
            split_count += 1
            for idx, chunk in enumerate(chunks):
                if _utf8_len(chunk) > allow:
                    chunk = _truncate_utf8(chunk, allow, suffix)
                    truncated_count += 1
                if idx == 0:
                    node.text = chunk
                    new_nodes.append(node)
                else:
                    try:
                        base_id = getattr(node, "id_", "") or ""
                        new_id = hashlib.md5(f"{base_id}-{idx}".encode("utf-8"), usedforsecurity=False).hexdigest()
                    except Exception:
                        new_id = None
                    new_nodes.append(TextNode(text=chunk, id_=new_id, metadata=dict(getattr(node, "metadata", {}) or {})))

        try:
            new_len = _utf8_len(node.get_content(metadata_mode=MetadataMode.ALL))
        except Exception:
            new_len = _utf8_len(node.text)
        print(f"⚠️  警告: 文本节点从 {original_len} bytes 处理至 {new_len} bytes")

    if dropped_meta_count > 0:
        print(f"⚠️  共剥离 {dropped_meta_count} 个节点的metadata（过长）")
    if split_count > 0:
        print(f"⚠️  共切分 {split_count} 个超长文本节点")
    if truncated_count > 0:
        print(f"⚠️  共截断 {truncated_count} 个超长文本节点")
    return new_nodes


def _build_node_parser(embed_model):
    """根据配置构建node parser：parent_child > 结构感知 > 注意力语义 > 语义感知 > 层次化 > 传统固定切块"""
    
    if NODE_PARSER_MODE == "parent_child":
        print(f"[切块策略] Parent-Child 两层结构切块")
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
        print(f"[切块策略] Transformer 注意力语义切块")
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
    from llama_index.core.schema import TextNode
    
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
    from llama_index.core.schema import TextNode
    
    print(f"[Parent-Child] 初始化两层结构切块器...")
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
        doc_node_id = hashlib.md5(f"doc-{doc_id}".encode('utf-8'), usedforsecurity=False).hexdigest()
        
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
            node_id = hashlib.md5(f"{doc_id}-pc-{idx}".encode('utf-8'), usedforsecurity=False).hexdigest()
            chunk_node_ids.append(node_id)
        
        # 第二遍：构建节点
        for idx, (chunk_text, chunk_meta) in enumerate(chunks):
            node_id = chunk_node_ids[idx]
            
            # 构建邻接关系
            prev_id = chunk_node_ids[idx - 1] if idx > 0 else None
            next_id = chunk_node_ids[idx + 1] if idx < len(chunks) - 1 else None
            
            # 查找同一 parent 内的邻居
            current_parent_idx = chunk_meta.get('parent_idx', 0)
            parent_first_child = None
            parent_last_child = None
            
            for other_idx, (_, other_meta) in enumerate(chunks):
                if other_meta.get('parent_idx') == current_parent_idx:
                    if parent_first_child is None:
                        parent_first_child = other_idx
                    parent_last_child = other_idx
            
            # 构建完整元数据
            merged_meta = {
                **chunk_meta,
                # 邻接关系
                'prev_id': prev_id or '',
                'next_id': next_id or '',
                'doc_node_id': doc_node_id,
                
                # Parent 内的位置信息
                'is_first_in_parent': (idx == parent_first_child),
                'is_last_in_parent': (idx == parent_last_child),
                
                # 全局位置
                'total_chunks_in_doc': len(chunks),
            }
            
            # 增强元数据
            enhanced_meta = enhance_chunk_metadata(
                chunk_text=chunk_text,
                chunk_meta=merged_meta,
                doc_text=text,
                file_path=file_path
            )
            
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
    from llama_index.core.schema import TextNode
    
    print(f"[SemanticSplitter] 初始化 Transformer 语义切块器...")
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
                current_section = section_info['section_num']
                
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


def main():
    # 1) 配置 LLM（用于回答）- 通过环境变量配置，便于与生成模型分离
    llm_model = os.getenv("OLLAMA_LLM_MODEL", "glm-4.7-flash:q8_0")
    Settings.llm = Ollama(model=llm_model, base_url=OLLAMA_BASE, request_timeout=120.0)

    # 2) 配置 Embedding（用于向量化）
    Settings.embed_model = OllamaEmbedding(
        model_name="qwen3-embedding:8b-q8_0", 
        base_url=OLLAMA_BASE,
        ollama_additional_kwargs={"num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "2048"))}
    )
    # 2.1) 计算嵌入维度（用于配置 Milvus）
    try:
        embed_dim = len(Settings.embed_model.get_query_embedding("test dimension"))
    except Exception:
        embed_dim = None

    # 2.2) 配置切块策略：parent_child > 结构感知 > 注意力语义 > 语义感知 > 层次化 > 固定切块
    node_parser = _build_node_parser(Settings.embed_model)
    if node_parser is not None and node_parser not in ("attention", "parent_child"):
        Settings.node_parser = node_parser
    use_structure_mode = (node_parser is None)  # None表示使用结构感知模式
    use_attention_mode = (node_parser == "attention")  # "attention"表示使用注意力语义模式
    use_parent_child_mode = (node_parser == "parent_child")  # "parent_child"表示使用两层结构模式

    # 3) 读取文档（递归遍历子目录，PDF 用 PyMuPDF 提高兼容性）
    # 注意：这里将 .doc 映射给 RTFReader，因为当前语料中的 IPC-2581C.doc 实为 RTF 格式
    file_extractor = {
        ".pdf": PyMuPDFReader(),
        ".docx": DocxReader(),
        ".doc": RTFReader(),
    }
    
    # 3.0) 如果启用编码修复，先修复文本文件编码
    if AUTO_FIX_ENCODING:
        print("🔧 正在检测并修复文件编码...")
        _fix_file_encodings(DATA_DIR)
    
    documents = SimpleDirectoryReader(
        DATA_DIR,
        recursive=True,
        file_extractor=file_extractor,
    ).load_data()

    # 3.1) 为每个文档设置稳定 ID（基于文件路径哈希），便于断点续跑和去重
    garbage_cleaned_count = 0
    
    for d in documents:
        src = d.metadata.get("file_path") or d.metadata.get("filename") or str(d.hash)
        d.id_ = hashlib.md5(src.encode("utf-8"), usedforsecurity=False).hexdigest()

        # 9.1) PCB 场景元数据：用于后续按板厂/工艺/EDA/产品线等过滤检索范围
        file_path = d.metadata.get("file_path") or d.metadata.get("filename")
        if isinstance(file_path, str) and file_path:
            inferred = _infer_metadata(file_path)
            for k, v in inferred.items():
                # 不覆盖 reader 已经给的字段
                d.metadata.setdefault(k, v)

        # 文本清洗（编码修复 + 乱码清理）
        if os.getenv("NORMALIZE_TEXT", "1") not in {"0", "false", "False"}:
            try:
                if hasattr(d, "text") and isinstance(getattr(d, "text"), str):
                    original_text = d.text
                    d.text = _normalize_text_for_embed(d.text)
                    # 统计清理效果
                    if original_text != d.text and CLEAN_OCR_GARBAGE:
                        garbage_cleaned_count += 1
            except Exception:
                pass
    
    if garbage_cleaned_count > 0:
        print(f"🧹 已清理 {garbage_cleaned_count} 个文档的OCR乱码")
    
    print(f"📚 共加载 {len(documents)} 个文档，开始向量化并写入 Milvus...")

    # 4) 连接 Milvus 向量库（Standalone 端口 19530）
    async def _init_store() -> MilvusVectorStore:
        return MilvusVectorStore(
            uri=MILVUS_URI,
            collection_name=COLLECTION,
            overwrite=True,   # 首次建库可改 True；后续建议做增量
            upsert_mode=True,  # 允许按 doc_id 覆盖，避免重复
            dim=embed_dim,
            embedding_field="embedding",
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
        vector_store = loop.run_until_complete(_init_store())
    else:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            vector_store = loop.run_until_complete(_init_store())
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 5) 建索引（会：切块 → embedding → 写入 Milvus）
    # 注意：先进行node parsing，然后截断过长文本，最后建索引
    from llama_index.core.ingestion import IngestionPipeline
    
    # 根据模式选择切块方式
    if use_parent_child_mode:
        print("📦 使用 Parent-Child 两层结构切块...")
        nodes = _parent_child_parse(documents)
        print(f"   生成 {len(nodes)} 个 children chunks")
    elif use_structure_mode:
        print("📐 使用结构感知切块...")
        nodes = _structure_aware_parse(documents)
        print(f"   生成 {len(nodes)} 个结构化chunks")
    elif use_attention_mode:
        print("🧠 使用 Transformer 注意力语义切块...")
        nodes = _attention_semantic_parse(documents)
        print(f"   生成 {len(nodes)} 个语义chunks")
    else:
        # 使用配置的node_parser
        nodes = Settings.node_parser.get_nodes_from_documents(documents, show_progress=True)
    
    # 截断超长文本（避免超过Milvus的65535字符限制）
    nodes = _truncate_text_nodes(nodes, max_bytes=60000, target_bytes=12000)
    
    # 统计chunk长度分布
    chunk_lengths = [len(n.text) for n in nodes if hasattr(n, 'text')]
    if chunk_lengths:
        avg_len = sum(chunk_lengths) / len(chunk_lengths)
        max_len = max(chunk_lengths)
        min_len = min(chunk_lengths)
        print(f"📊 Chunk统计: 数量={len(chunk_lengths)}, 平均长度={avg_len:.0f}, 最小={min_len}, 最大={max_len}")
    
    # 从nodes构建索引
    index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, show_progress=True)

    print(f"✅ Ingest done. docs={len(documents)} chunks={len(nodes)} collection={COLLECTION}")


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
                    except:
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
