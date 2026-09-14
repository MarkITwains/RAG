"""多模态处理：PDF 图片理解与表格结构化。

为什么需要
----------
PCB 标准与厂商资料里有大量「图」和「表」承载关键信息：
叠层结构图、阻抗曲线、工艺流程图、参数表、缺陷图谱。

这些内容在纯文本抽取后通常呈现为两种糟糕形态之一：

- **图**：完全丢失，或只剩一行「图 3 镀层厚度测试示意」
- **表**：被空格/制表符拼成一行行字符，列对齐关系丢失，模型很难正确解读对应关系

本模块做两件事：

1. **图片 → 文本**：从 PDF 抽出图片，调用 VLM（OpenAI 兼容 vision 接口）生成结构化描述，
   作为独立 chunk 参与检索与引用
2. **表格 → Markdown**：把「对齐块」还原成 Markdown 表格，列名与数值的对应关系不再丢失

设计取舍
--------
- VLM 走 OpenAI 兼容 ``/chat/completions``（``image_url`` 传 base64 data URL），
  与 LLM 后端解耦：可以本地 Ollama / 远端 API，也可以不开启（``MULTIMODAL_ENABLED=0``）
- 图片描述会写入 ``modality="image"`` 元数据，便于评测时区分图文来源
- 默认**关闭**：开启后入库会显著变慢（每张图一次 VLM 调用），按需打开
- 不导入 llama-index（延迟导入），可被单元测试轻量引用

用法::

    from pcb_rag.multimodal import text_tables_to_markdown, augment_documents_with_multimodal

    body = text_tables_to_markdown(raw_text)          # 表格结构化（零成本，可单独使用）
    docs = augment_documents_with_multimodal(docs)    # PDF 图片 → 描述 chunk
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "MULTIMODAL_ENABLED",
    "augment_documents_with_multimodal",
    "describe_image",
    "describe_multimodal",
    "extract_pdf_images",
    "extract_pdf_tables",
    "text_tables_to_markdown",
]


# ---------------------------------------------------------------------------
# 1. 配置
# ---------------------------------------------------------------------------
def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


MULTIMODAL_ENABLED = _env_bool("MULTIMODAL_ENABLED", False)
# 表格结构化：零成本，默认开启（只影响入库时的文本形态）
TABLE_MARKDOWN_ENABLED = _env_bool("TABLE_MARKDOWN_ENABLED", True)

# VLM（OpenAI 兼容 vision 接口）；base_url / api_key 留空时复用 LLM 后端配置
MULTIMODAL_VLM_BASE_URL = os.getenv("MULTIMODAL_VLM_BASE_URL", "").strip().rstrip("/")
MULTIMODAL_VLM_API_KEY = os.getenv("MULTIMODAL_VLM_API_KEY", "").strip()
MULTIMODAL_VLM_MODEL = os.getenv("MULTIMODAL_VLM_MODEL", "").strip()
MULTIMODAL_VLM_TIMEOUT = float(os.getenv("MULTIMODAL_VLM_TIMEOUT", "120"))
MULTIMODAL_MAX_TOKENS = int(os.getenv("MULTIMODAL_MAX_TOKENS", "512"))

# 图片抽取约束
MULTIMODAL_MAX_IMAGES_PER_DOC = int(os.getenv("MULTIMODAL_MAX_IMAGES_PER_DOC", "20"))
MULTIMODAL_MIN_IMAGE_PX = int(os.getenv("MULTIMODAL_MIN_IMAGE_PX", "120"))
MULTIMODAL_MIN_IMAGE_BYTES = int(os.getenv("MULTIMODAL_MIN_IMAGE_BYTES", "4096"))
MULTIMODAL_DESC_MAX_CHARS = int(os.getenv("MULTIMODAL_DESC_MAX_CHARS", "500"))

MULTIMODAL_IMAGE_PROMPT = os.getenv(
    "MULTIMODAL_IMAGE_PROMPT",
    "这是 PCB 制造技术文档中的一张插图。请用中文客观描述：\n"
    "1) 图的类型（结构剖面 / 工艺流程图 / 曲线 / 显微照片 / 示意图）；\n"
    "2) 图中标注的层次、材料、工艺步骤或坐标轴含义；\n"
    "3) 出现的关键数值与单位（如厚度、温度、时间、比例），逐个列出；\n"
    "4) 若图中含结论性文字或图题，一并摘录。\n"
    "只描述图中确实存在的内容，不要推测或补充常识。控制在 200 字以内。",
).strip()

# 表格结构化的判定参数
_TABLE_MIN_ROWS = int(os.getenv("TABLE_MIN_ROWS", "2"))
_TABLE_MIN_COLS = int(os.getenv("TABLE_MIN_COLS", "2"))

_COL_SPLIT_RE = re.compile(r"\s{2,}|\t+|\s*\|\s*")
_NUMERIC_RE = re.compile(r"^[\s]*[-+]?\d+(?:[.,]\d+)?\s*(?:%|mm|um|μm|mil|oz|℃|°C|MPa|N|V|A|s|min|h|g)?[\s]*$")


# ---------------------------------------------------------------------------
# 2. 表格结构化
# ---------------------------------------------------------------------------
def _split_columns(line: str) -> List[str]:
    """把一行文本切成列（连续空格 / 制表符 / 竖线都是分隔符）。"""

    if not line or not line.strip():
        return []
    if "\t" in line or re.search(r"\s{2,}", line) or "|" in line:
        cells = [c.strip() for c in _COL_SPLIT_RE.split(line.strip())]
        cells = [c for c in cells if c != ""]
        return cells
    return []


def _display_width(text: str) -> int:
    """估算显示宽度：CJK 字符按 2 列计，用于判断单元格是否「短」。"""

    return sum(2 if ord(ch) > 0x2E7F else 1 for ch in text)


def _looks_like_table_row(cells: Sequence[str]) -> bool:
    """判断一行是否像表格数据行：至少 2 列，且存在数值列或列普遍较短。

    用显示宽度而非字符数衡量：中文正文一句三十几字按字符数算很短、会误判成表格，
    按显示宽度算约 70 列，就能正确排除。
    """

    if len(cells) < _TABLE_MIN_COLS:
        return False
    if any(_NUMERIC_RE.match(c) for c in cells):
        return True
    widths = [max(_display_width(c), 1) for c in cells]
    return max(widths) <= 40 and sum(widths) <= 120


def _to_markdown_table(rows: Sequence[Sequence[str]]) -> str:
    """把二维单元格转成 Markdown 表格。"""

    if not rows:
        return ""
    width = max(len(r) for r in rows)
    normalized = [list(r) + [""] * (width - len(r)) for r in rows]
    header = normalized[0]
    body = normalized[1:]

    lines = [
        "| " + " | ".join(cell.replace("|", "\\|") or " " for cell in header) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(cell.replace("|", "\\|") or " " for cell in row) + " |")
    return "\n".join(lines)


def text_tables_to_markdown(
    text: str,
    *,
    min_rows: int = _TABLE_MIN_ROWS,
    min_cols: int = _TABLE_MIN_COLS,
) -> str:
    """把文本中的「对齐块」还原成 Markdown 表格，其余内容原样保留。

    典型输入（PDF 抽取后的参数表）：:

        项目        要求        试验方法
        镀层厚度    ≥0.8 μm     GB/T 4677
        附着力      无脱落      IPC-TM-650

    输出：标准 Markdown 表格，列与值的对应关系显式化。
    """

    if not text:
        return ""

    lines = text.split("\n")
    out: List[str] = []
    index = 0

    while index < len(lines):
        row = _split_columns(lines[index])
        if row and len(row) >= min_cols and _looks_like_table_row(row):
            block: List[List[str]] = [row]
            cursor = index + 1
            while cursor < len(lines):
                candidate = _split_columns(lines[cursor])
                if candidate and len(candidate) >= min_cols and _looks_like_table_row(candidate):
                    block.append(candidate)
                    cursor += 1
                else:
                    break

            if len(block) >= min_rows:
                widths = [len(r) for r in block]
                # 列数差异过大的「块」通常是误判，退化为原文
                if max(widths) - min(widths) <= 2:
                    out.append("")
                    out.append(_to_markdown_table(block))
                    out.append("")
                    index = cursor
                    continue

        out.append(lines[index])
        index += 1

    # 压缩连续空行
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(out))
    return result.strip("\n")


def extract_pdf_tables(path: str, *, max_tables: int = 50) -> List[Dict[str, Any]]:
    """用 PyMuPDF 的表格识别能力抽取 PDF 表格（不可用时返回空列表）。"""

    try:
        import fitz  # PyMuPDF
    except Exception:
        return []

    tables: List[Dict[str, Any]] = []
    try:
        with fitz.open(path) as doc:
            for page_index, page in enumerate(doc):
                finder = getattr(page, "find_tables", None)
                if not callable(finder):
                    continue
                try:
                    found = finder()
                except Exception:
                    continue
                for table in getattr(found, "tables", []) or []:
                    try:
                        rows = table.extract()
                    except Exception:
                        continue
                    cleaned = [[(cell or "").strip() for cell in row] for row in rows if row]
                    cleaned = [row for row in cleaned if any(row)]
                    if len(cleaned) < 2:
                        continue
                    tables.append({"page": page_index + 1, "rows": cleaned, "markdown": _to_markdown_table(cleaned)})
                    if len(tables) >= max_tables:
                        return tables
    except Exception:
        return tables

    return tables


# ---------------------------------------------------------------------------
# 3. 图片抽取
# ---------------------------------------------------------------------------
def extract_pdf_images(
    path: str,
    *,
    max_images: int = MULTIMODAL_MAX_IMAGES_PER_DOC,
    min_px: int = MULTIMODAL_MIN_IMAGE_PX,
    min_bytes: int = MULTIMODAL_MIN_IMAGE_BYTES,
) -> List[Dict[str, Any]]:
    """抽取 PDF 中的图片，返回 ``[{page, data, ext, width, height, index}, ...]``。

    会过滤掉装饰性小图（尺寸或体积过小），避免为无信息量的线条浪费 VLM 调用。
    """

    try:
        import fitz  # PyMuPDF
    except Exception:
        return []

    images: List[Dict[str, Any]] = []
    seen_xref: set = set()

    try:
        with fitz.open(path) as doc:
            for page_index, page in enumerate(doc):
                try:
                    raw_images = page.get_images(full=True)
                except Exception:
                    continue

                for image_index, info in enumerate(raw_images):
                    if len(images) >= max_images:
                        return images
                    xref = info[0] if info else None
                    if xref in seen_xref:
                        continue
                    seen_xref.add(xref)

                    try:
                        payload = doc.extract_image(xref)
                    except Exception:
                        continue
                    data = payload.get("image")
                    width = int(payload.get("width") or 0)
                    height = int(payload.get("height") or 0)
                    if not data or width < min_px or height < min_px or len(data) < min_bytes:
                        continue

                    images.append(
                        {
                            "page": page_index + 1,
                            "index": image_index,
                            "data": data,
                            "ext": str(payload.get("ext") or "png"),
                            "width": width,
                            "height": height,
                        }
                    )
    except Exception:
        return images

    return images


def is_vision_configured() -> bool:
    """判断 VLM 是否可用（未配置时跳过图片描述，仅保留图片占位信息）。"""

    if not MULTIMODAL_ENABLED:
        return False
    base = MULTIMODAL_VLM_BASE_URL or os.getenv("LLM_BASE_URL", "").strip().rstrip("/")
    model = MULTIMODAL_VLM_MODEL or os.getenv("VLM_MODEL", "").strip()
    return bool(base and model)


def _vision_endpoint() -> Tuple[str, str, str]:
    """返回 ``(url, api_key, model)``；base_url 兼容带或不带 /v1 的写法。"""

    base = MULTIMODAL_VLM_BASE_URL or os.getenv("LLM_BASE_URL", "").strip().rstrip("/")
    api_key = MULTIMODAL_VLM_API_KEY or os.getenv("LLM_API_KEY", "").strip()
    model = MULTIMODAL_VLM_MODEL or os.getenv("VLM_MODEL", "").strip() or os.getenv("LLM_MODEL", "").strip()

    if base.endswith("/chat/completions"):
        url = base
    else:
        url = f"{base}/chat/completions"
    return url, api_key, model


def describe_image(
    image_bytes: bytes,
    *,
    prompt: Optional[str] = None,
    ext: str = "png",
    model: Optional[str] = None,
) -> str:
    """调用 OpenAI 兼容 vision 接口描述图片；失败时返回空串（调用方决定降级策略）。"""

    if not image_bytes:
        return ""

    url, api_key, resolved_model = _vision_endpoint()
    resolved_model = (model or resolved_model).strip()
    if not is_vision_configured() or not resolved_model:
        return ""

    try:
        import requests
    except Exception:
        return ""

    mime = "image/jpeg" if ext.lower() in {"jpg", "jpeg"} else f"image/{ext.lower() or 'png'}"
    data_url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"

    payload = {
        "model": resolved_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt or MULTIMODAL_IMAGE_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "max_tokens": MULTIMODAL_MAX_TOKENS,
        "temperature": 0.1,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=MULTIMODAL_VLM_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        try:
            from pcb_rag.observability import counter, record_event

            counter("multimodal.describe_failed")
            record_event("multimodal.describe_failed", error=str(exc)[:200])
        except Exception:
            pass
        return ""

    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        return ""

    if isinstance(content, list):
        # 部分服务返回分段内容
        content = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))

    text = re.sub(r"\s+", " ", str(content)).strip()
    return text[:MULTIMODAL_DESC_MAX_CHARS]


# ---------------------------------------------------------------------------
# 4. 文档增强
# ---------------------------------------------------------------------------
def _doc_text(doc: Any) -> str:
    getter = getattr(doc, "get_content", None)
    if callable(getter):
        try:
            return str(getter())
        except Exception:
            pass
    return str(getattr(doc, "text", "") or "")


def _doc_metadata(doc: Any) -> Dict[str, Any]:
    metadata = getattr(doc, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _make_document(text: str, metadata: Dict[str, Any]) -> Any:
    """构造 llama-index Document；不可用时返回轻量对象（便于纯逻辑测试）。"""

    try:
        from llama_index.core import Document

        return Document(text=text, metadata=metadata)
    except Exception:
        return _SimpleDocument(text=text, metadata=metadata)


class _SimpleDocument:
    """llama-index 不可用时的兜底文档对象（仅承载 text / metadata）。"""

    __slots__ = ("text", "metadata", "id_", "doc_id")

    def __init__(self, text: str, metadata: Dict[str, Any]) -> None:
        self.text = text
        self.metadata = metadata
        self.id_ = ""
        self.doc_id = ""

    def get_content(self) -> str:
        return self.text

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return f"_SimpleDocument(chars={len(self.text)}, metadata_keys={sorted(self.metadata)})"


def _assign_stable_id(doc: Any, raw: str) -> None:
    """按内容派生稳定 ID：重复入库时 upsert 覆盖，而不是新增重复文档。"""

    doc_id = hashlib.md5(raw.encode("utf-8"), usedforsecurity=False).hexdigest()
    for attr in ("id_", "doc_id"):
        try:
            setattr(doc, attr, doc_id)
        except Exception:
            continue


def _inherit_doc_id(target: Any, source: Any) -> None:
    """把源文档的稳定 ID 复制到新文档（表格改写后仍能正确覆盖）。"""

    source_id = str(getattr(source, "doc_id", "") or getattr(source, "id_", "") or "")
    if not source_id:
        return
    for attr in ("id_", "doc_id"):
        try:
            setattr(target, attr, source_id)
        except Exception:
            continue


def augment_documents_with_multimodal(
    documents: Optional[Sequence[Any]],
    *,
    describe_images: Optional[bool] = None,
    tables_as_markdown: Optional[bool] = None,
    progress: bool = False,
) -> List[Any]:
    """对文档列表做多模态增强，返回**新列表**（原对象不被修改）。

    处理内容：

    1. 表格结构化：把正文里的对齐块转成 Markdown（``TABLE_MARKDOWN_ENABLED`` 控制）
    2. PDF 图片：抽取图片 → VLM 描述 → 追加为独立文档（``MULTIMODAL_ENABLED`` 控制）
    """

    if not documents:
        return []

    want_images = MULTIMODAL_ENABLED if describe_images is None else bool(describe_images)
    want_tables = TABLE_MARKDOWN_ENABLED if tables_as_markdown is None else bool(tables_as_markdown)
    vision_ready = want_images and is_vision_configured()

    result: List[Any] = []
    image_docs = 0
    table_docs = 0

    for doc in documents:
        text = _doc_text(doc)
        metadata = dict(_doc_metadata(doc))
        source_path = str(metadata.get("source_path") or metadata.get("file_path") or "")

        original_text = text
        if want_tables and text:
            converted = text_tables_to_markdown(text)
            if converted and converted != text:
                metadata["has_markdown_table"] = True
                table_docs += 1
                text = converted

        if text != original_text:
            rewritten = _make_document(text, metadata)
            _inherit_doc_id(rewritten, doc)
            result.append(rewritten)
        else:
            result.append(doc)

        if not want_images or not source_path.lower().endswith(".pdf"):
            continue

        try:
            images = extract_pdf_images(source_path)
        except Exception:
            images = []
        if not images:
            continue

        for image in images:
            description = ""
            if vision_ready:
                description = describe_image(image["data"], ext=image["ext"])
            if not description:
                # VLM 不可用时至少保留占位信息，让检索能定位到「这里有一张图」
                description = f"（第 {image['page']} 页存在一张插图，尺寸 {image['width']}×{image['height']}，尚未生成描述）"

            image_metadata = {
                **{k: v for k, v in metadata.items() if k in {"source_type", "source_path", "vendor", "eda", "doc_node_id"}},
                "modality": "image",
                "page": image["page"],
                "image_index": image["index"],
                "image_width": image["width"],
                "image_height": image["height"],
                "has_description": bool(vision_ready and "尚未生成描述" not in description),
            }
            image_text = (
                f"[图片内容] 文件：{os.path.basename(source_path) or '未知'}，第 {image['page']} 页\n"
                f"{description}"
            )
            image_doc = _make_document(image_text, image_metadata)
            _assign_stable_id(image_doc, f"{source_path}|img|{image['page']}|{image['index']}")
            result.append(image_doc)
            image_docs += 1

        if progress:
            print(f"[Multimodal] {os.path.basename(source_path)}: 抽取 {len(images)} 张图片")

    try:
        from pcb_rag.observability import counter

        counter("multimodal.image_docs", image_docs)
        counter("multimodal.table_converted", table_docs)
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# 5. 诊断
# ---------------------------------------------------------------------------
def describe_multimodal() -> Dict[str, Any]:
    """返回多模态配置摘要，供启动日志与 ``/health`` 展示。"""

    url, _api_key, model = _vision_endpoint()
    return {
        "enabled": MULTIMODAL_ENABLED,
        "vision_configured": is_vision_configured(),
        "vision_endpoint": url if is_vision_configured() else None,
        "vision_model": model if is_vision_configured() else None,
        "max_images_per_doc": MULTIMODAL_MAX_IMAGES_PER_DOC,
        "min_image_px": MULTIMODAL_MIN_IMAGE_PX,
        "table_markdown": TABLE_MARKDOWN_ENABLED,
    }
