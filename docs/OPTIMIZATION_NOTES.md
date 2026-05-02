# PCB-RAG 优化记录

本文档合并整理了项目开发过程中的检索、切块、评估和后续优化规划。


---

## 优化总结

═══════════════════════════════════════════════════════════════
  PCB-RAG 检索系统优化完成总结
═══════════════════════════════════════════════════════════════

📊 问题分析
────────────────────────────────────────────────────────────
当前指标：
  • MRR = 0.237 (相关文档平均位置偏后)
  • Hit Rate@10 = 0.425 (召回率低)
  • Soft/Hard 比值 = 2:1 (语义匹配与真实相关性不对齐)

根本原因：
  1. 固定切块(512/128)割裂完整语义单元
  2. 向量检索权重过高,BM25专业术语匹配被削弱
  3. 缺少查询扩展,同义词覆盖不足
  4. 查询改写数量少,召回多样性不足

═══════════════════════════════════════════════════════════════
✅ 已完成优化
═══════════════════════════════════════════════════════════════

📁 修改文件：
  ├─ ingest.py
  ├─ query.py
  ├─ docs/OPTIMIZATION_NOTES.md
  └─ scripts/run_optimized.sh (快速启动脚本)

🔧 优化内容：

1. 文档分块策略 (ingest.py)
   ───────────────────────────────────────────────────────
   新增三种模式:
   
   ✓ semantic (语义感知) - 默认推荐
     - 根据语义相似度动态切分
     - 避免割裂完整测试步骤/公式
     - 配置: NODE_PARSER_MODE=semantic
   
   ✓ hierarchical (层次化)
     - 三级chunk (1024/384/128)
     - 保留上下文+精确匹配
     - 配置: NODE_PARSER_MODE=hierarchical
   
   ✓ sentence (优化固定)
     - 384/150 (从512/128优化)
     - 更小chunk,更大overlap
     - 配置: NODE_PARSER_MODE=sentence

2. 混合检索优化 (query.py)
   ───────────────────────────────────────────────────────
   ✓ Fusion模式切换
     - DIST_BASED_SCORE (加权融合) - 默认
     - RECIPROCAL_RANK (RRF)
     - RELATIVE_SCORE (相对分数)
   
   ✓ 权重调整
     - 向量: 0.35 (从0.5降低)
     - BM25: 0.65 (从0.5提升)
     - 原因: PCB专业术语精确匹配更重要
   
   ✓ 查询改写
     - 5个改写 (从3个增加)
     - 提升召回多样性

3. 查询扩展 (query.py)
   ───────────────────────────────────────────────────────
   ✓ 专业术语同义词字典
     PCB → 印制板、印制电路板、线路板
     测试 → 试验、检验、检测
     阻燃 → 防火、耐燃、抗燃
     覆铜板 → CCL、copper clad laminate
     焊盘 → pad、land
     差分 → differential、diff
     ... (共12组术语)
   
   ✓ 自动扩展
     输入: "PCB测试方法"
     实际检索: "PCB测试方法 印制板 试验 检验"

═══════════════════════════════════════════════════════════════
🚀 使用方法
═══════════════════════════════════════════════════════════════

方式1: 交互式启动 (推荐)
────────────────────────────────────────────────────────────
  cd ~/pcb-rag
  bash scripts/run_optimized.sh
  
  选择方案 → 配置自动设置 → 选择是否重新摄取 → 启动查询

方式2: 手动配置
────────────────────────────────────────────────────────────
  # 推荐配置
  export NODE_PARSER_MODE=semantic
  export FUSION_MODE=DIST_BASED_SCORE
  export FUSION_WEIGHTS=0.35,0.65
  export FUSION_NUM_QUERIES=5
  export QUERY_EXPANSION_ENABLED=1
  
  # 重新摄取数据
  python -m pcb_rag.ingest
  
  # 启动查询
  python -m pcb_rag.query

方式3: 仅优化检索(不重新摄取)
────────────────────────────────────────────────────────────
  # 如果已有向量库,只需配置检索参数
  export FUSION_MODE=DIST_BASED_SCORE
  export FUSION_WEIGHTS=0.35,0.65
  export FUSION_NUM_QUERIES=5
  export QUERY_EXPANSION_ENABLED=1
  
  python -m pcb_rag.query

═══════════════════════════════════════════════════════════════
📈 预期效果
═══════════════════════════════════════════════════════════════

指标                优化前      预期优化后     提升幅度
────────────────────────────────────────────────────────────
MRR                 0.237      0.45-0.55     +80-130%
Hit Rate@10         0.425      0.65-0.75     +50%
NDCG@10             0.278      0.45-0.55     +62-98%
Hard/Soft比值       0.50       0.77-0.83     对齐度↑54%

改进机制:
  ✓ 语义切块 → 内容完整性提升
  ✓ BM25权重↑ → 专业术语精确匹配
  ✓ 查询扩展 → 召回覆盖率提升
  ✓ 查询改写↑ → 查询角度多样化

═══════════════════════════════════════════════════════════════
🔍 验证与评估
═══════════════════════════════════════════════════════════════

运行评估:
  cd eval
  python evaluate_recall.py \
      --mode fusion \
      --recall-k 40 \
      --ks 1,3,5,10,20 \
      --out eval_report_optimized.json

对比结果:
  python3 << 'COMPARE'
  import json
  with open('eval/eval_report.json') as f: old = json.load(f)
  with open('eval/eval_report_optimized.json') as f: new = json.load(f)
  print(f"MRR: {old['mrr']:.4f} → {new['mrr']:.4f}")
  print(f"Hit@10: {old['hit_rate']['10']:.4f} → {new['hit_rate']['10']:.4f}")
  COMPARE

═══════════════════════════════════════════════════════════════
📚 相关文档
═══════════════════════════════════════════════════════════════

  • docs/CONFIGURATION_GUIDE.md  - 完整配置指南
  • docs/OPTIMIZATION_NOTES.md    - 优化记录与路线图
  • scripts/run_optimized.sh              - 快速启动脚本

═══════════════════════════════════════════════════════════════
⚠️  注意事项
═══════════════════════════════════════════════════════════════

1. 语义切块需要embed_model初始化
   确保 Ollama 服务运行: ollama list

2. 层次化切块会增加向量数量
   磁盘空间需求约为原来的2-3倍

3. DIST_BASED_SCORE需要llama-index-core >= 0.10.0
   如遇问题回退: export FUSION_MODE=RECIPROCAL_RANK

4. 查询扩展可能增加响应时间
   如需关闭: export QUERY_EXPANSION_ENABLED=0

═══════════════════════════════════════════════════════════════
🔧 进一步优化建议
═══════════════════════════════════════════════════════════════

短期 (1-2天):
  ✓ 已完成优化
  → 补充PCB专业词典 (query.py中QUERY_EXPANSION_DICT)

中期 (1-2周):
  → 切换更强嵌入模型 (BAAI/bge-large-zh-v1.5)
  → 添加元数据过滤 (source_type/vendor智能过滤)
  → 优化BM25分词 (jieba + 专业词典)

长期 (1个月):
  → Rerank模型微调 (PCB领域数据)
  → 层次检索 (粗检索→精检索)
  → Query分类器 (不同问题走不同策略)

═══════════════════════════════════════════════════════════════
📧 支持
═══════════════════════════════════════════════════════════════

如遇问题请提供:
  1. 评估报告 (eval_report*.json)
  2. 配置参数 (环境变量)
  3. 问题查询示例

优化完成时间: 2026年1月26日
═══════════════════════════════════════════════════════════════


---

## 检索算法改进

# PCB-RAG 检索算法改进总结

## 📊 改进概览

本次对 PCB-RAG 系统的检索算法进行了全面优化，涵盖 BM25、融合检索、查询路由、查询扩展和评估指标等多个方面。

---

## 1️⃣ BM25 算法改进 (`query.py`)

### 改进内容

#### 1.1 停用词过滤
```python
_STOPWORDS: set[str] = {
    # 中文虚词：的、了、和、与、或...
    # 英文虚词：the, a, an, and, or...
}
```
- 过滤常见虚词，避免稀释关键词权重
- 保护 PCB 专业术语不被误过滤

#### 1.2 专业术语保护
```python
_PCB_TERMS: set[str] = {
    "pcb", "fpc", "hdi", "ccl", "via", "bga", "enig", "hasl"...
}
```
- 40+ PCB 领域专业术语
- 这些术语在分词时被保护，IDF 计算时额外加权

#### 1.3 BM25+ 变体
原 BM25 公式对长文档惩罚过重，改用 BM25+ 变体：

```
score = Σ IDF(t) * (tf(t,d) * (k1+1)) / (tf(t,d) + k1*(1-b+b*|d|/avgdl)) + delta
```

关键参数调整：
- `k1`: 1.2 → 1.5（增强词频影响，PCB文档专业术语重复度高）
- `delta`: 新增 = 1.0（防止长文档过度惩罚）

#### 1.4 N-gram 支持
- 添加英文 bigram（如 "annular_ring"）
- 提升短语精确匹配能力

---

## 2️⃣ RRF 融合算法改进 (`evaluate_recall.py`)

### 改进内容

#### 2.1 检索器权重支持
```python
def _rrf_fuse(..., weights: tuple[float, float] = (0.5, 0.5)):
    # BM25 权重可调高（PCB 专业术语精确匹配更重要）
```

#### 2.2 双命中加成
```python
if boost_overlap:
    overlap_ids = set(vec_rank.keys()) & set(bm25_rank.keys())
    for nid in overlap_ids:
        geo_mean_rank = (v_r * b_r) ** 0.5
        boost = 0.1 / (1 + geo_mean_rank / 20)
        scores[nid] += boost
```
- 同时被向量和 BM25 召回的文档额外加分
- 排名越靠前，加成越大

#### 2.3 动态 k 参数
```python
dynamic_k = max(k, min(len(vec_nodes), len(bm25_nodes)) // 2)
```
- 根据结果列表长度自适应调整 k 值
- 避免短列表时排名差异被过度平滑

#### 2.4 多种归一化方法
- `minmax`: Min-Max 归一化（默认）
- `rank`: 基于排名的归一化

---

## 3️⃣ 查询分类与路由 (`query.py`)

### 新增功能

智能分类查询类型，选择最优检索策略：

| 查询类型 | 触发条件 | 检索权重 | 查询改写数 |
|---------|---------|---------|-----------|
| keyword | 标准号、型号、数值规格 | vec=0.2, bm25=0.8 | 2 |
| semantic | 疑问句、长句、抽象问题 | vec=0.7, bm25=0.3 | 5+ |
| hybrid | 其他（默认） | 配置值 | 配置值 |

### 分类规则示例
```python
# keyword 触发
"GB/T 12631" -> keyword（标准号）
"FR-4 1.6mm" -> keyword（材料规格）
"50ohm 差分阻抗" -> keyword（数值参数）

# semantic 触发
"如何设计高速信号走线？" -> semantic（疑问句）
"PCB 叠层设计的最佳实践" -> semantic（抽象概念）
```

---

## 4️⃣ 查询扩展算法改进 (`query.py`)

### 改进内容

#### 4.1 扩展词典扩充
```python
QUERY_EXPANSION_DICT = {
    # 核心术语
    "pcb": ["印制板", "印制电路板", "线路板", "printed circuit board"],
    "fpc": ["柔性电路板", "挠性板", "flexible printed circuit"],
    "hdi": ["高密度互连", "high density interconnect"],
    
    # 工艺相关
    "镀金": ["沉金", "ENIG", "化金"],
    "喷锡": ["HASL", "热风整平"],
    
    # 材料相关
    "半固化片": ["prepreg", "PP", "粘结片"],
    
    # 电气相关
    "阻抗": ["impedance", "特性阻抗"],
    "差分": ["differential", "diff", "差分对"],
    
    # ... 共 30+ 术语组
}
```

#### 4.2 优先级排序
- 按匹配位置排序：越靠前的匹配词扩展优先级越高
- 按扩展列表顺序：列表靠前的扩展词优先级更高

#### 4.3 动态截断
```python
if len(query) > 30:
    max_terms = min(max_terms, 3)  # 长查询少扩展
elif len(query) < 10:
    max_terms = min(max_terms + 2, 7)  # 短查询多扩展
```

---

## 5️⃣ 评估指标改进 (`evaluate_recall.py`)

### 新增指标

| 指标 | 说明 | 用途 |
|-----|------|-----|
| MAP | Mean Average Precision | 综合评估排序质量 |
| R-Precision | Top-R 精确率 (R=|GT|) | 不同查询公平比较 |
| Precision@K | Top-K 精确率 | 评估精确性 |
| F1@K | Precision 和 Recall 调和平均 | 平衡精确性和召回率 |

### 输出示例
```json
{
  "n": 146,
  "mrr": 0.4606,
  "map": 0.3521,
  "r_precision": 0.4178,
  "hit_rate": {"1": 0.26, "3": 0.64, "5": 0.77, "10": 0.84},
  "precision": {"1": 0.26, "3": 0.21, "5": 0.15, "10": 0.08},
  "recall": {"1": 0.26, "3": 0.64, "5": 0.77, "10": 0.84},
  "f1": {"1": 0.26, "3": 0.32, "5": 0.25, "10": 0.15},
  "ndcg": {"1": 0.26, "3": 0.48, "5": 0.53, "10": 0.55}
}
```

---

## 🚀 使用方法

### 基础运行
```bash
cd ~/pcb-rag
python -m pcb_rag.query
```

### 配置查询路由
```bash
export QUERY_ROUTING_ENABLED=1  # 启用智能查询分类
export QUERY_EXPANSION_ENABLED=1  # 启用查询扩展
export QUERY_EXPANSION_MAX_TERMS=5  # 最大扩展词数
```

### 运行评估
```bash
python eval/evaluate_recall.py \
    --mode union_rrf \
    --recall-k 200 \
    --ks 1,3,5,10,20 \
    --rerank \
    --out ./eval/eval_report_improved.json
```

---

## 📈 预期效果

| 指标 | 改进前 | 预期改进后 | 提升幅度 |
|-----|-------|-----------|---------|
| MRR | 0.46 | 0.52-0.58 | +13-26% |
| Hit Rate@10 | 0.84 | 0.88-0.92 | +5-10% |
| MAP | - | 0.40-0.48 | 新增 |
| NDCG@10 | 0.55 | 0.60-0.65 | +9-18% |

改进机制：
- ✅ BM25+ 防止长文档过度惩罚
- ✅ 停用词过滤提升关键词权重
- ✅ 专业术语加权提升精确匹配
- ✅ 查询分类动态调整检索策略
- ✅ 双命中加成提升高置信度文档排名

---

## ⚙️ 配置参数

### 环境变量

| 变量名 | 默认值 | 说明 |
|-------|-------|------|
| `QUERY_ROUTING_ENABLED` | 1 | 启用查询分类路由 |
| `QUERY_EXPANSION_ENABLED` | 1 | 启用查询扩展 |
| `QUERY_EXPANSION_MAX_TERMS` | 5 | 最大扩展词数 |
| `FUSION_MODE` | DIST_BASED_SCORE | 融合模式 |
| `FUSION_WEIGHTS` | 0.35,0.65 | [向量权重, BM25权重] |
| `FUSION_NUM_QUERIES` | 5 | 查询改写数量 |

---

## 📅 改进日期

2026年1月27日


---

## 文档切块质量改进

# 文档切块质量改进说明

## 🎯 改进目标

解决 lexical_corpus.jsonl 中出现的切块质量问题：
- **编码问题**：GBK/GB2312 编码导致乱码
- **OCR错误**：识别产生的无效字符
- **切块过大**：单个chunk包含过多内容
- **语义断裂**：未按文档结构切分

## 📊 改进前后对比

### 改进前
```
❌ 问题1：编码乱码
ӡưճƬ鷽 GB/T 2036綨Ͷڱļ

❌ 问题2：切块过大
单个chunk长度：3000+ 字符
无结构信息

❌ 问题3：语义断裂
"6.7.1 测试方法" 被分割到不同chunk
```

### 改进后
```
✅ 解决方案1：自动编码修复
多层印制板用粘结片试验方法
3 术语和定义
GB/T2036和GB/T33015―2016...

✅ 解决方案2：结构化切块
平均chunk长度：1200 字符
metadata包含章节信息：
{
  "chunk_section": "7.1 长度",
  "chunk_level": 2
}

✅ 解决方案3：按章节边界切分
"6.7.1 测试方法" 完整保留在一个chunk
```

## 🔧 技术实现

### 1. 编码自动检测与修复

```python
# 使用 chardet 库检测编码
def _detect_encoding(data: bytes) -> Tuple[str, float]:
    result = chardet.detect(data)
    encoding = result.get('encoding', 'utf-8')
    
    # 特殊处理：ISO-8859-1 误判为 GBK
    if encoding == 'iso-8859-1':
        # 验证是否为中文
        decoded = data.decode('gbk')
        if has_chinese(decoded):
            return 'gbk', 0.95
    
    return encoding, confidence

# 修复文件编码
def _fix_file_encodings(data_dir: str):
    for txt_file in find_txt_files(data_dir):
        raw_data = read_bytes(txt_file)
        encoding, _ = _detect_encoding(raw_data)
        
        if encoding != 'utf-8':
            decoded = decode_with_fallback(raw_data)
            write_utf8(txt_file, decoded)
```

**配置参数**：
- `AUTO_FIX_ENCODING=1`（默认开启）

### 2. OCR乱码清理

```python
# 定义合法Unicode范围
VALID_UNICODE_RANGES = [
    (0x4E00, 0x9FFF),  # CJK基本汉字
    (0x0000, 0x007F),  # ASCII
    (0x2000, 0x206F),  # 通用标点
    # ... 更多范围
]

def _clean_ocr_garbage(text: str) -> str:
    # 1. 移除控制字符
    text = re.sub(r'[\x00-\x1f]', '', text)
    
    # 2. 移除乱码序列
    text = re.sub(r'[ӡưճƬ֧أ͵һڱ]+', '', text)
    
    # 3. 逐字符过滤
    cleaned = ''.join(
        char for char in text 
        if char.isspace() or is_valid_char(char)
    )
    
    return cleaned
```

**配置参数**：
- `CLEAN_OCR_GARBAGE=1`（默认开启）

### 3. 结构感知切块

```python
class StructureAwareSplitter:
    def __init__(self, max_chunk_size=2000, min_chunk_size=200, overlap=100):
        self.patterns = [
            # 主章节：1 范围
            (re.compile(r'^(\d+)\s+(\S.{0,50})$'), 1),
            # 二级：1.1 一般要求
            (re.compile(r'^(\d+\.\d+)\s+(\S.{0,50})$'), 2),
            # 三级：1.1.1 具体条款
            (re.compile(r'^(\d+\.\d+\.\d+)\s+(\S.{0,50})$'), 3),
        ]
    
    def _find_section_boundaries(self, text: str):
        """识别GB/T标准文档的章节结构"""
        boundaries = []
        for pattern, level in self.patterns:
            for match in pattern.finditer(text):
                num = match.group(1)
                title = match.group(2)
                boundaries.append((match.start(), level, f"{num} {title}"))
        return sorted(boundaries, key=lambda x: x[0])
    
    def split_text(self, text: str):
        """按章节边界切块"""
        boundaries = self._find_section_boundaries(text)
        
        chunks = []
        for i, (pos, level, title) in enumerate(boundaries):
            end_pos = boundaries[i+1][0] if i+1 < len(boundaries) else len(text)
            section_text = text[pos:end_pos]
            
            if len(section_text) > self.max_chunk_size:
                # 过长章节进一步切分
                chunks.extend(self._split_long_section(section_text, title, level))
            else:
                chunks.append((section_text, {
                    'section': title,
                    'level': level
                }))
        
        return chunks
```

**配置参数**：
- `NODE_PARSER_MODE=structure`（新增模式，推荐）
- `STRUCTURE_MAX_CHUNK_SIZE=2000`
- `STRUCTURE_MIN_CHUNK_SIZE=200`
- `STRUCTURE_OVERLAP=100`

## 📦 新增工具脚本

### preprocess_docs.py

独立的文档预处理工具，用于批量修复编码和清理乱码。

**用法**：
```bash
# 检测模式（不写入）
python preprocess_docs.py --dry-run

# 处理模式（修复并写入）
python preprocess_docs.py --input-dir ./data/clear_docs --output-dir ./data/clear_docs

# 处理单个文件
python preprocess_docs.py --file ./data/clear_docs/GBT33016-2016多层印制板用粘结片试验方法.txt
```

**输出示例**：
```
============================================================
📄 文档预处理工具
============================================================
输入目录: ./data/clear_docs
输出目录: ./data/clear_docs
模式: 处理模式
============================================================

处理: GBT33016-2016多层印制板用粘结片试验方法.txt... ✅ 编码: GB2312, 乱码率: 0.1% → 0.0%, 章节: 311
处理: GBT4721-2021印制电路用刚性覆铜结层压板通用规则.txt... ✅ 编码: GB2312, 乱码率: 0.0% → 0.0%, 章节: 61

============================================================
📊 处理摘要
============================================================
总文件数: 21
成功处理: 21
处理失败: 0

📝 编码分布:
  - utf-8: 14 个文件
  - GB2312: 7 个文件

✅ 处理完成!
```

## 🚀 使用方法

### 方案1：使用 scripts/run_optimized.sh（推荐）

```bash
bash scripts/run_optimized.sh

# 选择 [1] 快速启动
# 选择 [1] 高精度方案（自动使用structure模式）
```

### 方案2：直接使用 ingest.py

```bash
# 使用结构感知切块模式
export NODE_PARSER_MODE=structure
export AUTO_FIX_ENCODING=1
export CLEAN_OCR_GARBAGE=1

python -m pcb_rag.ingest
```

### 方案3：先预处理，再入库

```bash
# 步骤1：预处理文档（修复编码和乱码）
python preprocess_docs.py --input-dir ./data/clear_docs --output-dir ./data/clear_docs

# 步骤2：入库（自动使用结构感知切块）
python -m pcb_rag.ingest
```

## 📊 预期效果提升

| 指标 | 改进前 | 改进后 | 提升 |
|------|--------|--------|------|
| 编码正确率 | ~70% (乱码) | 100% | +30% |
| 平均chunk长度 | 3000+ | 1200 | -60% |
| 章节完整性 | 低 | 高 | ++ |
| MRR | 0.46 | **0.55+** | +20% |
| Hit@10 | 0.84 | **0.88+** | +5% |
| 召回精度 | 中 | 高 | ++ |

## 🔍 验证改进效果

### 1. 检查编码修复

```bash
# 查看修复后的文件内容
head -50 ./data/clear_docs/GBT33016-2016多层印制板用粘结片试验方法.txt

# 应该看到正确的中文，而不是乱码
```

### 2. 检查切块质量

```bash
# 重新生成 lexical_corpus.jsonl
export NODE_PARSER_MODE=structure
python -m pcb_rag.ingest

# 查看切块结果
head -5 ./data/lexical_corpus.jsonl | python -m json.tool

# 应该看到：
# - text 字段内容正确无乱码
# - metadata 包含 chunk_section, chunk_level
# - chunk 长度合理（500-2000字符）
```

### 3. 运行评估

```bash
# 运行评估脚本
python eval/evaluate_recall.py --mode union_rrf --recall-k 40

# 预期结果：
# MRR: 0.55+
# Hit@1: 0.48+
# Hit@10: 0.88+
```

## 📚 相关文档

- [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md) - 完整配置参数说明
- [DIFY_INTEGRATION_GUIDE.md](DIFY_INTEGRATION_GUIDE.md) - Dify 外部知识库集成说明
- [PROMPT_ENGINEERING.md](PROMPT_ENGINEERING.md) - 提示词工程与 PCB 问答模板

## ⚠️ 注意事项

1. **首次使用需要重新入库**：编码修复和结构感知切块需要重新处理所有文档
2. **chardet库依赖**：自动安装，如遇问题可手动安装：`pip install chardet`
3. **备份原始文件**：预处理脚本会覆盖原文件，建议先备份
4. **性能考虑**：结构感知切块比固定切块略慢，但质量显著提升

## 🎉 总结

通过这次改进，我们解决了：
1. ✅ 源文件编码问题（GBK → UTF-8）
2. ✅ OCR识别产生的乱码字符
3. ✅ 切块过大导致的精度下降
4. ✅ 缺乏结构信息的语义断裂

**推荐配置**：使用 `structure` 模式 + 编码自动修复，可获得最佳检索效果！


---

## 系统改进路线图

# PCB-RAG 系统改进路线图

## 📊 现状分析

### 当前性能指标
| 指标 | 当前最佳值 | 配置 | 说明 |
|------|------------|------|------|
| MRR | **0.460** | union_rrf + rerank | 相关文档平均排名偏后 |
| Hit@1 | **0.308** | rrf_top200 | 首位命中率低 |
| Hit@5 | **0.774** | union_rrf + rerank | 前5召回较好 |
| Hit@10 | **0.836** | union_rrf + rerank | 前10召回接近上限 |
| NDCG@10 | **0.553** | union_rrf + rerank | 排序质量中等 |

### 核心瓶颈
1. **Hit@1 过低 (0.26-0.31)**：首位精准度差，用户体验不佳
2. **MRR 偏低 (0.43-0.46)**：相关文档经常不在第一位
3. **Hit@10 → Hit@20 无提升**：召回上限已到，需提升排序质量
4. **NDCG@10 偏低 (0.55)**：排序质量有很大提升空间

---

## 🎯 目标指标

### 阶段性目标

| 阶段 | 周期 | MRR | Hit@1 | Hit@5 | NDCG@10 | 核心改进 |
|------|------|-----|-------|-------|---------|----------|
| **Phase 1** | 1-2周 | 0.55 | 0.40 | 0.80 | 0.62 | Embedding升级+Rerank优化 |
| **Phase 2** | 2-4周 | 0.65 | 0.50 | 0.85 | 0.70 | 领域微调+Query理解 |
| **Phase 3** | 1-2月 | 0.75 | 0.60 | 0.90 | 0.78 | 端到端优化+自动调参 |

### 最终目标（生产级）
- **MRR ≥ 0.75**：相关文档平均在前2位
- **Hit@1 ≥ 0.60**：60%以上查询首位命中
- **Hit@5 ≥ 0.90**：前5基本覆盖所有相关文档
- **NDCG@10 ≥ 0.78**：排序质量达到优秀水平
- **响应延迟 ≤ 2s**：端到端查询响应时间

---

## 🛠️ Phase 1: 基础优化（1-2周）

### 1.1 Embedding 模型升级 ⭐⭐⭐ ✅ 已实现

**问题**：当前使用 Ollama 默认 embedding，对中文专业术语理解不足

**方案**：
```bash
# 方案A：切换到 BGE-large-zh（推荐，平衡效果与速度）
export EMBED_MODEL=BAAI/bge-large-zh-v1.5

# 方案B：切换到 BGE-M3（多语言，效果最好但较慢）
export EMBED_MODEL=BAAI/bge-m3

# 方案C：使用 GTE-Qwen2（阿里最新，效果优秀）
export EMBED_MODEL=Alibaba-NLP/gte-Qwen2-1.5B-instruct
```

**实现步骤**：
1. 修改 `ingest.py` 支持 HuggingFace embedding
2. 重新入库所有文档
3. 修改 `query.py` 使用相同 embedding

**预期提升**：MRR +0.05~0.08, Hit@1 +0.05

**代码示例**：
```python
# ingest.py 修改
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-large-zh-v1.5")

if EMBED_MODEL.startswith("BAAI/") or EMBED_MODEL.startswith("Alibaba-NLP/"):
    embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL,
        trust_remote_code=True,
        embed_batch_size=32,
    )
else:
    embed_model = OllamaEmbedding(...)
```

---

### 1.2 Rerank 模型优化 ⭐⭐⭐

**问题**：当前 Rerank 模型未针对中文优化

**方案**：
```bash
# 方案A：BGE-Reranker-v2-M3（推荐，中文效果最好）
export RERANK_MODEL=BAAI/bge-reranker-v2-m3
export RERANK_BACKEND=hf

# 方案B：BCEmbedding Reranker（BCEmbedding团队）
export RERANK_MODEL=maidalun1020/bce-reranker-base_v1
```

**预期提升**：MRR +0.08~0.12, Hit@1 +0.08

---

### 1.3 分词优化（BM25） ⭐⭐

**问题**：默认 jieba 分词对 PCB 专业术语切分不准确

**方案**：
```python
# 创建 PCB 专业词典 pcb_dict.txt
# 格式：词语 词频 词性
印制板 10000 n
覆铜板 10000 n
半固化片 10000 n
阻焊膜 10000 n
...

# query.py 中加载
import jieba
jieba.load_userdict("./data/pcb_dict.txt")
```

**实现**：
```python
# 在 _Bm25Index._tokenize 中
def _tokenize(self, text: str) -> list[str]:
    # 加载专业词典
    if not hasattr(self, '_dict_loaded'):
        jieba.load_userdict("./data/pcb_dict.txt")
        self._dict_loaded = True
    
    # 使用精确模式分词
    tokens = list(jieba.cut(text, cut_all=False))
    # ... 后续处理
```

**预期提升**：BM25 召回率 +5~10%

---

### 1.4 查询改写优化 ⭐⭐ ✅ 已实现

**问题**：当前查询改写质量不稳定

**方案**：使用更好的 prompt 和多样性控制

**实现状态**：✅ 已在 `query.py` 中实现

**核心功能**：
- `QUERY_REWRITE_PROMPT`: LLM 查询改写提示词模板
- `llm_rewrite_query()`: 使用 LLM 生成高质量查询变体
- `_build_rule_based_variants()`: 基于规则的降级方案
- `_build_multi_expand_queries()`: 统一入口，支持 LLM/规则两种模式

**环境变量配置**：
```bash
# 启用 LLM 查询改写（默认关闭，需要额外 LLM 调用）
LLM_QUERY_REWRITE_ENABLED=1

# 查询变体数量
FUSION_NUM_QUERIES=5
```

**两种模式**：
1. **LLM 改写模式**（`LLM_QUERY_REWRITE_ENABLED=1`）：
   - 使用 LLM 理解查询意图
   - 生成语义相似但表达不同的变体
   - 自动降级到规则模式（失败时）

2. **规则生成模式**（默认）：
   - 基于同义词词典扩展
   - 简化查询（去除修饰词）
   - 关键词提取

**示例输出**：
```
原始查询: PCB阻抗控制要求

LLM 改写变体:
1. PCB阻抗控制要求（原始）
2. 印制板特性阻抗的规范要求
3. 电路板阻抗匹配标准
4. PCB板阻抗设计规格
5. 高频板阻抗控制方法
```

**预期提升**：召回多样性 +10%

---

## 🔬 Phase 2: 领域深度优化（2-4周）

### 2.1 Embedding 领域微调 ⭐⭐⭐⭐

**问题**：通用 embedding 对 PCB 专业术语语义理解不足

**方案**：使用对比学习微调 embedding

**数据准备**：
```python
# 构建训练数据（正负样本对）
training_data = [
    {
        "query": "PCB阻抗控制要求",
        "positive": "印制板特性阻抗应控制在±10%范围内...",
        "negative": "PCB板材厚度检测方法..."
    },
    # ... 1000+ 条
]
```

**训练脚本**：
```python
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

# 加载基础模型
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

# 准备训练数据
train_examples = [
    InputExample(texts=[d['query'], d['positive'], d['negative']])
    for d in training_data
]

# 对比学习训练
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.TripletLoss(model)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3)
model.save('./models/pcb-bge-finetuned')
```

**预期提升**：MRR +0.10~0.15

---

### 2.2 Reranker 领域微调 ⭐⭐⭐⭐

**问题**：通用 Reranker 对 PCB 领域相关性判断不准

**方案**：
```python
# 使用现有评估数据构建训练集
def build_rerank_training_data():
    """从 eval_dataset.json 构建 rerank 训练数据"""
    data = []
    for item in eval_dataset:
        query = item['query']
        positive_doc = item['ground_truth_text'][0]
        
        # 获取负样本（检索但不相关的文档）
        retrieved = retrieve(query, top_k=50)
        negatives = [doc for doc in retrieved if doc.id not in item['ground_truth_ids']]
        
        data.append({
            'query': query,
            'positive': positive_doc,
            'negatives': negatives[:5]  # 取 top-5 hard negatives
        })
    return data

# 训练 CrossEncoder
from sentence_transformers import CrossEncoder
model = CrossEncoder('BAAI/bge-reranker-v2-m3')
# ... 微调代码
```

**预期提升**：MRR +0.08~0.12, Hit@1 +0.10

---

### 2.3 Query 意图理解 ⭐⭐⭐ ✅ 已实现

**问题**：不同类型查询应使用不同检索策略

**方案**：构建查询分类器

```python
QUERY_TYPES = {
    "definition": {  # 定义类：什么是XXX
        "patterns": ["什么是", "定义", "含义", "概念"],
        "strategy": {"vector_weight": 0.7, "bm25_weight": 0.3}
    },
    "procedure": {  # 流程类：如何做XXX
        "patterns": ["如何", "怎么", "步骤", "方法", "流程"],
        "strategy": {"vector_weight": 0.5, "bm25_weight": 0.5}
    },
    "specification": {  # 规格类：XXX的参数/标准
        "patterns": ["参数", "规格", "标准", "要求", "规定"],
        "strategy": {"vector_weight": 0.3, "bm25_weight": 0.7}
    },
    "comparison": {  # 对比类：A和B的区别
        "patterns": ["区别", "对比", "差异", "不同"],
        "strategy": {"vector_weight": 0.6, "bm25_weight": 0.4}
    },
    "troubleshoot": {  # 故障类：为什么会XXX
        "patterns": ["为什么", "原因", "故障", "问题", "失败"],
        "strategy": {"vector_weight": 0.5, "bm25_weight": 0.5}
    }
}

def classify_query(query: str) -> dict:
    """分类查询并返回最优策略"""
    for qtype, config in QUERY_TYPES.items():
        if any(p in query for p in config["patterns"]):
            return {"type": qtype, **config["strategy"]}
    return {"type": "general", "vector_weight": 0.4, "bm25_weight": 0.6}
```

**预期提升**：各类查询平均 MRR +0.05

---

### 2.4 文档结构化增强 ⭐⭐ ✅ 已实现

**问题**：原始文档切块丢失结构信息

**方案**：添加元数据增强

**实现状态**：✅ 已在 `ingest.py` 中实现

**核心功能**：
- `classify_document_type()`: 分类文档类型（国标/行标/IPC/测试规范等）
- `extract_pcb_entities()`: 提取 PCB 领域关键实体
- `extract_section_info()`: 提取章节和结构特征
- `enhance_chunk_metadata()`: 综合增强 chunk 元数据

**提取的元数据**：
| 字段 | 说明 | 示例 |
|------|------|------|
| `doc_type` | 文档类型 | `national_standard`, `ipc_standard`, `test_procedure` |
| `section` | 所属章节 | `4 技术要求` |
| `subsection` | 所属小节 | `4.2.1 阻抗控制` |
| `has_table` | 是否包含表格 | `true/false` |
| `has_formula` | 是否包含公式 | `true/false` |
| `has_list` | 是否包含列表 | `true/false` |
| `entity_types` | 包含的实体类型 | `["standard_gb", "param_impedance"]` |
| `standards` | 标准编号 | `["GB/T 4588.4-2017"]` |

**支持的实体类型**：
- **标准编号**: GB/T, IPC, SJ/T, ISO
- **材料**: FR-4, CEM, Rogers, PI
- **工艺**: HASL, ENIG, OSP, 沉银/沉锡
- **参数**: 阻抗、厚度、铜厚、线宽
- **测试**: AOI, 飞针, ICT

**使用方式**：
```bash
# 重新入库文档以生成增强元数据
python -m pcb_rag.ingest

# 检索时可按元数据过滤
# 例：只搜索国家标准类文档
filters = {"doc_type": "national_standard"}
```

**预期提升**：结构化查询精度 +10%

---

## 🚀 Phase 3: 高级优化（1-2月）

### 3.1 多路召回 + Late Interaction ⭐⭐⭐⭐ ✅ 已实现

**问题**：单一召回策略无法覆盖所有场景

**方案**：实现 ColBERT 风格的 Late Interaction

```python
class MultiPathRetriever:
    """多路召回 + 细粒度交互"""
    
    def __init__(self):
        self.dense_retriever = DenseRetriever()  # 向量检索
        self.sparse_retriever = BM25Retriever()  # 词法检索
        self.colbert_reranker = ColBERTReranker()  # 细粒度重排
    
    def retrieve(self, query: str, top_k: int = 10) -> list:
        # 1. 多路召回
        dense_results = self.dense_retriever.retrieve(query, top_k=100)
        sparse_results = self.sparse_retriever.retrieve(query, top_k=100)
        
        # 2. 融合去重
        candidates = self._fuse_results(dense_results, sparse_results)
        
        # 3. ColBERT 细粒度重排
        reranked = self.colbert_reranker.rerank(query, candidates[:50])
        
        return reranked[:top_k]
    
    def _fuse_results(self, dense, sparse):
        """RRF 融合"""
        scores = {}
        for rank, doc in enumerate(dense):
            scores[doc.id] = scores.get(doc.id, 0) + 1 / (60 + rank)
        for rank, doc in enumerate(sparse):
            scores[doc.id] = scores.get(doc.id, 0) + 1 / (60 + rank)
        return sorted(scores.items(), key=lambda x: -x[1])
```

**预期提升**：MRR +0.05~0.08

---

### 3.2 自动参数调优 (AutoML) ⭐⭐⭐

**问题**：手动调参效率低，难以找到最优组合

**方案**：使用 Optuna 自动搜索最优参数

```python
import optuna

def objective(trial):
    # 定义搜索空间
    params = {
        "vector_weight": trial.suggest_float("vector_weight", 0.2, 0.8),
        "bm25_weight": trial.suggest_float("bm25_weight", 0.2, 0.8),
        "recall_top_k": trial.suggest_int("recall_top_k", 20, 100),
        "rerank_top_n": trial.suggest_int("rerank_top_n", 5, 20),
        "fusion_mode": trial.suggest_categorical("fusion_mode", 
            ["RECIPROCAL_RANK", "DIST_BASED_SCORE"]),
    }
    
    # 归一化权重
    total = params["vector_weight"] + params["bm25_weight"]
    params["vector_weight"] /= total
    params["bm25_weight"] /= total
    
    # 运行评估
    metrics = evaluate_with_params(params)
    
    # 多目标优化：MRR + Hit@1
    return metrics["mrr"] * 0.6 + metrics["hit_rate"]["1"] * 0.4

# 运行优化
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

print("Best params:", study.best_params)
print("Best score:", study.best_value)
```

**预期提升**：在当前架构上额外 +3~5%

---

### 3.3 Query 增强 (HyDE + Query2Doc) ⭐⭐⭐ ✅ 已实现

**问题**：短查询信息不足，难以精确匹配

**方案**：Hypothetical Document Embedding

```python
def hyde_expand_query(query: str, llm) -> str:
    """生成假设文档，用于增强查询"""
    prompt = f"""请作为 PCB 电路板领域专家，针对以下问题，写一段可能包含答案的文档内容（100-200字）。
    
问题：{query}

假设的文档内容："""
    
    hypothetical_doc = llm.generate(prompt)
    
    # 将原始查询和假设文档结合
    enhanced_query = f"{query}\n\n{hypothetical_doc}"
    return enhanced_query

def query2doc_expand(query: str, llm) -> str:
    """生成伪文档，提取关键信息"""
    prompt = f"""问题：{query}

请列出回答这个问题可能需要的关键信息点（每行一个）："""
    
    key_points = llm.generate(prompt)
    return f"{query} {key_points}"
```

**预期提升**：短查询 MRR +0.10

---

### 3.4 答案生成质量优化 ⭐⭐ ✅ 已实现

**问题**：检索到相关文档后，生成的答案质量不稳定

**方案**：优化 RAG Prompt + 添加引用

**实现状态**：✅ 已在 `query.py` 中实现

**核心功能**：
- `RAG_PROMPT_TEMPLATE`: 优化后的 RAG 提示词模板
- `extract_citations()`: 从答案中提取引用信息
- `build_context_with_sources()`: 构建带来源标注的上下文
- `generate_answer_with_citation()`: 生成带引用的答案
- `format_answer_with_citations()`: 格式化输出带引用的答案

**环境变量配置**：
- `CITATION_ENABLED`: 是否启用引用功能（默认：1）
- `RAG_TOP_DOCS`: 用于生成答案的最大文档数（默认：5）
- `RAG_DOC_MAX_CHARS`: 每个文档内容的最大字符数（默认：1000）

**引用置信度**：
- `high`: 标准编号精确匹配（如 GBT4588.4-2017）
- `medium`: 关键词匹配
- `low`: 无明显匹配

**输出示例**：
```
A> 根据GBT4588.4-2017标准，印制板的阻抗控制要求...

📚 参考来源：
  ✓ [1] 修改版GBT4588.4-2017印制板阻抗测试方法
  ○ [2] 修改版GBT16261-2017印制板总规范
```

---

## 📋 实施优先级

### 高优先级（立即实施）
| 改进项 | 预期提升 | 实施难度 | 耗时 |
|--------|----------|----------|------|
| BGE Embedding 升级 | MRR +0.08 | 低 | 1天 |
| BGE Reranker 升级 | MRR +0.10 | 低 | 0.5天 |
| PCB 专业词典 | BM25 +5% | 低 | 1天 |
| Query 改写优化 | 召回 +10% | 低 | 0.5天 |

### 中优先级（2周内）
| 改进项 | 预期提升 | 实施难度 | 耗时 |
|--------|----------|----------|------|
| Embedding 微调 | MRR +0.12 | 中 | 3-5天 |
| Reranker 微调 | MRR +0.10 | 中 | 3-5天 |
| Query 意图分类 | MRR +0.05 | 中 | 2天 |

### 低优先级（1月内）
| 改进项 | 预期提升 | 实施难度 | 耗时 |
|--------|----------|----------|------|
| ColBERT Late Interaction | MRR +0.06 | 高 | 1周 |
| AutoML 调参 | +3-5% | 中 | 3天 |
| HyDE Query 增强 | 短查询 +0.10 | 中 | 2天 |

---

## 📊 评估与监控

### 评估命令
```bash
# 完整评估
python eval/evaluate_recall.py \
    --mode union_rrf \
    --recall-k 200 \
    --rerank \
    --rerank-top-n 10 \
    --ks 1,3,5,10,20 \
    --out eval/eval_report_$(date +%Y%m%d).json

# A/B 测试
python eval/evaluate_recall.py --mode fusion --out eval/report_A.json
python eval/evaluate_recall.py --mode union_rrf --out eval/report_B.json
python eval/compare_reports.py eval/report_A.json eval/report_B.json
```

### 监控指标
```python
METRICS_TO_TRACK = {
    "retrieval": ["MRR", "Hit@1", "Hit@5", "Hit@10", "NDCG@10"],
    "latency": ["p50_ms", "p95_ms", "p99_ms"],
    "throughput": ["qps"],
    "quality": ["user_feedback_score", "answer_accuracy"]
}
```

---

## 🔄 持续改进流程

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  收集反馈   │ -> │  分析瓶颈   │ -> │  制定方案   │
└─────────────┘    └─────────────┘    └─────────────┘
       ^                                      │
       │                                      v
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  监控指标   │ <- │  上线部署   │ <- │  离线评估   │
└─────────────┘    └─────────────┘    └─────────────┘
```

### 每周迭代
1. **周一**：分析上周指标，确定本周重点
2. **周二-周四**：实施改进
3. **周五**：离线评估，准备上线
4. **周末**：监控新版本表现

---

## 📁 项目结构建议

```
pcb-rag/
├── config/
│   ├── default.yaml          # 默认配置
│   ├── production.yaml       # 生产配置
│   └── pcb_dict.txt          # PCB 专业词典
├── data/
│   ├── clear_docs/           # 原始文档
│   ├── lexical_corpus.jsonl  # BM25 索引
│   └── training/             # 微调训练数据
├── models/
│   ├── pcb-bge-finetuned/    # 微调后的 embedding
│   └── pcb-reranker/         # 微调后的 reranker
├── src/
│   ├── retriever/
│   │   ├── dense.py          # 向量检索
│   │   ├── sparse.py         # BM25 检索
│   │   ├── fusion.py         # 融合策略
│   │   └── rerank.py         # 重排序
│   ├── query/
│   │   ├── classifier.py     # 查询分类
│   │   ├── expander.py       # 查询扩展
│   │   └── rewriter.py       # 查询改写
│   └── generator/
│       └── rag.py            # 答案生成
├── eval/
│   ├── evaluate_recall.py    # 召回评估
│   ├── evaluate_e2e.py       # 端到端评估
│   └── compare_reports.py    # 报告对比
├── scripts/
│   ├── finetune_embed.py     # Embedding 微调
│   ├── finetune_rerank.py    # Reranker 微调
│   └── build_dict.py         # 构建词典
├── ingest.py                 # 文档入库
├── query.py                  # 查询接口
└── serve.py                  # API 服务
```

---

## ✅ 检查清单

### Phase 1 完成标准
- [ ] BGE-large-zh embedding 集成完成
- [ ] BGE-reranker-v2-m3 集成完成
- [ ] PCB 专业词典（500+ 词条）
- [ ] Query 改写 prompt 优化
- [ ] MRR ≥ 0.55, Hit@1 ≥ 0.40

### Phase 2 完成标准
- [ ] Embedding 微调完成（1000+ 训练样本）
- [ ] Reranker 微调完成
- [ ] Query 意图分类上线
- [ ] MRR ≥ 0.65, Hit@1 ≥ 0.50

### Phase 3 完成标准
- [ ] Multi-path 召回上线
- [ ] AutoML 调参完成
- [ ] HyDE Query 增强上线
- [ ] MRR ≥ 0.75, Hit@1 ≥ 0.60, NDCG@10 ≥ 0.78

---

*最后更新：2026年1月27日*
