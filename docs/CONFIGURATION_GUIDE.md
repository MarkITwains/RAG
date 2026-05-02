# PCB-RAG 配置参数完全指南

本文档详细说明 PCB-RAG 系统的所有可配置参数，帮助您根据实际需求进行最优配置。

---

## 📑 目录

1. [快速开始](#快速开始)
2. [服务配置](#服务配置)
3. [文档入库配置 (ingest.py)](#文档入库配置)
4. [检索查询配置 (query.py)](#检索查询配置)
5. [评估配置 (evaluate_recall.py)](#评估配置)
6. [推荐配置方案](#推荐配置方案)
7. [故障排除](#故障排除)

---

## 🚀 快速开始

### 最小配置（开箱即用）
```bash
# 只需确保 Ollama 和 Docker 可用
bash scripts/setup.sh

# 入库
bash scripts/run_ingest.sh

# 查询
bash scripts/run_query.sh
```

### 推荐配置（生产环境）
```bash
# 创建配置文件
cat > .env << 'EOF'
# 模型配置
OLLAMA_LLM_MODEL=glm-4.7-flash:q8_0
OLLAMA_NUM_CTX=4096

# 检索配置
FUSION_MODE=DIST_BASED_SCORE
FUSION_WEIGHTS=0.35,0.65
RECALL_TOP_K=40

# Rerank 配置
RERANK_ENABLED=1
RERANK_TOP_N=8

# 查询增强
QUERY_EXPANSION_ENABLED=1
QUERY_ROUTING_ENABLED=1
EOF

# 加载配置
source .env
```

---

## 🔧 服务配置

### Ollama 服务

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `OLLAMA_BASE` | `http://127.0.0.1:11434` | Ollama API 地址 |
| `OLLAMA_LLM_MODEL` | 自动选择 | 生成回答的 LLM 模型 |
| `OLLAMA_NUM_CTX` | `2048` | LLM 上下文窗口大小 |
| `OLLAMA_TIMEOUT` | `120` | API 请求超时（秒）|

**示例：**
```bash
export OLLAMA_LLM_MODEL="glm-4.7-flash:q8_0"
export OLLAMA_NUM_CTX=4096  # 处理长文档时增大
```

### Milvus 向量数据库

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `MILVUS_URI` | `http://127.0.0.1:19530` | Milvus 连接地址 |

---

## 📥 文档入库配置

### 切块策略 (`NODE_PARSER_MODE`)

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `structure` | 结构感知切块（**推荐**） | GB/T标准文档、有章节结构的文档 |
| `semantic` | 语义感知切块 | 技术文档、标准规范 |
| `hierarchical` | 层次化切块（三级） | 结构化文档、手册 |
| `sentence` | 固定大小切块 | 通用场景、快速入库 |

#### 结构感知切块参数（structure模式）

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `STRUCTURE_MAX_CHUNK_SIZE` | `2000` | 单个chunk最大字符数 |
| `STRUCTURE_MIN_CHUNK_SIZE` | `200` | 单个chunk最小字符数（太短会合并） |
| `STRUCTURE_OVERLAP` | `100` | chunk之间的重叠字符数 |

**特点：**
- 自动识别GB/T标准文档的章节结构（如 "6.7.1 测试方法"）
- 按章节边界切块，保持语义完整性
- 在metadata中记录章节信息（`chunk_section`, `chunk_level`）

#### 语义切块参数

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `NODE_PARSER_MODE` | `structure` | 切块模式 |
| `SEMANTIC_BUFFER_SIZE` | `1` | 句子缓冲数量 |
| `SEMANTIC_BREAKPOINT_THRESHOLD` | `90` | 相似度阈值百分位（0-100） |
| `SEMANTIC_CHUNK_SIZE` | `3000` | 单个 chunk 最大字符数 |

```bash
# 语义切块配置示例
export NODE_PARSER_MODE=semantic
export SEMANTIC_BUFFER_SIZE=1
export SEMANTIC_BREAKPOINT_THRESHOLD=90
export SEMANTIC_CHUNK_SIZE=3000
```

### 编码和文本清洗

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `AUTO_FIX_ENCODING` | `1` | 自动检测并修复文件编码（GBK→UTF-8） |
| `CLEAN_OCR_GARBAGE` | `1` | 清理OCR乱码字符 |
| `NORMALIZE_TEXT` | `1` | 规范化文本（空白符、换行等） |

```bash
# 编码修复配置（推荐保持默认）
export AUTO_FIX_ENCODING=1    # 自动修复GBK/GB2312编码
export CLEAN_OCR_GARBAGE=1    # 清理OCR识别错误产生的乱码
export NORMALIZE_TEXT=1       # 规范化空白符
```

#### 层次化切块参数

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `HIERARCHICAL_CHUNK_SIZES` | `1024,384,128` | 三级 chunk 大小 |

```bash
# 层次化切块配置示例
export NODE_PARSER_MODE=hierarchical
export HIERARCHICAL_CHUNK_SIZES=1024,384,128
```

#### 固定切块参数

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `CHUNK_SIZE` | `384` | chunk 大小 |
| `CHUNK_OVERLAP` | `150` | chunk 重叠字符数 |

```bash
# 固定切块配置示例
export NODE_PARSER_MODE=sentence
export CHUNK_SIZE=384
export CHUNK_OVERLAP=150
```

### 文本预处理

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `NORMALIZE_TEXT` | `1` | 是否清洗文本（去除多余空白） |

---

## 🔍 检索查询配置

### 混合检索（Fusion）

| 变量名 | 默认值 | 可选值 | 说明 |
|--------|--------|--------|------|
| `FUSION_MODE` | `DIST_BASED_SCORE` | `RECIPROCAL_RANK`, `DIST_BASED_SCORE`, `RELATIVE_SCORE` | 融合模式 |
| `FUSION_WEIGHTS` | `0.35,0.65` | `0.0-1.0,0.0-1.0` | [向量权重, BM25权重] |
| `FUSION_NUM_QUERIES` | `5` | 1-10 | 查询改写数量 |
| `RECALL_TOP_K` | `40` | 10-200 | 召回文档数量 |

**融合模式说明：**
- `RECIPROCAL_RANK`：RRF 倒数排名融合，对分数不敏感
- `DIST_BASED_SCORE`：加权分数融合（推荐），支持权重调节
- `RELATIVE_SCORE`：相对分数融合

```bash
# 高召回配置（BM25 主导）
export FUSION_MODE=DIST_BASED_SCORE
export FUSION_WEIGHTS=0.3,0.7
export FUSION_NUM_QUERIES=5
export RECALL_TOP_K=50
```

### 词法检索（BM25）

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `LEXICAL_ENABLED` | `1` | 是否启用 BM25 检索 |
| `LEXICAL_CACHE_PATH` | `./data/lexical_corpus.jsonl` | 词法索引缓存路径 |
| `LEXICAL_MAX_BYTES` | `8000` | 单文档最大字节数 |
| `LEXICAL_EXPORT_BATCH` | `2000` | 导出批次大小 |
| `LEXICAL_EXPORT_LIMIT` | `-1` | 导出数量限制（-1=全部） |

### 查询增强

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `QUERY_EXPANSION_ENABLED` | `1` | 是否启用同义词扩展 |
| `QUERY_EXPANSION_MAX_TERMS` | `5` | 最大扩展词数量 |
| `QUERY_ROUTING_ENABLED` | `1` | 是否启用智能查询分类 |

**查询分类说明：**
启用后，系统会自动判断查询类型并调整检索策略：
- `keyword` 查询 → BM25 权重提高到 0.8
- `semantic` 查询 → 向量权重提高到 0.7
- `hybrid` 查询 → 使用配置的默认权重

```bash
# 启用全部查询增强
export QUERY_EXPANSION_ENABLED=1
export QUERY_EXPANSION_MAX_TERMS=5
export QUERY_ROUTING_ENABLED=1
```

### 重排序（Rerank）

| 变量名 | 默认值 | 可选值 | 说明 |
|--------|--------|--------|------|
| `RERANK_ENABLED` | `1` | `0/1` | 是否启用重排序 |
| `RERANK_BACKEND` | `sbert` | `sbert`, `hf`, `qwen3vl` | 重排序后端 |
| `RERANK_MODEL` | `Qwen/Qwen3-VL-Reranker-2B` | - | 重排序模型 |
| `RERANK_TOP_N` | `8` | 1-20 | 重排序后保留的文档数 |

**重排序后端说明：**
- `sbert`：SentenceTransformer cross-encoder（快速，适合 CPU）
- `hf`：HuggingFace Transformers 直接加载
- `qwen3vl`：Qwen3-VL-Reranker 官方脚本（推荐，效果最好）

```bash
# 使用 Qwen3-VL Reranker
export RERANK_ENABLED=1
export RERANK_BACKEND=qwen3vl
export RERANK_MODEL=Qwen/Qwen3-VL-Reranker-2B
export RERANK_TOP_N=8
```

#### HuggingFace Rerank 详细配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `HF_RERANK_MODEL` | `Qwen/Qwen3-VL-Reranker-2B` | 模型 ID |
| `HF_RERANK_MAX_LENGTH` | `512` | 最大 token 长度 |
| `HF_RERANK_BATCH_SIZE` | `16` | 批处理大小 |
| `HF_RERANK_DEVICE` | `auto` | `auto/cuda/cpu` |
| `HF_RERANK_DTYPE` | `auto` | `auto/fp16/bf16/fp32` |
| `HF_DOWNLOAD_MIN_INTERVAL` | `1` | 下载进度刷新间隔（秒） |
| `HF_PREFETCH_SNAPSHOT` | `1` | 是否预下载模型 |

#### Qwen3-VL Rerank 详细配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `QWEN3VL_MAX_LENGTH` | `512` | 最大 token 长度 |
| `QWEN3VL_RERANK_INSTRUCTION` | `Given a search query...` | 重排序指令 |
| `QWEN3VL_ATTN_IMPL` | `` | 注意力实现（如 `flash_attention_2`） |

---

## 📊 评估配置

### 运行评估

```bash
python eval/evaluate_recall.py [OPTIONS]
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | `./eval/eval_dataset.json` | 评估数据集路径 |
| `--out` | `./eval/eval_report.json` | 输出报告路径 |
| `--llm` | 环境变量 | LLM 模型名称 |
| `--mode` | `fusion` | 检索模式 |
| `--recall-k` | `40` | 召回数量 |
| `--ks` | `1,3,5,10,20` | 评估的 K 值列表 |
| `--limit` | `0` | 评估条目限制（0=全部） |
| `--rerank` | `false` | 是否启用重排序 |
| `--rerank-top-n` | `0` | 重排序保留数量 |
| `--rerank-range` | `` | 重排序范围（如 `100,200`） |

### 检索模式 (`--mode`)

| 模式 | 说明 |
|------|------|
| `vector` | 仅向量检索 |
| `bm25` | 仅 BM25 检索 |
| `fusion` | 融合检索（RRF，固定 num_queries=3） |
| `fusion_rerank` | 融合检索 + 全局重排序 |
| `fusion_expand` | 融合检索 + 多查询扩展 |
| `fusion_score` | 分数归一化融合 |
| `union` | BM25+向量并集去重 |
| `rrf` | BM25+向量 RRF |
| `union_rrf` | 并集去重后分数融合（含多查询扩展） |

### Soft Match 配置

| 参数 | 环境变量 | 默认值 | 说明 |
|------|----------|--------|------|
| `--soft-match` | `SOFT_MATCH` | `none` | `none/llm/embed` |
| `--soft-llm-base-url` | `OLLAMA_BASE` | `http://127.0.0.1:11434` | LLM 服务地址 |
| `--soft-llm-model` | `OLLAMA_EVAL_JUDGE_MODEL` | `` | 判断模型 |
| `--soft-llm-timeout` | `OLLAMA_EVAL_JUDGE_TIMEOUT` | `60` | 超时（秒） |
| `--soft-cache` | `SOFT_MATCH_CACHE` | `./eval/softmatch_cache.jsonl` | 缓存路径 |
| `--embed-threshold` | `EMBED_MATCH_THRESHOLD` | `0.78` | Embedding 匹配阈值 |

```bash
# 完整评估示例
python eval/evaluate_recall.py \
    --mode union_rrf \
    --recall-k 200 \
    --ks 1,3,5,10,20 \
    --rerank \
    --rerank-top-n 10 \
    --soft-match embed \
    --embed-threshold 0.78 \
    --out ./eval/eval_report_full.json
```

### 构建测试集

```bash
python eval/build_golden_dataset.py [OPTIONS]
```

| 参数 | 环境变量 | 默认值 | 说明 |
|------|----------|--------|------|
| `--corpus` | `LEXICAL_CACHE_PATH` | `./data/lexical_corpus.jsonl` | 语料路径 |
| `--base-url` | `OLLAMA_BASE` | `http://127.0.0.1:11434` | Ollama 地址 |
| `--model` | `OLLAMA_EVAL_GEN_MODEL` | `` | 问题生成模型 |
| `--per-chunk` | `EVAL_QUERIES_PER_CHUNK` | `2` | 每个 chunk 生成的问题数 |
| `--limit` | `EVAL_CHUNK_LIMIT` | `200` | chunk 数量限制 |
| `--out` | - | `./eval/eval_dataset.json` | 输出路径 |

---

## 💡 推荐配置方案

### 方案 A：高精度配置（追求最佳效果）

```bash
# 入库配置
export NODE_PARSER_MODE=semantic
export SEMANTIC_CHUNK_SIZE=2000
export SEMANTIC_BREAKPOINT_THRESHOLD=85

# 检索配置
export FUSION_MODE=DIST_BASED_SCORE
export FUSION_WEIGHTS=0.35,0.65
export FUSION_NUM_QUERIES=5
export RECALL_TOP_K=50

# 查询增强
export QUERY_EXPANSION_ENABLED=1
export QUERY_ROUTING_ENABLED=1

# 重排序
export RERANK_ENABLED=1
export RERANK_BACKEND=qwen3vl
export RERANK_TOP_N=10
```

### 方案 B：高速度配置（追求响应速度）

```bash
# 入库配置
export NODE_PARSER_MODE=sentence
export CHUNK_SIZE=512
export CHUNK_OVERLAP=100

# 检索配置
export FUSION_MODE=RECIPROCAL_RANK
export FUSION_NUM_QUERIES=3
export RECALL_TOP_K=20

# 查询增强
export QUERY_EXPANSION_ENABLED=1
export QUERY_ROUTING_ENABLED=0

# 重排序（关闭或使用轻量模型）
export RERANK_ENABLED=0
```

### 方案 C：资源受限配置（低显存/CPU 环境）

```bash
# 入库配置
export NODE_PARSER_MODE=sentence
export CHUNK_SIZE=384
export CHUNK_OVERLAP=150

# 检索配置
export FUSION_MODE=RECIPROCAL_RANK
export FUSION_NUM_QUERIES=3
export RECALL_TOP_K=30
export LEXICAL_ENABLED=1

# 查询增强
export QUERY_EXPANSION_ENABLED=1
export QUERY_ROUTING_ENABLED=1

# 重排序（使用 CPU 友好的模型）
export RERANK_ENABLED=1
export RERANK_BACKEND=sbert
export RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
export RERANK_TOP_N=5
```

### 方案 D：标准/规范查询优化

```bash
# 针对 PCB 标准文档的特殊配置
export FUSION_MODE=DIST_BASED_SCORE
export FUSION_WEIGHTS=0.25,0.75  # BM25 权重更高
export QUERY_EXPANSION_ENABLED=1
export QUERY_ROUTING_ENABLED=1
```

---

## 🔧 故障排除

### 常见问题

#### 1. Milvus 连接失败
```bash
# 检查 Milvus 服务状态
docker ps | grep milvus

# 重启 Milvus
cd milvus && docker-compose restart
```

#### 2. Ollama 模型加载失败
```bash
# 检查可用模型
ollama list

# 拉取模型
ollama pull glm-4.7-flash:q8_0
```

#### 3. 显存不足
```bash
# 方案1：使用较小的 rerank 模型
export RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# 方案2：关闭 rerank
export RERANK_ENABLED=0

# 方案3：减小 batch size
export HF_RERANK_BATCH_SIZE=4
```

#### 4. 检索结果不相关
```bash
# 增加召回数量
export RECALL_TOP_K=100

# 调整融合权重（增加 BM25 权重）
export FUSION_WEIGHTS=0.2,0.8

# 启用查询扩展
export QUERY_EXPANSION_ENABLED=1
```

#### 5. 检索速度慢
```bash
# 减少召回数量
export RECALL_TOP_K=20

# 减少查询改写
export FUSION_NUM_QUERIES=2

# 关闭 rerank
export RERANK_ENABLED=0
```

### 性能调优建议

| 场景 | 关键参数 | 建议值 |
|------|----------|--------|
| 提高召回率 | `RECALL_TOP_K` | 50-100 |
| 提高精确度 | `RERANK_TOP_N` | 5-10 |
| 加快响应 | `FUSION_NUM_QUERIES` | 2-3 |
| 专业术语匹配 | `FUSION_WEIGHTS` | `0.25,0.75` |
| 语义理解 | `FUSION_WEIGHTS` | `0.6,0.4` |

---

## 📋 环境变量速查表

### 核心配置
```bash
# Ollama
OLLAMA_BASE=http://127.0.0.1:11434
OLLAMA_LLM_MODEL=glm-4.7-flash:q8_0
OLLAMA_NUM_CTX=4096
OLLAMA_TIMEOUT=120

# 检索
FUSION_MODE=DIST_BASED_SCORE
FUSION_WEIGHTS=0.35,0.65
FUSION_NUM_QUERIES=5
RECALL_TOP_K=40

# 查询增强
QUERY_EXPANSION_ENABLED=1
QUERY_ROUTING_ENABLED=1

# 重排序
RERANK_ENABLED=1
RERANK_BACKEND=qwen3vl
RERANK_TOP_N=8
```

### 入库配置
```bash
# 切块
NODE_PARSER_MODE=semantic
SEMANTIC_BUFFER_SIZE=1
SEMANTIC_BREAKPOINT_THRESHOLD=90
SEMANTIC_CHUNK_SIZE=3000

# 或使用固定切块
NODE_PARSER_MODE=sentence
CHUNK_SIZE=384
CHUNK_OVERLAP=150
```

### 评估配置
```bash
# Soft Match
SOFT_MATCH=embed
EMBED_MATCH_THRESHOLD=0.78

# 或使用 LLM Judge
SOFT_MATCH=llm
OLLAMA_EVAL_JUDGE_MODEL=glm-4.7-flash:q8_0
```

---

*最后更新：2026年1月27日*
