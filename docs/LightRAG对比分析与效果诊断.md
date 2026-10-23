# LightRAG 与 PCB-RAG 对比分析及效果诊断

**分析对象**：HKUDS/LightRAG `aaebfe1`（v1.5.8，2026-10-14）
**对照实现**：本仓库 `src/pcb_rag`（生产配置 `COLLECTION=pcb_kb_nv_query_passage_20261020`）
**数据来源**：`eval/reports/r1_*.json`、`r2_00`–`r2_12`、`embedding_query_passage_20261020.json`、`data/lexical_corpus_*.jsonl`、`.env`

---

## 一、结论摘要

本仓库的检索质量瓶颈不在融合算法，也不在查询扩展的层数，而在两处更基础的地方：**向量索引的 embedding 契约错误**，以及**评测基线建立在错误索引之上**。

已经发生的事实是：`nvidia/nemotron-3-embed-1b` 属于非对称 embedding，query 与 passage 必须以不同 `input_type` 编码。旧集合两侧编码方式一致，导致 query 向量与文档向量落在两个错位的子空间。配对实验显示，仅修正该项（`COLLECTION` 从 `pcb_kb` 换成 `pcb_kb_nv_query_passage_20261020`、`EMBED_API_PROVIDER=nvidia`、`EMBED_TRUNCATE=END`、query 侧走 `input_type=query`），纯向量排序的 MRR@20 由 0.068966 升至 0.393509，Hit@10 由 0.1625 升至 0.6375。

而 `r2_01`–`r2_08` 这八组消融实验全部在修正前的索引上运行，其向量路 MRR 与 `embedding_query_passage` 报告的 `before` 值同为 0.069。这批实验得出的「HyDE 负收益」「多查询扩展负收益」「路由负收益」结论，其归因对象是失真的向量召回路，不能作为功能取舍的依据。**在这批结论被重跑替换之前，不宜据此删除任何检索组件。**

与 LightRAG 的差距不在功能清单的长度，而在三个结构性选择：索引的语义单元（概念 vs 文本片段）、融合的失败模式（无分数依赖 vs RRF 加权）、上下文的预算方式（分段 30000 token vs 固定 5×1000 字符）。本仓库另有图检索模块，但 `GRAPH_RAG_ENABLED=0` 且 `data/graph/kg.json` 不存在，该能力当前未进入链路。

---

## 二、LightRAG 的核心机制

### 2.1 索引构建

LightRAG 的入库不是「切块 + 向量化」，而是「切块 + 用 LLM 把每个 chunk 拆成概念和关系 + 建三套索引」。

#### 阶段一：切块

按 token 计数切分，而非字符。默认 1200 tokens、overlap 100。token 计数用模型自带 tokenizer（`lightrag/utils.py` 的 `TiktokenTokenizer`），因此块长与 LLM 的实际上下文占用一致。可切换 semantic chunker（`lightrag/chunker/`），但默认路径是纯 token 窗口。

#### 阶段二：LLM 实体抽取

`operate.py:extract_entities`（3942 行起）对每个 chunk 发起抽取调用。提示词见 `prompt.py:56` 的 `entity_extraction_system_prompt`，输出两种记录：

```
entity<|#|>entity_name<|#|>entity_type<|#|>entity_description
relation<|#|>source_entity<|#|>target_entity<|#|>relationship_keywords<|#|>relationship_description
```

提示词中有四处设计值得注意：

- **实体类型表**：`default_entity_types_guidance` 给出 Person / Organization / Concept / Method / Data / Artifact 等 11 类，要求无法归类时落到 `Other`。类型是后续检索的路由依据之一，不是装饰字段。
- **一致性约束**：「Ensure consistent naming across the entire extraction process」。同一概念在不同 chunk 里的写法必须归一，否则后续合并会把同一实体拆成多个节点。
- **记录数硬上限**：`max_total_records` / `max_entity_records`，并要求「Do not try to fill the limit」。这避免 LLM 在信息稀疏的 chunk 上凑数，也是控制入库成本的手段。
- **章节语境**：`entity_extraction_section_context` 注入 `---Section Context---` 的标题路径（`h1 → h2 → h3`），并明确标注为不可信元数据、不得据其抽取。作用与本仓库的 `【上下文：...】` 前缀相同，但放在独立区块且带防注入声明。

**gleaning 机制**（`operate.py:4252` 起）：首次抽取完成后，把「已有结果」回灌给模型再问一轮「还有遗漏吗」，取第二次结果做并集。`entity_extract_max_gleaning` 控制轮数，默认 1 轮。这是用一次额外 LLM 调用换取召回，代价可控。

抽取输入受 `max_extract_input_tokens`（默认 20480）约束，超限则跳过 gleaning 而非截断输入。

#### 阶段三：合并与累积

同一实体在多个 chunk 出现时，`_merge_nodes_then_upsert`（2429 行）与 `_merge_edges_then_upsert`（2782 行）做三件事：

1. **描述用 LLM 摘要合并**（`_summarize_descriptions`，532 行），不是字符串拼接。合并后的 `description` 是一段连贯摘要，长度受 `summary_max_tokens` 控制。
2. **`degree` 累积**：实体出现次数即度数，成为检索排序信号。
3. **`weight` 累积**：关系强度，同样参与排序。

`_find_most_related_edges_from_entities`（6260 行）按 `(rank, weight)` 双键降序排列邻边，`rank` 即度数。这意味着**高频共现的概念天然靠前**，不需要额外的热度模型。

#### 阶段四：三套索引

| 索引 | 内容 | 用途 |
| --- | --- | --- |
| `entities_vdb` | `entity_name` + `description` 的向量 | local 模式检索入口 |
| `relationships_vdb` | `keywords` + `description` 的向量 | global 模式检索入口 |
| `chunks_vdb` | 原文 chunk 向量 | mix 模式的补充召回 |
| `graph` | 节点/边属性 + 邻接表 | 展开邻域、取证据 |

关键点在于：**被向量化的是「概念」而不是「文本片段」**。一个实体的 `description` 是 LLM 从该实体出现的所有 chunk 中归纳出的摘要。原本分散在 5 个 chunk 里关于「镀金层厚度」的表述，在索引里合并成一条带描述的实体记录。这是 LightRAG 在聚合类、多跳类问题上强于朴素 RAG 的根本原因——**聚合发生在索引期，而非检索期靠拼接相邻块来近似**。

### 2.2 检索策略

`_perform_kg_search`（5280 行）按 `query_param.mode` 分派：

| 模式 | 入口 | 检索对象 | 检索后动作 |
| --- | --- | --- | --- |
| `local` | `ll_keywords` | `entities_vdb` top_k=40 | 取实体属性 + 邻边（按 degree/weight） |
| `global` | `hl_keywords` | `relationships_vdb` top_k=40 | 取关系 + 两端实体 |
| `hybrid` | 两者 | 两套索引 | round-robin 交替合并 |
| `mix` | 三者 | 两套索引 + `chunks_vdb` | round-robin + chunk 向量补充 |
| `naive` | 原 query | `chunks_vdb` | 纯向量，等价于常规 RAG |

**融合方式是 round-robin 交替穿插，不是分数融合**：

```python
# operate.py:5436-5451，实体合并
for i in range(max(len(local_entities), len(global_entities))):
    if i < len(local_entities): ...      # 先取 local 第 i 个
    if i < len(global_entities): ...     # 再取 global 第 i 个
```

第 1 位来自 local，第 2 位来自 global，第 3 位又是 local。这个选择的意义比表面看起来大：**它不需要两路分数可比**。RRF 需要假设「两路的 rank 都大致反映相关性」，一旦某一路整体失效，它的高位结果会挤占另一路的真实信号。round-robin 只保证名额分配，失效路只能占掉属于自己的一半名额，不会越界污染。

**排序信号是三重叠加**：cosine 相似度（召回时）、degree（度数）、weight（关系强度）。后两者是索引期累积的图结构属性，在检索期免费获得。

**token 截断分两段**（`_apply_token_truncation`，5501 行）：实体 6000 tokens、关系 8000 tokens 分别截断。默认 `top_k=40`、`chunk_top_k=20`。

### 2.3 查询处理与生成

#### 关键词抽取即语义路由

`get_keywords_from_query`（4975 行）用一次 LLM 调用抽出两类关键词：

- `high_level_keywords`：概括性概念、主题、问题类型 → 送入 `relationships_vdb`
- `low_level_keywords`：具体实体、专有名词、术语、产品名 → 送入 `entities_vdb`

`hl_keywords` 为空时 `global` 分支自动跳过；两者皆空且 query 长度 < 50，则退化为把原 query 当 `ll_keywords`；query 长度 ≥ 50 且关键词为空，直接返回 `fail_response` 而不硬答。

这个二分法的实质是**用一次 LLM 调用决定走哪条检索路**，替代了本仓库的规则意图分类。它比规则分类更鲁棒的地方在于：分类结果直接就是检索输入（关键词本身），不需要「分类 → 映射到参数」这一步，也就没有映射错误。

#### 上下文组装：三段式 + 动态预算

`_build_context_str`（5873 行）产出四段结构：

```
Knowledge Graph Data (Entity):      ```json [{"entity": ..., "type": ..., "description": ...}] ```
Knowledge Graph Data (Relationship): ```json [{"src": ..., "tgt": ..., "description": ...}] ```
Document Chunks:                     ```json [{"reference_id": 1, "content": ...}] ```
Reference Document List:             [1] file_path
```

预算分配是动态的：

```python
available_chunk_tokens = max_total_tokens - (
    sys_prompt_tokens + kg_context_tokens + query_tokens + buffer_tokens
)   # buffer_tokens = 200
```

默认 `max_total_tokens=30000`。实体/关系段先按各自上限截断，剩余的全部给 chunk。**上下文窗口被当作可分配资源来管，而不是固定条数**。

#### 生成约束

`rag_response`（`prompt.py:334`）的关键约束：

- 「Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information」——把模型的自由度限定在措辞层面。
- 找不到答案时明确声明信息不足，不猜测。
- 强制 `### References` 段，每条 `* [n] Document Title`，最多 5 条，参考文献段后不得再输出内容。
- 响应语言必须与 query 一致。
- 调用时 `enable_cot=True`。

---

## 三、本仓库实际生效的链路

生产 `.env` 解析后的实际链路：

```
query
  → 元数据过滤（_extract_query_filters）
  → 规则意图识别 + 动态权重（QUERY_ROUTING_ENABLED=1）
  → 字面同义词扩展（_expand_query，字符串拼接）
  → HyDE（HYDE_ENABLED=1，QUERY_ENHANCE_ALL=1 → 每个查询都跑）
  → 分解 / step-back（均为 1）
  → 三路召回：向量(200) + HyDE向量(200) + BM25(200)
  → 加权 RRF（0.45 / 0.85 / 0.55，k=40）
  → API 精排（RERANK_TOP_N=10，输入上限 50）
  → chunk 邻居/父块扩展（CHUNK_EXPAND_ENABLED=1）
  → 取 5 篇 × 1000 字符
  → LLM 生成 + [n] 角标
```

未进入链路的模块：

| 模块 | 配置 | 实际状态 |
| --- | --- | --- |
| GraphRAG | `GRAPH_RAG_ENABLED=0` | 未启用；`data/graph/kg.json` 不存在 |
| 上下文压缩 | `CONTEXT_COMPRESSION_ENABLED=0` | 未启用 |
| Agentic 迭代检索 | `SELF_RAG_ENABLED=0` | 未启用 |
| MultiPath + ColBERT | `MULTIPATH_ENABLED=0` | 未启用 |

---

## 四、架构对比

| 环节 | LightRAG | 本仓库 |
| --- | --- | --- |
| 切块单位 | token（1200 / overlap 100） | 字符（child 800 / parent 3000 / overlap 80） |
| 索引语义单元 | 实体 + 关系 + chunk | 仅 chunk |
| 索引期 LLM 调用 | 每 chunk 1–2 次（含 gleaning） | 0（`GRAPH_EXTRACT_BACKEND=rule` 且未启用） |
| 同义实体归并 | LLM 摘要合并描述 + degree/weight 累积 | 无 |
| chunk 语境注入 | `---Section Context---` 独立区块 + 防注入声明 | `【上下文：文档 · 章节 · 类型】` 拼在正文首行 |
| 查询理解 | LLM 抽 hl/ll 关键词（结果即检索输入） | 规则意图分类 + 同义词字符串拼接 + HyDE + 分解 + stepback |
| 召回入口 | 实体库 / 关系库 / chunk 库 | chunk 库 + BM25（同质） |
| 融合 | round-robin 交替 + token 截断 | 加权 RRF（依赖三路 rank 质量） |
| 排序信号 | cosine + degree + weight | RRF rank + cross-encoder |
| 上下文结构 | 实体 JSON / 关系 JSON / chunk JSON / 引用表 | 纯文本 `[n] 文档 · 章节\n正文` |
| 上下文预算 | 30000 tokens 动态分配 | 5000 字符固定上限 |
| 引用溯源 | `reference_id` → chunk → 文件 | `[n]` 角标 + 标准号正则匹配 |
| 失败处理 | 关键词为空且 query 长 → 直接 fail_response | 无结果时返回空并提示 |

三处差距是结构性的，其余是可调参数：

1. **索引语义单元不同**。LightRAG 索引「概念」，本仓库索引「文本片段」。前者的聚合发生在索引期（同一实体的多段表述被 LLM 合并成一条描述），后者只能在检索期用相邻块扩展来近似。多跳、聚合、对比类问题在这一层就已分出高下，与融合算法无关。
2. **融合的失败模式不同**。round-robin 不依赖跨路分数可比，单路失效只损失自己的名额；加权 RRF 假设各路 rank 都部分有效，单路失效时其高位噪声会占据融合榜前列。本仓库 `FUSION_WEIGHTS=0.45,0.55`，向量路权重接近一半。
3. **上下文预算差 8 倍**。`RAG_TOP_DOCS=5` × `RAG_DOC_MAX_CHARS=1000` 硬上限 5000 字符，约 3000–4000 tokens；LightRAG 是 30000 tokens 且按段分配。本仓库 LLM 为 `qwen3.5:35b-a3b-q4_K_M`，`OLLAMA_NUM_CTX=8192`，上下文尚有近一倍冗余未被使用。

---

## 五、效果诊断

### 5.1 全部消融实验的基线已失效

`regression_v1.json`（80 条 query、26 个来源文档、5680 个 chunk）上的实测：

| 实验 | MRR | Hit@1 | Hit@5 | Hit@10 | nDCG@10 |
| --- | --- | --- | --- | --- | --- |
| `r2_00` BM25 单路 | **0.4891** | **0.3875** | 0.6000 | 0.6625 | 0.5265 |
| `r2_01` 向量单路 | 0.0690 | 0.0375 | 0.1000 | 0.1625 | 0.0880 |
| `r2_02` 向量 + 精排 | 0.2479 | 0.2000 | 0.3250 | 0.3250 | 0.2673 |
| `r2_03` 融合 + 精排 | 0.4811 | 0.3750 | 0.6125 | **0.6750** | 0.5286 |
| `r2_04` + 查询扩展 | 0.4634 | 0.3750 | 0.5875 | 0.6375 | 0.5056 |
| `r2_05` + 多查询扩展 | 0.4627 | 0.3625 | 0.5875 | 0.6500 | 0.5083 |
| `r2_06` + 意图路由 | 0.4622 | 0.3625 | 0.5875 | 0.6500 | 0.5078 |
| `r2_07` + 分解 / stepback | 0.4611 | 0.3625 | 0.5875 | 0.6500 | 0.5069 |
| `r2_08` 生产配置（全开） | 0.4435 | 0.3375 | 0.5750 | 0.6250 | 0.4880 |

`r2_01` 的 MRR 0.0690 与 `embedding_query_passage_20261020.json` 中 `before.mrr@20 = 0.068966` 逐位一致，两者指向同一份失效索引。因此 `r2_01` 至 `r2_08` 八组实验共用一个已经失真的向量召回路。

修正 embedding 契约后的对照：

| 指标 | before | after | 倍数 |
| --- | --- | --- | --- |
| MRR@20 | 0.068966 | 0.393509 | 5.71× |
| MRR@10 | 0.065556 | 0.386042 | 5.89× |
| Hit@1 | 0.0375 | 0.2375 | 6.33× |
| Hit@10 | 0.1625 | 0.6375 | 3.92× |
| Hit@200 | 0.4000 | 0.9375 | 2.34× |

该实验为「同一 query/passage 对的纯向量排序」，未叠加元数据过滤、精排、查询改写与 HyDE，因此排除了这些环节的干扰。Hit@200 从 0.40 升到 0.9375 尤其关键：**修正前有 60% 的查询，正确文档连 200 条候选都进不去**，此时任何融合、任何精排都无从发挥。

### 5.2 融合没有兑现价值

修前的状态下，融合 + 精排（MRR 0.4811）低于 BM25 单路（0.4891），差 0.008。Hit@1 少 1 条（31 → 30），Hit@10 多 1 条（53 → 54）。

这个结果的含义不是「融合有害」，而是**向量路对最终结果贡献接近零**。RRF 从失效的向量路召回 200 条噪声，精排阶段把它们重新打分，靠 cross-encoder 把 BM25 的正确结果捞回前列。精排掩盖了融合的问题，所以指标没有崩塌，但也因此掩盖了「混合检索架构未生效」这一事实。

`r2_02` 提供了交叉印证：向量 + 精排从 0.0690 升至 0.2479，精排单枪把失效向量路的指标提升 3.6 倍。精排的纠错能力足以吸收一路的噪声，这既是当前指标的护城河，也是问题被长期忽视的原因。

### 5.3 查询处理层层叠加、逐级递减

从 `r2_03` 到 `r2_08`，每叠加一层查询处理，MRR 单调下降：

```
0.4811 → 0.4634 → 0.4627 → 0.4622 → 0.4611 → 0.4435
（基础）（扩展）（多变体）（路由）（分解）（HyDE）
```

累计损失 0.0376 MRR，Hit@10 从 0.6750 降至 0.6250。生产配置（`r2_08`）反而是所有组合中最差的一档。

这一现象的机制在每条处理上都成立，且不依赖向量路是否修好：

- **`_expand_query` 是字符串拼接**（`query.py:987`，`return query + " " + " ".join(selected)`）。对 BM25 而言，追加同义词改变了查询词分布，原关键词的贡献被稀释；对 dense 而言，拼接后的长串 embedding 偏离原始意图。主流做法是生成**独立变体分别检索、在文档级融合**，而不是把变体塞进同一个查询串。
- **多查询变体（`FUSION_NUM_QUERIES=4`）** 在失效向量路上，每个变体返回 200 条噪声，RRF 对同一噪声文档累加多票，反而放大噪声。
- **HyDE 的开销与收益严重倒挂**。`r1_baseline_production.json` 的 `timing_mean.HyDE = 36.864s`，而总耗时 p50 = 36.187s——HyDE 是唯一的串行瓶颈。关掉它，延迟降至 `r2_07` 一档（约 18s），MRR 反而从 0.4435 升到 0.4611。

### 5.4 上下文预算过窄

`RAG_TOP_DOCS=5`、`RAG_DOC_MAX_CHARS=1000`，上限 5000 字符。而 chunk 的 p90 长度为 868 字符、max 976——**单条 chunk 几乎吃满单个文档槽位的 1000 字符预算，且被截断**。

实测语料 5680 个 chunk，长度分布 p10=274、p50=669、p90=868、max=976、均值 612。这意味着 5 个槽位在多数情况下只能容纳约 4.5 个完整 chunk。对于「某标准对 X 的全部要求」这类需要跨章节汇总的问题，5 个 chunk 的覆盖面远远不够，模型只能基于残缺证据作答或拒答。

LightRAG 的 30000 token 预算（约 45000 汉字）是这一项的 8 倍以上。

### 5.5 图检索未进入链路

`GRAPH_RAG_ENABLED=0`，`data/graph/kg.json` 不存在。README 中「GraphRAG 图检索」列为亮点，实际未生效。

即便启用，`GRAPH_EXTRACT_BACKEND=rule` 的抽取质量也与 LightRAG 不在同一量级。`extract_triples_rule`（`graph_rag.py:598`）依赖两类信号：

1. 句式正则（`_RULE_PATTERNS`），weight 1.2
2. 同句实体共现，关系名统一为 `mentioned_with`，weight 0.4

第二类是主体。它只能表达「这两个词出现在同一句」，无法表达「A 是 B 的上限」「A 适用于 B」这类语义。LightRAG 的 `relationship_description` 是 LLM 生成的自然语言说明，可直接作为证据进入 context；`mentioned_with` 只是一条无内容的边，除扩张候选集外无法提供可读事实。

### 5.6 元数据写入不一致

抽取 `data/lexical_corpus_pcb_kb_nv_query_passage_20261020.jsonl` 的 5680 条记录：

- `source_type` 全部为 `unknown`（5680/5680）
- `metadata` 仅 3 个字段：`source_type`、`source_path`、`eda`（仅 2877 条有值）
- 无 `parent_title` / `section` / `vendor` / `layer_count` / `copper_oz`

而 chunk 正文首行的语境前缀明确写着 `national_standard`：

```
【上下文：GBT2036-1994印制电路术语 · 5.2.6 照相缩小尺寸 photographic reduction dimension · national_standard】
```

分类信息存在于正文，却未写入 metadata。后果有三：

1. `_extract_query_filters` 按 `source_type` 过滤时全部落空；
2. `build_context_with_sources` 取 `metadata.get('parent_title')` 恒为空，引用只能退化为文件名，无法定位到章节；
3. `source_type` 是评测的分组维度（`by_source_type`），分组统计因此失真。

`eda` 字段仅覆盖 50.6% 的 chunk，按 `eda` 过滤会丢掉一半语料。

### 5.7 chunk 边界质量

抽样可见两类低信息密度块：

```
8.2.5.3 语句组成
<socket>语句包含以下内容：
<default>
<port>
<portgroup>
<powerdomain_group>
<swappable_port>
...
8.2.5.4 default语句
```

该类块有两个问题：其一，60% 以上内容是裸标签罗列，语义空洞，向量化后接近噪声；其二，「8.2.5.4 default语句」的标题落在本块末尾，其正文在下一块，标题与内容分离。

这与 README 已知问题 #7 相呼应：`parent_child` 模式下父块未单独入库，`parent_id` 指向同父块的首个子块，所谓「父级扩展」实际取回的是兄弟块而非父块内容。

### 5.8 评测集本身的偏向

80 条 query 的构成：中问中答 69 条（86.2%）、中问英答 11 条（13.8%）；query 长度中位 34 字。真值由 chunk 反推生成，问题文本保留了原文词汇。

这一构造使词汇重叠成为强信号，BM25 天然占优。它也解释了为何 BM25 单路 0.4891 显著高于修正后的向量 0.3935——**其中含有一部分评测集构造带来的偏差，不能全部归因于向量质量**。

但 embedding 修复带来的 5.71 倍提升是配对实验的结果（同一批 query-passage 对、同一排序方式、仅改编码侧），不受此偏差影响。结论的可靠性在于：偏差会让两个数字同时偏高或偏低，不会制造 5.7 倍的差距。

---

## 六、效果不佳的归因

按影响量级排序，每条附证据与可验证性：

| # | 归因 | 量级 | 证据 | 可控性 |
| --- | --- | --- | --- | --- |
| 1 | embedding query/passage 契约错误 | MRR 0.069 → 0.394 | 配对实验，Hit@200 0.40 → 0.9375 | 已修，需重跑验证 |
| 2 | 消融基线建立在失效索引上 | 八组实验结论失效 | `r2_01` MRR 与 `before` 逐位一致 | 重跑即可 |
| 3 | 查询处理层层稀释信号 | MRR 累计 −0.0376 | `r2_03`→`r2_08` 单调递减 | 可配置 |
| 4 | 上下文预算 5000 字符 | 聚合类问题无法作答 | `RAG_TOP_DOCS=5`、chunk p90=868 | 可配置 |
| 5 | 图检索未启用 / 抽取质量弱 | 多跳能力缺失 | `GRAPH_RAG_ENABLED=0`、`kg.json` 不存在 | 需建设 |
| 6 | 元数据写入不一致 | 过滤失效、溯源退化 | `source_type` 全 unknown | 需修 bug |
| 7 | chunk 边界与信息密度 | 部分块近似噪声 | 抽样见 5.7 | 需改切块 |
| 8 | 评测集偏向词汇匹配 | 指标绝对值偏高 | 86.2% 中问中答，真值由 chunk 反推 | 需换集 |

前四条构成本次问题的绝大部分。第 1 条已修复但未在完整链路上验证；第 3 条是「优化过度」的典型表现——组件本身未必有害，但在失效的召回路上叠加，只会放大噪声。

---

## 七、改进路线

### P0：重建可信基线（成本 0，1 天内）

在 `COLLECTION=pcb_kb_nv_query_passage_20261020` 上重跑 `r2_00`–`r2_12` 全系列，参数与首次一致（`--rrf-k 40 --recall-k 200`），产出新的基线表。

验收标准：`r2_01`（向量单路）MRR 应落在 0.35–0.42 区间。若仍低于 0.10，说明生产链路的 embedding 调用未走 `input_type=query`，需检查 `build_embed_model()` 与 `nvidia_embeddings.py` 的 query 侧实现，而非继续调检索参数。

在此之前，**不应据旧消融结论删除任何组件**——HyDE、多查询扩展、路由的负收益结论都建立在失效向量路上。

### P1：配置层收敛（成本 0，1–2 天）

| 项 | 现值 | 建议 | 依据 |
| --- | --- | --- | --- |
| `HYDE_ENABLED` | 1 | 0 | 省 36s/请求，`r2_07` 显示关闭后 MRR 更高 |
| `FUSION_WEIGHTS` | 0.45,0.55 | 待 P0 结果定；若向量 MRR ≈ 0.39，建议 0.35,0.65 | BM25 仍是主力，向量作补充 |
| `FUSION_NUM_QUERIES` | 4 | 1（先关多变体） | `r2_04`→`r2_05` 无增益 |
| `QUERY_DECOMPOSE_ENABLED` | 1 | 0 | `r2_07` 未显示增益 |
| `QUERY_STEP_BACK_ENABLED` | 1 | 0 | 同上 |
| `RAG_TOP_DOCS` | 5 | 8 | chunk p90=868，5 槽装不满 |
| `RAG_DOC_MAX_CHARS` | 1000 | 1500 | 消除对 p90 chunk 的截断 |
| `QUERY_ENHANCE_ALL` | 1 | 0 | 让增强只作用于短查询 |

P0 重跑后，上述各项应以新基线重新评估，而非直接照搬。特别是 `FUSION_WEIGHTS`，向量修复后其最优值必然上移。

### P2：修实现缺陷（1 周）

1. **元数据写入**（对应 5.6）
   `preprocess_docs.py` 或 `ingest.py` 的 `doc_metadata` 构造处，把已在正文前缀中的 `source_type`、`parent_title`、`section` 同步写入 metadata。验收：任取 100 条 chunk，`source_type != unknown` 的比例 ≥ 95%，`parent_title` 非空比例 ≥ 80%。

2. **`_expand_query` 改为独立变体检索**（对应 5.3）
   现实现把同义词拼进原串，稀释原词权重。改为：原查询与各变体分别检索，在文档级用 RRF 合并（此时 RRF 作用于「同一路的多个查询」，语义正确）。验收：`r2_04` 的 MRR 不低于无扩展版本。

3. **chunk 质量过滤**（对应 5.7）
   入库时对「裸标签罗列型」块（如连续 3 行以上匹配 `^<[a-z_]+>$` 且无中文正文）与纯表头块做标记，不单独成块，或合并入相邻块。验收：重新入库后，长度 p10 不低于 300，且抽样 20 条无纯标签块。

4. **父块入库**（对应 README 已知问题 #7）
   把 parent（3000 字符）也写入向量库，加 `node_level=parent` 字段。检索时先查 child，命中后用 `parent_id` 直接取父块文本替换。验收：对比开启前后，聚合类 query 的 Hit@5 提升可测。

### P3：架构级补齐（1–2 周，按需）

1. **三段式 context**（对应 4.3）
   在 `build_context_with_sources` 中，把上下文拆为「检索到的结构化事实」与「原文证据」两段，前者来自图谱或元数据，后者是 chunk 原文。即便不建图，也可把「标准号 + 条款号 + 参数值」抽成结构化行放在前面。验收：人工评估 20 条生成答案的事实准确率。

2. **融合改为配额制**（对应 4.2）
   替换纯 RRF 为「每路保底配额 + 剩余按 RRF 分配」。例如向量路至少保留 2 个名额、BM25 至少 5 个，剩余按分数排。这保留了 RRF 的排序能力，同时防止单路失效时被全量置换。验收：人为将向量路降级（用旧 collection）时，MRR 下降幅度应小于 0.02。

3. **LLM 实体抽取 + 图检索启用**（对应 5.5）
   `GRAPH_EXTRACT_BACKEND=llm`，提示词对齐 LightRAG 的 `entity_extraction_system_prompt`（类型表、一致性约束、记录上限、gleaning 一轮）。合并阶段用 LLM 摘要而非拼接，并累积 degree/weight。验收：5680 chunk 入库后，实体数、关系数、平均度数三项指标与 LightRAG 同规模语料处于同一量级；`GRAPH_RAG_ENABLED=1` 时聚合类问题 Hit@5 有可测提升。

   该项成本最高（每 chunk 1–2 次 LLM 调用），建议先在 500 chunk 子集上验证增益，再决定是否全量。

### 不建议做的事

- **不要在失效索引上继续调 `FUSION_WEIGHTS`、`FUSION_RRF_K`**。这些参数的作用是把三路排序折算成可比分数，前提是三路都有信号。向量路 MRR 0.069 时，调参只是在拟合噪声。
- **不要以「组件越多越好」的思路继续叠加**。`r2_03`→`r2_08` 已经证明叠加是负向的，问题不在组件本身，而在没有先确认每一层是否真的产生了正贡献。
- **不要用 `r2_xx` 的绝对指标与 LightRAG 论文数字比较**。数据集、语料、语言分布均不同，只有同集内配置对比才有意义。
- **不要为了对齐 LightRAG 而直接引入它的全套机制**。本仓库语料为 5680 chunk / 26 文档，属于中小规模、术语高度固定的领域语料，BM25 强是合理现象。LightRAG 的价值主要在多跳与聚合，应先确认这类问题在业务中的占比，再决定投入。

---

## 八、附：两处值得单独说明的实现细节

**其一，`_overlap_tail` 的修复思路**（`ingest.py:1189`）。原实现取末 80 字后又向前找句号截断，导致实际重叠接近 0；现改为向后补齐到句首，保证重叠落在 `[overlap, 2·overlap)`。这类「参数写了但实际不生效」的问题，在本仓库不是孤例（`FUSION_MODE` 空转、`parent_id` 指向首个子块、`source_type` 全 unknown 均属同类）。建议在 P2 阶段做一次「配置项 ↔ 实际生效点」的普查，成本低、收益直接。

**其二，`embedding_contract.py` 的设计值得保留并强化**。它把 `provider / model / dimension / query_input_type / document_input_type / truncate` 打成 profile 写入集合描述，入库前用 `validate_embedding_profile` 校验。本次问题的根因正是「两侧编码方式不一致但不报错」，该模块的存在使这类问题可被检测。建议补一条 CI 断言：query 侧与 document 侧的 `input_type` 必须不同（对非对称 provider），相同则直接失败。
