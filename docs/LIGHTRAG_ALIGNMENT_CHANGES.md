# 对齐 LightRAG 的实现变更记录

> 本文保留 2026-10-23 较早一轮实现的历史记录。后续七类问题修复、当前迁移方式与实际 WSL 测试结果见 [修复与验证报告](LIGHTRAG_FIXES_WSL_20261023.md)；本文中的测试数字、旧图配额和重建行为不代表当前最终状态。

**日期**：2026-10-23
**基准分析**：`docs/LightRAG对比分析与效果诊断.md`
**参照实现**：HKUDS/LightRAG `aaebfe1`（v1.5.8）
**验证环境**：llama-index-core 0.14.25 + llama-index-vector-stores-milvus

本次变更只修实现与配置，**未重跑任何评测**。所有涉及排序结果的调整，其收益需在新基线上复核后确认。

---

## 一、变更总览

| # | 模块 | 变更 | 是否需重建索引 |
| --- | --- | --- | --- |
| 1 | `ingest.py` | 修复 `source_type` 全线 unknown | 需要（重跑入库） |
| 2 | `ingest.py` | `_parent_child_parse` 未继承 doc 级元数据 | 需要 |
| 3 | `ingest.py` | 新增 chunk 质量评估与标记 | 需要 |
| 4 | `query.py` | 词法导出字段扩展 + 字段存在性过滤 | 否（自动重建缓存） |
| 5 | `fusion.py` | 新增配额制融合 | 否 |
| 6 | `context_budget.py` | 新增：token 预算与三段式上下文 | 否 |
| 7 | `query.py` / `dify_external_api.py` | 扩展串只喂 BM25；精排与 HyDE 改用原问题 | 否 |
| 8 | `graph_rag.py` | 图节点语义化（类型 / 描述 / 关系描述 / gleaning） | 需要（重建图谱） |
| 9 | `query.py` / `ingest.py` | 修复坏 `.venv` 链接导致导入失败 | 否 |
| 10 | `.env` / `.env.example` | 配置收敛 | 否 |

回滚方式：当时使用的 `.env.bak.before-lightrag-align-20261023` 已在 2026-10-27 的遗留文件清理中删除；代码级回滚仍按下列各节的开关说明逐项关闭，无需改代码。

---

## 二、元数据一致性

### 2.1 `source_type` 全线 unknown

**现象**：5680 条 chunk 的 `source_type` 全部为 `unknown`，而正文语境前缀里明确写着 `national_standard`。

**根因**：`_infer_source_type` 只按目录层级判断（`/标准/国家标准/`），实际语料平铺在 `clear_docs/` 下，匹配不到任何层级。

**修法**：保留目录优先，补一层文件名兜底。

```python
if re.match(r"^GB[/T\s]*\d", upper):   return "standard_national"
if re.match(r"^IPC[-\s]?[A-Z]*\d", upper): return "standard_international"
if re.match(r"^SJ[/T\s]*\d", upper):   return "standard_industry"
if re.match(r"^ISO[-\s]?\d", upper):   return "standard_international"
# 再退到 EDA 工具名 / 材料 / 教程 等关键词
```

**影响**：`_extract_query_filters` 按 `source_type` 过滤不再落空；评测 `by_source_type` 分组恢复意义；引用可定位到章节。

### 2.2 parent_child 模式未继承 doc 级元数据

**现象**：`_parent_child_parse` 构造的 metadata 为 `{**chunk_meta, ...}`，未并入 `doc_metadata`。而 `_structure_aware_parse` 与 `_attention_semantic_parse` 都是 `{**doc_metadata}`。

**后果**：生产用的 `NODE_PARSER_MODE=parent_child` 恰好是漏掉的那条路径，`source_type` / `vendor` / `layer_count` 等在 chunk 层全部丢失。

**修法**：改为 `{**doc_metadata, **chunk_meta, ...}`，并把 `parent_title` / `chunk_section` 归一出一份 `section` 字段，供下游统一取值。

### 2.3 字段名分裂

`enhance_chunk_metadata` 只写 `doc_type`（`national_standard`），而过滤与统计读 `source_type`（`standard_national`）。两套分类体系并存不是问题，问题是只有一套被写入。

**修法**：两个字段都保证有值，并在注释中写明各自用途（粗粒度来源 vs 细粒度文体）。

---

## 三、融合：配额制

### 3.1 动机

实测 `BM25 单路 MRR 0.4891` 高于 `融合 + 精排 0.4811`。纯加权 RRF 的隐含前提是"每一路 rank 都大致反映相关性"；某一路系统性失效时，它的高位结果仍会拿到 `w/(k+rank)`，把另一路的真实信号挤出榜单。

LightRAG 的 `hybrid` 模式用 round-robin 交替取数回避了这一点：不比较跨路分数，失效路最多占掉自己那份名额。

### 3.2 实现

`fusion.quota_rrf_fuse`：两阶段。

1. **轮转填充保底配额**：第 r 轮依次从各路取第 r 条，配额用尽即跳过。效果是同构于 round-robin。
2. **剩余名额按加权 RRF 补齐**：保留分数区分度。

返回的 `score` 与 `weighted_rrf_fuse` 同量纲，两套策略可直接横向对比。

### 3.3 开关

```
FUSION_QUOTA_ENABLED=1     # 0 则退回纯 RRF
FUSION_MIN_PER_ROUTE=1     # 每路保底条数
FUSION_QUOTAS=             # 显式配额，按 routes 顺序；留空则均分
```

`FUSION_QUOTAS` 故意不做"补零/截断"：顺序一旦错位，静默补零会让某一路彻底失去保底，比退回均分更危险。

### 3.4 验证

`tests/test_fusion.py::TestQuotaRrfFuse::test_failed_route_cannot_evict_the_other` 构造了"一路全是高位噪声、另一路只有一条正确结果"的场景，断言纯 RRF 会把正确结果挤出 top3、配额制能保住它。这是本次改动的核心语义，已有回归保护。

---

## 四、上下文：token 预算与三段式

### 4.1 问题

`RAG_TOP_DOCS=5 × RAG_DOC_MAX_CHARS=1000` 硬上限 5000 字符。而语料 chunk 的 p90 为 868 字符、max 976 —— 单块几乎吃满单槽且被截断，5 个槽位实际只能容纳约 4.5 个完整块。

LightRAG 的对应设计是 `max_total_tokens=30000`，先扣 system prompt 与实体/关系段，剩余全给 chunk。

### 4.2 实现

新增 `src/pcb_rag/context_budget.py`（纯逻辑，不依赖 llama-index）与 `query.build_context_with_budget`（适配层）。

预算切分：

```
引用表 = min(total × 0.06, 200)
固定开销 = 60
事实摘要 = total × 0.22
原文证据 = total - 以上三项        # 剩余额度全部给证据
```

若证据额度不足两条最小配额，放弃事实段，把窗口全部让给原文。

三段结构：

```
【事实摘要】
[1] GBT2036-1994 · 5.2.6 照相缩小尺寸 · national_standard · GB/T 2036
    该尺寸值是 1:1 比例的精确数值，并且必须在照相底图上标注。

【原文证据】
[1] GBT2036-1994 · 5.2.6 照相缩小尺寸
（完整正文）

【引用来源】
[1] GBT2036-1994印制电路术语.txt
```

三处编号严格对齐。

### 4.3 顺带的三个修正

- **去重**：多路召回会让同一 chunk 重复出现，此前不去重，预算紧张时等于用一半窗口重复同一段话；现在按 node id 去重。
- **正文取值**：`MetadataMode.EMBED` 会把 `source_path` / `doc_node_id` / `prev_id` 等键值拼进正文，既占预算又是模型读不懂的技术字段；改为 `MetadataMode.NONE`，定位信息由标题行承担。
- **低价值块降权**：`low_value=True` 的块排到末尾（只调顺序，不硬过滤）。

### 4.4 兼容

`build_context_with_sources` 默认走预算模式；**显式传 `max_docs` / `max_chars` 时仍走旧的固定截断**。评测脚本与测试依赖这一行为做可控对比，不宜被默认值悄悄改变。

`RAG_CONTEXT_MAX_TOKENS=0` 可整体回退。

### 4.5 验证

`tests/test_context_build.py` 用真实 `NodeWithScore` 覆盖：三段齐备、编号对齐、metadata 章节生效、去重、预算为 0 时返回空、12 条短块能突破旧的 5 条上限、超长块被截断而非丢弃。

---

## 五、查询扩展：拆分作用域

### 5.1 问题

`_expand_query` 是字符串拼接（`query + " " + 同义词`），拼接结果同时喂给向量检索、HyDE 与精排三处。

### 5.2 修法

按模型特性拆分：

| 环节 | 输入 | 理由 |
| --- | --- | --- |
| BM25 | `expanded_q` | 词袋模型，追同义词等价于扩展 OR 查询，通常有益 |
| 向量检索 | `clean_q` | dense 把整串编码成单向量，拼同义词会让向量偏离原意 |
| HyDE | `clean_q` | 语义是"为这个问题生成假设答案"，输入词表会让假设答案跑偏 |
| 精排 | `clean_q` | cross-encoder 做 query-doc 逐词交互，长 query 稀释关键短语权重 |

CLI 与 API 两侧同步修改。

---

## 六、图：实体语义化

### 6.1 问题

`GRAPH_EXTRACT_BACKEND=rule` 的抽取主体是"同句实体共现"，关系名统一为 `mentioned_with`（weight 0.4）。它只能表达"两个词出现在同一句"，无法表达"A 是 B 的上限"。除扩张候选集外提供不了可读事实。

### 6.2 变更

**数据结构**

- `Triple` 增加 `head_type` / `tail_type` / `head_desc` / `tail_desc` / `relation_desc`
- `GraphNode` 增加 `desc`（跨 chunk 合并的实体摘要）
- 边增加 `rd`（关系描述）

**类型归一**：提示词给出 11 类封闭集合（Standard / Material / Process / Structure / Parameter / Equipment / Software / Method / Defect / Component / Concept），`_normalize_entity_type` 兼容中英文与短语写法，避免同一类实体在图里碎成多种 type。

**描述合并**：`_merge_node_desc` 去重累加，上限 `GRAPH_NODE_DESC_CHARS=240`。不用简单拼接 —— 同一实体在几十个 chunk 出现时会膨胀成一篇长文，既撑大图文件又挤占检索时的上下文名额。

**gleaning**：首轮抽取后把已有三元组回灌，要求只补遗漏（参照 LightRAG 做法）。补充量不占用首轮配额，上限放宽到两倍，否则补出来的边会被直接截掉、白付一次调用。

**检索侧**：`build_neighborhood_text` 抽成纯函数，输出形如

```
[知识图谱·邻域] 「镀金层厚度」：金层最小厚度指标
相关事实：
GB/T 4677-2002 --规定--> 镀金层厚度（规定了镀层厚度测试要求）
  · GB/T 4677-2002：印制板测试方法国家标准
镀金层厚度 --测试方法为--> X射线荧光法
  · X射线荧光法：无损测厚方法
原文依据：镀金层厚度应不小于0.05μm / 采用X射线荧光法测量
```

无描述的图（规则抽取）产出的文本与旧版一致，只是少掉"有则附加"的说明，不会丢结构。

### 6.3 配置

```
GRAPH_RAG_ENABLED=1        # 图谱缺失时自动降级，不影响主链路
GRAPH_EXTRACT_BACKEND=llm  # 对齐 LightRAG 的关键
GRAPH_LLM_GLEANING=1
```

**成本提醒**：`llm` 模式下每个 chunk 1–2 次调用。5680 chunk 约 5700–11400 次请求。建议先在 500 chunk 子集上验证增益再全量。

---

## 七、健壮性修复

`.venv` 是一个指向 Linux 路径的坏符号链接。`_bootstrap_local_venv` 里的 `venv_dir.is_dir()` 在 Windows 上抛 `OSError: [WinError 1920]`，导致 `import pcb_rag.query` 直接失败 —— 整个模块不可用，且报错指向的是"路径无法访问"而非真正原因。

改为 `try/except OSError`，取不到就当作没有本地 venv。

同一问题会让 `pytest` 的收集阶段崩溃（遍历根目录时 stat `.venv`），运行测试需加 `--ignore=.venv`。

---

## 八、配置收敛

| 项 | 原值 | 新值 | 依据 |
| --- | --- | --- | --- |
| `HYDE_ENABLED` | 1 | 0 | 单请求 36.9s，是唯一串行瓶颈；旧消融显示负收益（待重跑确认） |
| `QUERY_ENHANCE_ALL` | 1 | 0 | 仅短查询启用增强 |
| `MULTI_EXPAND_ENABLED` | 1 | 0 | 旧消融无增益 |
| `FUSION_NUM_QUERIES` | 4 | 1 | 同上 |
| `QUERY_DECOMPOSE_ENABLED` | 1 | 0 | 逐级叠加后 MRR 递减 |
| `QUERY_STEP_BACK_ENABLED` | 1 | 0 | 同上 |
| `FUSION_WEIGHTS` | 0.45,0.55 | 0.4,0.6 | BM25 仍是主力，略向词法倾斜 |
| `GRAPH_RAG_ENABLED` | 0 | 1 | 图谱缺失会自动降级 |
| `GRAPH_EXTRACT_BACKEND` | rule | llm | 见第六节 |

**以上数值均为待验证的初始值**。旧消融实验全部跑在失效的向量索引上（`r2_01` 的 MRR 0.0690 与 embedding 报告的 `before` 逐位一致），其结论不能作为取舍依据。重跑后应逐项复核。

---

## 九、验证结果

```
134 passed  （test_fusion / test_context_budget / test_context_build /
             test_graph_rag / test_chunk_quality）
```

全量 `tests/` 为 `433 passed, 6 failed, 10 skipped`。6 个失败均与本次改动无关：

| 失败项 | 原因 |
| --- | --- |
| `test_experiment_records` ×4 | `scripts/experiment_queue.py` 依赖 `fcntl`，Windows 无此模块 |
| `test_security` ×2 | `.env` 中 `ACL_TRUST_HEADERS=1`，测试断言默认为 False |

这两类在 Linux / WSL 下应分别消失或需要独立判断。

---

## 十、待办

1. **在新集合上重跑 `r2_00`–`r2_12`**，重建基线。验收：`r2_01`（向量单路）MRR 应落在 0.35–0.42。
2. **重跑入库**（元数据修复 + 质量标记需要重新切块才生效）。
3. **建图**（`GRAPH_EXTRACT_BACKEND=llm`，建议先 500 chunk 子集）。
4. **父块入库**（README 已知问题 #7）本次未做：需把 parent 写入向量库并加 `node_level` 过滤，属重建索引级改动，应与第 2 项合并进行。
5. 按新基线复核第八节的全部数值。
