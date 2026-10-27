# Embedding 与 Rerank 双模型切换（2026-10-23）

## 1. 变更范围

| 项目 | 切换前 | 切换后 |
| --- | --- | --- |
| Embedding 模型 | `nvidia/nemotron-3-embed-1b` | `nvidia/nemotron-3.5-embed-8b` |
| Embedding 维度 | 2048 | 4096（模型原生维度） |
| Embedding 接口 | `https://integrate.api.nvidia.com/v1/embeddings` | 同左（未变） |
| Rerank 模型 | SiliconFlow `BAAI/bge-reranker-v2-m3` | `nvidia/nemotron-3.5-rerank-8b` |
| Rerank 接口 | `https://api.siliconflow.cn/v1/rerank` | `https://ai.api.nvidia.com/v1/retrieval/nvidia/nemotron-3.5-rerank-8b/reranking` |
| Rerank 线路格式 | Jina / 硅基流动兼容 | NVIDIA NIM `reranking` |
| 生产集合 | `pcb_kb_nv_query_passage_20261020` | `pcb_kb_nv35_embed8b_20261023` |

两项变更都改变检索语义：embedding 换了向量空间与维度，rerank 换了打分模型与分值量纲。新旧指标不构成可比序列，消融表须按新基线重测。

## 2. 接口契约（实测确认）

以下结论来自 2026-10-23 对两个接口的实际调用，不是文档推断。

### 2.1 Embeddings

- 原生输出 4096 维；`dimensions` 可取 1~4096，缩短后的向量由服务端重新做 L2 归一化。
- `input_type` 取 `query` / `passage`，必须与用途一致；错配会造成召回率大幅下降。
- `truncate` 仅接受 `END` / `NONE`。传 `START` 返回 HTTP 400：
  `Input should be 'END' or 'NONE'; got 'START'`。
- 单条输入上限 10240 token。
- 返回体为 OpenAI 兼容结构：`data[].embedding` / `data[].index` / `usage.prompt_tokens`，用量可直接入账。

因此 `EMBED_DIM` 在本模型下是**真实下发**的参数，不再只是「Milvus 建表用的期望值」：填 2048 就得到 2048 维截断向量，填 4096 得到原生向量。生产取 4096，不做截断。

### 2.2 Reranking

NIM 的检索重排接口与 `/v1/rerank` 兼容接口有三处硬性差异，直接沿用旧客户端会得到 400：

| 差异点 | `/v1/rerank` 兼容 | NIM `reranking` |
| --- | --- | --- |
| query | 字符串 | 对象 `{"text": "..."}`（不接受 `type` 字段） |
| passages | `documents: ["..."]` | `passages: [{"text": "..."}]`（不接受纯字符串） |
| 返回条数 | `top_n` 控制 | 不接受 `top_n`，服务端返回全部排序，客户端裁剪 |
| 分值 | `relevance_score`（0~1） | `logit`（未归一化，可为负） |
| 响应键 | `results` | `rankings` |

其他约束：passages 最多 512 条；query 与单条 passage 合计不超过 10240 token，超限时由 `truncate` 决定报错（`NONE`）或截断（`END`）。

适配后 `sources[].score` 仍对外暴露 0~1 的相关度：适配层按 `sigmoid(logit)` 归一化，排序结果与 logit 完全一致。若后续需要用 logit 做绝对阈值判定，应另设字段，不应改动 `score` 的量纲。

## 3. 代码改动

| 文件 | 改动 |
| --- | --- |
| `src/pcb_rag/nvidia_embeddings.py` | `expected_dimension > 0` 时把 `dimensions` 作为真实请求参数下发 |
| `src/pcb_rag/api_clients.py` | NVIDIA 分支的 `EMBED_TRUNCATE` 收紧为 `END` / `NONE`，并校验 `EMBED_DIM` 落在 1~4096 |
| `src/pcb_rag/api_clients.py` | `ApiReranker` 拆出 `_wire_payload` / `_parse_scores` 两个钩子，HTTP、限流、计费、重试逻辑保持单份 |
| `src/pcb_rag/api_clients.py` | 新增 `NvidiaApiReranker`：对象化请求体、`rankings[].logit` 解析、512 条上限、本地裁剪 |
| `src/pcb_rag/api_clients.py` | 新增 `RERANK_API_PROVIDER`（`auto` / `openai` / `nvidia`）与 `resolve_rerank_provider()`；`build_api_reranker()` 按线路分发 |
| `src/pcb_rag/api_clients.py` | `describe_backends()` 增补 rerank 线路、模型与地址，`/health` 可直接自查 |
| `src/pcb_rag/query.py` | 精排初始化日志补出实际线路类名 |

`auto` 判定为 `nvidia` 的条件是「官方域名 + 路径以 `/reranking` 结尾」，两者同时满足。自建 NIM 网关须显式设置 `RERANK_API_PROVIDER=nvidia`。

## 4. 配置改动

`.env` 与 `.env.eval` 同步更新：

```dotenv
COLLECTION=pcb_kb_nv35_embed8b_20261023
LEXICAL_CACHE_PATH=./data/lexical_corpus_pcb_kb_nv35_embed8b_20261023.jsonl
INGEST_MANIFEST_PATH=./data/ingest_manifest_pcb_kb_nv35_embed8b_20261023.json

EMBED_MODEL=nvidia/nemotron-3.5-embed-8b
EMBED_DIM=4096
EMBED_API_PROVIDER=nvidia
EMBED_TRUNCATE=END

RERANK_API_URL=https://ai.api.nvidia.com/v1/retrieval/nvidia/nemotron-3.5-rerank-8b/reranking
RERANK_API_MODEL=nvidia/nemotron-3.5-rerank-8b
RERANK_API_PROVIDER=nvidia
RERANK_TRUNCATE=END
RERANK_API_TIMEOUT=120
```

`RERANK_API_TIMEOUT` 由 60 秒提高到 120 秒：8B 重排模型在 50 条候选、单条 1200 字符的输入下明显慢于旧的 bge-reranker。

`.env.eval` 的 `RERANK_BUDGET_SCOPE` 改为 `rerun_20261023_nvidia_nemotron35`。计费口径按 scope 冻结，沿用旧 scope 会把上一代 reranker 的用量计入新实验。

## 5. 迁移步骤

换 embedding 模型必须重建向量集合，且**不能**写入旧集合：旧集合的 `description` 记录了上一套向量空间配置，`validate_embedding_profile()` 会拒绝混用。集合名、词法缓存路径、入库清单路径三者同步改名，避免旧缓存被误当成新语料的统计量。

块文本与 chunk id 保持不变，因此切块冻结、回归集标注与图谱节点引用继续有效。迁移只替换向量：

```bash
set -a; source .env.eval; set +a
python scripts/repair_embeddings.py \
  --source pcb_kb_nv_query_passage_20261020 \
  --target pcb_kb_nv35_embed8b_20261023 \
  --report eval/reports/embed_migration_20261023_nemotron35.json
```

该脚本从源集合逐行读取文本、按 `input_type=passage` 重新编码、写入新集合，并把完成位点落在 `data/embedding_migrations/<target>/state.json`，中断后可续跑。迁移完成后：

1. 对比源集合与目标集合的行数、id 与正文摘要，确认没有丢行；
2. 用冻结回归集跑一次向量单路检索，与换模型前的向量单路成绩并列留档；
3. 重建词法缓存（`LEXICAL_CACHE_PATH` 指向新文件），确认 BM25 统计量来自同一批文本；
4. 启动 API，`/health` 中 `embed_model`、`embed_dim`、`rerank_api_provider`、`rerank_model` 应与本文一致；
5. 确认无误后再删除或归档旧集合与旧清单；在此之前旧集合即回滚路径。

## 6. 回滚

回滚需三处同时回退，缺一不可：`.env` 中的 embedding 模型与维度、集合名与清单路径、rerank 接口与模型。仅回退模型而保留新集合，会因为向量空间不匹配被 `validate_embedding_profile()` 拒绝；仅回退集合而保留新模型，会得到错误的相似度。

脚本类文件（`scripts/prepare_guarded_experiments.py`）中固化了旧 SiliconFlow 配置的断言，属于历史批次的复现凭据，本次不修改。新批次应另建脚本或参数化，不要改动已归档批次的断言。

## 7. 迁移执行结果（2026-10-23 22:17）

`repair_embeddings.py` 全程 47 分 27 秒，分 89 个批次写完 5680 行，末尾校验「源集合与目标集合的 id、正文、元数据逐条一致」通过。

| 项 | 结果 |
| --- | --- |
| 新集合 `pcb_kb_nv35_embed8b_20261023` | 5680 行，4096 维，FLAT/IP |
| 集合 description | `nemotron-3.5-embed-8b` / 4096 / query+passage / truncate=END |
| 报告 | `eval/reports/embed_migration_20261023_nemotron35.json` |

向量单路基线（80 题冻结集，精确 IP 排序，无元数据过滤、无精排、无改写）：

| 指标 | 旧模型 2048（2026-10-20） | 新模型 4096 | 变化 |
| --- | --- | --- | --- |
| MRR@20 | 0.3935 | **0.4463** | +0.0527 |
| Hit@1 | 0.2375 | 0.3000 | +6.25pp |
| Hit@5 | 0.5875 | 0.6375 | +5.00pp |
| Hit@10 | 0.6375 | 0.7125 | +7.50pp |
| Hit@200 | 0.9375 | 0.9500 | +1.25pp |

两组数字来自不同报告（维度不同，脚本不计算配对 delta），但语料文本、题集、评测口径相同，可作跨报告对照，不能当作严格配对实验。Hit@200 已达 0.95，说明剩余误差主要不是"正确文档进不了候选池"，而是排序问题。

真实链路抽检（生产工厂 + 真实 Milvus + 真实接口）：4 道回归题向量 top8 命中 4/4，精排后 top5 仍 4/4，单次精排约 2 秒。词法缓存已按新集合预热，17.8 秒生成 8.47MB，API 首启不需再重建。

### 本次迁移不包含 2026-10-23 整理后的语料

入库清单 `ingest_manifest_pcb_kb_nv_query_passage_20261020.json` 记录的是 60 个 `data/clear_docs/` 文件，而当前 `DATA_DIR=./data/pcb_docs_clean` 有 66 个文件、264 万字符，其中 `04_工程语料/` 20 篇为当日新增。迁移只复制旧集合的行，不读盘上语料，因此上述基线描述的是**旧语料换新模型**。

若要让当日语料生效，须另做全量入库，不能增量覆盖：chunk id 由 `md5(file_id-序号)` 派生，而 `file_id = md5(文件路径)`（`ingest.py:3508`），`clear_docs` → `pcb_docs_clean` 的路径变化会让全部 id 改变，增量入库只会得到新旧两套并存的重复语料。
