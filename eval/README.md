# 评测体系

对 PCB-RAG 的检索与生成质量做量化评估。四个核心指标与生产参考阈值：

| 指标 | 度量对象 | 参考阈值 | 低于阈值时的排查方向 |
| --- | --- | --- | --- |
| Faithfulness | 答案与检索上下文的事实一致性 | ≥ 0.75 | 降低 temperature、强化约束提示、换指令遵循更强的模型 |
| Answer Relevancy | 是否真正回应了问题 | ≥ 0.80 | 高忠实度 + 低相关性通常是**被伪装成生成问题的检索问题** |
| Context Precision | 相关上下文是否排在前面 | ≥ 0.70 | 加/换 cross-encoder 重排 |
| Context Recall | 所需信息是否都被召回 | ≥ 0.80 | 增大 top-k、换 embedding、混合检索、查询改写 |

> 指标要**组合起来读**：`高忠实度 + 低精确率` 是检索问题；`低忠实度 + 上下文正确` 是生成问题；
> `低忠实度 + 答案却正确` 最危险——模型绕开了检索、直接用训练数据作答。

## 目录

```
eval/
├── build_golden_dataset.py   # 从语料生成黄金数据集（LLM 合成）
├── metrics.py                # 四指标实现（LLM-as-Judge，零额外依赖）
├── evaluate.py               # 评测主脚本
├── datasets/                 # 数据集（不进版本库）
└── reports/                  # 评测报告（不进版本库）
```

## 快速开始

```bash
# 1. 生成黄金数据集（需要已入库，语料缓存存在）
python eval/build_golden_dataset.py --limit 100 --per-chunk 1

# 2. 人工抽检 20~30 条，剔除不可回答或答案有误的样本

# 3. 评测
python eval/evaluate.py --dataset eval/datasets/golden.jsonl
```

## 数据集格式

JSONL，每行一条：

```json
{"id": "q-0001", "question": "4 层板的阻抗控制需要关注哪些参数？", "ground_truth": "...", "source_path": "GB-T 4588.4-2017.pdf"}
```

字段说明：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `question` | 是 | 用户问题 |
| `ground_truth` | 是 | 参考答案（用于 context_recall 与人工对照） |
| `id` | 否 | 样本编号 |
| `source_path` | 否 | 期望命中的来源文档（便于分析召回失败） |

你也可以完全不用自动生成脚本，直接手工编写数据集。

## 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--dataset` | `eval/datasets/golden.jsonl` | 数据集路径 |
| `--out` | `eval/reports/report.json` | 报告输出路径 |
| `--limit` | `0` | 评测条数上限（0=全部） |
| `--top-k` | `5` | 最终送入生成的上下文条数 |
| `--recall-k` | `50` | 精排前的召回条数 |
| `--no-rerank` | - | 关闭精排，用于对比重排收益 |
| `--retrieval-only` | - | 只评检索（context_precision / context_recall），更快更省 |
| `--metrics` | 全部 | 指定指标，逗号分隔 |

## 典型对比实验

```bash
# 基线
python eval/evaluate.py --out eval/reports/base.json

# 对比 1：关闭精排
python eval/evaluate.py --no-rerank --out eval/reports/no_rerank.json

# 对比 2：关闭 Contextual Retrieval 重新入库后再跑（改 .env 的 CONTEXTUAL_RETRIEVAL_MODE=off）
# 对比 3：关闭查询分解（QUERY_DECOMPOSE_ENABLED=0）
```

## 成本参考

用便宜模型当 judge 时，200 条样本、4 个指标的单次评测成本通常在 **1 美元以内**，
适合每次改动后都跑一遍。

## 说明

- 指标实现为自包含的 LLM-as-Judge（见 `metrics.py`），不依赖 `ragas` 等外部框架，
  避免引入额外的依赖树；如需切换到官方 RAGAS 口径，可自行替换 `metrics.py` 中的函数。
- LLM judge 存在偏差，建议定期人工抽检并与自动分数比对校准。

---

# 召回评测：`evaluate_recall.py`

本目录下有两套评测，**度量对象不同，不要混用**：

| 脚本 | 度量对象 | 指标 | 需要标注 |
| --- | --- | --- | --- |
| `evaluate.py` + `metrics.py` | **生成质量**（端到端） | Faithfulness / Answer Relevancy / Context Precision / Context Recall | 问题 + 参考答案 |
| `evaluate_recall.py` | **检索排序质量** | MRR / MAP / Hit@K / Recall@K / Precision@K / NDCG@K / F1@K（+ 软匹配） | 问题 + 真值 chunk_id |

只看 Hit@K 会掩盖"捞到了但排不对"的问题，因此**必须同时看 MRR 与 NDCG@K**。

## 数据格式

JSONL，每行一条：

```json
{"query": "问题文本",
 "ground_truth_ids": ["chunk_id_1"],
 "ground_truth_text": ["真值 chunk 原文"],
 "ground_truth_metadata": {"source_type": "standard", "source_path": "GB-T 4588.4-2017.txt"}}
```

- `ground_truth_ids` 用于 Hard Match（chunk 级精确命中）
- `ground_truth_text` 用于 Soft Match（语义命中判定）
- **本仓库数据集每条问题只绑定 1 个真值 chunk**，此时 `recall@K` 与 `hit_rate@K` 数值恒等，
  `recall` 字段没有独立信息量

## 无显卡 / 纯 API 部署

脚本本身不依赖本地 GPU，只要满足：OpenAI 兼容的 LLM + Embedding + Rerank，
以及可访问的 Milvus。`.env` 关键项：

```bash
LLM_BACKEND=api
EMBED_BACKEND=api
EMBED_MODEL=BAAI/bge-m3
EMBED_DIM=1024          # 必须显式填，Milvus 建表要用
RERANK_BACKEND=api      # 默认是 qwen3reranker（本地模型），纯 API 环境必须改
RERANK_API_URL=https://api.siliconflow.cn/v1/rerank
RERANK_API_MODEL=...
```

> `RERANK_BACKEND` 默认值是本地模型；不改会在无 GPU 机器上初始化失败并回退为"仅召回"，
> 此时 `--rerank` 实际不生效，指标会明显偏低。

## 常用命令

```bash
python -m py_compile eval/evaluate_recall.py            # 部署后先做语法自检

# 基线：融合 + 多查询扩展 + 重排（最接近线上）
python eval/evaluate_recall.py \
  --dataset ./eval/eval_dataset.json \
  --mode fusion_expand --rerank \
  --recall-k 200 --rerank-top-n 10 \
  --ks 1,3,5,10,20 \
  --out ./eval/reports/base.json

# 消融：关掉重排做对比
python eval/evaluate_recall.py --mode fusion_expand --no-rerank --out ./eval/reports/no_rerank.json

# 消融：RRF 平滑参数 k 扫描（线上默认 40，本脚本默认 60 —— 见下方注意事项）
for k in 10 20 40 60 80; do
  python eval/evaluate_recall.py --mode fusion_expand --rrf-k "$k" --out "./eval/reports/k_$k.json"
done

# 软匹配（语义命中，用 embedding 阈值判定）
python eval/evaluate_recall.py --mode fusion_expand --rerank \
  --soft-match embed --embed-threshold 0.78 \
  --out ./eval/reports/soft_embed.json
```

## 注意事项（读数前必看）

1. **`--rrf-k` 默认 60，而线上 `query.py` 的 `FUSION_RRF_K` 默认 40**，两者从未对齐。
   要复现线上行为需显式传 `--rrf-k 40`；要比较"评测口径"与"线上口径"的差异就分别跑。
2. **`--recall-k` 默认 40，而线上 `RECALL_TOP_K` 默认 200**。同样需要显式对齐。
3. **`--rerank-top-n 0`（默认）会沿用 `query.py` 的 `RERANK_TOP_N=200`**，即 200 条候选全部过重排。
   纯 API 部署下这是 200 条/次的远程调用，建议显式传 `--rerank-top-n 10`。
4. **Hit@K 不衡量顺序**：`Hit@10` 高只说明"捞到了"。同时报 `mrr` 与 `ndcg` 才有意义。
5. **真值由 chunk 反推问题生成**（`build_golden_dataset.py`），该 chunk 按定义相关，
   因此所有指标都会被系统性高估。**指标只用于同数据集内的配置对比，不代表线上召回率。**
6. **跨数据集不可比**：仓库内多版数据集互不重合，换集后的分数不能与旧集对比。
7. 软匹配的 LLM 判定默认复用 `LLM_MODEL`（与生成同源）。要消除同源偏差，
   用 `SOFT_JUDGE_*` 指向另一个厂商的模型。
8. `multipath_colbert` / `multipath_full` 需要本地 ColBERT 模型，纯 API 环境会打印提示
   并自动降级为 RRF 融合，不要把这些 mode 的分数当作 ColBERT 的效果。
