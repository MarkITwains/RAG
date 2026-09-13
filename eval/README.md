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
