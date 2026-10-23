# Embedding 修复后的六组实验重跑

本次按 `TestRule.md` 的六组要求及用户明确要求的重排用量保护协议运行，数据保存在：

```text
eval/reports/rerun_20261020_siliconflow_guarded/
```

集合为 `pcb_kb_nv_query_passage_20261020`；5680 个原切块、2048 维向量；60 个 TXT 文件；冻结回归集 80 题，全部真值 ID 存活。语料、缓存、数据集指纹和脱敏配置记录在 `manifest.json`。

当前重排已按用户最新要求恢复 SiliconFlow `BAAI/bge-reranker-v2-m3`，接口 `https://api.siliconflow.cn/v1/rerank`。旧 SiliconFlow 批次和旧 Voyage 批次均保留，新批次不复用旧测量结果。初始召回保持 200，但 API 输入硬上限改为 50，普通返回数为 10，正文不附带技术元数据。先通过小样本预检，再启动完整实验；细节见 [重排用量保护](RERANK_GUARDS_20261020.md)。

两条基线和 13 组独立消融共用最多 3 个并行槽位，每组独立进程和输出文件。基线与消融耗时从本批次开始即标注为 `shared_resources`，不能解释为隔离环境下的延迟；专门的并发压测仍独占队列，依赖闸门保持生效。主动间隔为 0，服务端 429、Retry-After 和有限重试全部记录。用户最新要求不限制 Rerank 总量，累计 token 和累计请求上限均设为 0，用量记录继续。服务端 TPM 限流后默认退避 60 秒再试，正常请求不主动等待，402 等错误仍会停止整队列。

用户进一步要求持续保持三组并行后，满足依赖的差异检查、语料统计、QA 生成、质量评估和 AI 审核也可补入这三个总槽位；专门压测和最终汇总仍独占。软匹配诊断实际只读取 Path A 的检索结果，因此前移到该基线完成后即可运行，不再等待无数据依赖的生成质量阶段。统计中的 BM25 构建耗时也属于共享资源测量。模型可用性预检失败的任务继续复查，不以假结果标记完成。

2026-10-20 用户指定将 QA 出题模型从 `qwen3.7-plus` 改为 `deepseek-v4-1-261017`。切换时尚未生成 golden 题目；答题和判卷模型各自保留。当前压测继续独占，结束后队列预检新模型并启动 100 题生成。切换记录保存在 manifest.policy_changes。

## 运行内容

1. 两条 80 题基线：评测脚本检索器和生产 `_retrieve_nodes`，按用户要求允许与独立消融并行，记录 MRR、统一截断的 MRR@10、Hit@1/3/5/10、NDCG@10、逐题结果和耗时。保留两条实现的默认参数差异，并单独记录超过 0.03 的差异；共享资源下的耗时不作为隔离延迟结论。
2. 13 行消融：BM25、向量、向量加重排、融合、同义扩展、多变体、路由、分解/Step-back、HyDE，以及 recall50、rerank50、rule、chunk expansion。每行独立进程，最多 3 行并行；最后一行按文档使用 top_k=5。
3. 两个接口、并发 1/4/8、每格 60 次请求，共 360 次测量，均关闭语义缓存。独立测试服务在 18000 端口，8000 端口生产服务保留。保存原始延迟、错误、trace ID、CPU/RSS、分段指标与服务端 HyDE/429 事件。
4. 新生成 100 题，AI 抽检 30 题后评测保留样本。生成器 `deepseek-v4-1-261017`、答题模型 `deepseek-v4-1-flash-260910`、judge `glm-5-3-260814`；三者不同。四指标沿用原提示和公式，独立 judge 调用并行，失败以不可判定记录。
5. 复用本轮 Path A 的真实检索结果，计算 0.70/0.78/0.85/0.90 阈值的软匹配，导出差异样本，AI 审核最多 20 条并进行 A/B/C/D 分类。
6. AI 编写 30 个拟真工程问题，十类各三题，包含口语、缩写、错字和多段证据需求。生产路径检索 top_k=5 后逐题进行 AI 证据充分性审核。

用户明确授权由 AI 替代人工环节。题目与审核均标注 AI 来源，不能将第六组称为真人来源测试，也不能将精确 ID 命中和 AI 证据审核混为相同指标。

## 断点、日志与可用性

每题写入 JSONL，阶段结果写 JSON；每个阶段有独立日志。模型参数从 WSL 用户目录的权限为 0600 的私有文件读取，报告内不保存 API Key 或 Token。

首轮热身出现上游 500 和长时间超时，诊断日志保留。队列在 LLM 阶段前使用原模型做简短可用性探测；不满足预检时等待五分钟后重试，独立的非 LLM 阶段可以先运行。未更换生产模型或降低测量标准。恢复后仍使用同一份冻结配置。

配置复核中发现一次试跑受 `.env` 加载顺序影响，使用了查询变体数 4。该段数据和日志已移到 `diagnostics/preflight_production_env4/`，不进入正式结果。正式运行严格导出 `.env.eval` 的参数（查询变体数为 3），并在实验进程中禁止自动回读生产 `.env`。每个阶段都保存实际生效的参数。

查看状态（WSL 项目目录）：

```bash
.venv/bin/python scripts/experiment_status.py
```

Windows PowerShell：

```powershell
wsl -d pure2404 -- bash -lc 'cd /mnt/d/myProject/rag_v1-main/rag_v1 && .venv/bin/python scripts/experiment_status.py'
```

队列在 tmux 会话 `pcb-rag-experiments` 中运行。关闭当前终端后会继续；电脑、WSL 和 Windows 的 12345 代理需要保持运行。

WSL 重启后可恢复：

```bash
cd /mnt/d/myProject/rag_v1-main/rag_v1
source "$HOME/.config/pcb-rag/proxy.sh"
tmux new-session -d -s pcb-rag-experiments \
  'cd /mnt/d/myProject/rag_v1-main/rag_v1 && . /home/pi/.config/pcb-rag/proxy.sh && exec .venv/bin/python scripts/experiment_queue.py --run-dir eval/reports/rerun_20261020_siliconflow_guarded >> eval/reports/rerun_20261020_siliconflow_guarded/queue.log 2>&1'
```

## 数据文件

- `progress.json`、`heartbeat.json`：当前阶段、逐题进度及完成状态。
- `retrieval/*.jsonl`：所有检索问题的原始排名、文本、元数据、耗时、错误和降级记录。
- `logs/`：每个阶段的独立日志。
- `load/`：360 次请求的延迟、状态码、服务端指标和资源采样。
- `datasets/`、`quality/`、`reviews/`：新测试集、生成质量和 AI 审核明细。
- 队列完成后生成 `REPORT.md`、`summary.json`、`ablation_summary.csv`、`retrieval_summary.csv`、`load_summary.csv`、`stage_timing_summary.csv`、`generation_quality_summary.csv`、`softmatch_summary.csv`。

API 后端无法提供远程 GPU 利用率和显存，这两项记录为不可用。冻结期间不重新入库，因此不伪造新的全量入库耗时。并行阶段耗时有重叠，不能相加当成总延迟。
