# Voyage Rerank 切换记录

这是历史记录。用户发现高用量后已停止本批次，并要求恢复 SiliconFlow；当前配置及保护机制见 [重排用量保护](RERANK_GUARDS_20261020.md)。旧 Voyage 批次的 42 次 HTTP 402 降级记录不能作为正常重排结果使用。本文中的原始文件路径与“保留/保存”均描述当时状态；Voyage 原始实验目录已于 2026-10-27 清理，现仅保留本文摘要。

2026-10-20 按用户要求将生产与评测的重排配置从 SiliconFlow `BAAI/bge-reranker-v2-m3` 切换为 Emrank 提供的 `voyage/rerank-2.5`。

- `RERANK_BACKEND=api`
- `RERANK_API_MODEL=voyage/rerank-2.5`
- `RERANK_API_URL=https://emrank.tech/v1/rerank`
- Key 仅写入被忽略的本地环境配置及 WSL 用户目录的私有配置文件，不进入报告。
- LLM、NVIDIA Embedding、5680 条向量集合及 80 题冻结回归集沿用原配置，不重新入库。

现有 HTTP 客户端直接兼容该接口：请求带 `model`、`query`、`documents`、`top_n`；响应通过每条结果的 `index` 和 `relevance_score` 映射回原候选。没有修改核心重排或检索实现。

## 接口验证

使用 WSL 的 12345 代理完成实际请求：中文语义测试把相关文档排在首位；200 条真实语料候选得到 200 个有效分数，索引完整，耗时约 3.03 秒；3 个同时发出的 20 候选请求全部返回 HTTP 200，单次约 1.52–1.57 秒。该小样本仅验证接口与并发可用性，不代表全量性能或持续配额。原始记录保存在新批次的 `diagnostics/reranker_preflight.json`。

## 实验与恢复

旧批次 `eval/reports/rerun_20261020_api_nvidia_fixed/` 停止后保留原始数据，并标注 `superseded`。新批次 `eval/reports/rerun_20261020_api_nvidia_voyage25/` 从两条基线开始重跑全部 27 步，旧模型数据不混入新结果。

用户明确授权尝试并行实验，并于 2026-10-20 18:29 放开基线与消融同时运行。调度器最多同时运行 3 组基线或独立消融；压测等其他步骤独占队列。已有评测脚本基线进程未重启，切换前完成的 22 题保留，整体耗时标为 `mixed_resources`；生产基线与消融标为 `shared_resources`。隔离延迟结论使用专门的压测。依赖闸门、配置隔离、断点记录和失败数据保留继续生效。每个进程原有的 6.2 秒最小重排间隔与 429 重试参数显式冻结。在线调度变更保存在 `diagnostics/baselines_parallel_policy.json` 和 manifest 的 `scheduling_changes`。

无参数运行 `.venv/bin/python scripts/experiment_status.py` 会自动选择最新且未被替代的批次。恢复后台队列的命令见 [实验执行文档](EXPERIMENT_RERUN_20261020.md)。不要同时启动第二个队列。

生产服务重启后从 `.env` 读取新配置，健康检查和带鉴权的真实检索应验证实际模型及无重排降级。切换前的 `.env` 和 `.env.eval` 备份保存在 WSL 私有 experiments 目录，不含在公开报告中。
