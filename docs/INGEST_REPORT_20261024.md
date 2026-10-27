# PCB 知识库清洗入库与建图报告（2026-10-24 批次）

## 一、结论

本批次全量入库已于 2026-10-26 18:19 完成原子发布，十一项校验全部通过。

| 项目 | 值 |
|---|---|
| 集合 | `pcb_kb_nv35_embed8b_clean_20261024` |
| revision | `f0adc4daf0b6c8d6697c54d8` |
| 文档数 | 66 |
| 阶段状态 | chunks / vectors / lexical / graph 均为 complete |
| pending | 已清空 |

## 二、数据规模

| 指标 | 数量 |
|---|---|
| 文档 | 66 |
| 父块 | 3,477 |
| 子块 | 5,747 |
| 快照节点合计 | 9,224 |
| 图谱实体 | 51,802 |
| 图谱关系（去重后） | 57,373 |
| 关系贡献记录 | 61,966 |
| 实体描述向量 | 51,802 |
| 关系描述向量 | 61,966 |
| 描述向量合计 | 113,768 |
| 嵌入维度 | 4,096 |

嵌入模型为 `nvidia/nemotron-3.5-embed-8b`，接口 `https://integrate.api.nvidia.com/v1`。

## 三、产物

### 3.1 快照文件

路径位于 `data/kb_state_pcb_kb_nv35_embed8b_clean_20261024_snapshots/f0adc4daf0b6c8d6697c54d8/`。

| 文件 | 大小 |
|---|---|
| `chunks.jsonl` | 29.3 MB |
| `lexical.jsonl` | 19.9 MB |
| `graph.json` | 127.5 MB |
| `graph.vectors.json` | 136.0 MB |
| `graph_parts/*.json` | 66 个分区 |

### 3.2 Milvus 集合

| 集合 | 行数 |
|---|---|
| `pcb_kb_nv35_embed8b_clean_20261024` | 5,747 |
| `pcb_kb_nv35_embed8b_clean_20261024_entities_cd439502e09d` | 51,802 |
| `pcb_kb_nv35_embed8b_clean_20261024_relations_17600a07d81d` | 61,966 |

## 四、时间线

| 时间 | 事件 |
|---|---|
| 2026-10-24 12:58 | 解析与向量写入启动 |
| 2026-10-24 13:59 | 子块向量写入完成，词法快照生成 |
| 2026-10-25 15:05 | 图谱分区 66/66 完成 |
| 2026-10-25 23:27 | 任务因会话终止中断，进度保留 |
| 2026-10-26 10:07 | 续跑，跳过解析、向量与词法阶段 |
| 2026-10-26 18:19 | 图向量完成，原子发布 |
| 2026-10-26 18:57 | 实体向量补数完成 |

图向量阶段最后一段实测 43.4 条/秒；此前因上游接口限流，长期处于 10 至 20 条/秒区间。

## 五、故障与处置

### 5.1 图向量阶段重试死循环

`build_graph_vectors` 以固定 32 条为一批嵌入，全部结果在内存聚合完成后才一次性写入 Milvus，中途无断点。上游接口出现 504 时，整个阶段归零，监督脚本的六次重试均在同一位置失败。

处置：在不修改任何参与指纹计算的源文件的前提下，通过 `.workbuddy/resume_graph_vectors.py` 在运行时替换该函数，实现批量 128、四路并发、指数退避、失败二分降批、逐窗口写入 Milvus 并记录断点。改造后稳态吞吐由 170 条/分钟提升至 768 条/分钟，峰值窗口达 2,000 条/分钟以上。

### 5.2 指纹变更导致阶段表被清空

首次改造时，启动脚本先导入 `pcb_rag.graph_semantic`、后导入 `pcb_rag.ingest`。`graph_rag.py` 在导入时读取 `GRAPH_RAG_ENABLED` 与 `GRAPH_EXTRACT_BACKEND`，而默认值由 `ingest.py` 在导入后设置，导入顺序改变使两项取值变为 `False` 与 `rule`，图阶段指纹随之变化。`RevisionStore.begin()` 判定 revision 变更（`f0adc4…` 变为 `41e13f…`），清空全部阶段记录，66 个分区记录丢失并开始重新切块。

处置：立即终止任务，以 `.workbuddy/_repair_state.py` 拦截 `run_ingestion` 取得真实 `docs` 与 `pipeline`（不写状态），验证重算 revision 与 `f0adc4daf0b6c8d6697c54d8` 逐字符一致后，按现有快照重建 69 个阶段记录（chunks、vectors、lexical 及 66 个分区）。原状态备份于 `data/kb_state_pcb_kb_nv35_embed8b_clean_20261024.json.repair-backup`。修订后启动器改为先导入 `ingest` 再打补丁，并设置 `INGEST_EXPECTED_REVISION` 护栏，使指纹异常在写入状态前即报错中止。

分区文件全程未受影响，无需重建。

### 5.3 Milvus 重启导致实体向量缺失

发布完成约一分钟后，Milvus 组件（etcd、minio、milvus）出现反复重建，WSL 实例本身未重启，磁盘使用率 3%、inode 使用率 1%，排除容量因素。重启期间未落盘的实体向量缺失 3,263 条，关系集合 61,966 条与子块向量 5,747 条均完好。

处置：`graph.vectors.json` 保留每条实体的原文，据此编写 `.workbuddy/_backfill_entity_vectors.py` 重新嵌入缺失项并写回原集合，脚本按缺失集合逐轮重算，具备幂等性。补数分两次完成，最终一轮写入 2,239 条，复核差异为 0。

## 六、校验结果

校验脚本为 `.workbuddy/_verify_publish.py`，一律按 id 集合比对，不采用 `row_count`（该统计不含未 flush 数据）。

| 序号 | 检查项 | 结果 |
|---|---|---|
| 1 | 活跃 revision 与预期一致 | 通过 |
| 2 | pending 已清空 | 通过 |
| 3 | 产物文件与阶段状态齐全 | 通过 |
| 4 | 图向量记录数（51,802 + 61,966） | 通过 |
| 5 | 嵌入身份一致 | 通过 |
| 6 | 子块数量 5,747 | 通过 |
| 7 | 子块向量 id 集合与快照一致 | 通过，差异 0 |
| 8 | 实体向量 id 集合与 payload 一致 | 通过，差异 0 |
| 9 | 关系向量 id 集合与 payload 一致 | 通过，差异 0 |
| 10 | 清单 66 篇且 revision 对齐 | 通过 |
| 11 | 图谱规模 57,373 条边 | 通过 |

## 七、遗留风险

1. **图向量阶段无断点**。`graph_semantic.build_graph_vectors` 仍为固定批量、全内存聚合的实现。改进需修改该文件，将改变 `GRAPH_IMPLEMENTATION` 指纹并触发重建，宜在下一次全量入库时一并处理。
2. **指纹计算依赖导入顺序**。`ingest.py` 在模块级设置 `GRAPH_*` 默认值，若其他入口先导入 `graph_rag`，默认值不生效且指纹随之改变。宜将默认值设置前移至 `graph_rag` 读取之前，或在入口处显式校验。
3. **Milvus 稳定性**。本批次发布后出现一轮组件重建，成因尚未确认，容器日志因运行账户不在 docker 组而无法读取。建议评估容器内存上限与集合加载策略，并保留仅加载当前 revision 的能力。
4. **长任务保活**。当前续跑依赖工具后台会话，会话终止即中断。建议改为 systemd 单元或容器化任务。
5. **补数脚本应固化**。`_backfill_entity_vectors.py` 目前为一次性脚本，建议并入运维脚本目录并补充测试。

## 八、复现与运维命令

以下命令在 WSL 项目根目录执行，虚拟环境为 `/home/pi/.local/share/pcb-rag/venv`。

```bash
# 全量入库（原流程，图谱阶段无断点）
bash .workbuddy/run_ingest_clean_20261024.sh

# 续跑图向量阶段（带断点、四路并发、失败降批）
bash .workbuddy/run_resume_graph_vectors.sh

# 发布后校验
/home/pi/.local/share/pcb-rag/venv/bin/python .workbuddy/_verify_publish.py

# 实体向量补数
bash .workbuddy/run_backfill_entity_vectors.sh
```

关键环境变量：`COLLECTION`、`INGEST_MANIFEST_PATH`、`LEXICAL_CACHE_PATH`、`GRAPH_LLM_WORKERS=12`、`GRAPH_LLM_GLEANING=0`、`INGEST_EXPECTED_REVISION=f0adc4daf0b6c8d6697c54d8`、`GRAPH_VECTOR_BATCH=128`、`GRAPH_VECTOR_WORKERS=4`、`GRAPH_VECTOR_TIMEOUT=20`。
