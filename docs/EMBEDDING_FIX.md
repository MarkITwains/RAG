# NVIDIA Embedding 输入模式修复

部署日期：2026-10-20。继续使用外部 NVIDIA API 的 `nvidia/nemotron-3-embed-1b`，不运行本地模型。

## 修复内容

此前 `build_embed_model()` 对 NVIDIA API 直接使用通用 `OpenAIEmbedding`，没有指定 `input_type`。新的 `NvidiaApiEmbedding` 为查询发送 `query`，为单篇和批量文档发送 `passage`。同步与异步接口使用相同约定；每次请求独立构造参数，避免并发调用互相覆盖模式。

请求显式使用 `encoding_format=float` 和 `truncate=END`。客户端检查向量维度、数值有效性和批量响应的输入索引，防止缺失、重复或乱序结果关联到错误文档。保留原客户端的换行归一化方式。

`.env`、`.env.eval` 当前配置为：

```dotenv
EMBED_BACKEND=api
EMBED_API_PROVIDER=nvidia
EMBED_MODEL=nvidia/nemotron-3-embed-1b
EMBED_DIM=2048
EMBED_TRUNCATE=END
COLLECTION=pcb_kb_nv_query_passage_20261020
```

`EMBED_API_PROVIDER=auto` 也能识别 NVIDIA 官方接口域名；自建 NVIDIA NIM 应显式使用 `nvidia`。其他 OpenAI 兼容服务仍走原来的客户端。

## 数据迁移和防混用

迁移脚本为 `scripts/repair_embeddings.py`。它读取旧集合已有实体，按原来的 `MetadataMode.EMBED` 恢复完整嵌入输入。切块正文、上下文前缀、ID、父子和邻接关系以及其他动态元数据全部保留，仅生成新的向量。

- 原集合：`pcb_kb`。
- 新集合：`pcb_kb_nv_query_passage_20261020`。
- 两个集合均为 5680 条、2048 维。
- 比对全部非向量字段的 SHA-256，确认两份语料完全一致；旧集合保留。
- 检索缓存和增量清单使用新集合专属文件，避免影响旧配置。

新集合的 description 保存向量生成配置，包括接口地址、模型、维度、查询与文档模式，以及文本构造规则。查询启动与入库都会校验该配置。已有数据但没有配置记录的集合，不能直接使用新的 NVIDIA 模式。全新空集合可在入库前登记配置。

迁移进度保存在 `data/embedding_migrations/<target>/`；中断后用同一条命令可继续。目标已存在但没有对应迁移记录时，脚本拒绝覆盖。脚本自身不会修改生产配置。

```bash
source "$HOME/.config/pcb-rag/proxy.sh"
.venv/bin/python scripts/repair_embeddings.py \
  --source pcb_kb \
  --target pcb_kb_nv_query_passage_20261020 \
  --batch-size 64 --min-interval 2 \
  --report eval/reports/embedding_query_passage_20261020.json
```

## 同题对照结果

使用冻结集 `eval/datasets/regression_v1.json` 的全部 80 题。所有配置使用同一批 5680 个实体，以精确 IP 排名比较；不加业务过滤、BM25、重排、改写或 HyDE。

- 原查询模式加原向量：MRR@20 为 **0.068966**，Hit@5 为 **0.1000**，Hit@200 为 **0.4000**。
- 仅修正查询、保留原向量：MRR@20 为 **0.096103**，Hit@5 为 **0.1375**，Hit@200 为 **0.5125**。
- 查询使用 query、文档重建为 passage：MRR@20 为 **0.393509**，Hit@5 为 **0.5875**，Hit@200 为 **0.9375**。

新向量排名同时通过独立矩阵计算和 Milvus 的 FLAT/IP 检索验证，两者的上述指标一致。该结果衡量这套冻结集上的向量检索变化；完整问答质量需要另行评估。

详细逐题排名、配置指纹和部署验证记录在 `eval/reports/embedding_query_passage_20261020.json`。该文件含内部问题文本，按项目已有规则不纳入 Git。

## 验证

WSL 中执行 `ACL_TRUST_HEADERS=0 REQUIRE_LLAMA_INDEX=1 .venv/bin/python -m pytest -q -rs`，358 项全部通过。新增测试覆盖同步、异步、批量、并发模式隔离、响应索引和维度检查、集合配置防混用，以及迁移时的语料一致性。真实 Milvus 的空集合配置登记也已验证。

API 已切到新集合，健康明细确认 `query` / `passage` 模式；完整中文问答返回正确答案和引用。新修改文件的 Ruff 检查通过。全仓扫描另有 5 处既有 lint 问题和 6 个既有测试文件的格式问题，本次没有修改那些无关文件。

## 回退

配置修改前曾使用 `.env.backup-embedding-20261020-151507` 和 `.env.eval.backup-embedding-20261020-151507`；这两份本地备份已在 2026-10-27 的遗留文件清理中删除，回退不再依赖备份文件。

如需回退，先停止 API，在 `.env` 与 `.env.eval` 中手工恢复旧集合 `pcb_kb` 及其配套缓存、清单，恢复 `EMBED_MODEL=nvidia/nemotron-3-embed-1b`、`EMBED_DIM=2048`，并将 `EMBED_API_PROVIDER` 显式设为 `openai`，然后启动 API。此处 `openai` 表示恢复原来的通用兼容请求方式；模型仍在 NVIDIA 外部接口。必须同时恢复旧集合和旧请求方式，不能让新的 query 模式查询旧向量。

原集合、原缓存及原增量清单均保留。切换集合后，API 需要重启以重新创建索引、模型客户端和内存缓存。
