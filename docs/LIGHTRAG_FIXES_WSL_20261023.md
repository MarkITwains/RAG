# LightRAG 对比问题修复与 WSL 验证

日期：2026-10-23。对应原审查 `LIGHTRAG_REVIEW_20261023.md` 的七类工程问题；原审查与历史评测数据保留。以下结果来自本次修复完成后的 WSL 验证。

## 已完成的修复

| 范围 | 当前实现 |
| --- | --- |
| 证据、引用与预算 | EvidenceBundle 保存最终顺序、实际入选文本和引用编号；去重、降权、裁剪后仍一致。完整请求计入 query、history、system、标题、引用及输出预留；引用编号有效性与事实支持程度分开报告。 |
| 可恢复入库与版本 | 内容及解析、向量、词法、图配置生成阶段指纹；先构建和校验快照，再原子发布。失败保留旧版，按阶段重试；丢失的图产物重建，完成的 chunk 嵌入复用。 |
| 图语义检索 | 实体描述、关系描述各有向量索引，支持 Milvus 或本地余弦检索；图可独立挽救基础零召回，命中回查原文并复核权限及版本后参与融合、精排。 |
| 图来源维护 | 每条关系保存独立来源贡献；删除一份文档不会删除其他来源支持的关系；合并、重跑幂等，空图持久化。 |
| 共享内核 | HTTP、SDK、CLI、评测调用 RetrievalPipeline；Self-RAG 的后续查询实际重新检索；查询扩展使用注入模型，降级状态来自本次请求。 |
| PCB 结构证据 | 保存真实 parent，子块单独嵌入；PDF 分页身份、表头/单位/脚注和续表关联；提取标准版本、条件、数值及单位。修复 OCR 清洗误删 μ、Ω、I、J。 |
| 运行与管理 | 总 deadline、有界并发/队列、断开取消、LLM/embedding/rerank 用量；tenant/user 会话归属；`/manage` 文档状态、失败重试、版本与图来源查询；收窄 lint 豁免并补齐 CI 依赖。 |

## 实际验证

WSL 发行版 `pure2404`，工作目录 `/mnt/d/myProject/rag_v1-main/rag_v1`，Python 3.12，解释器 `.venv/bin/python`。

- 全量 pytest：**594 passed，0 failed，3 warnings，99.65 秒**。显式启用真实 Milvus 集成测试；没有将该测试跳过后报告成功。
- JUnit 结果确认 **0 skipped、0 errors**。WSL `ruff check .`、`ruff format --check tests`、`compileall -q src eval scripts tests`、`pip check`、CLI/迁移命令 `--help` 均通过。
- WSL `python -m build --no-isolation` 成功构建 sdist 和 wheel；wheel 已检查包含共享检索模块与 `webui/manage.html`。安装包保存在 `.workbuddy/tmp/lightrag-fixes-20261023/dist/`。
- 命令：`PCB_RAG_TEST_MILVUS=1 REQUIRE_LLAMA_INDEX=1 RAG_VALIDATION_REPORT=.workbuddy/tmp/lightrag-fixes-20261023/milvus.json .venv/bin/python -m pytest -q --junitxml=.workbuddy/tmp/lightrag-fixes-20261023/pytest.xml --disable-warnings --maxfail=1 -o faulthandler_timeout=120`。
- Milvus 测试覆盖实际 `ingest.main()`、chunk/实体/关系向量索引、原父块、无变化跳过、失败后保留旧版、恢复不重嵌入 chunk、共享来源删除、新集合校验与热切换、HTTP 引用、最后来源删除后的空图读取。
- 使用确定性 LLM/embedding 和临时语料，不发送付费模型请求。临时集合自行清理，既有集合保留情况写入 `milvus.json`。
- 最终集成报告确认：测试创建的 **10 个临时集合已清理**，`production_collections_preserved=true`。
- 通用 embedding 的同步/异步/批量、失败计账、预算拦截另有 9 项回归，已包含在全量结果中。
- 管理页 HTTP 行为由全量回归覆盖；JavaScript 用 Windows Node.js v24.14.1 `node --check` 校验通过。没有宣称完成浏览器交互验收。
- 依赖版本记录在 `requirements/constraints-wsl-py312.txt`，包含 117 个解析后的依赖版本，不含机器本地 editable 路径；适用于 WSL/Linux Python 3.12 的 runtime、ollama、dev 配置。本地 torch 模型和模型权重不包含在该快照中。

本地验证产物：`.workbuddy/tmp/lightrag-fixes-20261023/pytest.xml`、`milvus.json`。此目录被 Git 忽略，公开可读的结果说明保存在本文件。

## 后续部署操作

本轮没有迁移生产语料、修改部署 `.env` 或切换生产集合。旧数据要使用新父块、元数据和图索引，需执行迁移：

```bash
# 原状态路径应与 API 使用的 KB_STATE_PATH 相同；按实际集合名替换示例。
python -m pcb_rag.migrate --state ./data/kb_state_pcb_kb.json --target-collection pcb_kb_v2
python -m pcb_rag.migrate --state ./data/kb_state_pcb_kb.json --target-collection pcb_kb_v2 --execute
# 核对生成的 migrations/pcb_kb_v2/state.validation.json 后激活；复用已完成阶段。
python -m pcb_rag.migrate --state ./data/kb_state_pcb_kb.json --target-collection pcb_kb_v2 --execute --publish
```

有评测真值时，同时传入 `--mapping-from 旧词法缓存.jsonl --truth-ids 旧chunk编号数组.json`。`--truth-ids` 必须是 JSON 字符串数组；要求的 ID 无法唯一映射时禁止激活。激活后 API 与入库保持相同状态文件，后续入库的 `COLLECTION` 对应新集合。`INGEST_OVERWRITE=1` 对非空集合报错，不能用于清空重建。

已有新版 chunk 快照且只需补图时，可用 `python -m pcb_rag.migrate --state 状态文件 --stage graph --execute`。此操作在成功后直接发布新 revision；源内容及切块配置未变时复用 chunk 向量。

## 验证边界

本轮证明工程回归与临时 Milvus 链路通过，没有运行新生产效果基线或同数据、同模型的 LightRAG 对跑，不能据此宣称准确率、召回率或速度提升。Ollama 客户端依赖已安装，本地模型推理未测试。同步供应商请求开始后可能继续执行；取消会停止后续阶段并保留工作槽至实际结束，迟到响应继续记录用量。
