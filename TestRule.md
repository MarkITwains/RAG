# PCB-RAG 评测与发布验收标准

- 标准版本：2.0
- 修订日期：2026-10-27
- 修订依据：`1fa6cc7` 及 2026-10-24 入库批次；每次实际评测必须另记代码 commit 与数据 revision。
- 适用范围：共享 `RetrievalPipeline`、`EvidenceBundle`、revision 发布、GraphRAG、NVIDIA embedding/rerank，以及 HTTP、SDK、CLI 和评测入口。
- 本文件是现行标准。文末历史结果只作对照，不是当前生产基线，也不能据此豁免任何闸门。

“必须”表示发布硬要求，“应”表示无正当理由不得偏离，“可”表示诊断或探索项。探索结果不得写成发布结论。

## 一、基本原则

1. **生产链路是效果事实来源。** 大批量评测可调用共享管线，发布前必须再用 `/retrieval` 与 `/api/ask` 做端到端验收，并覆盖流式响应、SDK/CLI 契约。
2. **先确认数据有效，再比较指标。** 数据集、语料、revision、模型或实际配置不一致时，不进入质量排名，也不计算“提升”。
3. **中间排名与最终证据分开报告。** 去重、父子扩展、裁剪后的 `EvidenceBundle` 才是生成和用户引用所见的证据；关闭扩展得到的排名只是诊断。
4. **执行完成不等于验收通过。** 请求失败、静默降级、判卷失败、样本缺失和阶段中断必须显式进入 verdict。
5. **质量、性能、可靠性分别验收。** 共享资源下的质量结果可保留，但其耗时不得用于性能结论。
6. **AI 与人工来源分开。** AI 出题、AI 审核必须标注；AI 拟真集不得称为真人问题集。

## 二、统一 verdict 与放行规则

每份正式报告必须包含一个机器可读 verdict：

| verdict | 含义 | 能否作为发布通过证据 |
| --- | --- | --- |
| `passed` | 必需阶段完整，全部硬门槛通过 | 可以 |
| `failed` | 执行完整，但质量、性能或可靠性未达门槛 | 不可以 |
| `invalid` | 数据、配置、请求、判卷或产物无效 | 不可以 |
| `incomplete` | 阶段未执行、被中断或必需样本缺失 | 不可以 |
| `inconclusive` | 样本量或置信区间不足以判定非劣/提升 | 不可以，需补样或扩大实验 |

具体要求：

- `passed` 的进程退出码必须为 0；其余 verdict 必须非零退出。调度器的 `complete`、`complete_with_failures` 只是执行状态，必须另映射为 verdict。
- 正式评测必须满足 `expected_n == attempted_n`；同时分别记录 `evaluated_n`、`failed_n`、`excluded_n`，不得用成功子集替代完整题集。
- 全功能基线中的请求错误、未解释 rerank 回退和未声明降级必须为 0。故障注入产生的降级进入独立韧性报告，不混入全功能质量基线。
- 不得删除失败题、超时题、难题或不再存活的题目后仍宣称使用原数据集。排除项必须逐条记录原因，并同时保留原始分母。
- 豁免必须包含 `scope`、`reason`、`approver`、`issue`、`created_at`、`expires_at`。过期自动失效。
- 数据无效、跨租户泄漏、revision 不一致、必需测试未执行和伪造/缺失产物不得豁免。任何豁免不得改变原始指标。

## 三、分层门禁与触发条件

| 门禁 | 内容 | 触发时机 |
| --- | --- | --- |
| G0 | 数据集、配置、代码与产物有效性 | 所有正式评测之前 |
| G1 | 静态检查、构建、单元与契约测试 | 每个 PR、每次发布 |
| G2 | 真实 Milvus、故障恢复、权限与版本发布 | 每日或发布前；入库/图/revision 变更必须执行 |
| G3 | 生产检索、最终证据、生成与 GraphRAG 效果 | 检索/模型/提示词/语料/切块变更及发布前 |
| G4 | 延迟、吞吐、错误率与资源用量 | 发布前；性能相关变更必须执行 |

- 纯文档或纯测试说明变更可只执行文档检查及受影响的 G1。
- 纯逻辑代码变更至少执行 G0+G1；共享检索、证据、API 或会话变更还必须执行相关 HTTP 契约与 G3 冒烟。
- embedding、rerank、切块、语料、GraphRAG、入库或发布实现变更必须执行 G0～G4，不能只跑向量单路后放行。
- 仅新增/修改语料时，必须同时跑稳定回归集和新增语料验收集；不必无条件重复全部参数扫描，但必须覆盖受影响链路和回滚。

## 四、G0：数据、配置与产物有效性

### 4.1 冻结清单

每次正式运行必须生成脱敏 `manifest.json`，至少包含：

- `schema_version`、`run_id`、`code_commit`、`dirty=false`、核心代码指纹；本地 dirty 运行只能作探索，不能成为发布证据。
- `collection`、活跃 `revision`、文档数、chunk/entity/relation 数量、语料非向量字段哈希、快照产物哈希。
- 数据集路径与 SHA-256、`expected_n`、数据集版本、排除项及原因。
- 实际生效的脱敏配置、环境文件摘要、`PipelineConfig.fingerprint`、入口和启动顺序。
- embedding 的 provider/model/dimension/query/passage/truncate；rerank 的 provider/model、输入候选上限、输出 top_n、truncate。
- 出题、答题、judge 的 backend/model/版本；图实现与图 revision；随机种子（后端支持时）。
- `timing_mode=exclusive|shared_resources`、并行度、热身规则和运行时补丁指纹。

密钥、token、私有问题正文和语料不得写入公开清单。任何影响产物或行为的运行时 monkeypatch 都必须版本化并纳入指纹；未声明补丁不得用于发布。

### 4.2 `regression_v3` 数据标准

现有 `regression_v2_clean20261024` 只能作为重标注输入，不能直接作为 v3 的已验收硬真值：

- 80 条中仅 27 条达到当前脚本的完整文本覆盖，53 条为部分探针匹配；部分覆盖不等于必然错误，但必须重新审核。
- 27 条完整覆盖项也只能优先进入复核，不能自动等同于“问题可回答且证据充分”。
- 当前数据集元数据仍为 `source_type=unknown`，v3 必须使用匹配后当前 chunk 的真实元数据。

v3 每题至少包含：

```text
id, query, answer, acceptable_evidence_sets, evidence_spans,
source_path, source_type, route, risk, authored_by, review_status
```

并满足：

1. JSON/JSONL 解析错误、重复问题 ID、空问题、空参考答案、必需字段缺失均为数据错误，不得静默跳过。
2. `acceptable_evidence_sets` 表示任一足以回答问题的证据集合；多跳题可标多个必需 chunk，同义或重复证据可标多个可替代集合。
3. `evidence_spans` 必须定位实际答案信息，记录当前 physical chunk ID、稳定 logical ID（存在时）、规范化文本哈希和审核人/审核来源。
4. 单证据硬命中题必须确认完整答案 span 位于目标 chunk；前缀探针匹配、部分覆盖或歧义映射不得直接进入该集合。
5. 多证据题、跨 chunk 题和父子块可替代题单独计量，不得强行压成单一 chunk ID。
6. 真值 ID 必须按目标 revision 做存活校验；映射必须同源、可解释且无未决歧义。旧 ID 存活率必须实测，不得写死。
7. `source_path/source_type/route/risk/authored_by/review_status` 必须可用于分片统计；必需元数据不得为 `unknown`。

同时维护两套视图：

- **稳定回归集**：题目与标注跨版本固定，用于同题配对比较。
- **当前语料验收集**：覆盖新增和修改文档。每篇新增/修改文档至少有一条已审核问题，并覆盖十类路由、口语/缩写/错字、表格、数值与单位、标准版本、多证据、证据冲突及不可回答问题。

稳定集与当前集分别出数，不得拼接后宣称历史提升。

### 4.3 配置与环境

- 正式实验从同一份冻结环境启动，每个配置使用独立进程；禁止同进程修改 `os.environ` 后假定模块常量已刷新。
- 显式记录 `RECALL_TOP_K`、`RERANK_TOP_N`、`FUSION_RRF_K`、`FUSION_NUM_QUERIES`、`RAG_TOP_DOCS`、图开关和最终扩展配置，不依赖脚本默认值暗示生产口径。
- 必须覆盖“干净环境”和“预设部署变量的污染环境”，验证 shell、dotenv、显式 CLI 参数的优先级及跨入口一致性。
- 冻结期间不得向被测 collection 写入。故障测试使用唯一临时 collection，绝不用生产库做清库或覆盖。

## 五、G1：工程回归标准

PR 与发布至少执行以下现有检查：

```bash
ruff check .
ruff format --check tests
python -m compileall -q src eval scripts tests
python -m build --no-isolation
PCB_RAG_LOAD_DOTENV=0 REQUIRE_LLAMA_INDEX=1 ACL_TRUST_HEADERS=0 \
  pytest -q -rs -m "not integration" \
  --cov=pcb_rag --cov=eval --cov-report=term-missing
```

通过条件：

- `0 failed`、`0 errors`、`0 unexpected skip`；显式 deselect 的 integration 必须另列，不得伪装成已执行。
- Ruff、格式、字节编译、构建均成功；新增 Python 文件及有行为变更的 Python 文件必须满足格式要求。历史脚本仅同步标准引用时，不强制顺带格式化整个文件。
- 关键依赖缺失必须失败，不能依靠 `importorskip` 静默消失。
- 配置测试必须覆盖模块导入顺序、环境污染、布尔值、CLI 覆盖和产物指纹。
- 评测代码自身必须有单元测试，至少覆盖指标公式、坏数据、空数据、失败退出码、脱敏和报告 schema。

覆盖率采用递进门槛，不用全仓一刀切替代行为测试：

- 首次在本标准下的干净提交记录总覆盖率基线，之后不得下降；不得通过删除测试或排除文件抬高比例。
- 新增/修改的关键逻辑行覆盖率应不低于 80%；`retrieval`、`evidence`、`ingestion`、`revisions`、`graph_semantic`、模型适配器和评测汇总逻辑单列观察。
- 覆盖率采集必须带 `--cov-fail-under` 或等价 CI 检查才具有门禁作用；阈值随已确认基线提升，不得事后调低以放行。

## 六、G2：真实存储、故障恢复与发布标准

真实 Milvus 检查必须在 nightly/release 执行，而非仅靠手工记录：

```bash
PCB_RAG_TEST_MILVUS=1 REQUIRE_LLAMA_INDEX=1 \
PCB_RAG_TEST_MILVUS_URI=<isolated_milvus_uri> \
RAG_VALIDATION_REPORT=<run_dir>/milvus.json \
  .venv/bin/python -m pytest -q -rs --junitxml=<run_dir>/pytest.xml
```

必须使用独立服务和唯一临时 collection，并验证临时集合清理、既有集合保留和报告实际生成。通过条件包括：

1. 现有入库、真实父块、无变化跳过、阶段失败保留旧版、恢复不重嵌入已完成 chunk、共享来源删除、集合切换和 HTTP 引用用例全部实际执行。
2. `ingest`、`graph_rag`、API、CLI、迁移入口按不同顺序导入时，同一输入产生相同有效配置和 revision；`INGEST_EXPECTED_REVISION` 不符必须在任何状态写入前失败。
3. 对 chunks、vectors、lexical、graph、graph_vectors、publish 分别注入异常和进程级中断；恢复必须幂等，无重复 ID，已完成阶段不得无意义重做。
4. 图向量必须在窗口级持久化断点后才可宣称“图向量阶段可恢复”。全量内存聚合、失败后从头重算不满足该项。
5. 发布后及 Milvus 重启后，按 ID 集合比对快照与子块、实体、关系向量，并复核正文、必需元数据、维度、嵌入身份和哈希；不得只信 `row_count`。
6. 图来源贡献、原文 revision 和 ACL 复核率必须为 100%；共享来源删除不影响其他来源，最后来源删除后不再返回该证据，跨租户泄漏必须为 0。
7. 热切换后 API/SDK/CLI 返回同一 revision，旧版在切换失败时仍可读取；回滚必须同时恢复集合、模型、缓存、清单和图产物。
8. 启用图块失败容忍时，失败块数量、比例、原因和清单必须进入报告；未审计失败块不得被静默记为成功发布。

## 七、G3：生产效果标准

### 7.1 基线与测量对象

新基线必须使用同一冻结的当前语料、当前 NVIDIA embedding/rerank、当前 GraphRAG 和生产配置。旧的向量单路结果不能替代完整融合+重排+图+最终证据基线。

模型或供应商变更后，先用固定的真实接口小样本做 canary，再启动全量实验：

- embedding 必须验证 query/passage、返回维度、有限数值、批量索引和 truncate；实际模型、维度和 collection description 必须一致。
- rerank 必须验证对象化请求、候选上限、完整索引、负 logit 和本地 top_n 裁剪；任何契约错误或未声明回退都使 canary 失败。
- sigmoid 只保持单调顺序，不保证跨模型分数可比。使用 `score_threshold` 前必须在已标注数据上校准分数分布，并分别报告阈值开启/关闭结果。

每次至少保留两类结果：

- **排名诊断**：向量/BM25/融合/精排中间顺序，用于定位候选召回和排序问题。
- **生产最终结果**：扩展、去重、预算裁剪后的证据及 `/retrieval`、`/api/ask` 实际返回，用于发布验收。

`evaluate_recall.py` 的独立实现和 Path A/Path B 对比仅保留为历史诊断。原“两路径 MRR 差 0.03”不再是生产放行主闸门，也不得用人工归因直接覆盖阈值。

### 7.2 指标与统计判定

检索至少报告：

- MRR@10、Hit@1/3/5/10、NDCG@10、空结果数、请求错误数、各降级数。
- 候选阶段另报 Recall@200；最终只保留10条时不得报告无意义的最终 Hit@20。
- 按 route、source_type、risk、单/多证据和新增/旧文档分片报告样本量及最差分片；分片少于30题只作探索。

候选与基线必须使用同题逐条配对结果：

- 报告每题 reciprocal rank、命中和 NDCG 差值，以及 paired bootstrap 95% 置信区间。
- 暂定非劣边界为 MRR@10 `-0.02`、Hit@5 `-2` 个百分点。正式跑前可评审调整，但不得看到结果后修改。
- 对 `candidate - baseline`：95% CI 下界高于负非劣边界才可判非劣；上界仍低于该边界为 `failed`；其余为 `inconclusive`。
- 宣称“提升”时，主要指标差值的 95% CI 下界必须大于 0，而不只是点估计为正。
- LLM 路由、HyDE、生成等随机或外部依赖关键配置至少重复 3 次并报告批间波动；确定性向量/BM25 计算可单次执行。

### 7.3 消融设计

逐级加法只用于观察特定路径，不再把每行差值解释为独立贡献。新矩阵包含：

1. 最小底座上的单项加入：vector、BM25、rerank、graph。
2. 完整生产配置上的 leave-one-out：分别关闭 BM25、rerank、graph、同义扩展、多变体、routing、decompose/step-back、HyDE、chunk expansion。
3. 参数诊断：recall50、rerank_top_n50、rule 模式；必要时增加 graph×rerank、expansion×parent 的交互项。
4. 每个配置独立进程、相同冻结集和 revision，保存实际生效参数、逐题结果和降级记录。

`shared_resources` 下可比较质量，但所有延迟比较必须标记不可用；性能只能取 G4 的独占测量。

### 7.4 GraphRAG 专项

必须在相同基础检索和重排配置上比较 graph off/on，并增加：

- 基础 vector+BM25 无有效证据题集上的 `graph_rescue_rate`。
- 多跳、多来源和需要关系限定的题目；报告图路对 MRR/NDCG/证据充分率的配对差值。
- 图证据来源、revision、ACL 和删除后不可见性；关键安全题中的无依据图证据必须为 0。
- 图构建/向量失败数、图路耗时和内存。图集合规模本身不得作为效果提升证据。

### 7.5 生成、引用与判卷

出题、答题和 judge 应使用不同模型或不同厂商，并记录确切身份。judge 必须用 20～30 条人工标注校准，报告一致率、分歧类型和失败数。

以下现有阈值作为首轮逐项门槛，不得用 overall 均值抵消单项不达标：

| 指标 | 首轮门槛 |
| --- | --- |
| Faithfulness | ≥ 0.75 |
| Answer Relevancy | ≥ 0.80 |
| Context Precision | ≥ 0.70 |
| Context Recall | ≥ 0.80 |

同时满足：

- 发布报告中每项指标必须达到其 `expected_n`；judge 失败重试后仍缺失则报告为 `invalid` 或 `incomplete`，不能只平均成功样本。
- 引用编号有效率必须为 100%，但编号有效不等于事实支持；另行报告 claim 支持率、citation coverage 和 `support_status`。
- 数值、单位、标准版本和适用条件关键题必须逐项正确，关键安全样本中的不支持引用为 0。
- 不可回答问题、证据冲突和知识不足必须单独评分，不得强迫模型回答后按普通 QA 计分。
- 自动指标阈值经首轮人工校准后方可调整为长期门槛；调整必须留版本记录。

### 7.6 软匹配与拟真问题

- 软匹配只作诊断，不得抵消硬真值或人工证据充分性失败。
- 保留 0.70/0.78/0.85/0.90 阈值扫描，审核软命中但硬未命中的样本，按 A/B/C/D 分类；judge 与生成模型应异源。
- `scripts/experiment_scenarios.json` 的30题是 AI 拟真鲁棒性集，不是真人集。
- 只有实际由人提出并人工标注证据充分性的题目，才可报告“真人 Hit@5”或真人使用质量。

## 八、G4：性能与负载标准

- 使用独立服务和独立端口，`timing_mode=exclusive`；不得与其他实验共享模型、Milvus 或压测队列。
- `/retrieval` 与 `/api/ask` 分别测并发 1/4/8；记录 p50/p95/p99、吞吐、错误率、超时、排队、各阶段耗时、降级和 token/费用。
- 至少热身3题且不计入正式样本；冷启动单独测量。缓存关闭用于基础负载，缓存开启另列，不能混算。
- 客户端与服务端指标交叉验证；远程 GPU 等不可得资源标记 `unavailable`，不得写成 0。
- 60 次/格仅为冒烟，不得作为发布级 p99 证据。正式报告必须给出分位数置信区间和有效样本量；样本不足时 p99 为 `inconclusive`。
- 首轮建立当前生产基线，不宣称满足未定义的绝对 SLO。后续版本在相同环境下比较，暂定 p95 比值的 95% CI 上界不得超过基线的 1.10 倍，且错误率不得上升。
- 产品绝对 SLO 确定后，必须同时满足绝对 SLO 与相对非劣门槛。

## 九、报告与留档

每个正式批次至少输出：

1. `manifest.json` 与数据校验报告。
2. G1/G2 执行结果、实际 skip/deselect、临时集合清理证明。
3. 生产基线、逐题结果、配对差值与置信区间。
4. 消融、GraphRAG、生成质量、judge 校准、降级与豁免明细。
5. 独占性能报告、资源可用性和原始请求明细。
6. 最终 `verdict`、未通过项、`incomplete_or_failed_steps` 和排除项。

原始问题、语料和私有报告可继续留在受控本地目录；脱敏摘要、schema、配置指纹和 verdict 必须纳入版本控制或不可变的受控制品库。不得因为原始数据不入 Git 就让公开结论完全无法核验。

## 十、当前实现差距与执行顺序

以下是标准生效前的明确差距，不得因本文件更新而视为已经修复：

- `eval/evaluate.py` 仍需把逐题异常、样本缺失和 `summary.invalid` 映射为非零退出及正式 verdict。
- `scripts/experiment_aux.py` 的固定80题和旧 Path A/B 闸门仍需参数化；新调度器必须支持 v3、多套数据集视图和 `expected_n`。
- `scripts/freeze_snapshot.py` 与 `scripts/remap_ground_truth.py` 仍生成历史 v1；部分探针匹配及写死的旧 ID 存活率不满足 v3，不能将其输出直接标为已验收真值。
- `scripts/run_r1_baseline.sh`、`scripts/run_r2_*.sh`、`scripts/run_reembed_and_eval.sh`、`scripts/run_g3_loadtest.sh`、`scripts/run_g4_g5.sh` 及相关评测脚本仍是历史六组协议实现；其中“历史版 TestRule”章节号和旧 G3/G4/G5 编号均不属于本版门禁。完成适配前只能作历史诊断，不得直接产出 v2 `passed`。
- 旧批次编排器写入 `protocol_sha256` 只证明记录了本文件哈希，不证明其实现满足本版协议；不得仅靠更新哈希把旧流程标为 v2 合规。
- CI 仍需把覆盖率递进门槛、真实 Milvus nightly/release 和报告 verdict 接入自动检查。
- `graph_semantic.build_graph_vectors` 仍需实现真正的窗口级断点与恢复，才能通过 G2 的图向量恢复项。
- `regression_v3` 的数据校验器、多证据指标、配对置信区间和新增语料验收集仍需实现。

执行顺序固定为：

1. 在干净提交上完成 G1，建立新的覆盖率与测试基线。
2. 完成 v3 重标注、数据校验和冻结清单；未通过 G0 不启动付费全量实验。
3. 完成 G2，优先关闭图向量断点、重启后 ID 一致性和发布回滚风险。
4. 运行当前双模型、66篇语料和 GraphRAG 的完整生产基线。
5. 基线有效后再做 leave-one-out 消融、关键配置重复实验和 G4；不得用旧成绩替代本轮基线。

## 十一、历史对照（非现行门槛）

| 批次 | 可保留的事实 | 使用限制 |
| --- | --- | --- |
| 2026-09-20 | 旧80题、旧模型和旧语料上的消融曾显示 BM25/重排贡献，HyDE 等收益有限 | 后续 embedding 修复和双模型切换已改变前提，不得继续作为当前最优配置结论 |
| 2026-10-20 | query/passage 配对修复后，旧60篇、5680块向量单路 MRR@20=0.393509、Hit@200=0.9375 | 只证明当时向量检索变化，不是完整生产或生成质量 |
| 2026-10-23 | nemotron-3.5-embed-8b 4096维迁移后，向量单路 MRR@20=0.4463、Hit@5=0.6375、Hit@200=0.9500 | 旧60篇、跨报告非严格配对；不含新 rerank/GraphRAG 的完整效果 |
| 2026-10-24～26 | 66篇、5747子块、51802实体、61966关系贡献，revision `f0adc4daf0b6c8d6697c54d8` 完成发布及11项运维校验 | 证明该次发布产物核验，不等于 G0～G4 全部通过，也不证明准确率提升 |
| 2026-10-20 SiliconFlow 重跑 | 留档包含完成的检索消融，但压测未收尾、生成质量未完成、软匹配和拟真题审核失败 | 批次为未完成状态，不得标记为 `passed` |

详细历史见 `docs/EMBEDDING_FIX.md`、`docs/MODEL_SWITCH_20261023.md`、`docs/INGEST_REPORT_20261024.md`、`docs/EXPERIMENT_RERUN_20261020.md` 和 `docs/LIGHTRAG_FIXES_WSL_20261023.md`。
