**PCB-RAG 对照 LightRAG：当前版本复核与改进路线**

复核日期：2026-10-23。本地版本：`e5ddcb5`。LightRAG 对照版本：官方主分支 `aaebfe1bb8d55deb6a4304d357a3bf4c92fcd40a`，提交时间为 2026-10-14。该提交通过 GitHub API 实时核实并下载源码；GitHub Latest Release 接口返回的是 `v1.5.7`，发布日期为 2026-09-02，报告以提交号界定比较范围。

建议将项目发展为**面向 PCB 工程问题、证据可追溯、索引可更新、可供 Agent 调用的知识检索内核**。优先完成证据与数据的一致性，再增加经过对照实验验证的图语义检索。保留现有 PCB 词法召回、结构处理、模型适配和 Dify 集成。

本次证据包含当前源码、白名单配置项、磁盘缓存、已有实验 JSON 和离线构造样例。没有在同一数据集上实际运行 LightRAG，因此下文不提供两项目的准确率或速度胜负结论。磁盘配置和历史进度也不等同于当前运行进程的状态。

**已经完成的部分，应以验证与接通为主**

本地已实现 NVIDIA query/passage embedding 契约、内容哈希增量清单、配额 RRF、token 预算上下文、图实体类型与描述、gleaning、结构化 Evidence/KnowledgeBase、流式问答、权限过滤、重排用量账本和多套评测工具。这些均不应再写成“从零新增”的待办。10 月 23 日的变更说明明确记载：本轮实现修改后尚未重跑效果评测。参见 [变更记录](D:/myProject/rag_v1-main/rag_v1/docs/LIGHTRAG_ALIGNMENT_CHANGES.md:1)。

实际生效状态还存在落差：

- 本地 `.env` 设置 `GRAPH_RAG_ENABLED=1`、`GRAPH_EXTRACT_BACKEND=llm`，配置的 `data/graph/kg.json` 不存在；按初始化代码，使用这份配置启动会进入无图降级。
- 当前集合对应的词法缓存有 5680 条记录，`source_type` 全部为 `unknown`，`section`、`parent_title`、`low_value` 均未导出。它能证明缓存尚未体现新数据模式；本次未连接 Milvus，不能仅据缓存断言实时数据库里的全部字段状态。
- 当前配置已关闭 HyDE、多查询扩展、分解与 Step-back；查询路由、配额融合仍开启。旧报告里的全增强配置不能当成当前配置。

以上白名单快照保存在 [evidence_snapshot.json](D:/myProject/rag_v1-main/rag_v1/.workbuddy/tmp/lightrag-review-20261023/evidence_snapshot.json:1)。图缺失的处理位置见 [dify_external_api.py](D:/myProject/rag_v1-main/rag_v1/src/pcb_rag/dify_external_api.py:764)。

**先修正现有实验结论的使用范围**

`eval/reports/rerun_20261020_siliconflow_guarded/retrieval/` 已保存 embedding 修复后的两条基线和 13 组消融结果。不能再把这一批描述为“尚未重跑”。这批结果也不能代替 10 月 23 日代码修改后的新基线。

统一使用 **MRR@10** 和 **Hit@10**，避免混合不同截断口径：

- BM25 单路：MRR@10 **0.482986**，Hit@10 **66.25%**。
- 向量单路：MRR@10 **0.370417**，Hit@10 **61.25%**。
- 向量＋精排：MRR@10 **0.499216**，Hit@10 **71.25%**。
- 向量＋BM25＋精排：MRR@10 **0.510660**，Hit@10 **73.75%**。
- 当时的完整增强配置：MRR@10 **0.472535**，Hit@10 **65.00%**。

五组均为 80 题，记录的数据集 SHA256 相同，集合为修复后的 `pcb_kb_nv_query_passage_20261020`；报告中的请求错误和重排降级计数均为 0，`invalid=false`。这些是已有文件中的测量结果，本次没有重复发送评测请求。

因此，“BM25 好于融合＋精排”不适用于这里的简洁融合配置。相较完整增强配置，简洁融合在这 80 题上高出 0.038125 MRR@10 和 8.75 个百分点 Hit@10。这个比较同时改变了多个组件，不能把差异全部归因于 HyDE；也未做显著性检验，不能直接推广到真实工程问题总体。

该批消融的耗时标为 `shared_resources`，不作为隔离环境性能结论。归档进度中，生成质量、最终汇总仍未完成，两个审核阶段记录为失败；该进度最后更新时间在 10 月 20 日，本次不据此判断后台任务是否仍在运行。

原始依据：[简洁融合结果](D:/myProject/rag_v1-main/rag_v1/eval/reports/rerun_20261020_siliconflow_guarded/retrieval/03_fusion_rerank.json:1)、[完整增强结果](D:/myProject/rag_v1-main/rag_v1/eval/reports/rerun_20261020_siliconflow_guarded/retrieval/08_hyde_production.json:1)、[归档进度](D:/myProject/rag_v1-main/rag_v1/eval/reports/rerun_20261020_siliconflow_guarded/progress.json:1)。

**1. P0：把索引版本与处理配置绑定，让实现改进真正进入数据**

`incremental.diff_documents()` 当前只比较文档内容指纹；清单条目只有 `path` 与 `hash`。切块器、元数据生成、Contextual Retrieval 或建图提示词改变时，原文件内容可能完全不变。默认增量入库会跳过这些文件，代码升级并不自动刷新已有索引。刚开启图功能也无法依靠“无变化的增量入库”补建全部存量图谱。

建议引入可检查的 `pipeline_fingerprint`，至少包含：源文档版本、解析/切块参数、元数据模式版本、embedding 契约、图抽取提示词版本。分别记录 chunk、向量、词法和图的完成版本，支持指定阶段的 backfill/rebuild。语义缓存也应使用知识库 revision 和检索配置作为隔离键的一部分。

对本次元数据更新，应先在新集合完成迁移与验证，再切换读取版本。单纯执行默认增量脚本不足以保证刷新。NVIDIA embedding 契约已有专门保护，应保留并将其纳入整体版本管理。

验收：内容未变时正常跳过；只改切块/元数据版本时能够准确重做对应阶段；图失败后能单独补建；四类存储暴露一致的 revision；新集合中的字段覆盖率和真值 ID 映射有检查记录。

代码依据：[incremental.py](D:/myProject/rag_v1-main/rag_v1/src/pcb_rag/incremental.py:85)、[入库计划](D:/myProject/rag_v1-main/rag_v1/src/pcb_rag/ingest.py:3620)、[embedding 契约](D:/myProject/rag_v1-main/rag_v1/src/pcb_rag/embedding_contract.py:20)。

**2. P0：让提示词、返回证据、引用和预算共用同一份最终结果**

`build_context_with_budget()` 会按 ID 去重、将低价值块后移、按预算裁剪，并按处理后的顺序重新编号。`extract_citations()` 仍枚举原始 `retrieved_docs`，生成、SDK 和 SSE 路径都把原始列表传给它。

离线样例已复现两种错位：原始 `[A低价值, B有效证据]` 变成提示词中的 `[B, A]` 后，答案 `[1]` 仍返回 A；原始 `[A, A, B]` 去重成 `[A, B]` 后，答案 `[2]` 仍返回 A。当前词法缓存尚无低价值标记，因此第一种问题尤其需要在新元数据入库前处理。这些是构造样例，不代表已经观察到真实用户回答发生错引。

建议上下文构建返回统一的 `EvidenceBundle`：最终入选证据、稳定 `citation_id`、实际使用文本、来源定位和预算统计。生成、SSE 首批来源、最终引用和 Agent 工具结果都消费这个对象。引用编号有效性与“该证据是否支持这句话”应分别判断，避免把出现 `[n]` 直接视为事实可信度高。

预算也需要收紧。当前只按字符粗估原文和事实段，没有逐项约束最终标题、引用表与完整提示词。离线构造 20 条带较长文件名的证据时，6000 预算渲染后按项目自己的估算达到 **6497**。这证明它不是严格上限，不代表当前模型已经超出真实窗口。

可借鉴 LightRAG 在截断后的 chunk 集合上生成 reference ID 的做法，再补上本项目自己的完整请求预算：模型 tokenizer 或明确标记的保守估算，扣除 system、query、history、输出预留后分配证据，最终序列化后复查。

验收：降权、去重、空块和截断均不改变引用指向；每个引用只能指向模型实际看过的证据；最终请求预算与配置上限一致。

代码依据：[上下文重排](D:/myProject/rag_v1-main/rag_v1/src/pcb_rag/query.py:1896)、[原始列表引用解析](D:/myProject/rag_v1-main/rag_v1/src/pcb_rag/query.py:1791)、[生成调用](D:/myProject/rag_v1-main/rag_v1/src/pcb_rag/query.py:2111)、[预算渲染](D:/myProject/rag_v1-main/rag_v1/src/pcb_rag/context_budget.py:185)。官方对照：[operate.py](D:/myProject/rag_v1-main/rag_v1/.workbuddy/tmp/lightrag-review-20261023/official/lightrag/operate.py:5993)。

**3. P1：让图具备实体和关系的语义召回能力**

LightRAG 的主要可借鉴点是三个不同的向量索引：实体、关系、原文 chunk。`local` 从实体入手，`global` 从关系入手，`hybrid` 组合二者，`mix` 再加入原文向量召回。轮转融合只是其中一个实现选择。

本项目已经有类型、描述和 gleaning；剩下的结构性差距是 `link_entities()` 仍用名称/词元字符串匹配，没有实体描述和关系描述的向量检索入口。HTTP 主路径在精排之后才附加图结果，而且依赖前面的检索结果非空；图路因而无法独立挽救零召回。

建议先给已有图模型补两个可替换的检索接口：`EntityRetriever` 和 `RelationRetriever`，继续复用现有 Milvus。召回实体/关系后，按来源 ID 取回可授权的原文证据，与向量/BM25 候选一起去重、精排或分配明确预算。图摘要保留为辅助事实，每条事实都能回到原文及文档版本。

从同一份 200–500 chunk 子集上比较“基础混合检索”与“基础混合＋图”，重点看跨文档比较、条件关联、别名表达和多段证据覆盖。记录额外抽取请求、入库时间和查询用量后，再决定是否扩大建图范围。子集规模是试验建议，并非已接受预算。

验收：未出现实体原名的语义问题也能链接；向量/BM25 为空时图路仍可工作；图返回的事实能回查证据；图开启后的增益按问题类型报告，简单问题的质量与延迟也有对照。

代码依据：[本地实体链接](D:/myProject/rag_v1-main/rag_v1/src/pcb_rag/graph_rag.py:1171)、[图追加位置](D:/myProject/rag_v1-main/rag_v1/src/pcb_rag/dify_external_api.py:1428)。官方对照：[三套索引](D:/myProject/rag_v1-main/rag_v1/.workbuddy/tmp/lightrag-review-20261023/official/lightrag/lightrag.py:1876)、[检索模式](D:/myProject/rag_v1-main/rag_v1/.workbuddy/tmp/lightrag-review-20261023/official/lightrag/operate.py:5280)。

**4. P1：先保证图谱来源维护正确，再做持续增量更新**

已复现两个直接关联入库更新的问题：

- 两份文档共同支持一条关系时，`remove_docs([其中一份])` 删除整条边，另一份文档的有效贡献一起消失。
- 删除最后一份支持文档后，如果本批没有新节点，`_update_knowledge_graph()` 因图已无边而不保存；旧 JSON 文件仍保留已删除关系，下次加载可恢复旧数据。

另有 `KnowledgeGraph.merge()` 只保留首个 chunk/doc 来源的问题已复现，但当前常规入库链路不调用该合并函数，应与前两个主路径问题区分优先级。

建议为每条事实保存独立来源贡献，如 `relation → [(doc_id, chunk_id, revision, evidence)]`。删除文档时减去该来源，只在贡献集合为空时删除关系，否则从剩余来源重算描述与权重。空图也必须持久化。LightRAG 的文档删除流程会检查剩余来源并重建仍有支持的实体/关系，这一思路适合借鉴。

入库过程进一步拆为解析、切块、向量、图抽取、发布等可恢复阶段，记录每阶段完成和失败状态。当前删除旧 chunk 与插入新 chunk 之间存在空窗，删除异常只返回 0，图失败又不会阻止清单保存；应增加失败状态、按阶段重试和发布前一致性检查。

热更新需要覆盖 BM25、图和语义缓存。本地已实现 BM25 重载标记；图目前在初始化时加载，检索缓存键也缺少知识库版本。可统一由 revision 通知失效或切换读取快照。

验收：删除 A 后 B 支持的关系仍存在；删除最后来源并重启后关系消失；任意阶段失败后可恢复；重跑不重复累计关系权重；新旧 revision 不混用。

代码依据：[共享关系删除](D:/myProject/rag_v1-main/rag_v1/src/pcb_rag/graph_rag.py:463)、[空图保存分支](D:/myProject/rag_v1-main/rag_v1/src/pcb_rag/ingest.py:3511)、[写入顺序](D:/myProject/rag_v1-main/rag_v1/src/pcb_rag/ingest.py:3723)、[缓存键](D:/myProject/rag_v1-main/rag_v1/src/pcb_rag/dify_external_api.py:1545)。官方对照：[保留剩余来源](D:/myProject/rag_v1-main/rag_v1/.workbuddy/tmp/lightrag-review-20261023/official/lightrag/lightrag.py:6497)、[文档状态](D:/myProject/rag_v1-main/rag_v1/.workbuddy/tmp/lightrag-review-20261023/official/lightrag/base.py:1317)。

**5. P1：将现有 KnowledgeBase 收敛为独立内核，并统一评测口径**

当前 `query.py` 为 5122 行，`ingest.py` 为 3837 行，HTTP 模块为 2384 行。文件长度本身不是故障，具体问题是 KnowledgeBase 反向导入 HTTP 模块、复用 `_app_state` 与私有函数，模型和配置又分布在模块全局状态中。

建议目标依赖方向：API、CLI、Agent SDK、评测适配层共同调用 `RetrievalPipeline`；内核通过显式 `PipelineConfig` 注入模型和存储。保留已经返回逐条 Evidence 的接口，把过滤、候选融合、精排、上下文和引用的最终决策放进内核。

已有生产路径评测可以继续使用；需要消除的是评测实现分叉。`eval/evaluate.py` 自建检索器并固定截断，不能自动代表 HTTP 完整链路。归档两条基线 MRR@10 为 0.509236 与 0.473125，但它们保留了参数/实现差异，不能把差值直接解释为同配置 bug。

还有一个具体入口差异：`/api/ask` 的 Self-RAG 适配器 `_NodeListRetriever.retrieve()` 忽略新问题，每轮都返回首次节点，无法完成其预期的“改写后重新检索”。如保留该功能，应调用真实管道并设置迭代、时间与用量上限。

验收：相同配置和候选输入下，SDK、HTTP、CLI、评测使用相同入选证据与引用映射；Self-RAG 改写问题后确实产生新的检索调用；降级字段反映本次请求各路实际成功、失败、跳过情况，而不只是初始化状态。

代码依据：[SDK 对 HTTP 的依赖](D:/myProject/rag_v1-main/rag_v1/src/pcb_rag/engine.py:194)、[评测检索器](D:/myProject/rag_v1-main/rag_v1/eval/evaluate.py:145)、[固定节点的 Self-RAG](D:/myProject/rag_v1-main/rag_v1/src/pcb_rag/dify_external_api.py:1723)。

**6. P1：强化 PCB 的条件、版本与结构证据**

图谱和上下文最终应服务于“什么条件下、哪份规范的哪个版本、要求哪个参数、引用哪条原文”。本地 `Triple` 有实体、关系和描述，但缺少独立的标准版本、适用条件、比较符、数值和单位字段。

建议对关键规范事实保留结构：标准/版本、章节、适用对象、工艺或环境条件、参数、比较符、原始数值与单位、原文定位。标准名称或参数相同但版本/适用范围不同的事实应并列呈现并标记冲突，避免合并描述时丢掉条件。单位归一化时保留原文值。

父子切块也需要完成语义：现在 `parent_id` 指向同一父块的第一个子块，扩展只补首个子块。应保存真正的 parent 文本，采用子块召回、按问题补齐父节，并对表格表头、单位、脚注和跨页延续保留关联。为原生 PDF、表格、参数边界与版本冲突单独建立验证样例。

验收：答案保留数值、单位与适用条件；表格问题能对应正确列头和脚注；跨版本规则不会混答；多证据问题能给出全部必要依据；无依据的问题有明确的不足说明。

代码依据：[父块锚点](D:/myProject/rag_v1-main/rag_v1/src/pcb_rag/ingest.py:3136)、[现有三元组字段](D:/myProject/rag_v1-main/rag_v1/src/pcb_rag/graph_rag.py:274)。

**7. P2：按实际使用方式补运维和管理能力**

LightRAG 提供文档索引管理和图探索界面，可作为后续产品化参照。本项目已具备聊天 UI、SSE、健康接口和日志；下一轮界面更有价值的内容是文档处理状态、失败原因与重试、知识库 revision、证据原文定位和图路径调试。

服务层优先加入总请求 deadline、全局并发/队列边界、阶段耗时与取消传播。现有重排用量保护应保留，逐步扩展到 LLM 和 embedding；是否启用付费累计上限由部署配置决定。CI 可逐步收窄核心文件的广泛 lint 豁免，并固定可复现依赖版本。

如果要服务多个用户，现有检索 ACL 之外还需要会话归属校验。`/api/sessions` 当前对持有服务 Token 的调用者返回整个会话列表，会话读取/删除只按 session_id 操作；应绑定 tenant/user 并逐请求授权。当前本地配置关闭 ACL，因此不能将此部署配置描述为已经实现租户隔离。单用户本地使用时，应按实际需求安排这项工作。

代码依据：[会话列表](D:/myProject/rag_v1-main/rag_v1/src/pcb_rag/dify_external_api.py:2280)、[依赖与静态检查配置](D:/myProject/rag_v1-main/rag_v1/pyproject.toml:1)。官方对照：[Server 与 WebUI](D:/myProject/rag_v1-main/rag_v1/.workbuddy/tmp/lightrag-review-20261023/official/docs/LightRAG-API-Server.md:1)。

**建议执行顺序与实验验收**

1. 修引用映射和完整预算边界，将本次离线构造样例转成正式回归测试；在图进入使用前修复来源删除与空图持久化。
2. 增加 pipeline/revision 标识，以新集合验证元数据迁移、真正父块与存量图 backfill，检查真值映射后切换。
3. 以简洁混合检索建立当前代码的新基线，先验证配额 RRF、路由、上下文策略的独立增益。
4. 同语料、同模型、同预算下增加实体/关系语义检索试验，再决定全量建图和路由策略。
5. 收敛共享内核与四种入口；补版本/参数/表格/多证据/无答案问题，最后完善任务管理和服务部署能力。

新基线应冻结源码 SHA、语料/数据集指纹、索引 revision、embedding 契约、rerank 模型与输入上限、最终上下文策略。保留现有 80 题作为回归锚点，另补真实需求导向的多证据和无答案样本。AI 生成或 AI 审核样本明确标注来源，避免称为真人测试。

同时报告检索排名（MRR/NDCG/Hit）、多证据覆盖、引用支持率、参数/单位正确性、忠实度、不可判定比例、隔离环境 p50/p95 和实际/估计用量。LightRAG 对跑可增加固定提交的 `naive`、`mix` 对照；如果解析或切块不同，报告中应标记这是整套系统对照，不能把差异全部归因于图检索。

**本次验证与材料**

- [离线复现脚本](D:/myProject/rag_v1-main/rag_v1/.workbuddy/tmp/lightrag-review-20261023/offline_checks.py:1) 使用真实轻量模块和从原文件 AST 隔离加载的函数，避免导入服务或调用模型。
- [离线结果](D:/myProject/rag_v1-main/rag_v1/.workbuddy/tmp/lightrag-review-20261023/offline_checks.json:1) 确认 6 个故障场景，另记录 1 个增量计划观察。观察断言通过表示故障得到复现，不表示对应业务行为正确。
- [证据快照](D:/myProject/rag_v1-main/rag_v1/.workbuddy/tmp/lightrag-review-20261023/evidence_snapshot.json:1) 保存脱敏配置、缓存统计和实验摘要，不包含用户问题正文或 API 密钥。
- [官方来源清单](D:/myProject/rag_v1-main/rag_v1/.workbuddy/tmp/lightrag-review-20261023/official/SOURCE.json:1) 保存固定提交的来源 URL 与下载文件校验值。

本机默认 Windows Python 缺少 pytest、LlamaIndex、FastAPI 等运行依赖，本轮未执行全量测试或在线性能评测。官方源码下载之外没有发送模型/数据库请求；上述建议的收益需要迁移数据后用新基线验证。
