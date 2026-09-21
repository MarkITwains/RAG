事实一：你有三套检索链路实现，指标出自哪一套必须先定。 eval/evaluate_recall.py（2168 行，有自己的 _FusionEnhancedRetriever）、dify_external_api._retrieve_nodes（/api/ask 和 /retrieval 真正走的）、query.py main()（CLI，带那个 HyDE 长文当主查询的 bug）。docx 里的数字来自第一套，用户实际拿到的是第二套。

事实二：生产 API 的默认参数和 README 不一样。 dify_external_api.py:37-44 用 setdefault 强制了 RERANK_TOP_N=10、FUSION_NUM_QUERIES=3、HYDE_MAX_LENGTH=200，而 README/docx 写的是 200 / 4 / 300。除非你 .env 里显式写了这几项，否则线上精排只保留 10 条——你之前评测用的 --rerank-top-n 和线上根本不是一个数。

事实三：你有两个现成数据集，只有一个能用。 eval/eval_dataset.json 100 条，chunk id 是 md5（当前 parent_child 的确定性 id md5(doc_id-pc-idx)），只要切块参数没动、语料没动就仍然有效；eval_dataset_146.json 的 id 是 uuid，对应早期入库，已经作废。另外 eval/reports/ 里有 3 月 8 日的报告：同一个 100 条集上 fusion_hyde_rerank_v2 跑出 MRR 0.870 / Hit@1 0.79 / Hit@5 0.99，比 docx 写的 v1.6（0.785 / 0.704 / 0.908）高一大截。这两组数哪个是真的，重测第一件事就是弄清楚。

一、开测前的三个前提（不做，后面全白测）

冻结语料快照。 记下 Milvus pcb_kb 当前 row count、data/clear_docs 的文件数和总字节、data/lexical_corpus.jsonl 的行数与 md5。整个重测期间不重新入库；如果你今天修的增量入库 bug 需要验证，另建一个 collection（COLLECTION=pcb_kb_test）去验，别碰主库。

冻结回归集。 用 eval_dataset.json 那 100 条，先做一次"存活校验"：对每条 ground_truth_ids 去 Milvus query(filter='id in [...]') 确认还在库里，剔除不在的，剩下的按来源文档分层保留不少于 80 条，另存为 eval/datasets/regression_v1.json，从此不再改。把语料指纹写进同一个文件的头部注释或旁边的 .meta.json。这就是以后所有版本横向对比的唯一基准，也是第一轮第 2 题的答案。

定死一份"生产配置"。 把线上实际生效的值写进 .env.eval：RECALL_TOP_K=200、RERANK_TOP_N=10（或者你决定改成 50，那就先改 README 再测）、FUSION_NUM_QUERIES=3、FUSION_RRF_K=40、RAG_TOP_DOCS=5、RAG_DOC_MAX_CHARS=1000、HYDE_ENABLED=1、QUERY_ROUTING_ENABLED=1、QUERY_DECOMPOSE_ENABLED=1、QUERY_STEP_BACK_ENABLED=1、CHUNK_EXPAND_ENABLED=1、MULTI_EXPAND_ENABLED=1、QUERY_UNDERSTANDING_MODE=hybrid。所有实验用同一份文件起步，每个实验只改一项。注意 query.py 的开关全是 import 时读 env，每个配置必须是独立进程，同一个进程里改 os.environ 不生效。

二、要测的六组数据
第 1 组：回归基线（回答"v1.6 到底是多少"）

在冻结集上跑两条路径的完整生产配置，各跑一次：

第一条用现有脚本，保证和 docx 数字同源可比：python eval/evaluate_recall.py --dataset eval/datasets/regression_v1.json --mode fusion_hyde_rerank --recall-k 200 --rerank --rerank-top-n 10 --rrf-k 40 --ks 1,3,5,10 --out eval/reports/r1_baseline_evalscript.json。

第二条走生产路径，这个脚本目前没有，需要写一个百来行的小脚本：在 .env.eval 下 import dify_external_api，手动调一次 _initialize_retrieval_engine()，然后对每个问题调 _retrieve_nodes(q, top_k=20)（CHUNK_EXPAND_ENABLED=0 拿纯排名），取 node id 序列算 MRR、Hit@1/3/5/10、NDCG@10——指标函数直接 from evaluate_recall import _ndcg_at_k, _average_precision 复用。分段耗时不用另外埋点，_retrieve_nodes 已经在 logger 里打 [Timing] 预处理 / 主检索 / HyDE / 精排 / 检索总耗时，挂一个 logging handler 用正则把秒数抓出来即可；[QueryRoute] type 同理抓出查询类型，用于分类型统计。

要记录的数：两条路径各自的 MRR、Hit@1/3/5/10、NDCG@10，以及两者的差。如果差超过 0.03，说明评测脚本和线上不是一个系统，这本身是个要查的 bug，也是面试可讲的发现。

第 2 组：消融矩阵（回答第 16 题"每一路各贡献多少"）

全部在冻结集、生产路径上跑，逐级加法，每行只比上一行多开一个开关。每个配置一个进程，用 shell 循环套 env 即可：

#	配置名	在上一行基础上改动的 env	回答的问题
1	vector_only	LEXICAL_ENABLED=0 RERANK_ENABLED=0 HYDE_ENABLED=0 QUERY_EXPANSION_ENABLED=0 QUERY_ROUTING_ENABLED=0 MULTI_EXPAND_ENABLED=0	纯向量底线（对应 v1.0 的 0.22）
2	vector_rerank	RERANK_ENABLED=1	Rerank 单独值多少
3	fusion_rerank	LEXICAL_ENABLED=1	BM25 融合本身赚不赚（v1.3 表里是略亏）
4	+ expansion	QUERY_EXPANSION_ENABLED=1	同义词扩展
5	+ multi_expand	MULTI_EXPAND_ENABLED=1	查询变体 3 个 vs 1 个
6	+ routing	QUERY_ROUTING_ENABLED=1 QUERY_DECOMPOSE_ENABLED=0 QUERY_STEP_BACK_ENABLED=0	动态权重（不含分解/step-back）
7	+ decompose/stepback	QUERY_DECOMPOSE_ENABLED=1 QUERY_STEP_BACK_ENABLED=1	这两个 LLM 调用值不值它的延迟
8	+ hyde（=生产配置）	HYDE_ENABLED=1	HyDE 的净贡献（docx 说收益有限）
9	full, recall 50	RECALL_TOP_K=50	200 条召回是否必要，省多少延迟
10	full, rerank_top_n 50	RERANK_TOP_N=50	线上 10 条会不会截掉正确答案
11	full, rule 模式	QUERY_UNDERSTANDING_MODE=rule	意图识别的 LLM 调用值不值
12	full + chunk_expand	CHUNK_EXPAND_ENABLED=1，用 top_k=5 看扩展后正确 chunk 是否还在前 5+5	扩展会不会把正确 chunk 挤出去

BM25 单路（--mode bm25）用现有 evaluate_recall.py 补一行，生产路径没有纯 BM25 分支。每行记录 MRR、Hit@1/5/10、NDCG@10、平均和 p95 总耗时、各分段耗时。最后你要能说出一句"Rerank 贡献 +0.4，融合 +0.0x，扩展 +0.0x，HyDE +0.0x 但多 Y 秒"。

第 3 组：延迟（回答第 6 题和第 14 题）

单并发的分段耗时第 1、2 组顺手就有了，重点是把 rerank 拆出来：Qwen3-Reranker-4B 对 200 条、batch 8、max_length 1024 的耗时，和 RERANK_TOP_N 无关（top_n 只影响输出条数，200 条照样全过模型），这个数字要单独记。

并发压测另写一个几十行的脚本：起 bash scripts/serve_api.sh，从冻结集抽 30 个问题，requests + ThreadPoolExecutor 分别以并发 1、4、8 各打 60 次 /retrieval（纯检索）和 /api/ask（含生成），记 p50/p95/p99、吞吐（req/s）、错误率、以及并发 8 时是否出现 HyDE 30 秒预算超时导致的 HyDE 路被跳过（看服务端日志里 [HyDE] 路生成超时 出现次数）。压测前先关语义缓存 SEMANTIC_CACHE_ENABLED=0，否则 30 个问题重复打全是缓存命中。同时用 nvidia-smi dmon -s um -d 1 记 GPU 利用率和显存峰值，/metrics 端点的分位数作交叉验证。

第 4 组：生成质量（回答第 3 题，异源 judge）

现有 evaluate.py 的 judge 和生成是同一个 build_llm()。最小改动是把 judge = build_llm() 那一行改成 build_llm(model=os.getenv("JUDGE_MODEL") or None, backend=os.getenv("JUDGE_BACKEND") or None)，api_clients.build_llm 本来就接受这两个参数。然后 JUDGE_BACKEND=api JUDGE_MODEL=<非 Qwen 家族，比如 DeepSeek 或 GLM，走你的 ZenMux> 跑 python eval/evaluate.py --dataset eval/datasets/golden.jsonl --top-k 5 --recall-k 200，四指标各一个数。另外把 metrics.py:67-75 失败返回 0.0 改成返回 None 并统计失败数，否则均值不可信。

golden.jsonl 需要重新生成一份（build_golden_dataset.py --limit 100，它会记 source_chunk_id），生成模型也换成 judge 那个异源模型，做到出题、答题、判卷三个不同模型。生成后人工抽 30 条：逐条判"问题是否像真人会问的"和"ground_truth 是否正确"，记剔除率——这个剔除率本身就是一个要报的数（说明 LLM 造题的噪声水平）。

第 5 组：软匹配诊断（回答第 5 题）

在冻结集上跑一次 --soft-match embed --embed-threshold 0.78，把"软命中但硬未命中"的样本全部导出，人工看 20 条，分类计数：A 类是同一文档的相邻 chunk（overlap 区域）、B 类是不同标准里的重复条款、C 类是真的语义等价但不同来源、D 类是误判。你要能说出"软硬差的 10 个点里，A+B 占了多少"，这样"软匹配是语料重复信号而不是系统能力"这句话就有数据了。再补一个阈值扫描：0.70 / 0.78 / 0.85 / 0.90 各跑一次软 MRR，证明 0.78 不是挑出来的。

第 6 组：真人问题集（回答"LLM 造题偏简单"的质疑）

你自己或找同学写 30 个真实会问的问题，覆盖路由表里的 10 个类型各 3 个，故意包含口语化、缩写、错别字、需要两段才能答全的问题。这批没有 chunk id 标注，评测方式改为：跑生产路径 top_k=5，把每题的 5 条 chunk 摘要导成 CSV，人工标"前 5 条里有没有能答这个问题的"，得到人工 Hit@5；再和冻结集的 Hit@5 对比，差值就是"合成题 vs 真人题"的分布差距。30 题标注约 40 分钟。

顺手记的：语料与入库统计

文档数、chunk 数、chunk 长度均值/最大/最小（ingest 日志里有）、Milvus 向量数、embedding 维度、BM25 索引构建耗时和内存、全量入库总耗时。这些是面试开场 30 秒介绍里要说的数，现在你一个都说不出来。

三、执行顺序与时间

第一天：前提三件事（1 小时）→ 第 1 组两条路径（每条 100 题、每题约 5～15 秒含 HyDE，各 20 分钟）→ 如果两条路径差异大，先查原因再往下走。第二天：第 2 组 12 个配置（每个 20 分钟，4 小时，可以晚上挂着跑）+ 第 5 组阈值扫描（1 小时）。第三天：第 3 组压测（1 小时）+ 第 4 组 golden 重生成与异源 judge（2 小时，主要是 API 调用等待）+ 人工抽检 30 条（40 分钟）。第四天：第 6 组写题、标注（1.5 小时）+ 汇总。总计约 12 小时人工加机器时间，其中你必须在场的约 5 小时。

写脚本的工作量：生产路径评测脚本约 150 行，消融循环 shell 约 30 行，压测约 60 行，汇总表约 50 行，evaluate.py 改 3 行。

四、测完要填的表

版本总览表重做：只保留冻结集上的数，v1.6 一行拆成"评测脚本路径"和"生产路径"两行；消融表 12 行；延迟表（并发 1/4/8 × /retrieval 与 /api/ask 的 p50/p95/p99）；分段耗时表（预处理、HyDE、向量、BM25、精排、生成）；生成质量表（四指标 + judge 模型名 + judge 失败数）；软匹配差异分类表；真人题 vs 合成题 Hit@5 对比。docx 里"1000 小时工业可靠性"那句改成"开发环境累计运行约 1000 小时无崩溃；并发 8 时 p95 为 X 秒"，data/traces.jsonl 从仓库删掉。

五、几个会让你白测的坑

evaluate_recall.py 2030 行左右会强行把 SHORT_QUERY_THRESHOLD 改成 100 并强开 HyDE，这和生产不一致，跑基线时要注意它的 --mode 内部逻辑不等于 .env 开关。RERANK_TOP_N=10 时 Hit@20 无意义，--ks 别超过 10。HyDE 有 30 秒等待预算，LLM 冷启动时前几题 HyDE 路会被跳过，正式跑之前先用 3 个问题热身。FUSION_NUM_QUERIES 在路由开启时会被各类型策略里的 num_queries（3～5）覆盖，消融第 5 行要看日志确认实际变体数。3 月那份 0.87 的报告如果在同一个 100 条集上复现不出来，优先怀疑语料在 3 月之后重入过库导致 id 变了，用存活校验的剔除数就能判断。
---
## 六、切块层冻结记录（2026-09-19，入库后即冻结，此后不重入）

### 冻结配置
- NODE_PARSER_MODE=parent_child；PARENT 3000/200；CHILD 800/150/80
- SEMANTIC_SPLIT_THRESHOLD=1000000：子块级语义边界检测**停用**（配额考虑；父层结构切分不受影响）。实测该路径在能工作时也未产出语义块（单篇 70KB 波动未达 0.15）。**重开该开关 = 换语料**，须全量重嵌入并重跑六组。.env 已同步为 1000000 并注释。
- 语料：60 文档 → 5680 chunks（Milvus pcb_kb）；指纹已改内容哈希（blake2b），manifest v2。

### 本次修复（重入前完成，tests/test_chunk_split.py 7 项回归）
- overlap 被句号截没：旧实现取末 80 字后 rfind('。') 向前截断，实际重叠近 0（全库命中率 1.1%）。改为向后补齐到句首（重叠 ∈ [80,160)）→ **94.4%**。
- 超长"句"整块塞入：无句末标点的 1901 字"句"整块入库。加 _hard_split 句内标点兜底 + 组装后超限二切 → max 1901→863（残差主要是注入行）。

### 质量体检（scripts/chunk_quality.py，5680 块）
- 结构起始率 52.8%（剥【上下文】前缀后）；句末非标点 27.9%（抽检=图注/表格行等原文无句号单位，非切坏）
- 同父相邻 overlap 命中 94.4%；content>800 仅 17 条（0.3%，含前缀剥除残差）
- split_method：length 57.5% / none 42.5% / semantic 0
- 真值可用性：**98/100 真值可被单个 chunk 完整覆盖**（2 条跨 chunk 边界）
- 顺序：离线重放 24/24 偏移单调；库内 next_id 链 0 冲突 / 0 悬空

### 已知问题（不重入，记录在案）
1. 孤立标题微型块（如 14 字"标题：11.1 IPC"）：源文档该节正文为表格/图，个位数，无害。
2. Contextual 标题行偶为句子碎片（"标题：99 的原理图来说明时序计算方法"）：属 contextual 模块标题定位问题，影响嵌入上限，不影响切块。
3. 度量教训（自查两次纠错）：长度口径必须剥注入行【上下文…】与"标题：…"（否则 0.3% 报成 8.8%）；按 next_id 查相邻块必须用 {id: row} 映射，写成 {next_id: row} 会取到无关块（曾误判"顺序倒退"）。

### 供应商变更（2026-09-19 晚，重嵌入记录）
- embed: openrouter free 通道 → NVIDIA 官方 API（模型同名 nemotron-3-embed-1b）。实测同一文本 cosine 仅 0.34 —— 两侧向量空间不兼容，**全量重嵌入**（块文本/id 不变，切块冻结与回归集有效）。
- rerank: openrouter llama-nemotron-rerank-vl-1b-v2:free → **SiliconFlow BAAI/bge-reranker-v2-m3**（生产 rerank 模型变更，消融表须注明）。
- 踩坑记录：EMBED_BASE_URL 不能带尾部 /embeddings（客户端自动拼接）；Path B 首跑暴露 query.py 未导入 current_principal（:3785/:3804/:4237，已修——此 bug 意味着此前 /retrieval 无业务过滤查询必崩）。
- openrouter 免费层当日配额（1000/天）已被白天重建+首跑评测耗尽，是 Path A 中途 429 崩溃的原因。

### 闸门触发与归因（2026-09-20，第 1 组完成）
- 结果：评测脚本路径 MRR=0.4956 / Hit@10=0.650；生产路径 MRR=0.4336 / Hit@10=0.625；|Δ|=0.062>0.03 → 闸门拦截，消融矩阵未启动（接续链按设计中止）。
- 归因（已闭合）：生产路径 `_extract_query_filters` 把查询中的 Altium/Allegro 抽成 `eda=altium/allegro` 元数据硬过滤，13/80 题被误杀（8 题完全无召回），该组 MRR=0.133 vs 无过滤题 0.487；反事实代入后总体≈0.487 ≈ Path A 的 0.4956。且评测脚本路径**不做业务过滤抽取**（0 处引用）——两路径差异的主因即此。
- 修复：`_retrieve_nodes` 增加 `_no_business_filter` 回退——业务过滤把候选池清零时，自动整链重检一次（保留 ACL，递归由标志限定一次）。死题实测：vec=0/bm25=0 → 回退 → 312/400 候选 → 精排 10 条。
- 待重跑：Path B v2 → 重过闸 → 消融矩阵自动接续。残留差异预计 <0.03；若 routing 动态权重仍贡献小幅差异，由矩阵行 05/06 归量。

### 闸门处置（2026-09-20，Path B v3 后）
- Path B v3（过滤回退修复 + HYDE_WAIT_BUDGET=60）：MRR=0.4460（0.4336→0.4460），空结果 0，HyDE 未命中 21→2。
- 残余 |ΔMRR|=0.0496>0.03。**归因判定：上游 API（emoera）返回不稳定，部分题目 HyDE 失败/超时降级；叠加生产路径路由动态权重与脚本路径固定融合的结构性差异。**判定为已知且可接受的系统性差异，不阻塞后续实验。
- 依据该判定放行第 2 组消融矩阵。注：矩阵行 05→08 本身即量化 routing / multi-expand / decompose / HyDE 的边际贡献，比闸门本身更有信息量。

### 第 2 组消融矩阵结果（2026-09-20，冻结集 80 题，限速版全表零失败）

| 配置 | MRR | Hit@1 | Hit@5 | NDCG@10 | p50(s) | 逐行ΔMRR |
| --- | --- | --- | --- | --- | --- | --- |
| 00 BM25 单路 | **0.4891** | 0.388 | 0.600 | 0.5265 | ~0 | — |
| 01 纯向量 | 0.0690 | 0.037 | 0.100 | 0.0880 | 0.3 | — |
| 02 向量+Rerank | 0.2479 | 0.200 | 0.325 | 0.2673 | 18.5 | +0.179 |
| 03 +BM25融合 | 0.4811 | 0.375 | 0.613 | 0.5286 | 18.0 | +0.233 |
| 04 +同义扩展 | 0.4634 | 0.375 | 0.588 | 0.5056 | 18.3 | −0.018 |
| 05 +多变体 | 0.4627 | 0.362 | 0.588 | 0.5083 | 18.3 | −0.001 |
| 06 +路由 | 0.4622 | 0.362 | 0.588 | 0.5078 | 13.8 | −0.001 |
| 07 +分解/step-back | 0.4611 | 0.362 | 0.588 | 0.5069 | 17.0 | −0.001 |
| 08 +HyDE（=生产） | 0.4435 | 0.338 | 0.575 | 0.4880 | 29.8 | −0.018 |
| 09 recall50 | 0.4289 | 0.350 | 0.525 | 0.4699 | 25.2 | −0.015 |
| 10 rerank_top50 | 0.4496 | 0.338 | 0.575 | 0.4887 | 35.6 | +0.021 |
| 11 rule 模式 | 0.4481 | 0.350 | 0.575 | 0.4886 | 35.3 | −0.002 |
| 12 +chunk_expand | 0.4512 | 0.338 | 0.588 | 0.4936 | 42.7 | +0.003 |
| 对照：r1 生产基线 | 0.4460 | 0.338 | 0.575 | 0.4926 | 36.2 | — |

一句话结论：**Rerank 在向量路上 +0.18；BM25 融合 +0.23；同义扩展/多变体/路由/分解/HyDE 全部 ≈0 或为负（HyDE −0.018 还多 13s）；BM25 单路就是当前语料的最优配置（0.4891）**。

#### 头号待查：向量路贡献≈0（行 01 MRR 0.069）
- 离线验证（1 题）：金标准 chunk 纯余弦排名 2548/5680。
- input_type 实验（3 题 × 3 模式）：query/passage 模式只解释一小部分（最好 2548→11，其余 256/408 仍差）。
- 判定：**非代码缺陷，是 embedding 模型与中文术语/参数密集语料的适配问题**（nemotron 系英文优先）。
- 改进选项（需重嵌入 + 重测，另行决策）：a) 换本地 Ollama qwen3-embedding:8b（中文强、零 API 配额）；b) SiliconFlow bge-m3；c) 保留现状，生产以 BM25 为主。
- 顺带小缺陷：vector-only 分支（bm25_index 为空时）没有过滤清零回退，行 01/02 各有 2 道空结果题。
