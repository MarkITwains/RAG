# 作为元件接入自研 Agent

`pcb_rag.engine` 是给 agent 用的程序接口。它与 HTTP 服务（`dify_external_api.py`）
共用同一套检索实现和同一个索引实例，区别只在于**返回结构化对象而不是 HTTP 响应**。

```python
from pcb_rag.engine import KnowledgeBase

kb = KnowledgeBase()
kb.ensure_ready()                       # 首次数十秒：连 Milvus、加载模型

hit = kb.retrieve("4 层板阻抗控制关注什么", top_k=8)
for ev in hit.evidence:
    print(ev.source_name, ev.section, ev.score)
```

---

## 两种粒度，别混用

| 方法 | 返回 | 谁来判断"够了没有" | 适用场景 |
| --- | --- | --- | --- |
| `retrieve()` | 逐条证据 | **你的 agent** | 需要精确引用、需要多轮补充检索、要把证据交给别的模型 |
| `answer()` | 已生成的答案 + 引用 | 元件内部 | 单轮问答、不需要 agent 介入判断 |

`retrieve()` 不生成答案，这是刻意的。把上下文拼成一个字符串交给 agent，等于把
"信息是否充分"的判断锁死在元件内部，agent 就失去了再检索一次的机会。

需要整块上下文时传 `with_context=True`；需要每条证据的完整正文时保持
`include_full_text=True`（默认）。

---

## 必须读 `degraded`

检索链路有多处降级，元件不会静默吞掉它们：

```python
hit = kb.retrieve(q)
if hit.degraded:
    # 例如 ["graph_unavailable", "rerank_unavailable"]
    logger.warning("检索链路降级：%s", hit.degraded)
```

这是 agent 做决策的关键信息。`graph_unavailable` 意味着多跳证据缺失，
`rerank_unavailable` 意味着排序只靠 RRF 融合分。看到降级时的正确反应通常是
放宽 top_k 或换一种检索表述，而不是把"证据不足"当成"知识库没有"。

`health()` 给出同样的信息，适合在 agent 启动时先探一次。

---

## 会话状态在 agent 这一侧

元件不保存任何会话状态。多轮对话由 agent 自己维护历史并传入：

```python
ans = kb.answer("那它对线宽有什么要求？", history=[
    {"role": "user", "content": "4 层板阻抗控制关注什么？"},
    {"role": "assistant", "content": "..."},
])
```

`history` 会被用于历史感知的查询改写（"那它" → 具体对象），最多取最近
`_MAX_HISTORY_TURNS` 轮。**元件不存历史的好处**是：agent 可以自由裁剪、压缩、
换一种记忆结构，元件不会在背后偷偷攒一份和 agent 不一致的上下文。

---

## 工具定义与分发

```python
specs = kb.tool_specs()                    # OpenAI function calling 格式
result = kb.call_tool(name, arguments)     # 返回 dict，直接 json.dumps 回灌
```

四个工具：

| 工具 | 作用 | 什么时候用 |
| --- | --- | --- |
| `search_knowledge` | 检索证据，不生成答案 | agent 要自己判断、要多次补充检索 |
| `ask_knowledge` | 检索 + 生成答案 | 直接把问题丢过去 |
| `explore_entity` | 查询图谱邻域 | 概念关联、多跳推理 |
| `knowledge_status` | 组件就绪状态 | 首次调用前探活 |

`call_tool` 会把 `top_k` 夹在 `[1, 30]`，防止模型给出离谱数值把上下文撑爆。
未知工具名返回 `{"error": ...}` 而不是抛异常 —— 模型幻觉出一个工具名是常态，
不该让 agent 崩掉。

`examples/agent_toolkit_demo.py --mode react` 给了一个不依赖原生 tool calling 的
文本协议循环实现，对不支持 function calling 的模型也能用。

---

## 运行时约束

**初始化是重操作。** 首次 `ensure_ready()` 要连 Milvus、加载 embedding 与 rerank
模型，数十秒。建议在 agent 进程启动时预热一次，不要放在首个请求路径上。

**并发安全。** `ensure_ready()` 有锁保护，重复调用是廉价空操作；检索本身只读
底层索引，可并发调用。若同进程内 HTTP 服务已初始化，元件直接复用，不会重复建索引。

**同步与异步。** `aretrieve` / `aanswer` / `acall_tool` 用 `asyncio.to_thread`
包装，不要在事件循环里直接调同步版本 —— 检索是阻塞 IO，会卡住整个循环。

**鉴权与元件无关。** HTTP 层的 Bearer Token 不参与元件调用。元件假设调用方已经
是可信进程内代码；跨进程暴露时应当自己套一层鉴权，而不是指望元件。

---

## 已知边界

| 限制 | 说明 |
| --- | --- |
| 单进程单索引 | `_app_state` 是模块级全局。多进程部署时每个进程各建一份索引，内存会翻倍 |
| 无流式 | `answer()` 返回完整答案。流式只在 HTTP 层的 `/api/ask/stream` 提供 |
| 图依赖建图 | `explore_entity` 需要先跑过入库建图；`kg.json` 不存在时返回 `graph_unavailable` |
| 降级不自动重试 | 元件只报告降级，不替你换策略。重试 / 放宽 top_k 属于 agent 的决策 |
| `score_threshold` 依赖精排 | 未启用精排时分数是 RRF 融合分（量级 ~0.01），按 0~1 的相似度设阈值会全部过滤掉 |

## 为什么还依赖 `dify_external_api`

`engine` 复用 HTTP 模块的 `_app_state` 与 `_retrieve_nodes`。这是过渡态：
检索编排有 400 余行，整体搬迁在没有集成测试的环境里风险高于收益。

好处是两种形态（Web / agent）共用同一个索引实例，不会出现两份词法索引互相不一致。
演进方向是把检索核心下沉到独立的 retrieval 层，届时 `engine` 的对外接口不变。

