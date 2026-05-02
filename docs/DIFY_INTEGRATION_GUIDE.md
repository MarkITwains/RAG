# 🔌 PCB-RAG 与 Dify 集成及 API 暴露指南

本文档介绍了如何将高度优化的 PCB-RAG（融合了 HyDE、多路召回、Rerank 等策略）接入 Dify，并最终通过 Dify 对外暴露标准对话 API，为后续 Altium Designer (AD) 等外部系统的接入打下基础。

## 1. 整体架构思路

为了最大程度保留我们对 RAG 检索链路的深度优化，同时利用 Dify 强大的提示词编排、LLM 接入和对话流管理能力，推荐采用 **“Dify 外部知识库 + Dify 对话 API”** 的架构：

1. **底层检索侧（PCB-RAG）**：使用 FastAPI 将现有的优化版 `query.py` 包装成符合 Dify 规范的 REST API（即 Dify 的“外部知识库 API”）。
2. **逻辑编排侧（Dify）**：在 Dify 中配置外部知识库，并创建 Chat 应用对接该知识库，利用大模型生成最终回复。
3. **接口服务侧（API 暴露）**：Dify 提供标准的 `/v1/chat-messages` 接口，等待未来的客户端（如 AD 插件）调用。

---

## 2. 核心步骤一：将 PCB-RAG 封装为 Dify 外部知识库 API

Dify 支持接入外部检索 API。本项目已提供 FastAPI 服务模块 `pcb_rag.dify_external_api`。

### 代码实现 (`dify_external_api.py`)

```python
from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel
from typing import List
import uvicorn

# 导入项目中现有的查询组件，例如您带有 HyDE 和 Rerank 的核心搜索函数
# 假设 query.py 中您的核心检索函数名为 perform_optimized_search
from pcb_rag.query import perform_optimized_search

app = FastAPI(title="PCB-RAG Dify External Knowledge API")

# 这是跟 Dify 约定的鉴权 Token，您需要在 Dify 后台配置时保持一致
API_TOKEN = os.getenv("DIFY_API_TOKEN", "change-me")

# -- Dify 请求数据模型 --
class RetrievalReq(BaseModel):
    knowledge_id: str
    query: str
    retrieval_setting: dict

# -- Dify 响应数据模型 --
class Record(BaseModel):
    content: str
    score: float
    title: str = ""
    metadata: dict = {}

class RetrievalRes(BaseModel):
    records: List[Record]

# 鉴权依赖
def verify_token(authorization: str = Header(None)):
    if not authorization or authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.post("/retrieval", response_model=RetrievalRes)
async def external_retrieval(req: RetrievalReq, _: str = Depends(verify_token)):
    user_query = req.query
    
    # 🔰 核心逻辑：调用您已优化好的 PCB-RAG 检索管道
    # 这里会自动过 HyDE、多路召回、BGE Rerank 等您的所有优化策略
    # 此处假设 perform_optimized_search 返回格式如: [{"text": "...", "score": 0.9, "source": "..."}]
    search_results = perform_optimized_search(query=user_query, top_k=5)
    
    # 将结果转换为 Dify 所需的格式
    records = []
    for doc in search_results:
        records.append(Record(
            content=doc.get("text", ""),
            score=doc.get("score", 0.0),
            title=doc.get("source", "未知文档"),
            metadata={"source": doc.get("source", "")}
        ))
        
    return RetrievalRes(records=records)

if __name__ == "__main__":
    # 启动 API 服务
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**运行该服务：**
```bash
bash scripts/serve_api.sh
```

---

## 3. 核心步骤二：在 Dify 中进行配置

### 3.1 接入外部知识库
1. 登录 Dify 平台。
2. 导航至左侧菜单的 **知识库 (Knowledge)** -> 取消选择或忽略创建本地知识库，选择 **接入外部知识库 (External API)**。
3. 填写配置项：
   - **名称**：PCB 硬件设计规范与经验
   - **URL**：`http://<your-server-host>:8000/retrieval`
   - **API Key**：`Bearer <your-dify-api-token>`（需包含 Bearer 前缀，需与服务端 `DIFY_API_TOKEN` 一致）
4. 保存测试，确认 Dify 成功连接到了您的 FastAPI。

### 3.2 创建并编排应用
1. 导航至 **工作室 (Studio)** -> **创建空白应用** -> 选择 **聊天助手 (Chatbot)**。
2. 在 **上下文 (Context)** 面板中，选择并添加刚才创建的“PCB 硬件设计规范与经验”外部知识库。
3. 在 **提示词 (Prompt)** 区域定义 AI 人设：
   > "你是一个资深的 PCB 硬件工程师和 Altium Designer 专家。请根据提供的上下文中提取有用的技术规范、布线规则或操作指南，来严谨地回答用户的问题。如果上下文中没有包含答案，请明确告知用户。"
4. 在右侧测试窗口提问，验证回答质量。

---

## 4. 核心步骤三：从 Dify 暴露对外的 API

一切在 Dify 就绪后，即可为未来的 AD (Altium Designer) 接入准备标准对话 API。

1. 在创建的 Chatbot 应用页面左侧菜单，点击 **访问 API (API Access)**。
2. 点击右上角 **API 密钥 -> 创建新密钥**，妥善保存这串具有访问权限的 `API Key` (如：`app-Xxxxxxxxxxxxx`)。
3. 后续客户端或任何软件（例如 AD），只需向该接口发送标准的 POST 请求即可进行问答。

### 未来客户端（如 AD 的 C# 插件或 Python 脚本）请求示例：

```bash
curl -X POST 'https://api.dify.ai/v1/chat-messages' \
--header 'Authorization: Bearer app-您的Dify_API_KEY' \
--header 'Content-Type: application/json' \
--data-raw '{
    "inputs": {},
    "query": "如何在AD中设置蛇形等长布线的阻抗控制？",
    "response_mode": "blocking",
    "conversation_id": "",
    "user": "ad_plugin_user_1"
}'
```

**Dify 响应数据结构（简版）**：
客户端只需解析 `answer` 字段即可展示给用户。
```json
{
    "event": "message",
    "message_id": "92xxxx-xxxx-xxxx",
    "conversation_id": "45xxxx-xxxx-xxxx",
    "mode": "chat",
    "answer": "根据规范，设置蛇形等长布线的阻抗控制，首先需要在 Design -> Rules 中...",
    "created_at": 1709290000
}
```

## 下一步
1. 按照第 2 步，在您的代码库中添加并调试 FastAPI 外部接入代码。
2. 保持端口通畅，完成 Dify 的配置和发版。
3. 当确认通过 API 工具 (如 Postman/curl) 能流畅获取结合了 RAG 上下文的高质量回答后，再进行具体的 Altium Designer 弹窗/侧边栏界面开发。