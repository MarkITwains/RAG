"""Conversation prompts shared by every generation entry point."""

import os

from pcb_rag.usage import model_complete
from pcb_rag.runtime import DeadlineExceeded, request_scope, run_stage

_HISTORY_REWRITE_PROMPT = """你是 PCB 领域的智能助手。用户的最新问题可能包含代词或省略，需要结合对话历史改写为一个独立、完整的检索查询。

规则：
1. 如果最新问题本身已经完整（不含代词/省略），直接原样输出。
2. 如果最新问题包含"它""这个""那个""上述"等指代词，或省略了主语，用对话历史中的具体实体替换。
3. 只输出改写后的查询，不要解释。
4. 保留原始问题的疑问意图。

对话历史（最近 {n} 轮）：
{history}

最新问题：{query}

改写后的查询："""


_RAG_PROMPT_WITH_HISTORY = """你是 PCB 电路板领域的技术专家。请根据以下参考文档和对话历史回答用户问题。

要求：
1. 只基于参考文档中的信息回答，不要编造
2. 如果文档中没有相关信息，明确说明"根据提供的文档无法回答"
3. 回答要准确、专业、简洁；数值保留原始单位、标准版本、适用条件，表格依据包含对应列头与脚注；不同版本或条件的要求分别说明，无法判定时明确说明证据不足
4. 参考文档带有编号 [1] [2]，引用某条信息时请在对应句子末尾标注同样的角标，例如：[1][2]
5. 如果用户的问题是对前一轮的追问，请结合对话历史进行回答

对话历史（最近 {n} 轮）：
{history}

参考文档：
{context}

用户问题：{query}

回答："""


_MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "5"))


def history_values(history):
    recent = [dict(message) for message in history if message.get("role") in {"user", "assistant"}][
        -_MAX_HISTORY_TURNS:
    ]
    lines = []
    for message in recent:
        label = "用户" if message["role"] == "user" else "助手"
        limit = 1000 if message["role"] == "user" else 300
        lines.append(label + "：" + str(message.get("content", ""))[:limit])
    return {"n": len(recent), "history": "\n".join(lines)}


def prepare_with_history(query, nodes, history, llm):
    from pcb_rag.query import prepare_generation

    recent = list(history or [])[-_MAX_HISTORY_TURNS:]
    while True:
        try:
            return prepare_generation(
                query,
                nodes,
                llm,
                template=_RAG_PROMPT_WITH_HISTORY if recent else None,
                values=history_values(recent) if recent else None,
            )
        except ValueError:
            if not recent:
                raise
            recent.pop(0)


def rewrite_query_with_history(query, chat_history, llm=None):
    if not chat_history:
        return query
    from pcb_rag.query import Settings, prepare_generation

    llm = llm or Settings.llm
    with request_scope(float(os.getenv("RAG_REQUEST_TIMEOUT", "120"))):
        try:
            request = prepare_generation(
                query, [], llm, template=_HISTORY_REWRITE_PROMPT, values=history_values(chat_history)
            )
            response = run_stage("history_rewrite", model_complete, llm, request.prompt)
            rewritten = str(getattr(response, "text", response)).strip()
            return rewritten if 3 < len(rewritten) < 500 else query
        except DeadlineExceeded:
            raise
        except Exception:
            from pcb_rag.runtime import current_control

            control = current_control()
            if control:
                control.degraded.append("history_rewrite_failed")
            return query


def generate_answer_with_history(query, retrieved_docs, chat_history, llm=None):
    from pcb_rag.query import CITATION_ENABLED, Settings

    llm = llm or Settings.llm
    request = prepare_with_history(query, retrieved_docs, chat_history, llm)
    bundle = request.evidence
    with request_scope(float(os.getenv("RAG_REQUEST_TIMEOUT", "120"))):
        if bundle.entries:
            response = run_stage("generation", model_complete, llm, request.prompt)
            answer = str(getattr(response, "text", response))
        else:
            answer = "根据提供的文档无法回答该问题，未找到预算内可用的相关证据。"
    return {
        "answer": answer,
        "evidence_bundle": bundle,
        "citations": bundle.citations(answer) if CITATION_ENABLED else [],
        "invalid_citations": bundle.invalid_citations(answer),
        "sources": [row["source_path"] for row in bundle.sources],
        "context_info": {
            "doc_count": len(bundle.entries),
            "sources_detail": bundle.sources,
            "total_chars": sum(row["char_count"] for row in bundle.entries),
            "budget": bundle.budget,
        },
    }
