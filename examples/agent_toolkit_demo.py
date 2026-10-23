"""把 PCB-RAG 当作元件接入自研 agent 的最小示例。

三种接入方式，按 agent 的复杂度递增：

    1. direct  —— 直接调用 retrieve / answer。适合单轮问答，不需要 agent 决策。
    2. tools   —— 打印 OpenAI function calling 定义，挂到支持 tool calling 的模型上。
    3. react   —— 文本协议的多轮循环。不依赖模型的 tool calling 能力，通配性最好。

运行前确保 Milvus 已启动、``.env`` 里的模型后端可用：

    python examples/agent_toolkit_demo.py --mode direct --query "4 层板阻抗控制关注什么"
    python examples/agent_toolkit_demo.py --mode tools
    python examples/agent_toolkit_demo.py --mode react --query "沉金和喷锡怎么选"

注意：首次调用会连接 Milvus 并加载 embedding / rerank 模型，耗时数十秒。
"""

from __future__ import annotations

import argparse
import json
import sys

from pcb_rag.env_loader import load_project_env

load_project_env()

from pcb_rag.engine import KnowledgeBase  # noqa: E402


# ---------------------------------------------------------------------------
# 方式一：直接调用
# ---------------------------------------------------------------------------
def mode_direct(query: str) -> None:
    """最少代码的用法：拿证据自己组织，或直接要一个带引用的答案。"""
    kb = KnowledgeBase()
    print(f"[状态] {json.dumps(kb.health(), ensure_ascii=False)}")

    hit = kb.retrieve(query, top_k=6)
    print(f"\n[检索] {len(hit.evidence)} 条证据，耗时 {hit.elapsed_ms} ms")
    if hit.degraded:
        print(f"[降级] {hit.degraded}")

    for ev in hit.evidence[:6]:
        print(f"  [{ev.index}] {ev.source_name} · {ev.section}  score={ev.score:.3f}  {ev.tokens}tok")
        print(f"      {ev.snippet[:100]}")

    ans = kb.answer(query)
    print(f"\n[答案] 耗时 {ans.elapsed_ms} ms")
    print(ans.answer)
    if ans.citations:
        print("\n[引用]")
        for c in ans.citations[:5]:
            print(f"  {json.dumps(c, ensure_ascii=False)}")


# ---------------------------------------------------------------------------
# 方式二：原生 function calling
# ---------------------------------------------------------------------------
def mode_tools() -> None:
    """打印可直接塞进 tools 参数的定义，并演示分发。"""
    kb = KnowledgeBase(auto_init=False)
    specs = kb.tool_specs()

    print(json.dumps(specs, ensure_ascii=False, indent=2))
    print("\n把上面这段放进请求的 tools 字段；模型返回 tool_calls 后，")
    print("取出 name 与 arguments，交给 kb.call_tool(name, arguments)：\n")
    print('    result = kb.call_tool(tool_call.function.name,')
    print('                            json.loads(tool_call.function.arguments))')
    print('    # result 是 dict，直接 json.dumps 回灌给模型')


# ---------------------------------------------------------------------------
# 方式三：文本协议循环
# ---------------------------------------------------------------------------
_SYSTEM = """你是 PCB 技术助手，可以调用一个知识库工具回答问题。

可用工具：
- search_knowledge(query, top_k)：检索原文证据，返回片段与出处。不生成答案。
- ask_knowledge(query)：检索并生成带引用的答案。
- explore_entity(entity)：查询概念在图谱中的邻域与关系说明。

规则：
1. 需要事实依据时必须调用工具，不要凭记忆回答。
2. 先查证据；证据不足时换一个更具体的查询再查一次。
3. 最多调用 3 次工具，然后给出结论。

每一步只输出一个 JSON，不要输出其他文字：
- 调用工具：{"tool": "search_knowledge", "arguments": {"query": "...", "top_k": 6}}
- 给出结论：{"final": "你的回答"}
"""


def _llm_complete(llm, prompt: str) -> str:
    resp = llm.complete(prompt)
    return (getattr(resp, "text", None) or str(resp)).strip()


def _parse_step(text: str) -> dict:
    """从模型输出里抠出第一个 JSON 对象。模型常带解释文字，不能直接 loads。"""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return {"final": text}


def mode_react(query: str, max_steps: int = 3) -> None:
    from pcb_rag.api_clients import build_llm

    kb = KnowledgeBase()
    llm = build_llm()

    transcript = [f"用户问题：{query}"]
    for step in range(1, max_steps + 1):
        prompt = _SYSTEM + "\n\n" + "\n".join(transcript) + "\n\n下一步："
        raw = _llm_complete(llm, prompt)
        decision = _parse_step(raw)

        if "final" in decision:
            print(f"\n[第 {step} 步] 给出结论\n")
            print(decision["final"])
            return

        tool = decision.get("tool", "")
        args = decision.get("arguments", {}) or {}
        print(f"\n[第 {step} 步] 调用 {tool}({json.dumps(args, ensure_ascii=False)})")

        result = kb.call_tool(tool, args)
        if "error" in result:
            observation = f"工具报错：{result['error']}"
        elif tool == "search_knowledge":
            rows = result.get("evidence", [])
            lines = [
                f"[{r['index']}] {r['source_name']} · {r['section']}"
                f"（score {r['score']:.3f}）\n    {r['snippet'][:220]}"
                for r in rows[:5]
            ]
            observation = f"检索到 {len(rows)} 条：\n" + ("\n".join(lines) if lines else "（无结果）")
            if result.get("degraded"):
                observation += f"\n注意：链路降级 {result['degraded']}"
        else:
            observation = json.dumps(result, ensure_ascii=False)[:1800]

        print(f"  观察：{observation[:300]}")
        transcript.append(f"上一步调用 {tool}，观察结果：\n{observation}")

    print("\n[达到步数上限] 用已有证据作答")
    transcript.append("请基于上述证据直接给出结论，只输出 {\"final\": \"...\"}")
    prompt = _SYSTEM + "\n\n" + "\n".join(transcript) + "\n\n下一步："
    print(_parse_step(_llm_complete(llm, prompt)).get("final", "（无结论）"))


def main() -> int:
    parser = argparse.ArgumentParser(description="PCB-RAG 元件接入示例")
    parser.add_argument("--mode", choices=["direct", "tools", "react"], default="direct")
    parser.add_argument("--query", default="4 层板的阻抗控制需要关注哪些参数？")
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    if args.mode == "direct":
        mode_direct(args.query)
    elif args.mode == "tools":
        mode_tools()
    else:
        mode_react(args.query, max_steps=args.steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
