"""Command-line adapter to the shared retrieval and evidence pipeline."""

from __future__ import annotations

import argparse
import json

from pcb_rag.engine import KnowledgeBase


def retrieve_for_cli(query, *, pipeline, top_k=10):
    """Small adapter used by both CLI and cross-entry regression tests."""
    return KnowledgeBase(auto_init=False, pipeline=pipeline).retrieve(query, top_k=top_k, with_context=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="PCB knowledge retrieval and cited answers")
    parser.add_argument("query", nargs="?")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--retrieve-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--self-rag", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.top_k <= 100:
        parser.error("--top-k must be between 1 and 100")
    kb = KnowledgeBase()
    history = []

    def execute(query):
        if args.retrieve_only:
            result = kb.retrieve(query, top_k=args.top_k, with_context=True)
        elif args.self_rag:
            from pcb_rag.query import self_rag_query
            from pcb_rag.runtime import request_scope

            class Retriever:
                def retrieve(self, value):
                    return kb.pipeline.retrieve_nodes(value, top_k=args.top_k)

            with request_scope(kb.pipeline.config.timeout):
                result = self_rag_query(query, Retriever(), kb.pipeline.llm)
            bundle = result.pop("evidence_bundle", None)
            result.pop("retrieved_docs", None)
            if bundle:
                result["evidence"] = bundle.sources
            print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result["answer"])
            return
        else:
            result = kb.answer(query, top_k=args.top_k, history=history)
            history.extend([{"role": "user", "content": query}, {"role": "assistant", "content": result.answer}])
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        elif args.retrieve_only:
            print(result.context or "未找到可用证据。")
        else:
            print(result.answer)
            for entry in result.citations:
                print(f"[{entry['index']}] {entry['source_name']} {entry.get('section', '')}")
        if result.degraded and not args.json:
            print("本次降级：" + ", ".join(result.degraded))

    if args.query:
        execute(args.query)
        return 0
    print("PCB-RAG（输入 exit/quit 退出，:clear 清空历史，:status 查看状态）")
    while True:
        try:
            query = input("\nQ> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in {"exit", "quit", ":q"}:
            break
        if query == ":clear":
            history.clear()
        elif query == ":status":
            print(json.dumps(kb.health(), ensure_ascii=False, indent=2))
        elif query.startswith(":model "):
            from pcb_rag.api_clients import build_llm

            kb.pipeline.state["llm"] = build_llm(query.removeprefix(":model ").strip())
        elif query:
            execute(query)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
