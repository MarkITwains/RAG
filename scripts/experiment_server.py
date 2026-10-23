#!/usr/bin/env python
"""Isolated production app with experiment-only stage measurements."""

from __future__ import annotations

import functools
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from scripts.experiment_common import isolate_project_env  # noqa: E402

isolate_project_env()

import pcb_rag.dify_external_api as api  # noqa: E402
from pcb_rag.observability import counter, observe, reset_metrics  # noqa: E402
from fastapi import Depends  # noqa: E402


def measured(function, name):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        counter(name + ".count")
        try:
            return function(*args, **kwargs)
        except Exception:
            counter(name + ".errors")
            raise
        finally:
            observe(name + ".ms", (time.perf_counter() - start) * 1000)

    return wrapper


def main():
    import uvicorn
    from llama_index.core.indices.vector_store.retrievers.retriever import VectorIndexRetriever
    from pcb_rag.api_clients import ApiReranker

    ApiReranker._request_scores = measured(ApiReranker._request_scores, "stage.rerank_request")
    VectorIndexRetriever._retrieve = measured(VectorIndexRetriever._retrieve, "stage.vector")
    api.LocalBM25Retriever._retrieve = measured(api.LocalBM25Retriever._retrieve, "stage.bm25")
    api._retrieve_nodes = measured(api._retrieve_nodes, "stage.retrieval")
    api.hyde_expand_query = measured(api.hyde_expand_query, "stage.hyde_fact")
    api._hyde_expand_v2 = measured(api._hyde_expand_v2, "stage.hyde_rule")
    api._generate_answer_with_history = measured(api._generate_answer_with_history, "stage.generation")

    @api.app.middleware("http")
    async def measurements(request, call_next):
        start = time.perf_counter()
        metric = "http." + request.method + request.url.path.replace("/", ".")
        try:
            response = await call_next(request)
            counter(metric + ".count")
            if response.status_code >= 400:
                counter(metric + ".errors")
            return response
        finally:
            observe(metric + ".ms", (time.perf_counter() - start) * 1000)

    @api.app.post("/_experiment/reset", dependencies=[Depends(api.verify_token)])
    def reset():
        reset_metrics()
        return {"status": "reset"}

    uvicorn.run(api.app, host="127.0.0.1", port=int(os.getenv("EXPERIMENT_PORT", "18000")), loop="asyncio")


if __name__ == "__main__":
    main()
