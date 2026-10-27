"""NVIDIA API embeddings with separate query/passage modes in every code path."""

from __future__ import annotations

import math
import time
from typing import Any, List

from llama_index.core.bridge.pydantic import Field
from llama_index.embeddings.openai import OpenAIEmbedding
from openai import APIConnectionError, APIStatusError, APITimeoutError

_RETRY_STATUS = {429, 500, 502, 503, 504}


def _retryable(error: BaseException) -> bool:
    """仅对网关抖动类错误重试；参数/鉴权错误应立即暴露。"""
    if isinstance(error, (APITimeoutError, APIConnectionError)):
        return True
    if isinstance(error, APIStatusError):
        return getattr(error, "status_code", 0) in _RETRY_STATUS
    return False


class NvidiaApiEmbedding(OpenAIEmbedding):
    """Use the OpenAI transport, with NVIDIA's retrieval-specific request body.

    Per-request modes are never stored in ``additional_kwargs``: query and
    ingestion requests can run concurrently on the same embedding instance.
    Float encoding is explicit, avoiding implicit SDK base64 transformations.
    """

    truncate: str = Field(default="END")
    expected_dimension: int = Field(default=0)

    @classmethod
    def class_name(cls) -> str:
        return "NvidiaApiEmbedding"

    def _payload(self, texts: List[str], input_type: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            # Match the existing OpenAIEmbedding newline normalization.
            "input": [text.replace("\n", " ") for text in texts],
            "encoding_format": "float",
            "extra_body": {"input_type": input_type, "truncate": self.truncate},
        }
        # The API accepts 1..native dimensions and re-normalizes shortened vectors.
        # Sending it explicitly keeps the request, the stored profile and the
        # Milvus field width in one place instead of relying on the model default.
        if self.expected_dimension > 0:
            payload["dimensions"] = self.expected_dimension
        return payload

    def _vectors(self, response: Any, count: int) -> List[List[float]]:
        rows = sorted(response.data, key=lambda item: item.index)
        if [row.index for row in rows] != list(range(count)):
            raise ValueError("NVIDIA embedding response has missing or duplicate input indices")
        vectors = []
        for row in rows:
            vector = row.embedding
            if not isinstance(vector, list) or not vector:
                raise ValueError("NVIDIA embedding response must contain nonempty float vectors")
            if self.expected_dimension and len(vector) != self.expected_dimension:
                raise ValueError(
                    f"NVIDIA embedding dimension changed: expected {self.expected_dimension}, got {len(vector)}"
                )
            if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in vector):
                raise ValueError("NVIDIA embedding response contains nonfinite or nonnumeric values")
            vectors.append(vector)
        return vectors

    def _embed(self, texts: List[str], input_type: str) -> List[List[float]]:
        if not texts:
            return []
        from pcb_rag.runtime import current_control, run_stage
        from pcb_rag.usage import finish_usage, reserve_usage
        ticket = reserve_usage("embedding", texts, self.model_name)
        def invoke():
            control = current_control()
            payload = self._payload(texts, input_type)
            if control:
                payload["timeout"] = min(float(self.timeout), control.remaining())
            response = None
            attempts = 0
            while True:
                try:
                    response = self._get_client().embeddings.create(**payload)
                    result = self._vectors(response, len(texts))
                except BaseException as error:
                    attempts += 1
                    # 上游网关偶发超时/5xx：有限退避重试，不改变检索语义，仅提高长跑成功率。
                    if attempts >= 3 or not _retryable(error):
                        finish_usage(ticket, response, error=True)
                        raise
                    if control:
                        control.remaining()
                    time.sleep(1.5 * attempts)
                    continue
                finish_usage(ticket, response)
                return result
        return run_stage("embedding", invoke)

    async def _aembed(self, texts: List[str], input_type: str) -> List[List[float]]:
        if not texts:
            return []
        import asyncio
        from pcb_rag.runtime import current_control
        from pcb_rag.usage import finish_usage, reserve_usage
        ticket = reserve_usage("embedding", texts, self.model_name)
        control = current_control()
        response = None
        attempts = 0
        payload = self._payload(texts, input_type)
        if control:
            payload["timeout"] = min(float(self.timeout), control.remaining())
        while True:
            try:
                call = self._get_aclient().embeddings.create(**payload)
                response = await asyncio.wait_for(call, control.remaining()) if control else await call
                result = self._vectors(response, len(texts))
            except BaseException as error:
                attempts += 1
                if attempts >= 3 or not _retryable(error):
                    finish_usage(ticket, response, error=True)
                    raise
                if control:
                    control.remaining()
                await asyncio.sleep(1.5 * attempts)
                continue
            finish_usage(ticket, response)
            return result

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed([query], "query")[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return (await self._aembed([query], "query"))[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed([text], "passage")[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return (await self._aembed([text], "passage"))[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts, "passage")

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return await self._aembed(texts, "passage")
