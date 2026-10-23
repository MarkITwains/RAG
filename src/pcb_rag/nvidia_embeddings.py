"""NVIDIA API embeddings with separate query/passage modes in every code path."""

from __future__ import annotations

import math
from typing import Any, List

from llama_index.core.bridge.pydantic import Field
from llama_index.embeddings.openai import OpenAIEmbedding


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
        return {
            "model": self.model_name,
            # Match the existing OpenAIEmbedding newline normalization.
            "input": [text.replace("\n", " ") for text in texts],
            "encoding_format": "float",
            "extra_body": {"input_type": input_type, "truncate": self.truncate},
        }

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
        response = self._get_client().embeddings.create(**self._payload(texts, input_type))
        return self._vectors(response, len(texts))

    async def _aembed(self, texts: List[str], input_type: str) -> List[List[float]]:
        if not texts:
            return []
        response = await self._get_aclient().embeddings.create(**self._payload(texts, input_type))
        return self._vectors(response, len(texts))

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
