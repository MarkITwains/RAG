"""Usage and request cancellation for embedding adapters returning only vectors.

These adapters expose estimates because their parent APIs discard provider
usage. NVIDIA's adapter records the original response instead.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar

from pcb_rag.runtime import current_control, run_stage
from pcb_rag.usage import finish_usage, reserve_usage

_tracking = ContextVar("embedding_usage_tracking", default=False)


class EmbeddingUsageMixin:
    def _track(self, texts, call, *args):
        if _tracking.get():
            return call(*args)
        ticket = reserve_usage("embedding", texts, self.model_name)

        def invoke():
            token = _tracking.set(True)
            try:
                result = call(*args)
            except BaseException:
                finish_usage(ticket, error=True)
                raise
            finally:
                _tracking.reset(token)
            finish_usage(ticket)
            return result

        return run_stage("embedding", invoke)

    async def _atrack(self, texts, call, *args):
        if _tracking.get():
            return await call(*args)
        ticket = reserve_usage("embedding", texts, self.model_name)
        token = _tracking.set(True)
        try:
            control = current_control()
            result = await asyncio.wait_for(call(*args), control.remaining()) if control else await call(*args)
        except BaseException:
            finish_usage(ticket, error=True)
            raise
        finally:
            _tracking.reset(token)
        finish_usage(ticket)
        return result

    def _get_query_embedding(self, query):
        return self._track([query], super()._get_query_embedding, query)

    def _get_text_embedding(self, text):
        return self._track([text], super()._get_text_embedding, text)

    def _get_text_embeddings(self, texts):
        return self._track(texts, super()._get_text_embeddings, texts)

    async def _aget_query_embedding(self, query):
        return await self._atrack([query], super()._aget_query_embedding, query)

    async def _aget_text_embedding(self, text):
        return await self._atrack([text], super()._aget_text_embedding, text)

    async def _aget_text_embeddings(self, texts):
        return await self._atrack(texts, super()._aget_text_embeddings, texts)
