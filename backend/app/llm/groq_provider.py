"""
app/llm/groq_provider.py
─────────────────────────
Groq LLM provider implementing BaseLLMProvider.

Wraps GroqClient without modifying it.
All generation calls delegate to GroqClient.generate().

Follows OllamaProvider structure exactly:
- Constructor receives a client instance, not settings directly.
- generate() catches all exceptions, re-raises as ProviderException.
- stream() yields full response as a single chunk (Phase 3, not true streaming).
- health_check() raises ProviderNotReadyException on failure.
- model_info() returns provider, model, base_url keys.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from app.core.exceptions import ProviderException, ProviderNotReadyException
from app.llm.base_provider import BaseLLMProvider
from app.llm.groq_client import GroqClient

logger = logging.getLogger(__name__)


class GroqProvider(BaseLLMProvider):
    """
    LLM provider backed by the Groq cloud API.

    Args:
        client: An initialised :class:`GroqClient` instance.
    """

    def __init__(self, client: GroqClient) -> None:
        self._client = client
        logger.debug("GroqProvider initialised")

    # ── BaseLLMProvider implementation ────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        """
        Generate text via Groq.

        Delegates to GroqClient.generate() and translates
        GroqException → ProviderException for provider-agnostic callers.
        """
        logger.info(
            "[PROVIDER] Groq generate",
            extra={
                "model": model,
                "prompt_len": len(prompt),
                "has_system": system is not None,
            },
        )

        try:
            return await self._client.generate(
                prompt=prompt,
                model=model,
                system=system,
                options=options,
            )
        except Exception as exc:
            raise ProviderException(
                f"Groq generation failed: {exc}",
                context={"model": model},
            ) from exc

    async def stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream tokens from Groq.

        Note: Streaming is stubbed in Phase 3. Full streaming implementation
        will be added in Phase 4 (Research Workflows). The full response is
        yielded as a single chunk.
        """
        # Phase 3: yield the full response as a single chunk
        result = await self.generate(
            prompt, system=system, model=model, options=options
        )
        yield result

    async def health_check(self) -> bool:
        """
        Verify Groq is reachable and the API key is valid.

        Raises:
            ProviderNotReadyException: If Groq is unavailable or key is invalid.
        """
        try:
            return await self._client.health_check()
        except Exception as exc:
            raise ProviderNotReadyException(
                f"Groq provider is not ready: {exc}"
            ) from exc

    async def model_info(self) -> dict[str, Any]:
        """Return Groq provider and model metadata."""
        models = []
        try:
            models = await self._client.list_models()
        except Exception:
            pass  # best-effort

        return {
            "provider": "groq",
            "model": self._client._default_model,
            "base_url": "https://api.groq.com/openai/v1",
            "available_models": [m.get("id") for m in models],
        }
