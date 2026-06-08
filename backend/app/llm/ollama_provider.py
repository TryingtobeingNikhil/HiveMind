"""
app/llm/ollama_provider.py
───────────────────────────
Ollama LLM provider implementing BaseLLMProvider.

Wraps the existing OllamaClient without modifying it.
All generation calls delegate to OllamaClient.generate().

Future providers (GroqProvider, DeepSeekProvider, OpenAIProvider)
follow the same pattern — implement BaseLLMProvider, wrap their SDK.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from app.core.exceptions import ProviderException, ProviderNotReadyException
from app.llm.base_provider import BaseLLMProvider
from app.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """
    LLM provider backed by a local Ollama instance.

    Args:
        client: An initialised :class:`OllamaClient` instance.
    """

    def __init__(self, client: OllamaClient) -> None:
        self._client = client
        logger.debug("OllamaProvider initialised")

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
        Generate text via Ollama.

        Delegates to OllamaClient.generate() and translates
        OllamaException → ProviderException for provider-agnostic callers.
        """
        logger.info(
            "[PROVIDER] Ollama generate",
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
                f"Ollama generation failed: {exc}",
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
        Stream tokens from Ollama.

        Note: Streaming is stubbed in Phase 2. Full streaming implementation
        will be added in Phase 4 (Research Workflows).
        """
        # Phase 2: yield the full response as a single chunk
        result = await self.generate(
            prompt, system=system, model=model, options=options
        )
        yield result

    async def health_check(self) -> bool:
        """
        Verify Ollama is reachable.

        Raises:
            ProviderNotReadyException: If Ollama is unavailable.
        """
        try:
            return await self._client.health_check()
        except Exception as exc:
            raise ProviderNotReadyException(
                f"Ollama provider is not ready: {exc}"
            ) from exc

    async def model_info(self) -> dict[str, Any]:
        """Return Ollama provider and model metadata."""
        models = []
        try:
            models = await self._client.list_models()
        except Exception:
            pass  # best-effort

        return {
            "provider": "ollama",
            "model": self._client._default_model,
            "base_url": self._client._base_url,
            "available_models": [m.get("name") for m in models],
        }
