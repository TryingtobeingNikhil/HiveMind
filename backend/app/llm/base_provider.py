"""
app/llm/base_provider.py
─────────────────────────
Abstract LLM provider interface.

All future LLM backends (Ollama, Groq, DeepSeek, OpenAI) implement this.
Agents and services call provider.generate(...) — never a specific client directly.

This ensures providers are swappable without any agent or service changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator


class BaseLLMProvider(ABC):
    """
    Abstract LLM provider interface.

    Implementations wrap a specific LLM backend (Ollama, Groq, etc.)
    and expose a unified interface for text generation.

    EXECUTION MODEL:
    All calls route through LLMExecutionManager.execute() in Phase 3.
    Providers themselves do NOT implement concurrency control.
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        """
        Generate a text response for the given prompt.

        Args:
            prompt:  User prompt.
            system:  Optional system prompt.
            model:   Override the default model name.
            options: Provider-specific model parameters.

        Returns:
            Generated text string.

        Raises:
            ProviderException: On generation failures.
        """
        ...  # pragma: no cover

    @abstractmethod
    async def stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream token chunks for the given prompt.

        Args:
            prompt:  User prompt.
            system:  Optional system prompt.
            model:   Override the default model name.
            options: Provider-specific model parameters.

        Yields:
            Text token chunks as they arrive.
        """
        ...  # pragma: no cover

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Verify the provider is reachable and can serve requests.

        Returns:
            True if healthy.

        Raises:
            ProviderNotReadyException: If the provider is unavailable.
        """
        ...  # pragma: no cover

    @abstractmethod
    async def model_info(self) -> dict[str, Any]:
        """
        Return metadata about the current model.

        Returns:
            Dict containing at minimum: ``provider``, ``model``.
        """
        ...  # pragma: no cover
