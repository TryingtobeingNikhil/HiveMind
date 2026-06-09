"""
app/embedding/base.py
──────────────────────
Abstract base class for all embedding providers.

All providers produce fixed-dimension float vectors.
Sequential execution only — no concurrent embedding pipelines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEmbeddingProvider(ABC):
    """
    Abstract interface for embedding providers.

    Implementations must be async-first. Batch methods loop sequentially
    — no concurrent embedding is performed.
    """

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """
        Embed a single text string.

        Args:
            text: Input text to embed.

        Returns:
            A float vector of fixed dimensionality.

        Raises:
            EmbeddingException: On provider errors.
        """
        ...  # pragma: no cover

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts sequentially.

        Args:
            texts: List of input texts.

        Returns:
            List of float vectors, aligned with ``texts``.

        Note:
            Implementations MUST NOT embed in parallel. Sequential looping
            is required to honour the single-active-execution model.
        """
        ...  # pragma: no cover

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Verify that the embedding provider is reachable and functional.

        Returns:
            True if the provider is ready.

        Raises:
            EmbeddingModelNotReadyException: If the provider is not available.
        """
        ...  # pragma: no cover

    @abstractmethod
    async def model_info(self) -> dict[str, Any]:
        """
        Return metadata about the embedding model.

        Returns:
            Dict containing at minimum: ``model``, ``dimensions``.
        """
        ...  # pragma: no cover
