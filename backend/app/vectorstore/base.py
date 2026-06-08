"""
app/vectorstore/base.py
────────────────────────
Abstract base class for all vector store implementations.

Defines the contract that MemoryService uses.
Future implementations (FAISSVectorStore, etc.) can be swapped
without any changes to the service layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.schemas.documents import DocumentChunk
from app.schemas.retrieval import RetrievedChunk


class BaseVectorStore(ABC):
    """
    Abstract vector store interface.

    All methods are async. Synchronous backends (e.g. ChromaDB)
    must be wrapped in ``asyncio.to_thread`` by the implementation.
    """

    @abstractmethod
    async def create_collection(self) -> None:
        """
        Ensure the configured collection exists.
        Called once during application startup.
        """
        ...  # pragma: no cover

    @abstractmethod
    async def upsert(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """
        Insert or update chunks with their embeddings.

        Args:
            chunks:     Chunks to store (IDs, content, metadata).
            embeddings: Embedding vectors aligned with ``chunks``.
        """
        ...  # pragma: no cover

    @abstractmethod
    async def query(
        self,
        query_embedding: list[float],
        n_results: int,
        where: dict[str, Any] | None = None,
        include_embeddings: bool = False,
    ) -> tuple[list[RetrievedChunk], list[list[float]] | None]:
        """
        Find the nearest neighbours for ``query_embedding``.

        Args:
            query_embedding:   Query vector.
            n_results:         Number of results to return.
            where:             Optional metadata filter dict.
            include_embeddings: If True, also return the stored vectors.

        Returns:
            A 2-tuple of:
              - Ordered list of :class:`RetrievedChunk` (best first).
              - Optional list of embedding vectors (aligned, None if not requested).
        """
        ...  # pragma: no cover

    @abstractmethod
    async def delete(self, document_id: str) -> None:
        """
        Remove all chunks belonging to ``document_id``.

        Args:
            document_id: The parent document whose chunks should be deleted.
        """
        ...  # pragma: no cover

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Verify the vector store is initialised and operational.

        Returns:
            True if healthy.
        """
        ...  # pragma: no cover

    @abstractmethod
    async def count(self, document_id: str | None = None) -> int:
        """
        Return the total number of stored chunks.

        Args:
            document_id: If provided, count only chunks for this document.
        """
        ...  # pragma: no cover
