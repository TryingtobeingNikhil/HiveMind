"""
app/reranking/base.py
──────────────────────
Abstract base class for retrieval rerankers.

The reranker sits between vector search and final result selection.
Pipeline:
    Query → Embedding Search → Top-20 → Reranker → Top-5

Architecture is pluggable. CrossEncoderReranker can be added in a
later phase without changing the MemoryService interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.retrieval import RetrievedChunk


class BaseReranker(ABC):
    """
    Abstract interface for retrieval rerankers.

    The reranker takes a query and a list of retrieved chunks and
    returns the same list re-ordered by a secondary relevance score.
    """

    @abstractmethod
    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        query_embedding: list[float] | None = None,
        chunk_embeddings: list[list[float]] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Rerank a list of retrieved chunks.

        Args:
            query:             The original search query.
            chunks:            Chunks from the initial vector search.
            query_embedding:   Pre-computed query vector (may be used by implementations).
            chunk_embeddings:  Pre-retrieved chunk vectors aligned with ``chunks``.

        Returns:
            Chunks sorted by descending rerank_score.
            Each chunk in the result has ``rerank_score`` populated.
        """
        ...  # pragma: no cover
