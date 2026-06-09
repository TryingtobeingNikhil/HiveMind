"""
app/reranking/cosine_reranker.py
──────────────────────────────────
Cosine similarity reranker.

Uses pre-retrieved chunk embeddings from the vector store to rerank
by cosine similarity between the query vector and each chunk vector.

No model weights required — pure numpy computation.
Deterministic output for a given input.

Fallback behaviour:
  If no embeddings are provided, returns the input list unchanged
  (ordering from initial vector search is preserved).
"""

from __future__ import annotations

import logging

import numpy as np

from app.reranking.base import BaseReranker
from app.schemas.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Returns a value in [-1, 1] (clipped to [0, 1] for non-negative embeddings).
    """
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a = float(np.linalg.norm(va))
    norm_b = float(np.linalg.norm(vb))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


class CosineReranker(BaseReranker):
    """
    Reranker using cosine similarity between the query and chunk embeddings.

    Requires pre-retrieved chunk embeddings (from vector store include_embeddings).
    When embeddings are unavailable, falls back to the original retrieval ordering.
    """

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        query_embedding: list[float] | None = None,
        chunk_embeddings: list[list[float]] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Rerank chunks by cosine similarity with the query embedding.

        Args:
            query:             Original search query (unused here — uses embeddings).
            chunks:            Candidates from vector search.
            query_embedding:   Pre-computed query vector.
            chunk_embeddings:  Pre-retrieved chunk vectors aligned with ``chunks``.

        Returns:
            Chunks sorted by ``rerank_score`` descending.
        """
        if not chunks:
            return []

        # Fallback: if we don't have embeddings to rerank with, return as-is
        if query_embedding is None or chunk_embeddings is None:
            logger.debug(
                "CosineReranker: no embeddings provided, returning original order"
            )
            for chunk in chunks:
                chunk.rerank_score = chunk.retrieval_score
            return chunks

        if len(chunk_embeddings) != len(chunks):
            logger.warning(
                "CosineReranker: embedding count mismatch, falling back to original order",
                extra={"chunks": len(chunks), "embeddings": len(chunk_embeddings)},
            )
            for chunk in chunks:
                chunk.rerank_score = chunk.retrieval_score
            return chunks

        logger.info(
            "CosineReranker: reranking",
            extra={"candidates": len(chunks)},
        )

        reranked: list[RetrievedChunk] = []
        for chunk, chunk_emb in zip(chunks, chunk_embeddings):
            if not chunk_emb:
                score = chunk.retrieval_score
            else:
                score = _cosine_similarity(query_embedding, chunk_emb)
                # Clip to [0, 1]
                score = max(0.0, min(1.0, score))
            # Create a copy with rerank_score set
            updated = chunk.model_copy(update={"rerank_score": score})
            reranked.append(updated)

        # Sort descending by rerank_score
        reranked.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)

        logger.info(
            "CosineReranker: complete",
            extra={
                "top_score": reranked[0].rerank_score if reranked else None,
                "bottom_score": reranked[-1].rerank_score if reranked else None,
            },
        )

        return reranked
