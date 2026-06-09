"""
tests/test_reranking.py
────────────────────────
Tests for the reranking layer.

CosineReranker uses numpy — no model weights or network required.
"""

from __future__ import annotations

import pytest

from app.reranking.cosine_reranker import CosineReranker, _cosine_similarity
from app.schemas.retrieval import RetrievedChunk


def _make_chunk(
    chunk_id: str,
    content: str = "test content",
    retrieval_score: float = 0.5,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-001",
        chunk_index=0,
        content=content,
        token_count=10,
        filename="test.txt",
        retrieval_score=retrieval_score,
    )


class TestCosineSimilarity:
    """Unit tests for the cosine similarity helper."""

    def test_identical_vectors_return_one(self) -> None:
        v = [1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors_return_zero(self) -> None:
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_zero_vector_returns_zero(self) -> None:
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_opposite_vectors_return_negative_one(self) -> None:
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(_cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_symmetric(self) -> None:
        a = [0.3, 0.7, 0.1]
        b = [0.9, 0.2, 0.5]
        assert abs(_cosine_similarity(a, b) - _cosine_similarity(b, a)) < 1e-6


class TestCosineReranker:
    """Tests for the CosineReranker."""

    async def test_empty_chunks_returns_empty(self) -> None:
        reranker = CosineReranker()
        result = await reranker.rerank("query", [], query_embedding=[1.0, 0.0])
        assert result == []

    async def test_fallback_when_no_embeddings(self) -> None:
        """When no embeddings provided, returns input order with rerank_score = retrieval_score."""
        reranker = CosineReranker()
        chunks = [_make_chunk("c1", retrieval_score=0.8), _make_chunk("c2", retrieval_score=0.6)]
        result = await reranker.rerank("query", chunks)
        assert len(result) == 2
        assert result[0].rerank_score == result[0].retrieval_score

    async def test_reranks_by_cosine_similarity(self) -> None:
        """Chunk most similar to query should rank first."""
        reranker = CosineReranker()
        # Query points in direction of [1, 0]
        query_embedding = [1.0, 0.0]

        # Chunk A: aligns with query → high similarity
        # Chunk B: orthogonal to query → low similarity
        chunk_a = _make_chunk("chunk_a", retrieval_score=0.5)
        chunk_b = _make_chunk("chunk_b", retrieval_score=0.7)  # higher initial score

        chunk_embeddings = [
            [1.0, 0.0],   # chunk_a — identical to query
            [0.0, 1.0],   # chunk_b — orthogonal to query
        ]

        result = await reranker.rerank(
            "query",
            [chunk_a, chunk_b],
            query_embedding=query_embedding,
            chunk_embeddings=chunk_embeddings,
        )

        # chunk_a should rank first despite lower retrieval_score
        assert result[0].chunk_id == "chunk_a"
        assert result[1].chunk_id == "chunk_b"

    async def test_rerank_scores_populated(self) -> None:
        reranker = CosineReranker()
        query_embedding = [1.0, 0.0]
        chunks = [_make_chunk("c1"), _make_chunk("c2")]
        chunk_embeddings = [[0.9, 0.1], [0.1, 0.9]]

        result = await reranker.rerank(
            "query",
            chunks,
            query_embedding=query_embedding,
            chunk_embeddings=chunk_embeddings,
        )

        for chunk in result:
            assert chunk.rerank_score is not None
            assert 0.0 <= chunk.rerank_score <= 1.0

    async def test_fallback_on_embedding_count_mismatch(self) -> None:
        reranker = CosineReranker()
        chunks = [_make_chunk("c1"), _make_chunk("c2")]
        chunk_embeddings = [[1.0, 0.0]]  # only 1 embedding for 2 chunks

        result = await reranker.rerank(
            "query",
            chunks,
            query_embedding=[1.0, 0.0],
            chunk_embeddings=chunk_embeddings,
        )

        # Should fall back gracefully
        assert len(result) == 2

    async def test_result_is_sorted_descending(self) -> None:
        reranker = CosineReranker()
        query_embedding = [1.0, 0.0]
        chunks = [_make_chunk(f"c{i}") for i in range(5)]
        chunk_embeddings = [
            [0.1, 0.9],
            [0.5, 0.5],
            [1.0, 0.0],
            [0.8, 0.2],
            [0.3, 0.7],
        ]

        result = await reranker.rerank(
            "query",
            chunks,
            query_embedding=query_embedding,
            chunk_embeddings=chunk_embeddings,
        )

        scores = [r.rerank_score for r in result]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]
