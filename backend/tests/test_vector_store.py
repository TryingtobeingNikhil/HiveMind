"""
tests/test_vector_store.py
───────────────────────────
Tests for ChromaVectorStore.

ChromaDB is mocked via unittest.mock.patch — no real ChromaDB instance needed.
Tests verify the asyncio.to_thread wrapping and result parsing logic.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.exceptions import VectorStoreException, VectorStoreNotReadyException
from app.schemas.documents import DocumentChunk, FileType
from app.vectorstore.chroma_store import ChromaVectorStore


def _make_settings() -> MagicMock:
    s = MagicMock()
    s.chroma_persist_path.return_value = None
    # Make it a Path-like object
    from pathlib import Path
    s.chroma_persist_path = Path("/tmp/test_chroma")
    s.chroma_collection_name = "test_collection"
    return s


def _make_chunk(
    chunk_id: str = "doc-001_chunk_0",
    doc_id: str = "doc-001",
    idx: int = 0,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        chunk_index=idx,
        token_count=50,
        content=f"Test content for chunk {idx}",
        filename="test.txt",
    )


_FAKE_QUERY_RESULTS = {
    "ids": [["chunk_0", "chunk_1"]],
    "documents": [["Content A", "Content B"]],
    "metadatas": [
        [
            {"document_id": "doc-001", "chunk_index": 0, "token_count": 50, "filename": "a.txt"},
            {"document_id": "doc-002", "chunk_index": 1, "token_count": 60, "filename": "b.txt"},
        ]
    ],
    "distances": [[0.1, 0.3]],  # cosine distance: lower = better
}


class TestChromaVectorStore:
    def _make_store_with_mock_collection(self) -> tuple[ChromaVectorStore, MagicMock]:
        settings = _make_settings()
        store = ChromaVectorStore(settings)
        # Inject a mock collection directly
        mock_collection = MagicMock()
        store._collection = mock_collection
        return store, mock_collection

    # ── Not ready guard ────────────────────────────────────────────────────────

    async def test_upsert_raises_if_not_initialised(self) -> None:
        settings = _make_settings()
        store = ChromaVectorStore(settings)
        with pytest.raises(VectorStoreNotReadyException):
            await store.upsert([_make_chunk()], [[0.1, 0.2]])

    async def test_query_raises_if_not_initialised(self) -> None:
        settings = _make_settings()
        store = ChromaVectorStore(settings)
        with pytest.raises(VectorStoreNotReadyException):
            await store.query([0.1, 0.2], n_results=5)

    # ── Upsert ────────────────────────────────────────────────────────────────

    async def test_upsert_calls_collection(self) -> None:
        store, mock_col = self._make_store_with_mock_collection()
        chunks = [_make_chunk("c1", "doc-001", 0), _make_chunk("c2", "doc-001", 1)]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]

        await store.upsert(chunks, embeddings)

        mock_col.upsert.assert_called_once()
        call_kwargs = mock_col.upsert.call_args[1]
        assert call_kwargs["ids"] == ["c1", "c2"]
        assert len(call_kwargs["embeddings"]) == 2

    async def test_upsert_raises_on_mismatch(self) -> None:
        store, _ = self._make_store_with_mock_collection()
        chunks = [_make_chunk()]
        embeddings = [[0.1], [0.2]]  # wrong count
        with pytest.raises(VectorStoreException, match="Mismatch"):
            await store.upsert(chunks, embeddings)

    async def test_upsert_empty_is_noop(self) -> None:
        store, mock_col = self._make_store_with_mock_collection()
        await store.upsert([], [])
        mock_col.upsert.assert_not_called()

    # ── Query ─────────────────────────────────────────────────────────────────

    async def test_query_returns_retrieved_chunks(self) -> None:
        store, mock_col = self._make_store_with_mock_collection()
        mock_col.query.return_value = _FAKE_QUERY_RESULTS

        chunks, embeddings = await store.query([0.1, 0.2], n_results=5)

        assert len(chunks) == 2
        assert embeddings is None
        assert chunks[0].chunk_id == "chunk_0"
        assert chunks[1].chunk_id == "chunk_1"

    async def test_query_converts_distance_to_similarity(self) -> None:
        store, mock_col = self._make_store_with_mock_collection()
        mock_col.query.return_value = _FAKE_QUERY_RESULTS

        chunks, _ = await store.query([0.1, 0.2], n_results=5)

        # distance 0.1 → similarity 0.9
        assert abs(chunks[0].retrieval_score - 0.9) < 0.001
        # distance 0.3 → similarity 0.7
        assert abs(chunks[1].retrieval_score - 0.7) < 0.001

    async def test_query_with_include_embeddings(self) -> None:
        store, mock_col = self._make_store_with_mock_collection()
        results_with_embeddings = {
            **_FAKE_QUERY_RESULTS,
            "embeddings": [[[0.1, 0.2], [0.3, 0.4]]],
        }
        mock_col.query.return_value = results_with_embeddings

        chunks, chunk_embeddings = await store.query(
            [0.1, 0.2], n_results=5, include_embeddings=True
        )

        assert chunk_embeddings is not None
        assert len(chunk_embeddings) == 2

    async def test_query_empty_results(self) -> None:
        store, mock_col = self._make_store_with_mock_collection()
        mock_col.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        chunks, _ = await store.query([0.1, 0.2], n_results=5)
        assert chunks == []

    # ── Delete ────────────────────────────────────────────────────────────────

    async def test_delete_calls_collection(self) -> None:
        store, mock_col = self._make_store_with_mock_collection()
        await store.delete("doc-001")
        mock_col.delete.assert_called_once_with(where={"document_id": "doc-001"})

    # ── Health check ──────────────────────────────────────────────────────────

    async def test_health_check_true_when_initialised(self) -> None:
        store, mock_col = self._make_store_with_mock_collection()
        mock_col.count.return_value = 42
        result = await store.health_check()
        assert result is True

    async def test_health_check_false_when_not_initialised(self) -> None:
        settings = _make_settings()
        store = ChromaVectorStore(settings)
        result = await store.health_check()
        assert result is False

    # ── Count ─────────────────────────────────────────────────────────────────

    async def test_count_returns_total(self) -> None:
        store, mock_col = self._make_store_with_mock_collection()
        mock_col.count.return_value = 99
        count = await store.count()
        assert count == 99

    async def test_count_by_document_id(self) -> None:
        store, mock_col = self._make_store_with_mock_collection()
        mock_col.get.return_value = {"ids": ["c1", "c2"]}
        count = await store.count("doc-001")
        assert count == 2
