"""
tests/test_memory_service.py
─────────────────────────────
Tests for MemoryService.

All dependencies (embedding, vector store, repos, reranker) are mocked.
Tests verify the orchestration logic without any real I/O.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.exceptions import DocumentNotFoundException, IngestionException
from app.memory.memory_service import MemoryService
from app.schemas.documents import DocumentMetadata, DocumentStatus, FileType, IngestionResult
from app.schemas.retrieval import RetrievedChunk


def _make_settings() -> MagicMock:
    s = MagicMock()
    s.retrieval_top_k = 5
    s.retrieval_candidate_k = 20
    s.reranker_enabled = True
    s.chunk_token_size = 10
    s.chunk_overlap_tokens = 2
    s.tokenizer_model = "bert-base-uncased"
    s.embedding_model = "nomic-embed-text"
    return s


def _make_metadata(
    doc_id: str = "doc-001",
    status: DocumentStatus = DocumentStatus.UPLOADED,
    filename: str = "test.txt",
) -> DocumentMetadata:
    return DocumentMetadata(
        document_id=doc_id,
        filename=filename,
        file_type=FileType.TXT,
        file_path="/tmp/test.txt",
        status=status,
    )


def _make_retrieved_chunk(chunk_id: str = "c1", score: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-001",
        chunk_index=0,
        content="test content",
        token_count=10,
        filename="test.txt",
        retrieval_score=score,
    )


def _make_service(
    doc_repo: MagicMock,
    metrics_repo: MagicMock,
    embedding: MagicMock,
    vector_store: MagicMock,
    reranker: MagicMock | None = None,
) -> MemoryService:
    settings = _make_settings()
    service = MemoryService(
        settings=settings,
        embedding=embedding,
        vector_store=vector_store,
        doc_repo=doc_repo,
        metrics_repo=metrics_repo,
        reranker=reranker,
    )
    # Replace the chunker with a mock
    mock_chunker = MagicMock()
    from app.schemas.documents import DocumentChunk
    mock_chunker.chunk.return_value = [
        DocumentChunk(
            chunk_id="doc-001_chunk_0",
            document_id="doc-001",
            chunk_index=0,
            token_count=5,
            content="test chunk content",
            filename="test.txt",
        )
    ]
    service._chunker = mock_chunker
    return service


class TestMemoryServiceStoreDocument:
    async def test_store_document_calls_repo_create(self) -> None:
        doc_repo = MagicMock()
        doc_repo.create = AsyncMock(return_value=_make_metadata())
        metrics_repo = MagicMock()
        embedding = MagicMock()
        vector_store = MagicMock()

        service = _make_service(doc_repo, metrics_repo, embedding, vector_store)
        metadata = _make_metadata()
        result = await service.store_document(metadata)

        doc_repo.create.assert_called_once_with(metadata)
        assert result.document_id == "doc-001"


class TestMemoryServiceProcessDocument:
    def _make_full_service_with_temp_file(self) -> tuple[MemoryService, MagicMock, str]:
        """Build a service with a real temp file for ingestion."""
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        tmp.write("This is test content for ingestion pipeline testing.")
        tmp.flush()
        tmp_path = tmp.name
        tmp.close()

        metadata = DocumentMetadata(
            document_id="doc-001",
            filename="test.txt",
            file_type=FileType.TXT,
            file_path=tmp_path,
            status=DocumentStatus.UPLOADED,
        )

        doc_repo = MagicMock()
        doc_repo.get = AsyncMock(return_value=metadata)
        doc_repo.update_status = AsyncMock()

        metrics_repo = MagicMock()
        metrics_repo.record = AsyncMock()

        embedding = MagicMock()
        embedding.embed_batch = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        vector_store = MagicMock()
        vector_store.upsert = AsyncMock()

        service = _make_service(doc_repo, metrics_repo, embedding, vector_store)
        return service, doc_repo, tmp_path

    async def test_process_document_sets_processing_then_processed(self) -> None:
        service, doc_repo, tmp_path = self._make_full_service_with_temp_file()
        try:
            result = await service.process_document("doc-001")
            assert result.status == DocumentStatus.PROCESSED

            calls = [call.args[1] for call in doc_repo.update_status.call_args_list]
            assert DocumentStatus.PROCESSING in calls
            assert DocumentStatus.PROCESSED in calls
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    async def test_process_document_raises_for_missing_doc(self) -> None:
        doc_repo = MagicMock()
        doc_repo.get = AsyncMock(side_effect=DocumentNotFoundException("not found"))
        metrics_repo = MagicMock()
        embedding = MagicMock()
        vector_store = MagicMock()

        service = _make_service(doc_repo, metrics_repo, embedding, vector_store)

        with pytest.raises(DocumentNotFoundException):
            await service.process_document("nonexistent-id")


class TestMemoryServiceRetrieve:
    def _make_retrieve_service(
        self,
        vector_results: list[RetrievedChunk],
        reranker: MagicMock | None = None,
    ) -> MemoryService:
        doc_repo = MagicMock()
        metrics_repo = MagicMock()
        metrics_repo.record = AsyncMock()

        embedding = MagicMock()
        embedding.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3])

        vector_store = MagicMock()
        vector_store.query = AsyncMock(return_value=(vector_results, None))

        return _make_service(doc_repo, metrics_repo, embedding, vector_store, reranker)

    async def test_retrieve_returns_results(self) -> None:
        chunks = [_make_retrieved_chunk("c1"), _make_retrieved_chunk("c2")]
        service = self._make_retrieve_service(chunks)

        results = await service.retrieve("test query", top_k=5)
        assert len(results) == 2

    async def test_retrieve_embeds_query(self) -> None:
        chunks = [_make_retrieved_chunk()]
        service = self._make_retrieve_service(chunks)

        await service.retrieve("test query")
        service._embedding.embed_text.assert_called_once_with("test query")

    async def test_retrieve_applies_score_threshold(self) -> None:
        chunks = [
            _make_retrieved_chunk("c1", score=0.9),
            _make_retrieved_chunk("c2", score=0.3),
        ]
        service = self._make_retrieve_service(chunks)

        results = await service.retrieve("query", score_threshold=0.5)
        assert len(results) == 1
        assert results[0].chunk_id == "c1"

    async def test_retrieve_limits_to_top_k(self) -> None:
        chunks = [_make_retrieved_chunk(f"c{i}") for i in range(10)]
        service = self._make_retrieve_service(chunks)

        results = await service.retrieve("query", top_k=3)
        assert len(results) == 3

    async def test_retrieve_records_metrics(self) -> None:
        chunks = [_make_retrieved_chunk()]
        service = self._make_retrieve_service(chunks)

        await service.retrieve("query")
        service._metrics_repo.record.assert_called_once()


class TestMemoryServiceHealth:
    async def test_health_returns_dict(self) -> None:
        doc_repo = MagicMock()
        metrics_repo = MagicMock()
        embedding = MagicMock()
        vector_store = MagicMock()
        vector_store.health_check = AsyncMock(return_value=True)
        vector_store.count = AsyncMock(return_value=42)

        service = _make_service(doc_repo, metrics_repo, embedding, vector_store)
        status = await service.health()

        assert "vector_store" in status
        assert status["vector_store"] == "healthy"
        assert status["chunk_count"] == 42
