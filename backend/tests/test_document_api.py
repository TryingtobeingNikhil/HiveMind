"""
tests/test_document_api.py
───────────────────────────
Integration tests for the document and retrieval API endpoints.

All services are mocked via dependency_overrides.
Tests use FastAPI's TestClient — no real Ollama or ChromaDB.
"""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.schemas.documents import (
    DocumentListResponse,
    DocumentMetadata,
    DocumentStatus,
    FileType,
    IngestionResult,
    UploadResponse,
)
from app.schemas.retrieval import (
    ContextPackage,
    ContextSource,
    RetrievedChunk,
    TokenUsage,
)


def _make_test_settings() -> Settings:
    return Settings(
        APP_ENV="testing",
        OLLAMA_BASE_URL="http://localhost:11434",
        UPLOADS_DIR="/tmp/test_uploads",
        DATABASE_PATH="/tmp/test_docs.db",
        CHROMA_PERSIST_DIR="/tmp/test_chroma",
        TOKENIZER_MODEL="bert-base-uncased",
    )


def _make_metadata(doc_id: str = "test-doc-001") -> DocumentMetadata:
    return DocumentMetadata(
        document_id=doc_id,
        filename="test.txt",
        file_type=FileType.TXT,
        file_path=f"/tmp/test_uploads/{doc_id}_test.txt",
        status=DocumentStatus.UPLOADED,
    )


def _make_retrieved_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1",
        document_id="doc-001",
        chunk_index=0,
        content="test retrieved content",
        token_count=10,
        filename="test.txt",
        retrieval_score=0.85,
        rerank_score=0.9,
    )


@pytest.fixture
def mock_memory_service() -> MagicMock:
    service = MagicMock()
    service.store_document = AsyncMock(return_value=_make_metadata())
    service.process_document = AsyncMock(
        return_value=IngestionResult(
            document_id="test-doc-001",
            filename="test.txt",
            status=DocumentStatus.PROCESSED,
            chunk_count=3,
            token_count=150,
            char_count=500,
        )
    )
    service.list_documents = AsyncMock(return_value=[_make_metadata()])
    service.get_document = AsyncMock(return_value=_make_metadata())
    service.delete_document = AsyncMock(return_value=None)
    service.retrieve = AsyncMock(return_value=[_make_retrieved_chunk()])
    service.health = AsyncMock(
        return_value={"vector_store": "healthy", "chunk_count": 42}
    )
    return service


@pytest.fixture
def mock_context_builder() -> MagicMock:
    builder = MagicMock()
    builder.build.return_value = ContextPackage(
        query="test query",
        formatted_context="[Source: test.txt | Chunk 0]\ntest content",
        sources=[
            ContextSource(
                chunk_id="c1",
                document_id="doc-001",
                filename="test.txt",
                chunk_index=0,
                retrieval_score=0.85,
                rerank_score=0.9,
            )
        ],
        token_usage=TokenUsage(
            retrieved_context=50,
            total_used=100,
            budget_available=4096,
            budget_remaining=3996,
        ),
        chunk_count=1,
    )
    return builder


@pytest.fixture
def mock_llm_manager() -> MagicMock:
    manager = MagicMock()
    manager.stats = {"is_busy": False, "total_calls": 0}
    return manager


@pytest.fixture
def test_client(
    mock_memory_service: MagicMock,
    mock_context_builder: MagicMock,
    mock_llm_manager: MagicMock,
) -> TestClient:
    """Build a TestClient with all Phase 2 services mocked."""
    from app.dependencies.providers import (
        get_context_builder,
        get_llm_manager,
        get_memory_service,
        settings_provider,
    )
    from app.main import create_app

    settings = _make_test_settings()
    app = create_app(settings=settings)

    # Override dependencies
    app.dependency_overrides[settings_provider] = lambda: settings
    app.dependency_overrides[get_memory_service] = lambda: mock_memory_service
    app.dependency_overrides[get_context_builder] = lambda: mock_context_builder
    app.dependency_overrides[get_llm_manager] = lambda: mock_llm_manager

    return TestClient(app, raise_server_exceptions=True)


# ── Document upload ───────────────────────────────────────────────────────────


class TestDocumentUpload:
    def test_upload_txt_file_succeeds(self, test_client: TestClient) -> None:
        content = b"This is a test document content."
        response = test_client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.txt", io.BytesIO(content), "text/plain")},
        )
        assert response.status_code == 201
        data = response.json()["data"]
        assert data["filename"] == "test.txt"
        assert data["file_type"] == "txt"

    def test_upload_md_file_succeeds(self, test_client: TestClient) -> None:
        content = b"# Title\n\nMarkdown content."
        response = test_client.post(
            "/api/v1/documents/upload",
            files={"file": ("readme.md", io.BytesIO(content), "text/markdown")},
        )
        assert response.status_code == 201
        data = response.json()["data"]
        assert data["file_type"] == "md"

    def test_upload_unsupported_format_returns_422(self, test_client: TestClient) -> None:
        content = b"docx content"
        response = test_client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.docx", io.BytesIO(content), "application/octet-stream")},
        )
        assert response.status_code in (400, 422, 503)

    def test_upload_empty_file_returns_error(self, test_client: TestClient) -> None:
        response = test_client.post(
            "/api/v1/documents/upload",
            files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
        )
        assert response.status_code in (400, 422, 503)


# ── Document ingest ───────────────────────────────────────────────────────────


class TestDocumentIngest:
    def test_ingest_returns_result(self, test_client: TestClient) -> None:
        response = test_client.post(
            "/api/v1/documents/ingest",
            json={"document_id": "test-doc-001"},
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["document_id"] == "test-doc-001"
        assert data["status"] == DocumentStatus.PROCESSED.value

    def test_ingest_missing_document_id_returns_error(
        self, test_client: TestClient
    ) -> None:
        response = test_client.post(
            "/api/v1/documents/ingest",
            json={},
        )
        assert response.status_code in (400, 422, 503)


# ── Document list + get ───────────────────────────────────────────────────────


class TestDocumentList:
    def test_list_documents(self, test_client: TestClient) -> None:
        response = test_client.get("/api/v1/documents")
        assert response.status_code == 200
        data = response.json()["data"]
        assert "documents" in data
        assert "total" in data

    def test_get_document_by_id(self, test_client: TestClient) -> None:
        response = test_client.get("/api/v1/documents/test-doc-001")
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["document_id"] == "test-doc-001"


# ── Document delete ───────────────────────────────────────────────────────────


class TestDocumentDelete:
    def test_delete_document(self, test_client: TestClient) -> None:
        response = test_client.delete("/api/v1/documents/test-doc-001")
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["document_id"] == "test-doc-001"


# ── Retrieval search ──────────────────────────────────────────────────────────


class TestRetrievalSearch:
    def test_search_returns_results(self, test_client: TestClient) -> None:
        response = test_client.post(
            "/api/v1/retrieval/search",
            json={"query": "test query", "top_k": 5},
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["chunk_id"] == "c1"

    def test_search_with_score_threshold(self, test_client: TestClient) -> None:
        response = test_client.post(
            "/api/v1/retrieval/search",
            json={"query": "query", "top_k": 3, "score_threshold": 0.5},
        )
        assert response.status_code == 200

    def test_search_empty_query_returns_error(self, test_client: TestClient) -> None:
        response = test_client.post(
            "/api/v1/retrieval/search",
            json={"query": "", "top_k": 5},
        )
        assert response.status_code == 422


# ── Context building ──────────────────────────────────────────────────────────


class TestContextBuild:
    def test_build_context_returns_package(self, test_client: TestClient) -> None:
        response = test_client.post(
            "/api/v1/retrieval/context",
            json={
                "query": "What is the execution model?",
                "system_prompt": "You are a helpful assistant.",
                "top_k": 5,
            },
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert "formatted_context" in data
        assert "sources" in data
        assert "token_usage" in data


# ── Memory health ─────────────────────────────────────────────────────────────


class TestMemoryHealth:
    def test_memory_health_returns_status(self, test_client: TestClient) -> None:
        response = test_client.get("/api/v1/memory/health")
        assert response.status_code == 200
        data = response.json()["data"]
        assert "memory" in data
        assert "llm_execution_manager" in data
