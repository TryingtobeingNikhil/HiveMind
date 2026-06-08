"""
app/memory/memory_service.py
─────────────────────────────
MemoryService — the shared memory layer for all future agents.

This service is the ONLY entry point through which agents and the
orchestrator interact with the document memory system.

Pipeline:
    store_document()
      ↓ ingest (extract text)
      ↓ chunk (token-aware)
      ↓ embed (Ollama embeddings)
      ↓ upsert (ChromaDB)
      ↓ persist metadata (SQLite)

    retrieve()
      ↓ embed query
      ↓ vector search (top-20 candidates)
      ↓ rerank (cosine similarity)
      ↓ return top-K RetrievedChunks

Do NOT create a MemoryAgent — that is reserved for Phase 3.
Future agents interact ONLY through this service.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from app.chunking.token_chunker import TokenAwareChunker
from app.core.config import Settings
from app.core.exceptions import (
    DocumentNotFoundException,
    DocumentParsingException,
    IngestionException,
    RetrievalException,
)
from app.db.document_repository import DocumentRepository
from app.db.metrics_repository import MetricsRepository
from app.embedding.base import BaseEmbeddingProvider
from app.ingestion.factory import get_ingestor_for_type
from app.reranking.base import BaseReranker
from app.schemas.documents import (
    DocumentChunk,
    DocumentMetadata,
    DocumentStatus,
    IngestionResult,
)
from app.schemas.retrieval import RetrievalMetrics, RetrievedChunk
from app.vectorstore.base import BaseVectorStore
from app.core.telemetry import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class MemoryService:
    """
    Shared memory layer for Open Deep Research.

    Coordinates document ingestion, embedding, vector storage,
    and retrieval. Future agents call this service directly —
    they never access the vector store, chunker, or embedding
    provider independently.

    Args:
        settings:        Application settings.
        embedding:       Embedding provider instance.
        vector_store:    Vector store instance.
        doc_repo:        Document metadata repository.
        metrics_repo:    Retrieval metrics repository.
        reranker:        Optional reranker instance.
    """

    def __init__(
        self,
        settings: Settings,
        embedding: BaseEmbeddingProvider,
        vector_store: BaseVectorStore,
        doc_repo: DocumentRepository,
        metrics_repo: MetricsRepository,
        reranker: BaseReranker | None = None,
    ) -> None:
        self._settings = settings
        self._embedding = embedding
        self._vector_store = vector_store
        self._doc_repo = doc_repo
        self._metrics_repo = metrics_repo
        self._reranker = reranker
        self._chunker = TokenAwareChunker(settings)

    # ── Ingestion ─────────────────────────────────────────────────────────────

    async def store_document(self, metadata: DocumentMetadata) -> DocumentMetadata:
        """
        Persist a new document metadata record (status=UPLOADED).

        Called immediately after a file is saved to disk.
        Ingestion is triggered separately by process_document().

        Args:
            metadata: Pre-populated document metadata.

        Returns:
            The persisted metadata record.
        """
        return await self._doc_repo.create(metadata)

    async def process_document(self, document_id: str) -> IngestionResult:
        """
        Run the full ingestion pipeline for an uploaded document.

        Steps:
            1. Load metadata from repository
            2. Set status → PROCESSING
            3. Ingest file (extract text)
            4. Chunk text (token-aware)
            5. Embed chunks (sequential)
            6. Upsert to vector store
            7. Update metadata → PROCESSED

        Args:
            document_id: ID of a previously uploaded document.

        Returns:
            An :class:`IngestionResult` with counts and status.
        """
        logger.info(
            "[MEMORY] Starting document processing",
            extra={"document_id": document_id},
        )

        # 1. Load metadata
        try:
            metadata = await self._doc_repo.get(document_id)
        except DocumentNotFoundException:
            raise

        if metadata.status == DocumentStatus.DELETED:
            raise IngestionException(
                f"Cannot process deleted document '{document_id}'.",
                context={"document_id": document_id},
            )

        # 2. Mark as PROCESSING
        await self._doc_repo.update_status(document_id, DocumentStatus.PROCESSING)

        try:
            # 3. Ingest (extract text)
            ingestor = get_ingestor_for_type(metadata.file_type)
            file_path = Path(metadata.file_path)
            document = await ingestor.ingest(file_path, metadata)

            logger.info(
                "[MEMORY] Document ingested",
                extra={
                    "document_id": document_id,
                    "char_count": len(document.content),
                },
            )

            # 4. Chunk
            chunks: list[DocumentChunk] = self._chunker.chunk(
                document.content, metadata
            )

            logger.info(
                "[MEMORY] Document chunked",
                extra={
                    "document_id": document_id,
                    "chunk_count": len(chunks),
                },
            )

            # 5. Embed (sequential loop)
            chunk_texts = [c.content for c in chunks]
            embeddings = await self._embedding.embed_batch(chunk_texts)

            # Filter out chunks with empty embeddings
            valid_pairs = [
                (chunk, emb)
                for chunk, emb in zip(chunks, embeddings)
                if emb
            ]
            if not valid_pairs:
                raise IngestionException(
                    "All chunk embeddings were empty — embedding provider may be misconfigured.",
                    context={"document_id": document_id},
                )

            valid_chunks, valid_embeddings = zip(*valid_pairs)
            valid_chunks_list = list(valid_chunks)
            valid_embeddings_list = list(valid_embeddings)

            logger.info(
                "[MEMORY] Chunks embedded",
                extra={
                    "document_id": document_id,
                    "embedded_count": len(valid_chunks_list),
                    "dimensions": len(valid_embeddings_list[0]) if valid_embeddings_list else 0,
                },
            )

            # 6. Upsert to vector store
            await self._vector_store.upsert(valid_chunks_list, valid_embeddings_list)

            logger.info(
                "[MEMORY] Chunks stored in vector store",
                extra={"document_id": document_id},
            )

            # Count tokens for the full document
            token_count = sum(c.token_count for c in chunks)
            char_count = len(document.content)

            # 7. Update status → PROCESSED
            await self._doc_repo.update_status(
                document_id,
                DocumentStatus.PROCESSED,
                char_count=char_count,
                token_count=token_count,
                chunk_count=len(valid_chunks_list),
                processed_at=datetime.now(timezone.utc),
            )

            logger.info(
                "[MEMORY] Document processing complete",
                extra={
                    "document_id": document_id,
                    "chunk_count": len(valid_chunks_list),
                    "token_count": token_count,
                },
            )

            return IngestionResult(
                document_id=document_id,
                filename=metadata.filename,
                status=DocumentStatus.PROCESSED,
                chunk_count=len(valid_chunks_list),
                token_count=token_count,
                char_count=char_count,
            )

        except (DocumentParsingException, IngestionException):
            # Record failure in metadata
            import traceback
            error_msg = traceback.format_exc()[-500:]  # truncate to 500 chars
            await self._doc_repo.update_status(
                document_id,
                DocumentStatus.FAILED,
                error_message=str(error_msg),
            )
            raise
        except Exception as exc:
            await self._doc_repo.update_status(
                document_id,
                DocumentStatus.FAILED,
                error_message=str(exc)[:500],
            )
            raise IngestionException(
                f"Unexpected error processing document '{document_id}': {exc}",
                context={"document_id": document_id},
            ) from exc

    # ── Retrieval ─────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float = 0.0,
        metadata_filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the most relevant chunks for a query.

        Pipeline:
            embed query → vector search (top-20) → rerank → top-K

        Args:
            query:            The search query string.
            top_k:            Number of final results (default from settings).
            score_threshold:  Minimum similarity score to include a result.
            metadata_filters: Optional ChromaDB where-clause filters.

        Returns:
            Ordered list of :class:`RetrievedChunk` (highest score first).
        """
        effective_top_k = top_k or self._settings.retrieval_top_k
        candidate_k = self._settings.retrieval_candidate_k

        logger.info(
            "[MEMORY] Retrieval started",
            extra={
                "query_len": len(query),
                "top_k": effective_top_k,
                "candidate_k": candidate_k,
            },
        )

        start_time = time.monotonic()

        with tracer.start_as_current_span("memory.retrieve") as span:
            span.set_attribute("query", query[:100])
            span.set_attribute("top_k", effective_top_k)
            try:
                # 1. Embed query
                query_embedding = await self._embedding.embed_text(query)

                # 2. Vector search — retrieve more candidates than needed for reranking
                include_embeddings = (
                    self._reranker is not None and self._settings.reranker_enabled
                )
                candidates, chunk_embeddings = await self._vector_store.query(
                    query_embedding=query_embedding,
                    n_results=candidate_k,
                    where=metadata_filters,
                    include_embeddings=include_embeddings,
                )

                # 3. Apply score threshold
                if score_threshold > 0.0:
                    candidates = [
                        c for c in candidates if c.retrieval_score >= score_threshold
                    ]

                # 4. Rerank if enabled
                if (
                    self._reranker is not None
                    and self._settings.reranker_enabled
                    and candidates
                ):
                    logger.info(
                        "[MEMORY] Reranking",
                        extra={"candidates": len(candidates)},
                    )
                    candidates = await self._reranker.rerank(
                        query=query,
                        chunks=candidates,
                        query_embedding=query_embedding,
                        chunk_embeddings=chunk_embeddings,
                    )

                # 5. Trim to top_k
                results = candidates[:effective_top_k]

            except Exception as exc:
                raise RetrievalException(
                    f"Retrieval failed for query: {exc}",
                    context={"query": query[:100]},
                ) from exc

        elapsed_ms = (time.monotonic() - start_time) * 1000

        logger.info(
            "[MEMORY] Retrieval complete",
            extra={
                "results": len(results),
                "latency_ms": round(elapsed_ms, 2),
            },
        )

        # Persist metrics asynchronously (best-effort)
        try:
            metrics = RetrievalMetrics(
                query=query,
                retrieved_chunk_ids=[c.chunk_id for c in results],
                retrieval_scores=[c.retrieval_score for c in results],
                rerank_scores=(
                    [c.rerank_score for c in results if c.rerank_score is not None]
                    or None
                ),
                retrieval_latency_ms=round(elapsed_ms, 2),
                total_results=len(results),
            )
            await self._metrics_repo.record(metrics)
        except Exception as exc:
            logger.warning(
                "[MEMORY] Failed to record retrieval metrics",
                extra={"error": str(exc)},
            )

        return results

    # ── Document management ───────────────────────────────────────────────────

    async def get_document(self, document_id: str) -> DocumentMetadata:
        """Return document metadata by ID."""
        return await self._doc_repo.get(document_id)

    async def list_documents(
        self,
        status: DocumentStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[DocumentMetadata]:
        """Return a paginated list of document metadata records."""
        return await self._doc_repo.list_all(status=status, limit=limit, offset=offset)

    async def delete_document(self, document_id: str) -> None:
        """
        Delete a document: remove chunks from vector store + soft-delete metadata.

        Args:
            document_id: Document to delete.
        """
        logger.info(
            "[MEMORY] Deleting document",
            extra={"document_id": document_id},
        )

        # Verify document exists
        await self._doc_repo.get(document_id)

        # Remove from vector store
        try:
            await self._vector_store.delete(document_id)
        except Exception as exc:
            logger.warning(
                "[MEMORY] Failed to delete from vector store",
                extra={"document_id": document_id, "error": str(exc)},
            )

        # Soft delete in DB
        await self._doc_repo.soft_delete(document_id)

        logger.info(
            "[MEMORY] Document deleted",
            extra={"document_id": document_id},
        )

    async def health(self) -> dict:
        """Return health status for all memory subsystems."""
        vector_ok = await self._vector_store.health_check()
        try:
            chunk_count = await self._vector_store.count()
        except Exception:
            chunk_count = -1

        return {
            "vector_store": "healthy" if vector_ok else "unhealthy",
            "chunk_count": chunk_count,
            "embedding_model": self._settings.embedding_model,
            "reranker_enabled": self._settings.reranker_enabled,
            "reranker": type(self._reranker).__name__ if self._reranker else None,
        }
