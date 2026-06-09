"""
app/vectorstore/chroma_store.py
─────────────────────────────────
ChromaDB vector store implementation.

ChromaDB runs in-process with SQLite persistence.
No separate server required.

All ChromaDB calls are synchronous; they are wrapped in
asyncio.to_thread() to avoid blocking the event loop.

Collection uses cosine distance for similarity scoring.
retrieval_score = 1 - distance  (0 = worst, 1 = best match)
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.core.exceptions import VectorStoreException, VectorStoreNotReadyException
from app.schemas.documents import DocumentChunk
from app.schemas.retrieval import RetrievedChunk
from app.vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB-backed vector store with cosine similarity.

    Args:
        settings: Application settings (persist_dir, collection_name).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._persist_dir: Path = settings.chroma_persist_path
        self._collection_name: str = settings.chroma_collection_name
        self._client: Any = None
        self._collection: Any = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def create_collection(self) -> None:
        """
        Initialise the ChromaDB client and get-or-create the collection.
        Must be called before any other method.
        """
        try:
            await asyncio.to_thread(self._sync_create_collection)
            logger.info(
                "ChromaDB collection ready",
                extra={
                    "collection": self._collection_name,
                    "persist_dir": str(self._persist_dir),
                },
            )
        except Exception as exc:
            raise VectorStoreNotReadyException(
                f"Failed to initialise ChromaDB: {exc}",
                context={"persist_dir": str(self._persist_dir)},
            ) from exc

    def _sync_create_collection(self) -> None:
        """Synchronous ChromaDB initialisation (runs in thread pool)."""
        import chromadb  # type: ignore[import-untyped]

        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},  # cosine distance for HNSW index
        )

    def _ensure_ready(self) -> None:
        if self._collection is None:
            raise VectorStoreNotReadyException(
                "ChromaVectorStore is not initialised. Call create_collection() first."
            )

    # ── Upsert ────────────────────────────────────────────────────────────────

    async def upsert(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """
        Insert or update chunks in the collection.

        Args:
            chunks:     Chunk objects (IDs, content, metadata).
            embeddings: Embedding vectors (aligned with chunks).
        """
        self._ensure_ready()

        if len(chunks) != len(embeddings):
            raise VectorStoreException(
                f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings."
            )

        if not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [
            {
                "document_id": c.document_id,
                "chunk_index": c.chunk_index,
                "token_count": c.token_count,
                "filename": c.filename,
            }
            for c in chunks
        ]

        logger.info(
            "Upserting chunks to ChromaDB",
            extra={"collection": self._collection_name, "count": len(chunks)},
        )

        try:
            await asyncio.to_thread(
                self._collection.upsert,
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info(
                "Upsert complete",
                extra={"collection": self._collection_name, "count": len(chunks)},
            )
        except Exception as exc:
            raise VectorStoreException(
                f"ChromaDB upsert failed: {exc}",
                context={"collection": self._collection_name},
            ) from exc

    # ── Query ─────────────────────────────────────────────────────────────────

    async def query(
        self,
        query_embedding: list[float],
        n_results: int,
        where: dict[str, Any] | None = None,
        include_embeddings: bool = False,
    ) -> tuple[list[RetrievedChunk], list[list[float]] | None]:
        """
        Return the top-N nearest chunks for a query embedding.
        """
        self._ensure_ready()

        include = ["documents", "metadatas", "distances"]
        if include_embeddings:
            include.append("embeddings")

        logger.debug(
            "Querying ChromaDB",
            extra={"n_results": n_results, "include_embeddings": include_embeddings},
        )

        try:
            results: dict[str, Any] = await asyncio.to_thread(
                self._sync_query,
                query_embedding=query_embedding,
                n_results=n_results,
                where=where,
                include=include,
            )
        except Exception as exc:
            raise VectorStoreException(
                f"ChromaDB query failed: {exc}"
            ) from exc

        retrieved = self._parse_results(results)
        chunk_embeddings: list[list[float]] | None = None

        if include_embeddings and "embeddings" in results:
            raw_embs = results["embeddings"]
            if raw_embs and len(raw_embs) > 0:
                chunk_embeddings = raw_embs[0]  # ChromaDB nests results per query

        logger.debug(
            "Query complete",
            extra={"results": len(retrieved)},
        )

        return retrieved, chunk_embeddings

    def _sync_query(
        self,
        query_embedding: list[float],
        n_results: int,
        where: dict[str, Any] | None,
        include: list[str],
    ) -> dict[str, Any]:
        """Synchronous ChromaDB query (runs in thread pool)."""
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": include,
        }
        if where:
            kwargs["where"] = where
        return self._collection.query(**kwargs)

    @staticmethod
    def _parse_results(results: dict[str, Any]) -> list[RetrievedChunk]:
        """
        Convert raw ChromaDB query results to RetrievedChunk instances.

        ChromaDB returns results nested per-query (outer list = one per query).
        We always issue single queries, so we index [0] into each field.
        Distances are cosine distances [0, 2]; convert to similarity [0, 1].
        """
        ids_list: list[list[str]] = results.get("ids", [[]])
        docs_list: list[list[str]] = results.get("documents", [[]])
        metas_list: list[list[dict]] = results.get("metadatas", [[]])
        dists_list: list[list[float]] = results.get("distances", [[]])

        if not ids_list or not ids_list[0]:
            return []

        ids = ids_list[0]
        docs = docs_list[0] if docs_list else [""] * len(ids)
        metas = metas_list[0] if metas_list else [{}] * len(ids)
        dists = dists_list[0] if dists_list else [1.0] * len(ids)

        chunks: list[RetrievedChunk] = []
        for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists):
            # Cosine distance from ChromaDB ∈ [0, 2] when using cosine space
            # (0 = identical, 2 = opposite). Convert to similarity [0, 1]:
            similarity = max(0.0, min(1.0, 1.0 - dist))
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    document_id=meta.get("document_id", ""),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    content=doc,
                    token_count=int(meta.get("token_count", 0)),
                    filename=meta.get("filename", ""),
                    retrieval_score=similarity,
                    metadata=meta,
                )
            )

        return chunks

    # ── Delete ────────────────────────────────────────────────────────────────

    async def delete(self, document_id: str) -> None:
        """Remove all chunks for the given document."""
        self._ensure_ready()

        logger.info(
            "Deleting chunks from ChromaDB",
            extra={"document_id": document_id, "collection": self._collection_name},
        )

        try:
            await asyncio.to_thread(
                self._collection.delete,
                where={"document_id": document_id},
            )
            logger.info(
                "Chunks deleted", extra={"document_id": document_id}
            )
        except Exception as exc:
            raise VectorStoreException(
                f"ChromaDB delete failed for document '{document_id}': {exc}",
                context={"document_id": document_id},
            ) from exc

    # ── Health check ──────────────────────────────────────────────────────────

    async def health_check(self) -> bool:
        """Return True if the collection is initialised and accessible."""
        if self._collection is None:
            return False
        try:
            await asyncio.to_thread(self._collection.count)
            return True
        except Exception:
            return False

    # ── Count ─────────────────────────────────────────────────────────────────

    async def count(self, document_id: str | None = None) -> int:
        """Return the total chunk count in the collection."""
        self._ensure_ready()

        try:
            if document_id is None:
                return await asyncio.to_thread(self._collection.count)
            else:
                results = await asyncio.to_thread(
                    self._collection.get,
                    where={"document_id": document_id},
                )
                return len(results.get("ids", []))
        except Exception as exc:
            raise VectorStoreException(
                f"ChromaDB count failed: {exc}"
            ) from exc
