"""
app/db/document_repository.py
──────────────────────────────
Repository for document metadata persistence.

All database access is isolated here — services interact through this
repository interface, never with aiosqlite directly.

Uses the repository pattern so the storage backend can be swapped
without changing any service-layer code.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import aiosqlite

from app.core.exceptions import DatabaseException, DocumentNotFoundException
from app.schemas.documents import DocumentMetadata, DocumentStatus, FileType

logger = logging.getLogger(__name__)


class DocumentRepository:
    """
    Async repository for :class:`DocumentMetadata` records.

    All methods raise :class:`DatabaseException` on unexpected database errors.
    """

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._db = conn

    # ── Create ────────────────────────────────────────────────────────────────

    async def create(self, metadata: DocumentMetadata) -> DocumentMetadata:
        """
        Persist a new document record.

        Args:
            metadata: The document metadata to insert.

        Returns:
            The same metadata object (for chaining convenience).
        """
        try:
            await self._db.execute(
                """
                INSERT INTO documents
                    (document_id, filename, file_type, file_path, status,
                     char_count, token_count, chunk_count, created_at,
                     processed_at, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata.document_id,
                    metadata.filename,
                    metadata.file_type.value,
                    metadata.file_path,
                    metadata.status.value,
                    metadata.char_count,
                    metadata.token_count,
                    metadata.chunk_count,
                    metadata.created_at.isoformat(),
                    metadata.processed_at.isoformat() if metadata.processed_at else None,
                    metadata.error_message,
                ),
            )
            await self._db.commit()
            logger.debug(
                "Document record created",
                extra={"document_id": metadata.document_id},
            )
            return metadata
        except Exception as exc:
            raise DatabaseException(
                f"Failed to create document record: {exc}",
                context={"document_id": metadata.document_id},
            ) from exc

    # ── Read ──────────────────────────────────────────────────────────────────

    async def get(self, document_id: str) -> DocumentMetadata:
        """
        Retrieve a document by its ID.

        Raises:
            DocumentNotFoundException: If no record exists with that ID.
        """
        try:
            cursor = await self._db.execute(
                "SELECT * FROM documents WHERE document_id = ?",
                (document_id,),
            )
            row = await cursor.fetchone()
        except Exception as exc:
            raise DatabaseException(
                f"Failed to query document: {exc}",
                context={"document_id": document_id},
            ) from exc

        if row is None:
            raise DocumentNotFoundException(
                f"Document '{document_id}' not found.",
                context={"document_id": document_id},
            )

        return self._row_to_metadata(dict(row))

    async def list_all(
        self,
        status: DocumentStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[DocumentMetadata]:
        """
        Return a paginated list of document records.

        Args:
            status: Optional filter by DocumentStatus.
            limit:  Maximum records to return.
            offset: Number of records to skip.
        """
        try:
            if status is not None:
                cursor = await self._db.execute(
                    "SELECT * FROM documents WHERE status = ? LIMIT ? OFFSET ?",
                    (status.value, limit, offset),
                )
            else:
                cursor = await self._db.execute(
                    "SELECT * FROM documents LIMIT ? OFFSET ?",
                    (limit, offset),
                )
            rows = await cursor.fetchall()
        except Exception as exc:
            raise DatabaseException(f"Failed to list documents: {exc}") from exc

        return [self._row_to_metadata(dict(r)) for r in rows]

    # ── Update ────────────────────────────────────────────────────────────────

    async def update_status(
        self,
        document_id: str,
        status: DocumentStatus,
        *,
        error_message: str | None = None,
        char_count: int | None = None,
        token_count: int | None = None,
        chunk_count: int | None = None,
        processed_at: datetime | None = None,
    ) -> None:
        """Update mutable fields on an existing document record."""
        try:
            now = processed_at or (
                datetime.now(timezone.utc)
                if status in (DocumentStatus.PROCESSED, DocumentStatus.FAILED)
                else None
            )
            await self._db.execute(
                """
                UPDATE documents
                SET status        = ?,
                    error_message = ?,
                    char_count    = COALESCE(?, char_count),
                    token_count   = COALESCE(?, token_count),
                    chunk_count   = COALESCE(?, chunk_count),
                    processed_at  = COALESCE(?, processed_at)
                WHERE document_id = ?
                """,
                (
                    status.value,
                    error_message,
                    char_count,
                    token_count,
                    chunk_count,
                    now.isoformat() if now else None,
                    document_id,
                ),
            )
            await self._db.commit()
            logger.debug(
                "Document status updated",
                extra={"document_id": document_id, "status": status.value},
            )
        except Exception as exc:
            raise DatabaseException(
                f"Failed to update document status: {exc}",
                context={"document_id": document_id},
            ) from exc

    # ── Delete ────────────────────────────────────────────────────────────────

    async def soft_delete(self, document_id: str) -> None:
        """Mark a document as DELETED (soft delete — record is retained)."""
        await self.update_status(document_id, DocumentStatus.DELETED)

    async def hard_delete(self, document_id: str) -> None:
        """Permanently remove a document record from the database."""
        try:
            await self._db.execute(
                "DELETE FROM documents WHERE document_id = ?", (document_id,)
            )
            await self._db.commit()
            logger.debug(
                "Document record deleted", extra={"document_id": document_id}
            )
        except Exception as exc:
            raise DatabaseException(
                f"Failed to delete document: {exc}",
                context={"document_id": document_id},
            ) from exc

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _row_to_metadata(row: dict[str, Any]) -> DocumentMetadata:
        """Convert a raw database row dict to a DocumentMetadata instance."""
        processed_at = None
        if row.get("processed_at"):
            processed_at = datetime.fromisoformat(row["processed_at"])

        return DocumentMetadata(
            document_id=row["document_id"],
            filename=row["filename"],
            file_type=FileType(row["file_type"]),
            file_path=row["file_path"],
            status=DocumentStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            processed_at=processed_at,
            char_count=row.get("char_count"),
            token_count=row.get("token_count"),
            chunk_count=row.get("chunk_count"),
            error_message=row.get("error_message"),
        )
