"""
app/schemas/documents.py
────────────────────────
Pydantic schemas for the document ingestion and processing pipeline.

All public-facing types use strict typing and validation.
Internal types (DocumentChunk) are used between services, not directly exposed.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ── Document status enum ──────────────────────────────────────────────────────


class DocumentStatus(str, Enum):
    """
    Lifecycle state of a document in the ingestion pipeline.

    Transitions:
        UPLOADED → PROCESSING → PROCESSED
                             → FAILED
        any → DELETED
    """

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    DELETED = "deleted"


# ── File type enum ────────────────────────────────────────────────────────────


class FileType(str, Enum):
    """Supported document file types."""

    PDF = "pdf"
    TXT = "txt"
    MARKDOWN = "md"

    @classmethod
    def from_extension(cls, ext: str) -> "FileType":
        """Derive FileType from a file extension (without leading dot)."""
        ext = ext.lower().lstrip(".")
        mapping = {"pdf": cls.PDF, "txt": cls.TXT, "md": cls.MARKDOWN}
        if ext not in mapping:
            raise ValueError(f"Unsupported file extension: .{ext}")
        return mapping[ext]

    @classmethod
    def supported_extensions(cls) -> list[str]:
        return ["pdf", "txt", "md"]


# ── Document metadata ─────────────────────────────────────────────────────────


class DocumentMetadata(BaseModel):
    """
    Persisted metadata record for an ingested document.

    Stored in the SQLite documents table.
    Returned by GET /api/v1/documents/{id}.
    """

    document_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique document identifier (UUID)",
    )
    filename: str = Field(..., description="Original filename as uploaded")
    file_type: FileType = Field(..., description="Detected file type")
    file_path: str = Field(..., description="Server-side storage path for the raw file")
    status: DocumentStatus = Field(
        default=DocumentStatus.UPLOADED,
        description="Current lifecycle status",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of initial upload (UTC)",
    )
    processed_at: datetime | None = Field(
        default=None,
        description="Timestamp of completed processing (UTC)",
    )
    char_count: int | None = Field(
        default=None, description="Total character count of extracted text"
    )
    token_count: int | None = Field(
        default=None, description="Total token count using the configured tokenizer"
    )
    chunk_count: int | None = Field(
        default=None, description="Number of chunks produced during ingestion"
    )
    error_message: str | None = Field(
        default=None, description="Error detail if status is FAILED"
    )

    class Config:
        use_enum_values = False  # Keep enum instances in responses


# ── Document (with content) ───────────────────────────────────────────────────


class Document(BaseModel):
    """
    Full document representation including extracted text content.

    Used internally between the ingestion and chunking layers.
    Not serialised to the database (content lives in files + vector store).
    """

    metadata: DocumentMetadata
    content: str = Field(..., description="Full extracted text content")


# ── Document chunk ────────────────────────────────────────────────────────────


class DocumentChunk(BaseModel):
    """
    A single token-aware chunk produced by the chunking pipeline.

    Stored in ChromaDB with embeddings.
    Each chunk carries full source attribution for retrieval.
    """

    chunk_id: str = Field(
        description="Unique chunk identifier: '{document_id}_chunk_{index}'"
    )
    document_id: str = Field(..., description="Parent document identifier")
    chunk_index: int = Field(..., description="Zero-based position in the document", ge=0)
    token_count: int = Field(..., description="Exact token count for this chunk", gt=0)
    content: str = Field(..., description="Decoded text content of this chunk")
    filename: str = Field(default="", description="Source filename (for attribution)")

    @field_validator("chunk_id", mode="before")
    @classmethod
    def default_chunk_id(cls, v: str | None, info: Any) -> str:
        if v:
            return v
        values = info.data if hasattr(info, "data") else {}
        doc_id = values.get("document_id", "unknown")
        idx = values.get("chunk_index", 0)
        return f"{doc_id}_chunk_{idx}"

    class Config:
        populate_by_name = True


# ── Ingestion result ──────────────────────────────────────────────────────────


class IngestionResult(BaseModel):
    """
    Result returned after completing the ingestion pipeline for a document.

    Returned by POST /api/v1/documents/ingest.
    """

    document_id: str
    filename: str
    status: DocumentStatus
    chunk_count: int = Field(default=0)
    token_count: int = Field(default=0)
    char_count: int = Field(default=0)
    error_message: str | None = None


# ── Upload response ───────────────────────────────────────────────────────────


class UploadResponse(BaseModel):
    """Returned by POST /api/v1/documents/upload."""

    document_id: str
    filename: str
    file_type: FileType
    status: DocumentStatus
    message: str = "File uploaded successfully. Call /ingest to process it."


# ── Document list response ────────────────────────────────────────────────────


class DocumentListResponse(BaseModel):
    """Returned by GET /api/v1/documents."""

    documents: list[DocumentMetadata]
    total: int
