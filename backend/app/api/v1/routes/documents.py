"""
app/api/v1/routes/documents.py
────────────────────────────────
Document management endpoints.

POST /api/v1/documents/upload  — Upload a file (save to disk, record metadata)
POST /api/v1/documents/ingest  — Run ingestion pipeline on an uploaded document
GET  /api/v1/documents         — List all documents
GET  /api/v1/documents/{id}    — Get document metadata by ID
DELETE /api/v1/documents/{id}  — Delete document + chunks
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile, status
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    DocumentNotFoundException,
    DocumentParsingException,
    IngestionException,
    ValidationException,
)
from app.dependencies.providers import MemoryServiceDep, SettingsDep
from app.schemas.common import APIResponse
from app.schemas.documents import (
    DocumentListResponse,
    DocumentMetadata,
    DocumentStatus,
    FileType,
    IngestionResult,
    UploadResponse,
)

router = APIRouter(prefix="/documents", tags=["Documents"])
logger = logging.getLogger(__name__)


# ── Upload ────────────────────────────────────────────────────────────────────


@router.post(
    "/upload",
    response_model=APIResponse[UploadResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document file",
)
async def upload_document(
    file: UploadFile,
    settings: SettingsDep,
    memory: MemoryServiceDep,
) -> APIResponse[UploadResponse]:
    """
    Accept a file upload, save it to disk, and create a metadata record.

    Supported formats: PDF, TXT, MD

    Returns a document_id. Call POST /ingest to start processing.
    """
    if not file.filename:
        raise ValidationException("Uploaded file must have a filename.")

    # Detect and validate file type
    ext = Path(file.filename).suffix.lstrip(".").lower()
    supported = FileType.supported_extensions()
    if ext not in supported:
        raise ValidationException(
            f"Unsupported file type '.{ext}'. Supported: {supported}",
            context={"filename": file.filename, "extension": ext},
        )

    try:
        file_type = FileType.from_extension(ext)
    except ValueError as exc:
        raise ValidationException(str(exc)) from exc

    # Read file content
    try:
        content = await file.read()
    except Exception as exc:
        raise IngestionException(
            f"Failed to read uploaded file: {exc}",
            context={"filename": file.filename},
        ) from exc

    if len(content) == 0:
        raise ValidationException(
            "Uploaded file is empty.",
            context={"filename": file.filename},
        )

    # Generate document ID and save to uploads dir
    document_id = str(uuid.uuid4())
    uploads_dir = settings.uploads_dir_path
    uploads_dir.mkdir(parents=True, exist_ok=True)

    safe_filename = f"{document_id}_{file.filename}"
    file_path = uploads_dir / safe_filename

    try:
        file_path.write_bytes(content)
    except Exception as exc:
        raise IngestionException(
            f"Failed to save uploaded file: {exc}",
            context={"filename": file.filename},
        ) from exc

    # Create metadata record
    metadata = DocumentMetadata(
        document_id=document_id,
        filename=file.filename,
        file_type=file_type,
        file_path=str(file_path),
        status=DocumentStatus.UPLOADED,
    )
    await memory.store_document(metadata)

    logger.info(
        "Document uploaded",
        extra={
            "document_id": document_id,
            "filename": file.filename,
            "size_bytes": len(content),
        },
    )

    return APIResponse(
        data=UploadResponse(
            document_id=document_id,
            filename=file.filename,
            file_type=file_type,
            status=DocumentStatus.UPLOADED,
        )
    )


# ── Ingest ────────────────────────────────────────────────────────────────────


@router.post(
    "/ingest",
    response_model=APIResponse[IngestionResult],
    summary="Run the ingestion pipeline on an uploaded document",
)
async def ingest_document(
    body: dict,
    memory: MemoryServiceDep,
) -> APIResponse[IngestionResult]:
    """
    Trigger the full ingestion pipeline for a previously uploaded document.

    Body: `{"document_id": "<uuid>"}`

    Pipeline: ingest → chunk → embed → upsert to vector store → update status
    """
    document_id: str = body.get("document_id", "").strip()
    if not document_id:
        raise ValidationException(
            "Request body must contain 'document_id'.",
        )

    result = await memory.process_document(document_id)

    logger.info(
        "Document ingested",
        extra={
            "document_id": document_id,
            "chunk_count": result.chunk_count,
            "status": result.status.value,
        },
    )

    return APIResponse(data=result)


# ── List ──────────────────────────────────────────────────────────────────────


@router.get(
    "",
    response_model=APIResponse[DocumentListResponse],
    summary="List all documents",
)
async def list_documents(
    memory: MemoryServiceDep,
    limit: int = 100,
    offset: int = 0,
) -> APIResponse[DocumentListResponse]:
    """Return a paginated list of all document metadata records."""
    documents = await memory.list_documents(limit=limit, offset=offset)
    return APIResponse(
        data=DocumentListResponse(documents=documents, total=len(documents))
    )


# ── Get by ID ─────────────────────────────────────────────────────────────────


@router.get(
    "/{document_id}",
    response_model=APIResponse[DocumentMetadata],
    summary="Get document metadata by ID",
)
async def get_document(
    document_id: str,
    memory: MemoryServiceDep,
) -> APIResponse[DocumentMetadata]:
    """Return metadata for a single document."""
    metadata = await memory.get_document(document_id)
    return APIResponse(data=metadata)


# ── Delete ────────────────────────────────────────────────────────────────────


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a document and its vector store chunks",
)
async def delete_document(
    document_id: str,
    memory: MemoryServiceDep,
) -> APIResponse[dict]:
    """
    Delete a document:
      1. Remove all chunks from the vector store.
      2. Soft-delete the metadata record (status → DELETED).
    """
    await memory.delete_document(document_id)
    return APIResponse(
        data={"document_id": document_id, "status": DocumentStatus.DELETED.value}
    )
