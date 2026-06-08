"""
app/ingestion/pdf_ingestor.py
──────────────────────────────
PDF document ingestor using pypdf.

Extracts text from all pages sequentially.
Pages are joined with a newline separator.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.core.exceptions import DocumentParsingException
from app.ingestion.base import BaseIngestor
from app.schemas.documents import Document, DocumentMetadata

logger = logging.getLogger(__name__)


class PDFIngestor(BaseIngestor):
    """Ingestor for PDF files using pypdf."""

    async def ingest(self, file_path: Path, metadata: DocumentMetadata) -> Document:
        """
        Extract text from a PDF file.

        Args:
            file_path: Absolute path to the PDF file.
            metadata:  Pre-populated document metadata.

        Returns:
            A :class:`Document` with extracted text.

        Raises:
            DocumentParsingException: If the file is corrupt or has no extractable text.
        """
        logger.info(
            "Ingesting PDF",
            extra={"document_id": metadata.document_id, "path": str(file_path)},
        )

        try:
            from pypdf import PdfReader  # type: ignore[import-untyped]
        except ImportError as exc:
            raise DocumentParsingException(
                "pypdf is not installed. Add 'pypdf' to requirements.txt."
            ) from exc

        try:
            reader = PdfReader(str(file_path))
        except Exception as exc:
            raise DocumentParsingException(
                f"Cannot open PDF '{metadata.filename}': {exc}",
                context={"document_id": metadata.document_id, "path": str(file_path)},
            ) from exc

        pages: list[str] = []
        for page_num, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
                pages.append(text)
            except Exception as exc:
                logger.warning(
                    "Failed to extract text from PDF page",
                    extra={
                        "document_id": metadata.document_id,
                        "page": page_num,
                        "error": str(exc),
                    },
                )

        raw_text = "\n".join(pages)
        normalised = self._normalise_whitespace(raw_text)

        if not normalised:
            raise DocumentParsingException(
                f"No extractable text found in PDF '{metadata.filename}'. "
                "The file may be scanned or contain only images.",
                context={"document_id": metadata.document_id, "pages": len(reader.pages)},
            )

        logger.info(
            "PDF ingestion complete",
            extra={
                "document_id": metadata.document_id,
                "pages": len(reader.pages),
                "char_count": len(normalised),
            },
        )

        return Document(metadata=metadata, content=normalised)
