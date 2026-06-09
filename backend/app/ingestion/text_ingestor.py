"""
app/ingestion/text_ingestor.py
───────────────────────────────
Plain text and Markdown document ingestor.

Reads the file as UTF-8 text. No special parsing is applied to Markdown —
the raw Markdown syntax is preserved so the LLM can interpret it.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.core.exceptions import DocumentParsingException
from app.ingestion.base import BaseIngestor
from app.schemas.documents import Document, DocumentMetadata

logger = logging.getLogger(__name__)


class TextIngestor(BaseIngestor):
    """Ingestor for .txt and .md files."""

    async def ingest(self, file_path: Path, metadata: DocumentMetadata) -> Document:
        """
        Read and normalise a plain text or Markdown file.

        Args:
            file_path: Absolute path to the text file.
            metadata:  Pre-populated document metadata.

        Returns:
            A :class:`Document` with the file content.

        Raises:
            DocumentParsingException: If the file cannot be read or is empty.
        """
        logger.info(
            "Ingesting text file",
            extra={
                "document_id": metadata.document_id,
                "path": str(file_path),
                "file_type": metadata.file_type.value,
            },
        )

        try:
            raw_text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Fallback: try latin-1 for legacy files
            try:
                raw_text = file_path.read_text(encoding="latin-1")
                logger.warning(
                    "File read with latin-1 fallback encoding",
                    extra={"document_id": metadata.document_id},
                )
            except Exception as exc:
                raise DocumentParsingException(
                    f"Cannot decode '{metadata.filename}': {exc}",
                    context={"document_id": metadata.document_id},
                ) from exc
        except FileNotFoundError as exc:
            raise DocumentParsingException(
                f"File not found: '{file_path}'",
                context={"document_id": metadata.document_id},
            ) from exc
        except Exception as exc:
            raise DocumentParsingException(
                f"Cannot read '{metadata.filename}': {exc}",
                context={"document_id": metadata.document_id},
            ) from exc

        normalised = self._normalise_whitespace(raw_text)

        if not normalised:
            raise DocumentParsingException(
                f"File '{metadata.filename}' is empty or contains only whitespace.",
                context={"document_id": metadata.document_id},
            )

        logger.info(
            "Text ingestion complete",
            extra={
                "document_id": metadata.document_id,
                "char_count": len(normalised),
            },
        )

        return Document(metadata=metadata, content=normalised)
