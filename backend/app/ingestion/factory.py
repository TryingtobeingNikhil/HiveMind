"""
app/ingestion/factory.py
─────────────────────────
Ingestor factory — selects the correct ingestor for a given file type.

Usage:
    ingestor = get_ingestor_for_type(FileType.PDF)
    document = await ingestor.ingest(file_path, metadata)
"""

from __future__ import annotations

from app.core.exceptions import DocumentParsingException
from app.ingestion.base import BaseIngestor
from app.ingestion.pdf_ingestor import PDFIngestor
from app.ingestion.text_ingestor import TextIngestor
from app.schemas.documents import FileType

_INGESTOR_MAP: dict[FileType, type[BaseIngestor]] = {
    FileType.PDF: PDFIngestor,
    FileType.TXT: TextIngestor,
    FileType.MARKDOWN: TextIngestor,
}


def get_ingestor_for_type(file_type: FileType) -> BaseIngestor:
    """
    Return the appropriate ingestor instance for the given file type.

    Args:
        file_type: The :class:`FileType` enum value.

    Returns:
        An instantiated :class:`BaseIngestor` subclass.

    Raises:
        DocumentParsingException: If no ingestor is registered for the type.
    """
    ingestor_cls = _INGESTOR_MAP.get(file_type)
    if ingestor_cls is None:
        raise DocumentParsingException(
            f"No ingestor registered for file type: {file_type.value}",
            context={"file_type": file_type.value, "supported": list(_INGESTOR_MAP.keys())},
        )
    return ingestor_cls()
