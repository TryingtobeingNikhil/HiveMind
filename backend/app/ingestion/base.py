"""
app/ingestion/base.py
─────────────────────
Abstract base ingestor contract.

All file-type-specific ingestors implement this interface.
The factory selects the correct ingestor at runtime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from app.schemas.documents import Document, DocumentMetadata


class BaseIngestor(ABC):
    """
    Abstract ingestor for a specific document file type.

    Subclasses implement ``ingest()`` to:
      1. Load raw bytes from disk.
      2. Extract text content.
      3. Normalise whitespace.
      4. Return a typed :class:`Document`.
    """

    @abstractmethod
    async def ingest(self, file_path: Path, metadata: DocumentMetadata) -> Document:
        """
        Load and extract text from the file at ``file_path``.

        Args:
            file_path: Absolute path to the stored file.
            metadata:  Pre-populated metadata record for this document.

        Returns:
            A :class:`Document` containing the metadata and extracted text.

        Raises:
            DocumentParsingException: If the file cannot be parsed.
        """
        ...  # pragma: no cover

    @staticmethod
    def _normalise_whitespace(text: str) -> str:
        """
        Collapse excessive blank lines and strip leading/trailing whitespace.

        Preserves paragraph structure (single blank line between paragraphs)
        without reducing the text to a single line.
        """
        import re

        # Collapse 3+ consecutive newlines to 2 (one blank line)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Strip leading/trailing whitespace per line (preserves indentation structure)
        lines = [line.rstrip() for line in text.splitlines()]
        return "\n".join(lines).strip()
