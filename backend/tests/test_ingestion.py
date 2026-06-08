"""
tests/test_ingestion.py
────────────────────────
Tests for the document ingestion layer.

All file I/O is mocked — no real files required.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.core.exceptions import DocumentParsingException
from app.ingestion.base import BaseIngestor
from app.ingestion.factory import get_ingestor_for_type
from app.ingestion.pdf_ingestor import PDFIngestor
from app.ingestion.text_ingestor import TextIngestor
from app.schemas.documents import DocumentMetadata, DocumentStatus, FileType


def _make_metadata(
    filename: str = "test.txt",
    file_type: FileType = FileType.TXT,
) -> DocumentMetadata:
    return DocumentMetadata(
        document_id="doc-test-001",
        filename=filename,
        file_type=file_type,
        file_path="/tmp/test.txt",
        status=DocumentStatus.UPLOADED,
    )


# ── Factory tests ─────────────────────────────────────────────────────────────


class TestIngestorFactory:
    def test_returns_pdf_ingestor_for_pdf(self) -> None:
        ingestor = get_ingestor_for_type(FileType.PDF)
        assert isinstance(ingestor, PDFIngestor)

    def test_returns_text_ingestor_for_txt(self) -> None:
        ingestor = get_ingestor_for_type(FileType.TXT)
        assert isinstance(ingestor, TextIngestor)

    def test_returns_text_ingestor_for_md(self) -> None:
        ingestor = get_ingestor_for_type(FileType.MARKDOWN)
        assert isinstance(ingestor, TextIngestor)


# ── TextIngestor tests ────────────────────────────────────────────────────────


class TestTextIngestor:
    def test_ingest_plain_text(self) -> None:
        ingestor = TextIngestor()
        metadata = _make_metadata()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Hello world\n\nThis is a test document.")
            tmp_path = Path(f.name)

        try:
            import asyncio
            doc = asyncio.get_event_loop().run_until_complete(
                ingestor.ingest(tmp_path, metadata)
            )
            assert "Hello world" in doc.content
            assert "test document" in doc.content
            assert doc.metadata.document_id == "doc-test-001"
        finally:
            tmp_path.unlink()

    def test_ingest_markdown(self) -> None:
        ingestor = TextIngestor()
        metadata = _make_metadata("test.md", FileType.MARKDOWN)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write("# Title\n\nParagraph with **bold** text.")
            tmp_path = Path(f.name)

        try:
            import asyncio
            doc = asyncio.get_event_loop().run_until_complete(
                ingestor.ingest(tmp_path, metadata)
            )
            assert "# Title" in doc.content
        finally:
            tmp_path.unlink()

    def test_ingest_empty_file_raises(self) -> None:
        ingestor = TextIngestor()
        metadata = _make_metadata()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("   ")  # whitespace only
            tmp_path = Path(f.name)

        try:
            import asyncio
            with pytest.raises(DocumentParsingException, match="empty or contains only whitespace"):
                asyncio.get_event_loop().run_until_complete(
                    ingestor.ingest(tmp_path, metadata)
                )
        finally:
            tmp_path.unlink()

    def test_ingest_missing_file_raises(self) -> None:
        ingestor = TextIngestor()
        metadata = _make_metadata()

        import asyncio
        with pytest.raises(DocumentParsingException):
            asyncio.get_event_loop().run_until_complete(
                ingestor.ingest(Path("/nonexistent/path.txt"), metadata)
            )

    def test_normalise_excessive_whitespace(self) -> None:
        ingestor = TextIngestor()
        metadata = _make_metadata()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Line one\n\n\n\n\nLine two")
            tmp_path = Path(f.name)

        try:
            import asyncio
            doc = asyncio.get_event_loop().run_until_complete(
                ingestor.ingest(tmp_path, metadata)
            )
            # Should collapse 5 blank lines to 2
            assert "\n\n\n" not in doc.content
        finally:
            tmp_path.unlink()


# ── FileType enum tests ───────────────────────────────────────────────────────


class TestFileType:
    def test_from_extension_pdf(self) -> None:
        assert FileType.from_extension("pdf") == FileType.PDF

    def test_from_extension_txt(self) -> None:
        assert FileType.from_extension("txt") == FileType.TXT

    def test_from_extension_md(self) -> None:
        assert FileType.from_extension("md") == FileType.MARKDOWN

    def test_from_extension_case_insensitive(self) -> None:
        assert FileType.from_extension("PDF") == FileType.PDF

    def test_from_extension_with_dot(self) -> None:
        assert FileType.from_extension(".txt") == FileType.TXT

    def test_from_extension_unsupported_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            FileType.from_extension("docx")

    def test_supported_extensions(self) -> None:
        exts = FileType.supported_extensions()
        assert "pdf" in exts
        assert "txt" in exts
        assert "md" in exts


# ── BaseIngestor whitespace normalisation ─────────────────────────────────────


class TestBaseIngestorNormalisation:
    def test_collapse_multiple_newlines(self) -> None:
        class _ConcreteIngestor(BaseIngestor):
            async def ingest(self, file_path, metadata):  # type: ignore
                ...

        result = _ConcreteIngestor._normalise_whitespace("a\n\n\n\nb")
        assert result == "a\n\nb"

    def test_strip_trailing_whitespace(self) -> None:
        class _ConcreteIngestor(BaseIngestor):
            async def ingest(self, file_path, metadata):  # type: ignore
                ...

        result = _ConcreteIngestor._normalise_whitespace("  hello  \n  world  ")
        # Line-level rstrip should remove trailing spaces
        assert result == "hello\n  world"  # leading spaces preserved
