"""
tests/test_chunking.py
───────────────────────
Tests for the token-aware chunking pipeline.

The tokenizer is mocked to avoid network access and HuggingFace downloads
in CI. All chunk structure and logic is tested deterministically.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.core.exceptions import DocumentParsingException
from app.schemas.documents import DocumentMetadata, DocumentStatus, FileType


def _make_settings(chunk_size: int = 10, overlap: int = 2):
    """Create a minimal Settings mock."""
    s = MagicMock()
    s.chunk_token_size = chunk_size
    s.chunk_overlap_tokens = overlap
    s.tokenizer_model = "bert-base-uncased"
    return s


def _make_metadata(doc_id: str = "doc-001") -> DocumentMetadata:
    return DocumentMetadata(
        document_id=doc_id,
        filename="test.txt",
        file_type=FileType.TXT,
        file_path="/tmp/test.txt",
        status=DocumentStatus.UPLOADED,
    )


def _make_mock_tokenizer(word_count_per_token: int = 1):
    """
    Build a mock tokenizer that treats each space-separated word as one token.
    Encode: split on spaces → list of ints
    Decode: join ints as placeholder words
    """
    def encode(text: str, add_special_tokens: bool = False) -> list[int]:
        words = text.strip().split()
        return list(range(len(words)))  # each word = 1 token ID

    def decode(
        token_ids: list[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        # Return a reconstructed string using generic word placeholders
        # that ensures non-empty output for non-empty input
        if not token_ids:
            return ""
        return " ".join(f"word{i}" for i in token_ids)

    mock = MagicMock()
    mock.encode.side_effect = encode
    mock.decode.side_effect = decode
    mock.vocab_size = 30522
    return mock


class TestTokenAwareChunker:
    """Tests for TokenAwareChunker using a mocked tokenizer."""

    def _make_chunker(self, chunk_size: int = 10, overlap: int = 2):
        from app.chunking.token_chunker import TokenAwareChunker

        settings = _make_settings(chunk_size, overlap)
        chunker = TokenAwareChunker(settings)
        chunker._tokenizer = _make_mock_tokenizer()
        return chunker

    def test_basic_chunking_produces_chunks(self) -> None:
        """Documents longer than chunk_size produce multiple chunks."""
        chunker = self._make_chunker(chunk_size=5, overlap=1)
        metadata = _make_metadata()
        # 15 words → should produce multiple chunks
        content = " ".join(f"word{i}" for i in range(15))
        chunks = chunker.chunk(content, metadata)
        assert len(chunks) > 1

    def test_short_document_produces_one_chunk(self) -> None:
        """Documents shorter than chunk_size produce exactly one chunk."""
        chunker = self._make_chunker(chunk_size=20, overlap=2)
        metadata = _make_metadata()
        content = "hello world foo bar"  # 4 words
        chunks = chunker.chunk(content, metadata)
        assert len(chunks) == 1

    def test_chunk_ids_are_unique(self) -> None:
        chunker = self._make_chunker(chunk_size=5, overlap=1)
        metadata = _make_metadata()
        content = " ".join(f"word{i}" for i in range(20))
        chunks = chunker.chunk(content, metadata)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_ids_contain_document_id(self) -> None:
        chunker = self._make_chunker(chunk_size=5, overlap=1)
        metadata = _make_metadata("my-doc-123")
        content = " ".join(f"word{i}" for i in range(15))
        chunks = chunker.chunk(content, metadata)
        for chunk in chunks:
            assert "my-doc-123" in chunk.chunk_id

    def test_chunk_index_is_sequential(self) -> None:
        chunker = self._make_chunker(chunk_size=5, overlap=1)
        metadata = _make_metadata()
        content = " ".join(f"word{i}" for i in range(20))
        chunks = chunker.chunk(content, metadata)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunk_has_positive_token_count(self) -> None:
        chunker = self._make_chunker(chunk_size=5, overlap=1)
        metadata = _make_metadata()
        content = " ".join(f"word{i}" for i in range(15))
        chunks = chunker.chunk(content, metadata)
        for chunk in chunks:
            assert chunk.token_count > 0

    def test_empty_content_raises(self) -> None:
        chunker = self._make_chunker()
        metadata = _make_metadata()
        with pytest.raises(DocumentParsingException, match="empty document"):
            chunker.chunk("   ", metadata)

    def test_chunk_preserves_document_id(self) -> None:
        chunker = self._make_chunker(chunk_size=10, overlap=2)
        metadata = _make_metadata("special-doc")
        content = " ".join(f"word{i}" for i in range(12))
        chunks = chunker.chunk(content, metadata)
        for chunk in chunks:
            assert chunk.document_id == "special-doc"

    def test_chunk_preserves_filename(self) -> None:
        chunker = self._make_chunker(chunk_size=10, overlap=2)
        metadata = _make_metadata()
        content = " ".join(f"word{i}" for i in range(12))
        chunks = chunker.chunk(content, metadata)
        for chunk in chunks:
            assert chunk.filename == "test.txt"

    def test_count_tokens(self) -> None:
        chunker = self._make_chunker()
        # Mock tokenizer: each space-separated word = 1 token
        count = chunker.count_tokens("hello world foo")
        assert count == 3

    def test_overlap_produces_more_chunks_than_no_overlap(self) -> None:
        chunker_overlap = self._make_chunker(chunk_size=5, overlap=2)
        chunker_no_overlap = self._make_chunker(chunk_size=5, overlap=0)
        metadata = _make_metadata()
        content = " ".join(f"word{i}" for i in range(20))

        chunks_overlap = chunker_overlap.chunk(content, metadata)
        chunks_no_overlap = chunker_no_overlap.chunk(content, metadata)

        assert len(chunks_overlap) >= len(chunks_no_overlap)
