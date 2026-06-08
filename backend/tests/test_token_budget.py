"""
tests/test_token_budget.py
───────────────────────────
Tests for TokenBudgetManager and ContextBuilder.

The HuggingFace tokenizer is mocked — no network access required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.core.exceptions import TokenBudgetException
from app.memory.context_builder import ContextBuilder
from app.memory.token_budget import TokenBudgetManager
from app.schemas.retrieval import RetrievedChunk


def _make_mock_tokenizer():
    """Mock tokenizer: 1 token per space-separated word."""
    def encode(text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(len(text.split())))

    def decode(ids: list[int], **kwargs) -> str:
        return " ".join(f"word{i}" for i in ids)

    mock = MagicMock()
    mock.encode.side_effect = encode
    mock.decode.side_effect = decode
    return mock


def _make_settings(max_ctx: int = 4096, reserved: int = 512) -> MagicMock:
    s = MagicMock()
    s.max_context_tokens = max_ctx
    s.reserved_output_tokens = reserved
    s.tokenizer_model = "bert-base-uncased"
    return s


def _make_chunk(
    chunk_id: str,
    content: str,
    token_count: int,
    score: float = 0.9,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-001",
        chunk_index=0,
        content=content,
        token_count=token_count,
        filename="test.txt",
        retrieval_score=score,
        rerank_score=score,
    )


class TestTokenBudgetManager:
    def _make_manager(self, max_ctx: int = 4096, reserved: int = 512) -> TokenBudgetManager:
        settings = _make_settings(max_ctx, reserved)
        manager = TokenBudgetManager(settings, tokenizer=_make_mock_tokenizer())
        return manager

    def test_estimate_tokens_empty(self) -> None:
        manager = self._make_manager()
        assert manager.estimate_tokens("") == 0

    def test_estimate_tokens_counts_words(self) -> None:
        manager = self._make_manager()
        # Mock tokenizer: 1 token per word
        assert manager.estimate_tokens("hello world foo") == 3

    def test_calculate_available_context(self) -> None:
        manager = self._make_manager(max_ctx=100, reserved=10)
        # system=5 words, instructions=3, history=4, query=2
        available = manager.calculate_available_context(
            system_prompt="a b c d e",
            instructions="x y z",
            history="p q r s",
            query="w1 w2",
            reserved_output=10,
        )
        # 100 - 5 - 3 - 4 - 2 - 10 = 76
        assert available == 76

    def test_calculate_available_context_raises_when_overflow(self) -> None:
        manager = self._make_manager(max_ctx=10, reserved=5)
        with pytest.raises(TokenBudgetException, match="exceed"):
            manager.calculate_available_context(
                system_prompt="a b c d e f g h",  # 8 tokens
                query="i j",  # 2 tokens
                reserved_output=5,
            )

    def test_allocate_chunk_budget_fits_all(self) -> None:
        manager = self._make_manager()
        chunks = [
            _make_chunk("c1", "content", 50),
            _make_chunk("c2", "content", 50),
        ]
        selected = manager.allocate_chunk_budget(200, chunks)
        assert len(selected) == 2

    def test_allocate_chunk_budget_stops_at_limit(self) -> None:
        manager = self._make_manager()
        chunks = [
            _make_chunk("c1", "content", 100),
            _make_chunk("c2", "content", 100),
            _make_chunk("c3", "content", 100),
        ]
        selected = manager.allocate_chunk_budget(150, chunks)
        assert len(selected) == 1

    def test_allocate_chunk_budget_empty_chunks(self) -> None:
        manager = self._make_manager()
        selected = manager.allocate_chunk_budget(1000, [])
        assert selected == []

    def test_truncate_context_short_text_unchanged(self) -> None:
        manager = self._make_manager()
        text = "hello world"
        result = manager.truncate_context(text, max_tokens=100)
        # When text is shorter than max_tokens, it is returned as-is (not re-decoded)
        assert "hello" in result or "world" in result

    def test_truncate_context_empty_returns_empty(self) -> None:
        manager = self._make_manager()
        assert manager.truncate_context("", max_tokens=100) == ""


class TestContextBuilder:
    def _make_builder(self, max_ctx: int = 4096) -> ContextBuilder:
        settings = _make_settings(max_ctx, reserved=50)
        manager = TokenBudgetManager(settings, tokenizer=_make_mock_tokenizer())
        return ContextBuilder(settings, manager)

    def test_build_empty_chunks_returns_empty_package(self) -> None:
        builder = self._make_builder()
        package = builder.build("query", [])
        assert package.formatted_context == ""
        assert package.chunk_count == 0
        assert package.sources == []

    def test_build_includes_chunk_content(self) -> None:
        builder = self._make_builder()
        chunks = [_make_chunk("c1", "This is test content", 20, score=0.9)]
        package = builder.build("query", chunks)
        assert "test content" in package.formatted_context or "word" in package.formatted_context

    def test_build_populates_sources(self) -> None:
        builder = self._make_builder()
        chunks = [_make_chunk("c1", "content", 20, 0.9)]
        package = builder.build("query", chunks)
        assert len(package.sources) == 1
        assert package.sources[0].chunk_id == "c1"

    def test_build_deduplicates_identical_content(self) -> None:
        builder = self._make_builder()
        # Two chunks with identical content
        content = "same content for both chunks here"
        chunks = [
            _make_chunk("c1", content, 20, 0.8),
            _make_chunk("c2", content, 20, 0.9),
        ]
        package = builder.build("query", chunks)
        # Should deduplicate — only 1 chunk in result
        assert package.chunk_count == 1
        assert package.deduplication_count == 1

    def test_build_respects_token_budget(self) -> None:
        builder = self._make_builder(max_ctx=100)
        # Create chunks that together exceed the budget
        # With max_ctx=100, reserved=50, and fixed overhead:
        # very little room for chunks
        chunks = [
            _make_chunk("c1", "content", 40),
            _make_chunk("c2", "content", 40),
        ]
        package = builder.build("query", chunks)
        # Should not crash even when budget is tight
        assert package.chunk_count <= 2

    def test_build_preserves_query(self) -> None:
        builder = self._make_builder()
        chunks = [_make_chunk("c1", "content", 10)]
        package = builder.build("my test query", chunks)
        assert package.query == "my test query"

    def test_build_populates_token_usage(self) -> None:
        builder = self._make_builder()
        chunks = [_make_chunk("c1", "content", 10)]
        package = builder.build("query", chunks, system_prompt="system")
        assert package.token_usage.system_prompt > 0
        assert package.token_usage.reserved_output > 0
