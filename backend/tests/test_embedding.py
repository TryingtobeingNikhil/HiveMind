"""
tests/test_embedding.py
────────────────────────
Tests for the OllamaEmbeddingProvider.

HTTP client is mocked via unittest.mock.AsyncMock.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.core.exceptions import EmbeddingException, EmbeddingModelNotReadyException
from app.embedding.ollama_embedding import OllamaEmbeddingProvider


def _make_settings(
    base_url: str = "http://localhost:11434",
    model: str = "nomic-embed-text",
) -> MagicMock:
    s = MagicMock()
    s.ollama_api_url = base_url
    s.embedding_model = model
    s.embedding_timeout = 30.0
    return s


_FAKE_EMBEDDING = [0.1, 0.2, 0.3, 0.4, 0.5]


def _make_provider() -> tuple[OllamaEmbeddingProvider, AsyncMock]:
    settings = _make_settings()
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    # Ensure aclose is an async mock
    mock_client.aclose = AsyncMock()
    provider = OllamaEmbeddingProvider(settings, http_client=mock_client)
    return provider, mock_client


class TestOllamaEmbeddingProvider:
    """Tests for OllamaEmbeddingProvider using AsyncMock."""

    async def test_embed_text_returns_vector(self) -> None:
        provider, mock_client = _make_provider()
        req = httpx.Request("POST", "http://localhost/api/embeddings")
        mock_client.post.return_value = httpx.Response(200, json={"embedding": _FAKE_EMBEDDING}, request=req)
        
        result = await provider.embed_text("Hello world")
        
        assert result == _FAKE_EMBEDDING
        mock_client.post.assert_called_once()
        assert "/api/embeddings" in mock_client.post.call_args[0][0]
        await provider.aclose()

    async def test_embed_text_raises_on_empty(self) -> None:
        provider, mock_client = _make_provider()
        
        with pytest.raises(EmbeddingException, match="empty text"):
            await provider.embed_text("   ")
            
        mock_client.post.assert_not_called()
        await provider.aclose()

    async def test_embed_batch_sequential(self) -> None:
        provider, mock_client = _make_provider()
        req = httpx.Request("POST", "http://localhost/api/embeddings")
        mock_client.post.return_value = httpx.Response(200, json={"embedding": _FAKE_EMBEDDING}, request=req)
        
        results = await provider.embed_batch(["text one", "text two", "text three"])
        
        assert len(results) == 3
        for r in results:
            assert r == _FAKE_EMBEDDING
        assert mock_client.post.call_count == 3
        await provider.aclose()

    async def test_embed_batch_empty_input_returns_empty(self) -> None:
        provider, mock_client = _make_provider()
        
        results = await provider.embed_batch([])
        
        assert results == []
        mock_client.post.assert_not_called()
        await provider.aclose()

    async def test_embed_text_raises_on_connection_error(self) -> None:
        provider, mock_client = _make_provider()
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")
        
        with pytest.raises(EmbeddingException):
            await provider.embed_text("test")
            
        await provider.aclose()

    async def test_embed_text_raises_on_404(self) -> None:
        provider, mock_client = _make_provider()
        req = httpx.Request("POST", "http://localhost/api/embeddings")
        mock_client.post.return_value = httpx.Response(404, request=req)
        
        with pytest.raises(EmbeddingException, match="not found"):
            await provider.embed_text("test")
            
        await provider.aclose()

    async def test_embed_text_raises_on_empty_embedding_response(self) -> None:
        provider, mock_client = _make_provider()
        req = httpx.Request("POST", "http://localhost/api/embeddings")
        mock_client.post.return_value = httpx.Response(200, json={"embedding": []}, request=req)
        
        with pytest.raises(EmbeddingException, match="empty embedding"):
            await provider.embed_text("test")
            
        await provider.aclose()

    async def test_health_check_passes(self) -> None:
        provider, mock_client = _make_provider()
        req = httpx.Request("GET", "http://localhost/api/tags")
        mock_client.get.return_value = httpx.Response(
            200,
            json={"models": [{"name": "nomic-embed-text:latest"}]},
            request=req,
        )
        
        result = await provider.health_check()
        
        assert result is True
        mock_client.get.assert_called_once()
        assert "/api/tags" in mock_client.get.call_args[0][0]
        await provider.aclose()

    async def test_health_check_fails_if_model_not_pulled(self) -> None:
        provider, mock_client = _make_provider()
        req = httpx.Request("GET", "http://localhost/api/tags")
        mock_client.get.return_value = httpx.Response(
            200,
            json={"models": [{"name": "llama3.2:3b"}]},
            request=req,
        )
        
        with pytest.raises(EmbeddingModelNotReadyException):
            await provider.health_check()
            
        await provider.aclose()

    async def test_model_info_returns_dict(self) -> None:
        provider, _ = _make_provider()
        info = await provider.model_info()
        assert "model" in info
        assert info["provider"] == "ollama"
        await provider.aclose()
