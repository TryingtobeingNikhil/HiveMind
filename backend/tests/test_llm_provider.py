"""
tests/test_llm_provider.py
───────────────────────────
Tests for BaseLLMProvider / OllamaProvider.

OllamaClient is mocked — no live Ollama connection required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.exceptions import ProviderException, ProviderNotReadyException
from app.llm.ollama_provider import OllamaProvider


def _make_mock_client(
    generate_return: str = "LLM response",
    health_return: bool = True,
) -> MagicMock:
    client = MagicMock()
    client.generate = AsyncMock(return_value=generate_return)
    client.health_check = AsyncMock(return_value=health_return)
    client.list_models = AsyncMock(return_value=[{"name": "llama3.2:3b"}])
    client._default_model = "llama3.2:3b"
    client._base_url = "http://localhost:11434"
    return client


class TestOllamaProvider:
    async def test_generate_returns_string(self) -> None:
        client = _make_mock_client(generate_return="Hello from Ollama")
        provider = OllamaProvider(client)
        result = await provider.generate("What is AI?")
        assert result == "Hello from Ollama"

    async def test_generate_passes_prompt_to_client(self) -> None:
        client = _make_mock_client()
        provider = OllamaProvider(client)
        await provider.generate("test prompt", system="be helpful")
        client.generate.assert_called_once()
        call_kwargs = client.generate.call_args[1]
        assert call_kwargs["prompt"] == "test prompt"
        assert call_kwargs["system"] == "be helpful"

    async def test_generate_wraps_exception_as_provider_exception(self) -> None:
        client = _make_mock_client()
        from app.core.exceptions import OllamaGenerationError
        client.generate = AsyncMock(side_effect=OllamaGenerationError("fail"))
        provider = OllamaProvider(client)

        with pytest.raises(ProviderException):
            await provider.generate("prompt")

    async def test_health_check_returns_true(self) -> None:
        client = _make_mock_client(health_return=True)
        provider = OllamaProvider(client)
        result = await provider.health_check()
        assert result is True

    async def test_health_check_raises_provider_exception_on_failure(self) -> None:
        client = _make_mock_client()
        from app.core.exceptions import OllamaConnectionError
        client.health_check = AsyncMock(
            side_effect=OllamaConnectionError("Ollama not reachable")
        )
        provider = OllamaProvider(client)

        with pytest.raises(ProviderNotReadyException):
            await provider.health_check()

    async def test_model_info_returns_dict(self) -> None:
        client = _make_mock_client()
        provider = OllamaProvider(client)
        info = await provider.model_info()
        assert info["provider"] == "ollama"
        assert "model" in info

    async def test_stream_yields_response(self) -> None:
        client = _make_mock_client(generate_return="streamed result")
        provider = OllamaProvider(client)

        chunks = []
        async for chunk in provider.stream("hello"):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0] == "streamed result"
