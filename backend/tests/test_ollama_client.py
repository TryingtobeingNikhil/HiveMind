"""
tests/test_ollama_client.py
────────────────────────────
Unit tests for OllamaClient.

Strategy: Build a respx.MockRouter, wrap it in a MockTransport, inject that
transport into an httpx.AsyncClient, then inject the client into OllamaClient.
This guarantees all HTTP traffic is intercepted without touching real sockets.

No live Ollama server required.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from httpx import Response

from app.core.config import Settings
from app.core.exceptions import (
    OllamaConnectionError,
    OllamaGenerationError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
)
from app.llm.ollama_client import OllamaClient

MOCK_BASE = "http://mock-ollama:11434"


def build_client(settings: Settings, router: respx.MockRouter) -> OllamaClient:
    """
    Construct an OllamaClient backed by a respx MockRouter.

    Uses httpx.MockTransport(router.handler) — the non-deprecated pattern.
    """
    transport = httpx.MockTransport(router.handler)
    http_client = httpx.AsyncClient(
        base_url=MOCK_BASE,
        transport=transport,
        timeout=httpx.Timeout(5.0),
        headers={"Content-Type": "application/json"},
    )
    return OllamaClient(settings, http_client=http_client)


# ── health_check() ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestOllamaHealthCheck:

    async def test_health_check_returns_true_when_server_responds(
        self, test_settings: Settings
    ) -> None:
        router = respx.MockRouter(assert_all_called=False)
        router.get(f"{MOCK_BASE}/api/tags").mock(
            return_value=Response(200, json={"models": []})
        )
        client = build_client(test_settings, router)
        result = await client.health_check()
        assert result is True

    async def test_health_check_raises_connection_error_when_unreachable(
        self, test_settings: Settings
    ) -> None:
        router = respx.MockRouter(assert_all_called=False)
        router.get(f"{MOCK_BASE}/api/tags").mock(
            side_effect=httpx.ConnectError("refused")
        )
        client = build_client(test_settings, router)
        with pytest.raises(OllamaConnectionError):
            await client.health_check()

    async def test_health_check_raises_timeout_error(
        self, test_settings: Settings
    ) -> None:
        router = respx.MockRouter(assert_all_called=False)
        router.get(f"{MOCK_BASE}/api/tags").mock(
            side_effect=httpx.TimeoutException("timed out")
        )
        client = build_client(test_settings, router)
        with pytest.raises(OllamaTimeoutError):
            await client.health_check()

    async def test_health_check_raises_on_non_200_status(
        self, test_settings: Settings
    ) -> None:
        router = respx.MockRouter(assert_all_called=False)
        router.get(f"{MOCK_BASE}/api/tags").mock(return_value=Response(500))
        client = build_client(test_settings, router)
        with pytest.raises(OllamaConnectionError):
            await client.health_check()


# ── list_models() ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestOllamaListModels:

    async def test_list_models_returns_model_list(
        self, test_settings: Settings
    ) -> None:
        models_payload = [
            {"name": "llama3.2:3b", "size": 1_000_000},
            {"name": "mistral:7b", "size": 5_000_000},
        ]
        router = respx.MockRouter(assert_all_called=False)
        router.get(f"{MOCK_BASE}/api/tags").mock(
            return_value=Response(200, json={"models": models_payload})
        )
        client = build_client(test_settings, router)
        result = await client.list_models()
        assert len(result) == 2
        assert result[0]["name"] == "llama3.2:3b"

    async def test_list_models_returns_empty_when_no_models(
        self, test_settings: Settings
    ) -> None:
        router = respx.MockRouter(assert_all_called=False)
        router.get(f"{MOCK_BASE}/api/tags").mock(
            return_value=Response(200, json={"models": []})
        )
        client = build_client(test_settings, router)
        result = await client.list_models()
        assert result == []

    async def test_list_models_raises_on_connect_error(
        self, test_settings: Settings
    ) -> None:
        router = respx.MockRouter(assert_all_called=False)
        router.get(f"{MOCK_BASE}/api/tags").mock(
            side_effect=httpx.ConnectError("refused")
        )
        client = build_client(test_settings, router)
        with pytest.raises(OllamaConnectionError):
            await client.list_models()


# ── generate() ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestOllamaGenerate:

    async def test_generate_returns_response_text(
        self, test_settings: Settings
    ) -> None:
        router = respx.MockRouter(assert_all_called=False)
        router.post(f"{MOCK_BASE}/api/generate").mock(
            return_value=Response(
                200,
                json={
                    "model": "test-model",
                    "response": "Hello from Ollama!",
                    "done": True,
                    "eval_count": 5,
                    "eval_duration": 100_000_000,
                },
            )
        )
        client = build_client(test_settings, router)
        result = await client.generate(prompt="Say hello")
        assert result == "Hello from Ollama!"

    async def test_generate_uses_default_model(
        self, test_settings: Settings
    ) -> None:
        captured: list[dict] = []

        def capture(request: httpx.Request) -> httpx.Response:
            import json
            captured.append(json.loads(request.content))
            return httpx.Response(200, json={"response": "ok", "done": True})

        router = respx.MockRouter(assert_all_called=False)
        router.post(f"{MOCK_BASE}/api/generate").mock(side_effect=capture)
        client = build_client(test_settings, router)
        await client.generate(prompt="test")
        assert captured[0]["model"] == "test-model"

    async def test_generate_raises_model_not_found_on_404(
        self, test_settings: Settings
    ) -> None:
        router = respx.MockRouter(assert_all_called=False)
        router.post(f"{MOCK_BASE}/api/generate").mock(return_value=Response(404))
        client = build_client(test_settings, router)
        with pytest.raises(OllamaModelNotFoundError):
            await client.generate(prompt="test")

    async def test_generate_raises_generation_error_on_error_payload(
        self, test_settings: Settings
    ) -> None:
        router = respx.MockRouter(assert_all_called=False)
        router.post(f"{MOCK_BASE}/api/generate").mock(
            return_value=Response(200, json={"error": "model crashed"})
        )
        client = build_client(test_settings, router)
        with pytest.raises(OllamaGenerationError):
            await client.generate(prompt="test")

    async def test_generate_raises_on_non_200_status(
        self, test_settings: Settings
    ) -> None:
        router = respx.MockRouter(assert_all_called=False)
        router.post(f"{MOCK_BASE}/api/generate").mock(return_value=Response(503))
        client = build_client(test_settings, router)
        with pytest.raises(OllamaGenerationError):
            await client.generate(prompt="test")

    async def test_generate_raises_connection_error_when_unreachable(
        self, test_settings: Settings
    ) -> None:
        router = respx.MockRouter(assert_all_called=False)
        router.post(f"{MOCK_BASE}/api/generate").mock(
            side_effect=httpx.ConnectError("refused")
        )
        client = build_client(test_settings, router)
        with pytest.raises(OllamaConnectionError):
            await client.generate(prompt="test")

    async def test_generate_with_custom_model(
        self, test_settings: Settings
    ) -> None:
        captured: list[dict] = []

        def capture(request: httpx.Request) -> httpx.Response:
            import json
            captured.append(json.loads(request.content))
            return httpx.Response(200, json={"response": "custom", "done": True})

        router = respx.MockRouter(assert_all_called=False)
        router.post(f"{MOCK_BASE}/api/generate").mock(side_effect=capture)
        client = build_client(test_settings, router)
        await client.generate(prompt="test", model="mistral:7b")
        assert captured[0]["model"] == "mistral:7b"

    async def test_generate_with_system_prompt(
        self, test_settings: Settings
    ) -> None:
        captured: list[dict] = []

        def capture(request: httpx.Request) -> httpx.Response:
            import json
            captured.append(json.loads(request.content))
            return httpx.Response(200, json={"response": "resp", "done": True})

        router = respx.MockRouter(assert_all_called=False)
        router.post(f"{MOCK_BASE}/api/generate").mock(side_effect=capture)
        client = build_client(test_settings, router)
        await client.generate(prompt="hello", system="You are a helpful assistant.")
        assert captured[0].get("system") == "You are a helpful assistant."


# ── Client lifecycle ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestOllamaClientLifecycle:

    async def test_async_context_manager(self, test_settings: Settings) -> None:
        """Verify the async context manager opens and closes without error."""
        async with OllamaClient(test_settings) as c:
            assert c is not None
