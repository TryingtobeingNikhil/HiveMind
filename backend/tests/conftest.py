"""
tests/conftest.py
─────────────────
Shared pytest fixtures for Open Deep Research backend tests.

Fixtures:
    test_settings     — Settings instance with testing overrides
    mock_ollama_client — OllamaClient with health_check() patched as AsyncMock
    mock_app          — FastAPI app with DI overrides, no lifespan
    async_client      — httpx.AsyncClient (via ASGITransport) targeting mock_app
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.dependencies.providers import get_ollama_client, settings_provider
from app.llm.ollama_client import OllamaClient
from app.main import create_app


# ── Test Settings ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """
    Return a Settings instance configured for the testing environment.

    No .env file required — all values are set via constructor kwargs.
    """
    return Settings(
        APP_NAME="open-deep-research-test",
        APP_ENV="testing",
        APP_VERSION="0.0.0-test",
        DEBUG=False,
        LOG_LEVEL="DEBUG",
        LOG_FORMAT="text",
        OLLAMA_BASE_URL="http://mock-ollama:11434",
        OLLAMA_MODEL="test-model",
        OLLAMA_TIMEOUT=5.0,
        OLLAMA_MAX_RETRIES=0,
        OLLAMA_RETRY_WAIT=0.0,
        PORT=8001,
    )


# ── Mock Ollama Client ────────────────────────────────────────────────────────


@pytest.fixture()
def mock_ollama_client(test_settings: Settings) -> OllamaClient:
    """
    OllamaClient with all async methods pre-mocked via AsyncMock.

    - health_check() → returns True (healthy)
    - list_models()  → returns empty list
    - generate()     → returns empty string

    Individual tests override these defaults as needed.
    """
    client = OllamaClient.__new__(OllamaClient)
    client._settings = test_settings
    client._base_url = test_settings.ollama_api_url
    client._default_model = test_settings.ollama_model
    client.health_check = AsyncMock(return_value=True)
    client.list_models = AsyncMock(return_value=[])
    client.generate = AsyncMock(return_value="")
    client.aclose = AsyncMock()
    return client  # type: ignore[return-value]


@pytest.fixture()
def mock_ollama_unhealthy(test_settings: Settings) -> OllamaClient:
    """
    OllamaClient that raises OllamaConnectionError on health_check().
    """
    from app.core.exceptions import OllamaConnectionError

    client = OllamaClient.__new__(OllamaClient)
    client._settings = test_settings
    client._base_url = test_settings.ollama_api_url
    client._default_model = test_settings.ollama_model
    client.health_check = AsyncMock(
        side_effect=OllamaConnectionError("Cannot reach Ollama at http://mock-ollama:11434")
    )
    client.list_models = AsyncMock(return_value=[])
    client.generate = AsyncMock(return_value="")
    client.aclose = AsyncMock()
    return client  # type: ignore[return-value]


# ── FastAPI test app ──────────────────────────────────────────────────────────


@pytest.fixture()
def mock_app(test_settings: Settings, mock_ollama_client: OllamaClient):
    """
    FastAPI application with:
     - Settings overridden to test_settings
     - OllamaClient overridden to mock_ollama_client (no real HTTP)
     - app.state pre-seeded so request handlers don't fail
     - Lifespan skipped (no startup/shutdown I/O)
    """
    application = create_app(settings=test_settings)

    # Dependency injection overrides
    application.dependency_overrides[settings_provider] = lambda: test_settings
    application.dependency_overrides[get_ollama_client] = lambda: mock_ollama_client

    # Seed app.state directly (bypasses lifespan startup)
    application.state.ollama_client = mock_ollama_client
    application.state.ready = True

    return application


@pytest.fixture()
def mock_app_unhealthy(test_settings: Settings, mock_ollama_unhealthy: OllamaClient):
    """
    FastAPI application variant where Ollama is configured as unreachable.
    Used by tests that need the /ready endpoint to return 503.
    """
    application = create_app(settings=test_settings)

    application.dependency_overrides[settings_provider] = lambda: test_settings
    application.dependency_overrides[get_ollama_client] = lambda: mock_ollama_unhealthy

    application.state.ollama_client = mock_ollama_unhealthy
    application.state.ready = True

    return application


# ── Async HTTP test clients ───────────────────────────────────────────────────


@pytest_asyncio.fixture()
async def async_client(mock_app) -> AsyncClient:
    """
    Async HTTP test client targeting the healthy mock FastAPI application.

    Uses ASGITransport (httpx >= 0.28 removed the deprecated `app=` kwarg).
    """
    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest_asyncio.fixture()
async def async_client_unhealthy(mock_app_unhealthy) -> AsyncClient:
    """Async HTTP test client targeting the unhealthy mock application."""
    transport = ASGITransport(app=mock_app_unhealthy)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
