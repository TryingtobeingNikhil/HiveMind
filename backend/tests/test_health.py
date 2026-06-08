"""
tests/test_health.py
──────────────────────
Tests for GET /health, GET /ready, GET /api/v1/health, GET /api/v1/ready.

Uses:
  - async_client           → app with healthy mock Ollama (200 responses)
  - async_client_unhealthy → app with unreachable mock Ollama (503 responses)
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
class TestRootHealthEndpoint:
    """Tests for GET /health (root liveness check)."""

    async def test_health_returns_200(self, async_client: AsyncClient) -> None:
        response = await async_client.get("/health")
        assert response.status_code == 200

    async def test_health_returns_healthy_status(self, async_client: AsyncClient) -> None:
        response = await async_client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    async def test_health_response_has_correct_content_type(
        self, async_client: AsyncClient
    ) -> None:
        response = await async_client.get("/health")
        assert "application/json" in response.headers["content-type"]


@pytest.mark.asyncio
class TestV1HealthEndpoint:
    """Tests for GET /api/v1/health."""

    async def test_v1_health_returns_200(self, async_client: AsyncClient) -> None:
        response = await async_client.get("/api/v1/health")
        assert response.status_code == 200

    async def test_v1_health_returns_healthy_status(self, async_client: AsyncClient) -> None:
        response = await async_client.get("/api/v1/health")
        data = response.json()
        assert data["status"] == "healthy"


@pytest.mark.asyncio
class TestRootReadyEndpoint:
    """Tests for GET /ready (root readiness shortcut)."""

    async def test_ready_returns_200_when_ollama_healthy(
        self, async_client: AsyncClient
    ) -> None:
        response = await async_client.get("/ready")
        assert response.status_code == 200

    async def test_ready_returns_healthy_status_when_ollama_reachable(
        self, async_client: AsyncClient
    ) -> None:
        response = await async_client.get("/ready")
        data = response.json()
        assert data["status"] == "healthy"

    async def test_ready_returns_503_when_ollama_unreachable(
        self, async_client_unhealthy: AsyncClient
    ) -> None:
        response = await async_client_unhealthy.get("/ready")
        assert response.status_code == 503

    async def test_ready_includes_components(self, async_client: AsyncClient) -> None:
        response = await async_client.get("/ready")
        data = response.json()
        assert "components" in data
        assert isinstance(data["components"], list)
        assert len(data["components"]) >= 1

    async def test_ready_includes_app_version(self, async_client: AsyncClient) -> None:
        response = await async_client.get("/ready")
        data = response.json()
        assert "app_version" in data

    async def test_ready_ollama_component_present(
        self, async_client: AsyncClient
    ) -> None:
        response = await async_client.get("/ready")
        data = response.json()
        names = [c["name"] for c in data["components"]]
        assert "ollama" in names

    async def test_ready_unhealthy_status_when_ollama_down(
        self, async_client_unhealthy: AsyncClient
    ) -> None:
        response = await async_client_unhealthy.get("/ready")
        data = response.json()
        assert data["status"] == "unhealthy"
        ollama_component = next(
            (c for c in data["components"] if c["name"] == "ollama"), None
        )
        assert ollama_component is not None
        assert ollama_component["status"] == "unhealthy"


@pytest.mark.asyncio
class TestV1ReadyEndpoint:
    """Tests for GET /api/v1/ready."""

    async def test_v1_ready_returns_200_when_healthy(
        self, async_client: AsyncClient
    ) -> None:
        response = await async_client.get("/api/v1/ready")
        assert response.status_code == 200

    async def test_v1_ready_returns_503_when_unhealthy(
        self, async_client_unhealthy: AsyncClient
    ) -> None:
        response = await async_client_unhealthy.get("/api/v1/ready")
        assert response.status_code == 503
