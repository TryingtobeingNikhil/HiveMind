"""
app/services/health_service.py
───────────────────────────────
HealthService — orchestrates readiness checks for all application dependencies.

Responsibilities:
- Probe the Ollama server and measure round-trip latency.
- Aggregate component statuses into a single ReadinessResponse.
- Future phases can register additional components (vector DB, message bus, …)
  without changing the API route.
"""

from __future__ import annotations

import logging
import time

from app.llm.ollama_client import OllamaClient
from app.schemas.health import ComponentStatus, HealthStatus, ReadinessResponse
from app.queue.redis_queue import ResearchTaskQueue

logger = logging.getLogger(__name__)


class HealthService:
    """
    Aggregates readiness checks across all required system components.

    Usage::

        service = HealthService(ollama_client=client, app_version="0.1.0")
        response = await service.check_readiness()
    """

    def __init__(
        self,
        ollama_client: OllamaClient,
        app_version: str = "0.1.0",
    ) -> None:
        self._ollama = ollama_client
        self._app_version = app_version

    async def check_readiness(self, task_queue: ResearchTaskQueue | None = None) -> ReadinessResponse:
        """
        Run all component probes and return an aggregated ReadinessResponse.

        The overall status is:
          - ``healthy``  — all components passed.
          - ``degraded`` — at least one component failed (reserved for future use).
          - ``unhealthy``— a required component is unreachable.

        Returns:
            :class:`ReadinessResponse` with per-component details.
        """
        components: list[ComponentStatus] = []

        # ── Ollama probe ──────────────────────────────────────────────────────
        ollama_status = await self._probe_ollama()
        components.append(ollama_status)

        # ── Future components go here ──────────────────────────────────────────
        # e.g. vector_db_status = await self._probe_vector_db()
        # components.append(vector_db_status)

        # ── Aggregate ─────────────────────────────────────────────────────────
        any_unhealthy = any(c.status == HealthStatus.UNHEALTHY for c in components)
        overall = HealthStatus.UNHEALTHY if any_unhealthy else HealthStatus.HEALTHY

        response = ReadinessResponse(
            status=overall,
            components=components,
            app_version=self._app_version,
            redis_reachable=await task_queue.health() if task_queue else None,
        )

        logger.info(
            "Readiness check completed",
            extra={
                "overall_status": overall,
                "components": [
                    {"name": c.name, "status": c.status} for c in components
                ],
            },
        )
        return response

    async def _probe_ollama(self) -> ComponentStatus:
        """Probe Ollama and return its ComponentStatus."""
        start = time.monotonic()
        try:
            await self._ollama.health_check()
            latency_ms = round((time.monotonic() - start) * 1000, 2)
            return ComponentStatus(
                name="ollama",
                status=HealthStatus.HEALTHY,
                latency_ms=latency_ms,
                detail="Ollama server is reachable",
            )
        except Exception as exc:
            latency_ms = round((time.monotonic() - start) * 1000, 2)
            logger.warning(
                "Ollama readiness probe failed",
                extra={"error": str(exc)},
            )
            return ComponentStatus(
                name="ollama",
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                detail=str(exc),
            )
