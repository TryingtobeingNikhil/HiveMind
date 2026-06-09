"""
app/schemas/health.py
─────────────────────
Pydantic models for /health and /ready responses.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class HealthStatus(StrEnum):
    """Possible status values for health and readiness checks."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentStatus(BaseModel):
    """Status of a single dependency or sub-component."""

    name: str = Field(..., description="Component identifier")
    status: HealthStatus
    latency_ms: float | None = Field(
        default=None,
        description="Round-trip latency in milliseconds (when measurable)",
    )
    detail: str | None = Field(default=None, description="Human-readable status detail")
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """
    Response for ``GET /health``.

    A 200 here means the application process is alive — not necessarily
    that dependencies are reachable (see ReadinessResponse for that).
    """

    status: HealthStatus = HealthStatus.HEALTHY


class ReadinessResponse(BaseModel):
    """
    Response for ``GET /ready``.

    Aggregates the status of all required dependencies.
    HTTP 200 is returned only when ``status == "healthy"``.
    HTTP 503 is returned when any component is unhealthy.
    """

    status: HealthStatus
    components: list[ComponentStatus] = Field(default_factory=list)
    app_version: str | None = None
    redis_reachable: bool | None = None

    @property
    def is_ready(self) -> bool:
        """Return True only when all components are healthy."""
        return self.status == HealthStatus.HEALTHY
