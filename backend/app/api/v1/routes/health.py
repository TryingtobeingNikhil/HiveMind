"""
app/api/v1/routes/health.py
────────────────────────────
Health and readiness endpoints for API v1.

Routes:
    GET /api/v1/health  → Simple liveness check (always 200 if process is up)
    GET /api/v1/ready   → Readiness check (probes Ollama, returns 200 or 503)
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.dependencies.providers import HealthServiceDep, SettingsDep
from app.schemas.health import HealthResponse, HealthStatus, ReadinessResponse

router = APIRouter(tags=["Health"])
logger = logging.getLogger(__name__)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check",
    description=(
        "Returns HTTP 200 with `{\"status\": \"healthy\"}` if the application "
        "process is running. Does not probe any downstream dependencies."
    ),
)
async def health() -> HealthResponse:
    """Liveness check — confirms the application process is alive."""
    return HealthResponse(status=HealthStatus.HEALTHY)


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness check",
    description=(
        "Probes all required dependencies (Ollama, …) and returns HTTP 200 "
        "when all are healthy, or HTTP 503 when any dependency is unreachable."
    ),
    responses={
        200: {"description": "All dependencies healthy"},
        503: {"description": "One or more dependencies unavailable"},
    },
)
async def ready(
    request: Request,
    health_service: HealthServiceDep,
    settings: SettingsDep,
) -> JSONResponse:
    """
    Readiness check — probes all required downstream dependencies.

    Returns 200 when all components are healthy, 503 otherwise.
    Load balancers and orchestrators should use this endpoint to decide
    whether to route traffic to this instance.
    """
    logger.debug("Readiness check requested")
    task_queue = getattr(request.app.state, "task_queue", None)
    response = await health_service.check_readiness(task_queue=task_queue)
    http_status = 200 if response.is_ready else 503
    return JSONResponse(
        status_code=http_status,
        content=response.model_dump(mode="json"),
    )
