"""
app/api/v1/routes/memory.py
─────────────────────────────
Memory system health endpoint.

GET /api/v1/memory/health — Health check for all memory subsystems
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter

from app.dependencies.providers import LLMManagerDep, MemoryServiceDep
from app.schemas.common import APIResponse

router = APIRouter(prefix="/memory", tags=["Memory"])
logger = logging.getLogger(__name__)


@router.get(
    "/health",
    response_model=APIResponse[dict[str, Any]],
    summary="Memory system health",
)
async def memory_health(
    memory: MemoryServiceDep,
    llm_manager: LLMManagerDep,
) -> APIResponse[dict[str, Any]]:
    """
    Report the health of all Phase 2 memory subsystems.

    Returns status for:
      - vector store
      - embedding model
      - reranker
      - LLM execution manager
    """
    memory_status = await memory.health()
    manager_stats = llm_manager.stats

    health_data: dict[str, Any] = {
        "memory": memory_status,
        "llm_execution_manager": manager_stats,
        "status": "healthy",
    }

    if memory_status.get("vector_store") != "healthy":
        health_data["status"] = "degraded"

    logger.debug("Memory health check", extra=health_data)

    return APIResponse(data=health_data)
