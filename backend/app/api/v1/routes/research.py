"""
app/api/v1/routes/research.py
───────────────────────────────
Research orchestration endpoints — Phase 3.

Endpoints:
  POST /api/v1/research/start               — Start a research session
  GET  /api/v1/research/history             — List past sessions
  GET  /api/v1/research/{session_id}        — Get full WorkflowState
  GET  /api/v1/research/{session_id}/status — Get lightweight status
  GET  /api/v1/research/{session_id}/report — Get the final report

Design notes:
  - POST /start runs synchronously in Phase 3 (no background tasks).
  - Ordering of routes matters: /history must be registered before
    /{session_id} to avoid FastAPI treating "history" as a session_id.
  - All async methods on dependencies are properly awaited.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from app.core.exceptions import SessionNotFoundException
from app.dependencies.providers import (
    OrchestratorDep,
    ResearchRepositoryDep,
)

router = APIRouter(prefix="/research", tags=["Research"])
logger = logging.getLogger(__name__)


# ── POST /start ───────────────────────────────────────────────────────────────


class StartResearchRequest:
    """Inline request model to avoid an extra schemas file."""

    def __init__(self, query: str) -> None:
        self.query = query


from pydantic import BaseModel, Field


class StartResearchBody(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000, description="Research query")


@router.post(
    "/start",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start a research session",
    response_description="Session created and pipeline completed",
)
async def start_research(
    body: StartResearchBody,
    orchestrator: OrchestratorDep,
) -> JSONResponse:
    """
    Start a research session.

    Runs the full pipeline synchronously (Phase 3 design):
      PLANNING → RESEARCHING → CRITIQUING → REPORTING → COMPLETED

    Returns 202 with session metadata immediately after completion.
    The full WorkflowState is retrievable via GET /{session_id}.
    """
    session_id = str(uuid4())

    logger.info(
        "Research session starting",
        extra={"session_id": session_id, "query_len": len(body.query)},
    )

    state = await orchestrator.run(query=body.query, session_id=session_id)

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "session_id": state.session_id,
            "status": state.status,
            "created_at": state.created_at.isoformat(),
        },
    )


# ── GET /history ──────────────────────────────────────────────────────────────
# IMPORTANT: This route must be registered BEFORE /{session_id} to prevent
# FastAPI from matching the literal string "history" as a session_id parameter.


@router.get(
    "/history",
    status_code=status.HTTP_200_OK,
    summary="List research session history",
)
async def list_research_history(
    repo: ResearchRepositoryDep,
    limit: int = 20,
    offset: int = 0,
) -> JSONResponse:
    """
    Return a paginated list of research sessions (most recent first).

    Query params:
      limit:  Maximum sessions to return (default 20).
      offset: Sessions to skip (default 0).
    """
    states = await repo.list_all(limit=limit, offset=offset)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=[
            {
                "session_id": s.session_id,
                "query": s.query,
                "status": s.status,
                "current_stage": s.current_stage.value,
                "created_at": s.created_at.isoformat(),
                "completed_at": s.completed_at.isoformat() if s.completed_at else None,
            }
            for s in states
        ],
    )


# ── GET /{session_id} ─────────────────────────────────────────────────────────


@router.get(
    "/{session_id}",
    status_code=status.HTTP_200_OK,
    summary="Get full WorkflowState for a session",
)
async def get_research_session(
    session_id: str,
    repo: ResearchRepositoryDep,
) -> JSONResponse:
    """
    Return the full WorkflowState for a research session.

    Returns 404 if the session does not exist.
    """
    try:
        state = await repo.get(session_id)
    except SessionNotFoundException:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": {"message": f"Session '{session_id}' not found"}},
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=state.model_dump(mode="json"),
    )


# ── GET /{session_id}/status ──────────────────────────────────────────────────


@router.get(
    "/{session_id}/status",
    status_code=status.HTTP_200_OK,
    summary="Get lightweight status for a session",
)
async def get_research_status(
    session_id: str,
    repo: ResearchRepositoryDep,
) -> JSONResponse:
    """
    Return a lightweight status payload for a research session.

    Returns 404 if the session does not exist.
    """
    try:
        state = await repo.get(session_id)
    except SessionNotFoundException:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": {"message": f"Session '{session_id}' not found"}},
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "session_id": state.session_id,
            "status": state.status,
            "current_stage": state.current_stage.value,
            "error": state.error,
        },
    )


# ── GET /{session_id}/report ──────────────────────────────────────────────────


@router.get(
    "/{session_id}/report",
    status_code=status.HTTP_200_OK,
    summary="Get the final research report",
)
async def get_research_report(
    session_id: str,
    repo: ResearchRepositoryDep,
) -> JSONResponse:
    """
    Return the final ResearchReport for a session.

    Returns:
      200 — report is available.
      404 — session not found.
      422 — session failed before a report was generated.
    """
    try:
        state = await repo.get(session_id)
    except SessionNotFoundException:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": {"message": f"Session '{session_id}' not found"}},
        )

    if state.report is None:
        if state.status == "failed":
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "error": {
                        "message": (
                            f"Session '{session_id}' failed before a report "
                            f"was generated. Error: {state.error}"
                        )
                    }
                },
            )
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "error": {
                    "message": (
                        f"Report not yet available for session '{session_id}'. "
                        f"Current stage: {state.current_stage.value}"
                    )
                }
            },
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=state.report.model_dump(mode="json"),
    )
