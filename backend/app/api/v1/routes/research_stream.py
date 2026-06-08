"""
app/api/v1/routes/research_stream.py
────────────────────────────────────
SSE streaming endpoint for Phase 5.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.core.exceptions import SessionNotFoundException
from app.dependencies.providers import ResearchRepositoryDep, SettingsDep

router = APIRouter(prefix="/research", tags=["Research"])
logger = logging.getLogger(__name__)


@router.get(
    "/{session_id}/stream",
    response_class=StreamingResponse,
    summary="Stream research session updates",
    description=(
        "Returns a Server-Sent Events (SSE) stream yielding the complete "
        "WorkflowState JSON representation whenever the state updates. "
        "Closes automatically when status is 'completed' or 'failed'."
    ),
)
async def research_stream(
    session_id: str,
    request: Request,
    repo: ResearchRepositoryDep,
    settings: SettingsDep,
) -> StreamingResponse:
    """Stream WorkflowState updates to the client via SSE."""
    logger.info("SSE connection requested", extra={"session_id": session_id})

    # Verify session exists
    try:
        await repo.get(session_id)
    except SessionNotFoundException:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    async def event_generator() -> AsyncGenerator[str, None]:
        poll_interval = settings.sse_poll_interval_seconds
        last_updated_at = None

        try:
            while True:
                if await request.is_disconnected():
                    logger.info("SSE client disconnected", extra={"session_id": session_id})
                    break

                try:
                    state = await repo.get(session_id)
                except SessionNotFoundException:
                    break

                # Only yield if state updated
                if last_updated_at is None or state.updated_at > last_updated_at:
                    payload = state.model_dump(mode="json")
                    yield f"data: {json.dumps(payload)}\n\n"
                    last_updated_at = state.updated_at

                    if state.status in ("completed", "failed"):
                        logger.info("SSE stream completed", extra={"session_id": session_id, "status": state.status})
                        break

                await asyncio.sleep(poll_interval)
        except asyncio.CancelledError:
            logger.info("SSE stream cancelled", extra={"session_id": session_id})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )
