"""
app/api/v1/routes/agents.py
────────────────────────────
Agent API endpoints — Phase 1 exposes a single echo endpoint for framework validation.

Routes:
    POST /api/v1/agents/echo  → Run SimpleEchoAgent on provided input

This endpoint validates the full stack:
    HTTP request → schema validation → agent receive_task → execute → return result
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from app.agents.base.echo_agent import SimpleEchoAgent
from app.schemas.agent import AgentResult, AgentTask, AgentTaskRequest

router = APIRouter(prefix="/agents", tags=["Agents"])
logger = logging.getLogger(__name__)


@router.post(
    "/echo",
    response_model=AgentResult,
    summary="Echo agent",
    description=(
        "Runs the SimpleEchoAgent on the provided input. "
        "This endpoint exists solely to validate the agent framework is "
        "wired correctly end-to-end. Not used in production workflows."
    ),
    responses={
        200: {"description": "Agent executed successfully"},
        422: {"description": "Invalid request body"},
        500: {"description": "Agent execution error"},
    },
)
async def echo_agent(request_body: AgentTaskRequest) -> AgentResult:
    """
    Submit input to the SimpleEchoAgent and return its result.

    The agent receives a task, executes (echoes the input), and returns
    a fully typed AgentResult.
    """
    task = AgentTask(
        input=request_body.input,
        parameters=request_body.parameters,
    )

    agent = SimpleEchoAgent()
    agent.receive_task(task)

    logger.info(
        "Running echo agent",
        extra={"task_id": str(task.task_id), "input_length": len(task.input)},
    )

    result = await agent.run()
    return result
