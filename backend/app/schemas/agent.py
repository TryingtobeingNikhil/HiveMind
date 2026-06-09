"""
app/schemas/agent.py
────────────────────
Pydantic models for agent task submission and result retrieval.

These schemas form the contract between the API layer and the agent framework.
Future agents can extend AgentTask with domain-specific payloads.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class AgentState(StrEnum):
    """State machine values for an agent lifecycle."""

    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


class AgentTask(BaseModel):
    """
    Input envelope for submitting work to an agent.

    Agents receive this model via ``receive_task()``.
    """

    task_id: UUID = Field(default_factory=uuid4, description="Unique task identifier")
    input: str = Field(..., min_length=1, description="Task input — prompt or instruction")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional agent-specific parameters",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="UTC timestamp when this task was created",
    )


class AgentResult(BaseModel):
    """
    Output envelope returned by an agent after ``execute()``.
    """

    task_id: UUID = Field(..., description="Matches the originating AgentTask.task_id")
    agent_id: str = Field(..., description="Identifier of the agent that ran the task")
    agent_name: str = Field(..., description="Human-readable agent name")
    state: AgentState = Field(..., description="Final state of the agent")
    output: Any = Field(default=None, description="Agent-produced output (type varies by agent)")
    error: str | None = Field(default=None, description="Error message if state == error")
    metadata: dict[str, Any] = Field(default_factory=dict)
    completed_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="UTC timestamp when execution completed",
    )


class AgentTaskRequest(BaseModel):
    """HTTP request body for the agent echo endpoint."""

    input: str = Field(..., min_length=1, description="Input to send to the agent")
    parameters: dict[str, Any] = Field(default_factory=dict)
