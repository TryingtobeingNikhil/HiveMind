"""
app/agents/base/agent.py
────────────────────────
Abstract BaseAgent — the foundational contract for all agents in Open Deep Research.

Design:
- Every agent subclass must implement ``execute()``.
- ``receive_task()`` validates and stores the incoming AgentTask.
- ``return_result()`` wraps execution output into a typed AgentResult.
- An internal state machine tracks: IDLE → RUNNING → DONE | ERROR.

Future agents (research, summariser, critic, …) inherit from BaseAgent
and override only the ``execute()`` method.

EXECUTION MODEL CONSTRAINT
──────────────────────────────
Agents are logical roles, NOT parallel workers.

All agent execution is strictly sequential under an external orchestrator
(introduced in Phase 3). The system assumes:

  - Only ONE agent is active (RUNNING state) at any point in time.
  - Only ONE LLM call is in-flight across the entire system at any moment.
  - No shared-state concurrency between agents is expected or supported.
  - Agents must NOT assume or require parallel execution.

IMPORTANT: Agents must assume single-active-execution context controlled
by an external orchestrator. Do NOT introduce concurrency primitives
(locks, semaphores, queues) inside BaseAgent or its subclasses.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from app.core.exceptions import AgentExecutionError, AgentNotReadyError
from app.schemas.agent import AgentResult, AgentState, AgentTask

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all Open Deep Research agents.

    Subclasses must implement:
        async def execute(self) -> Any

    EXECUTION MODEL
    ───────────────
    Agents are logical roles, not parallel workers. At any given moment
    only ONE agent will hold RUNNING state system-wide. Execution order
    is controlled exclusively by the Phase 3 orchestrator.

    Agents MUST NOT:
      - spawn background tasks or threads
      - make concurrent LLM calls
      - assume shared state with other simultaneously-running agents

    Example::

        class MyAgent(BaseAgent):
            agent_id   = "my-agent-v1"
            agent_name = "My Agent"

            async def execute(self) -> str:
                assert self.current_task is not None
                return f"processed: {self.current_task.input}"
    """

    # Subclasses define these as class-level attributes
    agent_id: str
    agent_name: str

    def __init__(self) -> None:
        if not hasattr(self, "agent_id") or not hasattr(self, "agent_name"):
            raise TypeError(
                f"{type(self).__name__} must define class attributes "
                "'agent_id' and 'agent_name'."
            )
        self._state: AgentState = AgentState.IDLE
        self._current_task: AgentTask | None = None
        self._last_result: AgentResult | None = None
        self._logger = logging.getLogger(
            f"{__name__}.{type(self).__name__}"
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def state(self) -> AgentState:
        """Current state of the agent."""
        return self._state

    @property
    def current_task(self) -> AgentTask | None:
        """The task this agent is currently holding (may be None if IDLE)."""
        return self._current_task

    @property
    def last_result(self) -> AgentResult | None:
        """The most recently produced result, or None if never run."""
        return self._last_result

    # ── Task lifecycle ────────────────────────────────────────────────────────

    def receive_task(self, task: AgentTask) -> None:
        """
        Accept an incoming task and transition the agent to IDLE-ready state.

        This method validates the task and stores it so that ``execute()`` can
        access it via ``self.current_task``.

        Args:
            task: A validated :class:`AgentTask` instance.
        """
        self._current_task = task
        self._state = AgentState.IDLE
        self._logger.info(
            "Task received",
            extra={
                "agent_id": self.agent_id,
                "task_id": str(task.task_id),
                "input_length": len(task.input),
            },
        )

    @abstractmethod
    async def execute(self) -> Any:
        """
        Execute the task held in ``self.current_task``.

        SEQUENTIAL EXECUTION CONSTRAINT:
        This method is invoked by the external orchestrator one agent at a
        time. It must complete fully before the next agent is activated.
        Do NOT launch background tasks or make concurrent LLM calls from
        within this method.

        Implementations must:
          - Access the task via ``self.current_task``.
          - Return a result that will be wrapped into ``AgentResult.output``.
          - Raise any exception on failure (it will be caught by ``run()``).

        Returns:
            Any serialisable output produced by this agent.
        """
        ...  # pragma: no cover

    async def run(self) -> AgentResult:
        """
        Orchestrate the full agent lifecycle: validate → execute → return result.

        This is the primary entry point called by the API layer or orchestrator.
        Under the sequential execution model, ``run()`` is always called by the
        orchestrator for ONE agent at a time — never concurrently.

        Returns:
            A typed :class:`AgentResult` wrapping the execution output.

        Raises:
            AgentNotReadyError:  If no task has been received yet.
            AgentExecutionError: If ``execute()`` raises an unhandled exception.
        """
        if self._current_task is None:
            raise AgentNotReadyError(
                f"Agent '{self.agent_id}' has no task. Call receive_task() first.",
                context={"agent_id": self.agent_id},
            )

        task = self._current_task
        self._state = AgentState.RUNNING
        self._logger.info(
            "Agent execution started",
            extra={"agent_id": self.agent_id, "task_id": str(task.task_id)},
        )

        try:
            output = await self.execute()
            self._state = AgentState.DONE
            result = self.return_result(output=output)
            self._last_result = result
            self._logger.info(
                "Agent execution completed",
                extra={"agent_id": self.agent_id, "task_id": str(task.task_id)},
            )
            return result

        except AgentExecutionError:
            # Already a typed exception — re-raise as-is
            self._state = AgentState.ERROR
            raise

        except Exception as exc:
            self._state = AgentState.ERROR
            error_msg = f"{type(exc).__name__}: {exc}"
            self._logger.exception(
                "Agent execution failed",
                extra={"agent_id": self.agent_id, "task_id": str(task.task_id)},
            )
            result = self.return_result(error=error_msg)
            self._last_result = result
            raise AgentExecutionError(
                f"Agent '{self.agent_id}' failed: {error_msg}",
                context={"agent_id": self.agent_id, "task_id": str(task.task_id)},
            ) from exc

    def return_result(
        self,
        *,
        output: Any = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResult:
        """
        Wrap execution output into a typed :class:`AgentResult`.

        Args:
            output:   The output produced by ``execute()``.
            error:    Error message if execution failed.
            metadata: Optional additional metadata to include.

        Returns:
            A fully populated :class:`AgentResult`.
        """
        if self._current_task is None:
            raise AgentNotReadyError("Cannot return result: no task was received.")

        return AgentResult(
            task_id=self._current_task.task_id,
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            state=self._state,
            output=output,
            error=error,
            metadata=metadata or {},
        )

    # ── Utility ───────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset the agent to its initial IDLE state, clearing any held task."""
        self._current_task = None
        self._state = AgentState.IDLE
        self._logger.debug("Agent reset", extra={"agent_id": self.agent_id})

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} "
            f"id={self.agent_id!r} "
            f"state={self._state.value!r}>"
        )
