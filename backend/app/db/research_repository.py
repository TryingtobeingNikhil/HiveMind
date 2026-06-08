"""
app/db/research_repository.py
──────────────────────────────
Repository for research session persistence.

Follows the exact same pattern as DocumentRepository and MetricsRepository:
  - Constructor receives an aiosqlite.Connection (shared singleton from app.state).
  - All methods are async.
  - JSON fields serialised as strings (model.model_dump_json()).
  - JSON fields deserialised via Model.model_validate_json() on read.
  - DatabaseException raised on all unexpected DB errors.
  - SessionNotFoundException raised when a session is not found.

Tables used (created by app/db/database.py):
  research_sessions        — WorkflowState persistence
  agent_execution_metrics  — per-agent timing records

Phase 4 additions:
  - iteration column (INTEGER NOT NULL DEFAULT 1)
  - evaluations_json column (TEXT)
  - _serialise_evaluations / _deserialise_evaluations helpers
  - iteration and evaluations_json included in INSERT/UPDATE/SELECT
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import aiosqlite

from app.core.exceptions import DatabaseException, SessionNotFoundException
from app.schemas.research import (
    CriticResult,
    EvaluationResult,
    ResearchPlan,
    ResearchReport,
    ResearchResult,
    WorkflowStage,
    WorkflowState,
)

logger = logging.getLogger(__name__)


class ResearchRepository:
    """
    Async repository for :class:`WorkflowState` records.

    All methods receive and return fully typed Pydantic models.
    JSON serialisation is an internal concern of this class.
    """

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._db = conn

    # ── Session CRUD ──────────────────────────────────────────────────────────

    async def create(self, state: WorkflowState) -> WorkflowState:
        """
        Persist a new research session.

        Args:
            state: The initial WorkflowState to persist.

        Returns:
            The same state object (for chaining convenience).

        Raises:
            DatabaseException: On any database error.
        """
        try:
            await self._db.execute(
                """
                INSERT INTO research_sessions
                    (session_id, query, status, current_stage,
                     plan_json, results_json, critic_json, report_json,
                     error, created_at, updated_at, completed_at,
                     iteration, evaluations_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    state.session_id,
                    state.query,
                    state.status,
                    state.current_stage.value,
                    state.plan.model_dump_json() if state.plan else None,
                    self._serialise_results(state.results),
                    state.critic_result.model_dump_json() if state.critic_result else None,
                    state.report.model_dump_json() if state.report else None,
                    state.error,
                    state.created_at.isoformat(),
                    state.updated_at.isoformat(),
                    state.completed_at.isoformat() if state.completed_at else None,
                    state.iteration,
                    self._serialise_evaluations(state.evaluations),
                ),
            )
            await self._db.commit()
            logger.debug(
                "Research session created",
                extra={"session_id": state.session_id, "status": state.status},
            )
            return state
        except Exception as exc:
            raise DatabaseException(
                f"Failed to create research session: {exc}",
                context={"session_id": state.session_id},
            ) from exc

    async def update(self, state: WorkflowState) -> None:
        """
        Update all mutable fields of an existing research session.

        Args:
            state: The updated WorkflowState to persist.

        Raises:
            DatabaseException: On any database error.
        """
        try:
            now = datetime.now(timezone.utc)
            state = state.model_copy(update={"updated_at": now})

            await self._db.execute(
                """
                UPDATE research_sessions
                SET status           = ?,
                    current_stage    = ?,
                    plan_json        = ?,
                    results_json     = ?,
                    critic_json      = ?,
                    report_json      = ?,
                    error            = ?,
                    updated_at       = ?,
                    completed_at     = ?,
                    iteration        = ?,
                    evaluations_json = ?
                WHERE session_id = ?
                """,
                (
                    state.status,
                    state.current_stage.value,
                    state.plan.model_dump_json() if state.plan else None,
                    self._serialise_results(state.results),
                    state.critic_result.model_dump_json() if state.critic_result else None,
                    state.report.model_dump_json() if state.report else None,
                    state.error,
                    now.isoformat(),
                    state.completed_at.isoformat() if state.completed_at else None,
                    state.iteration,
                    self._serialise_evaluations(state.evaluations),
                    state.session_id,
                ),
            )
            await self._db.commit()
            logger.debug(
                "Research session updated",
                extra={
                    "session_id": state.session_id,
                    "status": state.status,
                    "stage": state.current_stage.value,
                },
            )
        except Exception as exc:
            raise DatabaseException(
                f"Failed to update research session: {exc}",
                context={"session_id": state.session_id},
            ) from exc

    async def get(self, session_id: str) -> WorkflowState:
        """
        Retrieve a research session by ID.

        Args:
            session_id: The session identifier.

        Returns:
            The fully deserialised WorkflowState.

        Raises:
            SessionNotFoundException: If no session exists with that ID.
            DatabaseException: On any database error.
        """
        try:
            cursor = await self._db.execute(
                "SELECT * FROM research_sessions WHERE session_id = ?",
                (session_id,),
            )
            row = await cursor.fetchone()
        except Exception as exc:
            raise DatabaseException(
                f"Failed to query research session: {exc}",
                context={"session_id": session_id},
            ) from exc

        if row is None:
            raise SessionNotFoundException(
                f"Research session '{session_id}' not found."
            )

        return self._row_to_state(dict(row))

    async def list_all(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> list[WorkflowState]:
        """
        Return a paginated list of research sessions (most recent first).

        Args:
            limit:  Maximum records to return.
            offset: Number of records to skip.

        Returns:
            List of WorkflowState objects.
        """
        try:
            cursor = await self._db.execute(
                """
                SELECT * FROM research_sessions
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )
            rows = await cursor.fetchall()
        except Exception as exc:
            raise DatabaseException(f"Failed to list research sessions: {exc}") from exc

        return [self._row_to_state(dict(r)) for r in rows]

    # ── Agent metrics ─────────────────────────────────────────────────────────

    async def record_agent_metric(
        self,
        *,
        session_id: str,
        agent_name: str,
        stage: str,
        execution_time: float,
        model_used: str | None,
        started_at: datetime,
        completed_at: datetime,
    ) -> None:
        """
        Persist a single agent execution timing record.

        Args:
            session_id:     Parent session identifier.
            agent_name:     Name of the agent (e.g. "planner_agent").
            stage:          WorkflowStage value string.
            execution_time: Wall-clock execution time in seconds.
            model_used:     Model name used for this agent call (if known).
            started_at:     UTC datetime when the agent started.
            completed_at:   UTC datetime when the agent completed.
        """
        try:
            await self._db.execute(
                """
                INSERT INTO agent_execution_metrics
                    (id, session_id, agent_name, stage,
                     execution_time, model_used, started_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    session_id,
                    agent_name,
                    stage,
                    execution_time,
                    model_used,
                    started_at.isoformat(),
                    completed_at.isoformat(),
                ),
            )
            await self._db.commit()
            logger.debug(
                "Agent metric recorded",
                extra={
                    "session_id": session_id,
                    "agent_name": agent_name,
                    "execution_time": round(execution_time, 3),
                },
            )
        except Exception as exc:
            # Best-effort — metric recording failure must not fail the session
            logger.warning(
                "Failed to record agent metric",
                extra={
                    "session_id": session_id,
                    "agent_name": agent_name,
                    "error": str(exc),
                },
            )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _serialise_results(results: list[ResearchResult]) -> str | None:
        """Serialise a list of ResearchResult to a JSON string."""
        if not results:
            return None
        import json
        return json.dumps([r.model_dump(mode="json") for r in results])

    @staticmethod
    def _deserialise_results(raw: str | None) -> list[ResearchResult]:
        """Deserialise a JSON string back to a list of ResearchResult."""
        if not raw:
            return []
        import json
        return [ResearchResult.model_validate(item) for item in json.loads(raw)]

    @staticmethod
    def _serialise_evaluations(evals: list[EvaluationResult]) -> str | None:
        """Serialise a list of EvaluationResult to a JSON string."""
        if not evals:
            return None
        import json
        return json.dumps([e.model_dump(mode="json") for e in evals])

    @staticmethod
    def _deserialise_evaluations(raw: str | None) -> list[EvaluationResult]:
        """Deserialise a JSON string back to a list of EvaluationResult."""
        if not raw:
            return []
        import json
        return [EvaluationResult.model_validate(item) for item in json.loads(raw)]

    @staticmethod
    def _row_to_state(row: dict[str, Any]) -> WorkflowState:
        """Convert a raw database row dict to a WorkflowState instance."""
        plan = (
            ResearchPlan.model_validate_json(row["plan_json"])
            if row.get("plan_json")
            else None
        )
        results = ResearchRepository._deserialise_results(row.get("results_json"))
        critic_result = (
            CriticResult.model_validate_json(row["critic_json"])
            if row.get("critic_json")
            else None
        )
        report = (
            ResearchReport.model_validate_json(row["report_json"])
            if row.get("report_json")
            else None
        )
        completed_at = (
            datetime.fromisoformat(row["completed_at"])
            if row.get("completed_at")
            else None
        )
        # Phase 4 fields — gracefully handle rows from pre-Phase-4 DBs
        iteration = int(row["iteration"]) if row.get("iteration") is not None else 1
        evaluations = ResearchRepository._deserialise_evaluations(
            row.get("evaluations_json")
        )

        return WorkflowState(
            session_id=row["session_id"],
            query=row["query"],
            status=row["status"],  # type: ignore[arg-type]
            current_stage=WorkflowStage(row["current_stage"]),
            plan=plan,
            results=results,
            critic_result=critic_result,
            report=report,
            error=row.get("error"),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            completed_at=completed_at,
            iteration=iteration,
            evaluations=evaluations,
        )
