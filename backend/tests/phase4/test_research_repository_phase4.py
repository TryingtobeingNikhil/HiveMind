"""
tests/phase4/test_research_repository_phase4.py
─────────────────────────────────────────────────
Tests for ResearchRepository Phase 4 additions.

Coverage:
  - create() with iteration=2 and evaluations persists correctly
  - update() with evaluations persists correctly
  - get() deserialises iteration and evaluations correctly
  - evaluations_json=NULL → returns empty list (not error)
  - iteration column defaults to 1 on missing value

All tests use an in-memory aiosqlite database for real I/O.
"""

from __future__ import annotations

from datetime import datetime, timezone

import aiosqlite
import pytest

from app.core.exceptions import SessionNotFoundException
from app.db.database import (
    _CREATE_AGENT_EXECUTION_METRICS_TABLE,
    _CREATE_AGENT_METRICS_IDX,
    _CREATE_RESEARCH_SESSIONS_IDX,
    _CREATE_RESEARCH_SESSIONS_TABLE,
)
from app.db.research_repository import ResearchRepository
from app.schemas.research import (
    EvaluationResult,
    ResearchPlan,
    ResearchResult,
    ResearchTask,
    WorkflowStage,
    WorkflowState,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
async def db():
    """Provide an in-memory aiosqlite connection with Phase 3 + Phase 4 tables."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row

    for ddl in [
        _CREATE_RESEARCH_SESSIONS_TABLE,
        _CREATE_AGENT_EXECUTION_METRICS_TABLE,
        _CREATE_RESEARCH_SESSIONS_IDX,
        _CREATE_AGENT_METRICS_IDX,
    ]:
        await conn.executescript(ddl)
    await conn.commit()

    yield conn
    await conn.close()


@pytest.fixture
def repo(db) -> ResearchRepository:
    return ResearchRepository(db)


def _make_state(
    session_id: str = "sess-001",
    iteration: int = 1,
    evaluations: list[EvaluationResult] | None = None,
) -> WorkflowState:
    now = datetime.now(timezone.utc)
    return WorkflowState(
        session_id=session_id,
        query="What is AI?",
        status="running",
        current_stage=WorkflowStage.PLANNING,
        plan=None,
        results=[],
        critic_result=None,
        report=None,
        error=None,
        created_at=now,
        updated_at=now,
        completed_at=None,
        iteration=iteration,
        evaluations=evaluations or [],
    )


def _make_evaluation(
    session_id: str = "sess-001",
    iteration: int = 1,
    sufficient: bool = False,
    confidence: float = 0.3,
) -> EvaluationResult:
    gap_task = ResearchTask(
        task_id="gap-t1",
        query="Missing topic A",
        priority=2,
    )
    return EvaluationResult(
        session_id=session_id,
        iteration=iteration,
        sufficient=sufficient,
        confidence_score=confidence,
        confidence_threshold=0.5,
        gaps=["Missing topic A"],
        gap_tasks=[gap_task],
        reasoning="Research gaps found.",
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_with_iteration_persists_correctly(repo, db):
    """create() with iteration=2 stores it in the database."""
    state = _make_state("s1", iteration=2)
    await repo.create(state)

    cursor = await db.execute(
        "SELECT iteration FROM research_sessions WHERE session_id = ?", ("s1",)
    )
    row = await cursor.fetchone()
    assert row is not None
    assert row["iteration"] == 2


@pytest.mark.asyncio
async def test_create_with_evaluations_persists_correctly(repo):
    """create() with evaluations list stores them as JSON and round-trips correctly."""
    eval_result = _make_evaluation("s2", iteration=1)
    state = _make_state("s2", iteration=1, evaluations=[eval_result])
    await repo.create(state)

    retrieved = await repo.get("s2")
    assert len(retrieved.evaluations) == 1
    assert retrieved.evaluations[0].session_id == "s2"
    assert retrieved.evaluations[0].iteration == 1
    assert retrieved.evaluations[0].sufficient is False
    assert retrieved.evaluations[0].confidence_score == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_update_with_evaluations_persists_correctly(repo):
    """update() with evaluations persists correctly and replaces existing list."""
    state = _make_state("s3", iteration=1)
    await repo.create(state)

    eval_result = _make_evaluation("s3", iteration=1)
    updated = state.model_copy(
        update={
            "iteration": 1,
            "evaluations": [eval_result],
            "status": "completed",
            "current_stage": WorkflowStage.COMPLETED,
        }
    )
    await repo.update(updated)

    retrieved = await repo.get("s3")
    assert len(retrieved.evaluations) == 1
    assert retrieved.evaluations[0].gaps == ["Missing topic A"]
    assert len(retrieved.evaluations[0].gap_tasks) == 1
    assert retrieved.evaluations[0].gap_tasks[0].task_id == "gap-t1"


@pytest.mark.asyncio
async def test_get_deserialises_iteration_correctly(repo):
    """get() returns correct iteration value from database."""
    state = _make_state("s4", iteration=3)
    await repo.create(state)

    retrieved = await repo.get("s4")
    assert retrieved.iteration == 3


@pytest.mark.asyncio
async def test_get_with_null_evaluations_returns_empty_list(repo, db):
    """evaluations_json=NULL in DB → get() returns [] not error."""
    state = _make_state("s5", iteration=1, evaluations=[])
    await repo.create(state)

    # Manually set evaluations_json to NULL to simulate a pre-Phase-4 row
    await db.execute(
        "UPDATE research_sessions SET evaluations_json = NULL WHERE session_id = ?",
        ("s5",),
    )
    await db.commit()

    retrieved = await repo.get("s5")
    assert retrieved.evaluations == []
    assert isinstance(retrieved.evaluations, list)


@pytest.mark.asyncio
async def test_get_default_iteration_is_1(repo, db):
    """Rows without iteration column (or with DEFAULT) return iteration=1."""
    state = _make_state("s6", iteration=1)
    await repo.create(state)

    retrieved = await repo.get("s6")
    assert retrieved.iteration == 1


@pytest.mark.asyncio
async def test_evaluations_gap_tasks_serialise_correctly(repo):
    """EvaluationResult.gap_tasks round-trip through serialisation."""
    gap_task_1 = ResearchTask(task_id="gt-1", query="What is deep learning?", priority=2)
    gap_task_2 = ResearchTask(task_id="gt-2", query="How do LLMs work?", priority=2)
    eval_result = EvaluationResult(
        session_id="s7",
        iteration=1,
        sufficient=False,
        confidence_score=0.2,
        confidence_threshold=0.5,
        gaps=["deep learning", "LLMs"],
        gap_tasks=[gap_task_1, gap_task_2],
        reasoning="Major gaps found.",
    )
    state = _make_state("s7", iteration=1, evaluations=[eval_result])
    await repo.create(state)

    retrieved = await repo.get("s7")
    assert len(retrieved.evaluations) == 1
    assert len(retrieved.evaluations[0].gap_tasks) == 2

    retrieved_tasks = {t.task_id: t for t in retrieved.evaluations[0].gap_tasks}
    assert "gt-1" in retrieved_tasks
    assert retrieved_tasks["gt-1"].query == "What is deep learning?"
    assert retrieved_tasks["gt-1"].priority == 2
    assert "gt-2" in retrieved_tasks


@pytest.mark.asyncio
async def test_multiple_evaluations_round_trip(repo):
    """Multiple EvaluationResults in the list all survive a round-trip."""
    evals = [
        _make_evaluation("s8", iteration=1, sufficient=False, confidence=0.2),
        EvaluationResult(
            session_id="s8",
            iteration=2,
            sufficient=True,
            confidence_score=0.8,
            confidence_threshold=0.5,
            gaps=[],
            gap_tasks=[],
            reasoning="Threshold met on iteration 2.",
        ),
    ]
    state = _make_state("s8", iteration=2, evaluations=evals)
    await repo.create(state)

    retrieved = await repo.get("s8")
    assert len(retrieved.evaluations) == 2
    assert retrieved.evaluations[0].sufficient is False
    assert retrieved.evaluations[1].sufficient is True
    assert retrieved.evaluations[1].reasoning == "Threshold met on iteration 2."


@pytest.mark.asyncio
async def test_create_and_update_iteration_change(repo):
    """Updating iteration from 1 → 2 is persisted correctly."""
    state = _make_state("s9", iteration=1)
    await repo.create(state)

    updated = state.model_copy(update={"iteration": 2})
    await repo.update(updated)

    retrieved = await repo.get("s9")
    assert retrieved.iteration == 2
