"""
tests/phase3/test_research_repository.py
──────────────────────────────────────────
Tests for ResearchRepository.

Coverage:
  - create session → persists correctly (SQL called)
  - get session → returns WorkflowState with deserialized models
  - session not found → raises SessionNotFoundException
  - update session → updated_at changes
  - list_all → returns list of states
  - record_agent_metric → best-effort, no exception on DB failure

All tests use an in-memory aiosqlite database for real I/O.
"""

from __future__ import annotations

from datetime import datetime, timezone

import aiosqlite
import pytest

from app.core.exceptions import SessionNotFoundException
from app.db.database import (
    _CREATE_AGENT_EXECUTION_METRICS_TABLE,
    _CREATE_RESEARCH_SESSIONS_TABLE,
    _CREATE_AGENT_METRICS_IDX,
    _CREATE_RESEARCH_SESSIONS_IDX,
)
from app.db.research_repository import ResearchRepository
from app.schemas.research import (
    CriticResult,
    ResearchPlan,
    ResearchReport,
    ResearchResult,
    ResearchTask,
    WorkflowStage,
    WorkflowState,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
async def db():
    """Provide an in-memory aiosqlite connection with Phase 3 tables created."""
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


def _make_state(session_id: str = "sess-001") -> WorkflowState:
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
    )


def _make_plan(session_id: str = "sess-001") -> ResearchPlan:
    return ResearchPlan(
        session_id=session_id,
        original_query="What is AI?",
        tasks=[ResearchTask(task_id="t1", query="Define AI", priority=1)],
        estimated_complexity="low",
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_persists_session(repo, db):
    """create() inserts a record into the database."""
    state = _make_state("s1")
    await repo.create(state)

    cursor = await db.execute(
        "SELECT session_id FROM research_sessions WHERE session_id = ?", ("s1",)
    )
    row = await cursor.fetchone()
    assert row is not None
    assert row["session_id"] == "s1"


@pytest.mark.asyncio
async def test_get_returns_workflow_state(repo):
    """get() returns a fully deserialised WorkflowState."""
    state = _make_state("s2")
    await repo.create(state)

    retrieved = await repo.get("s2")

    assert isinstance(retrieved, WorkflowState)
    assert retrieved.session_id == "s2"
    assert retrieved.query == "What is AI?"
    assert retrieved.status == "running"
    assert retrieved.current_stage == WorkflowStage.PLANNING


@pytest.mark.asyncio
async def test_get_raises_session_not_found(repo):
    """get() with unknown session_id → raises SessionNotFoundException."""
    with pytest.raises(SessionNotFoundException):
        await repo.get("nonexistent-session")


@pytest.mark.asyncio
async def test_update_persists_changes(repo):
    """update() changes status and stage in DB."""
    state = _make_state("s3")
    await repo.create(state)

    updated_state = state.model_copy(
        update={
            "status": "completed",
            "current_stage": WorkflowStage.COMPLETED,
        }
    )
    await repo.update(updated_state)

    retrieved = await repo.get("s3")
    assert retrieved.status == "completed"
    assert retrieved.current_stage == WorkflowStage.COMPLETED


@pytest.mark.asyncio
async def test_update_persists_plan(repo):
    """update() with a plan → plan deserialised correctly on get()."""
    state = _make_state("s4")
    await repo.create(state)

    plan = _make_plan("s4")
    updated = state.model_copy(update={"plan": plan, "current_stage": WorkflowStage.RESEARCHING})
    await repo.update(updated)

    retrieved = await repo.get("s4")
    assert retrieved.plan is not None
    assert retrieved.plan.session_id == "s4"
    assert len(retrieved.plan.tasks) == 1
    assert retrieved.plan.tasks[0].task_id == "t1"


@pytest.mark.asyncio
async def test_update_persists_results(repo):
    """update() with results → results deserialised correctly on get()."""
    state = _make_state("s5")
    await repo.create(state)

    results = [
        ResearchResult(
            task_id="t1", task_query="Q?", findings="F",
            sources=["doc-001"], confidence=0.8
        )
    ]
    updated = state.model_copy(update={"results": results})
    await repo.update(updated)

    retrieved = await repo.get("s5")
    assert len(retrieved.results) == 1
    assert retrieved.results[0].task_id == "t1"
    assert retrieved.results[0].confidence == 0.8


@pytest.mark.asyncio
async def test_list_all_returns_sessions(repo):
    """list_all() returns all sessions in reverse chronological order."""
    await repo.create(_make_state("sA"))
    await repo.create(_make_state("sB"))
    await repo.create(_make_state("sC"))

    states = await repo.list_all(limit=10, offset=0)

    assert len(states) >= 3
    ids = [s.session_id for s in states]
    assert "sA" in ids
    assert "sB" in ids
    assert "sC" in ids


@pytest.mark.asyncio
async def test_list_all_respects_limit(repo):
    """list_all() respects the limit parameter."""
    for i in range(5):
        await repo.create(_make_state(f"limit-{i}"))

    states = await repo.list_all(limit=2, offset=0)
    assert len(states) == 2


@pytest.mark.asyncio
async def test_record_agent_metric_persists(repo, db):
    """record_agent_metric() inserts a record into agent_execution_metrics."""
    state = _make_state("s-metric")
    await repo.create(state)

    now = datetime.now(timezone.utc)
    await repo.record_agent_metric(
        session_id="s-metric",
        agent_name="planner_agent",
        stage="planning",
        execution_time=1.23,
        model_used="llama3",
        started_at=now,
        completed_at=now,
    )

    cursor = await db.execute(
        "SELECT * FROM agent_execution_metrics WHERE session_id = ?",
        ("s-metric",),
    )
    row = await cursor.fetchone()
    assert row is not None
    assert row["agent_name"] == "planner_agent"
    assert abs(row["execution_time"] - 1.23) < 0.001


@pytest.mark.asyncio
async def test_get_with_critic_result_and_report(repo):
    """Fully populated WorkflowState round-trips through create/get."""
    state = _make_state("s-full")
    await repo.create(state)

    plan = _make_plan("s-full")
    results = [
        ResearchResult(task_id="t1", task_query="Q", findings="F",
                       sources=[], confidence=0.9)
    ]
    critic = CriticResult(
        approved=True, issues=[], suggestions=[],
        overall_quality="good", reviewed_task_ids=["t1"]
    )
    report = ResearchReport(
        session_id="s-full", query="What is AI?", summary="Summary",
        sections=[], citations=[], confidence_score=0.9, critic_quality="good"
    )

    full_state = state.model_copy(
        update={
            "plan": plan,
            "results": results,
            "critic_result": critic,
            "report": report,
            "status": "completed",
            "current_stage": WorkflowStage.COMPLETED,
        }
    )
    await repo.update(full_state)

    retrieved = await repo.get("s-full")
    assert retrieved.plan is not None
    assert len(retrieved.results) == 1
    assert retrieved.critic_result is not None
    assert retrieved.report is not None
    assert retrieved.report.summary == "Summary"
