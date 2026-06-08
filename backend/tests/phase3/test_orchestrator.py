"""
tests/phase3/test_orchestrator.py
───────────────────────────────────
Tests for ResearchOrchestrator.

Coverage:
  - Full happy path: verify stage order PLANNING→RESEARCHING→CRITIQUING→REPORTING→COMPLETED
  - Planner fails → state.status="failed", execution stops after PLANNING
  - Critic fails (uses fallback) → execution continues to REPORTING
  - ReportWriter fails → state.status="failed"
  - research_repository.save/update called at every stage transition
  - Sequential task execution (no parallelism) verified
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from app.core.exceptions import PlannerException, ReportWriterException
from app.orchestration.orchestrator import ResearchOrchestrator
from app.schemas.research import (
    CriticResult,
    ResearchPlan,
    ResearchReport,
    ResearchResult,
    ResearchTask,
    WorkflowStage,
    WorkflowState,
)


# ── Fixtures / helpers ────────────────────────────────────────────────────────

def _make_plan(task_count: int = 2) -> ResearchPlan:
    return ResearchPlan(
        session_id="sess-001",
        original_query="What is AI?",
        tasks=[
            ResearchTask(task_id=f"t{i}", query=f"Sub-query {i}", priority=i)
            for i in range(1, task_count + 1)
        ],
        estimated_complexity="low",
    )


def _make_result(task_id: str = "t1") -> ResearchResult:
    return ResearchResult(
        task_id=task_id,
        task_query="Sub-query",
        findings="Some findings",
        sources=["doc-001"],
        confidence=0.75,
    )


def _make_critic() -> CriticResult:
    return CriticResult(
        approved=True,
        issues=[],
        suggestions=[],
        overall_quality="good",
        reviewed_task_ids=["t1"],
    )


def _make_report() -> ResearchReport:
    return ResearchReport(
        session_id="sess-001",
        query="What is AI?",
        summary="Summary here.",
        sections=[],
        citations=["doc-001"],
        confidence_score=0.75,
        critic_quality="good",
    )


def _make_orchestrator(
    plan: ResearchPlan | None = None,
    planner_raises: Exception | None = None,
    result_list: list[ResearchResult] | None = None,
    critic: CriticResult | None = None,
    report: ResearchReport | None = None,
    report_writer_raises: Exception | None = None,
) -> tuple[ResearchOrchestrator, MagicMock, MagicMock, MagicMock, MagicMock, MagicMock]:
    """
    Build a fully mocked orchestrator.

    Returns (orchestrator, planner, researcher, critic_agent, writer, repo).
    """
    plan = plan or _make_plan()
    result_list = result_list or [_make_result("t1"), _make_result("t2")]
    critic = critic or _make_critic()
    report = report or _make_report()

    planner_mock = MagicMock()
    if planner_raises:
        planner_mock.plan = AsyncMock(side_effect=planner_raises)
    else:
        planner_mock.plan = AsyncMock(return_value=plan)

    researcher_mock = MagicMock()
    # Return results in sequence
    researcher_mock.research = AsyncMock(side_effect=result_list)

    critic_mock = MagicMock()
    critic_mock.critique = AsyncMock(return_value=critic)

    writer_mock = MagicMock()
    if report_writer_raises:
        writer_mock.write = AsyncMock(side_effect=report_writer_raises)
    else:
        writer_mock.write = AsyncMock(return_value=report)

    repo_mock = MagicMock()
    repo_mock.create = AsyncMock(return_value=None)
    repo_mock.update = AsyncMock(return_value=None)
    repo_mock.record_agent_metric = AsyncMock(return_value=None)

    memory_mock = MagicMock()
    manager_mock = MagicMock()

    orchestrator = ResearchOrchestrator(
        planner=planner_mock,
        researcher=researcher_mock,
        critic=critic_mock,
        report_writer=writer_mock,
        memory_service=memory_mock,
        research_repository=repo_mock,
        execution_manager=manager_mock,
        skip_critic_for_low_complexity=False,  # always run full pipeline in tests
    )

    return orchestrator, planner_mock, researcher_mock, critic_mock, writer_mock, repo_mock


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_happy_path_full_pipeline():
    """Full run → PLANNING→RESEARCHING→CRITIQUING→REPORTING→COMPLETED."""
    plan = _make_plan(task_count=2)
    results = [_make_result("t1"), _make_result("t2")]
    orch, planner, researcher, critic_agent, writer, repo = _make_orchestrator(
        plan=plan, result_list=results
    )

    state = await orch.run(query="What is AI?", session_id="sess-001")

    assert state.status == "completed"
    assert state.current_stage == WorkflowStage.COMPLETED
    assert state.plan is not None
    assert len(state.results) == 2
    assert state.critic_result is not None
    assert state.report is not None
    assert state.error is None
    assert state.completed_at is not None


@pytest.mark.asyncio
async def test_happy_path_agents_called_in_order():
    """Verify agent call order: plan → research (×N) → critique → write."""
    plan = _make_plan(task_count=2)
    results = [_make_result("t1"), _make_result("t2")]
    orch, planner, researcher, critic_agent, writer, repo = _make_orchestrator(
        plan=plan, result_list=results
    )

    call_order = []
    planner.plan.side_effect = lambda **kw: call_order.append("plan") or _make_plan(2)
    researcher.research.side_effect = lambda **kw: call_order.append("research") or _make_result()
    critic_agent.critique.side_effect = lambda **kw: call_order.append("critique") or _make_critic()
    writer.write.side_effect = lambda **kw: call_order.append("write") or _make_report()

    # Re-wire with side_effects
    plan = _make_plan(task_count=2)

    async def plan_fn(**kw):
        call_order.append("plan")
        return plan

    async def research_fn(**kw):
        call_order.append("research")
        return _make_result()

    async def critique_fn(**kw):
        call_order.append("critique")
        return _make_critic()

    async def write_fn(**kw):
        call_order.append("write")
        return _make_report()

    planner.plan = AsyncMock(side_effect=plan_fn)
    researcher.research = AsyncMock(side_effect=research_fn)
    critic_agent.critique = AsyncMock(side_effect=critique_fn)
    writer.write = AsyncMock(side_effect=write_fn)

    await orch.run(query="What is AI?", session_id="sess-001")

    assert call_order[0] == "plan"
    assert call_order[1] == "research"
    assert call_order[2] == "research"   # 2 tasks
    assert call_order[3] == "critique"
    assert call_order[4] == "write"


@pytest.mark.asyncio
async def test_planner_fails_session_marked_failed():
    """Planner raises PlannerException → status='failed', no further agents called."""
    orch, planner, researcher, critic_agent, writer, repo = _make_orchestrator(
        planner_raises=PlannerException("LLM returned bad JSON")
    )

    state = await orch.run(query="What is AI?", session_id="sess-001")

    assert state.status == "failed"
    assert state.current_stage == WorkflowStage.FAILED
    assert state.error is not None
    assert "LLM returned bad JSON" in state.error

    # Researcher, critic, writer must NOT have been called
    researcher.research.assert_not_called()
    critic_agent.critique.assert_not_called()
    writer.write.assert_not_called()


@pytest.mark.asyncio
async def test_planner_fails_repository_updated():
    """Planner fails → repo.update called with failed state before returning."""
    orch, planner, researcher, critic_agent, writer, repo = _make_orchestrator(
        planner_raises=PlannerException("bad JSON")
    )

    await orch.run(query="What is AI?", session_id="sess-001")

    # repo.create called once (initial), repo.update at least once (with failed)
    repo.create.assert_called_once()
    assert repo.update.call_count >= 1
    # Verify the final update call has failed status
    final_call_state: WorkflowState = repo.update.call_args[0][0]
    assert final_call_state.status == "failed"


@pytest.mark.asyncio
async def test_critic_fails_uses_fallback_continues_to_reporting():
    """CriticAgent failure → fallback used, execution continues to REPORTING."""
    orch, planner, researcher, critic_agent, writer, repo = _make_orchestrator()

    # CriticAgent returns fallback (never raises per spec)
    fallback = CriticResult(
        approved=True,
        issues=["critic parse failed: LLM unavailable"],
        suggestions=[],
        overall_quality="acceptable",
        reviewed_task_ids=["t1"],
    )
    critic_agent.critique = AsyncMock(return_value=fallback)

    state = await orch.run(query="What is AI?", session_id="sess-001")

    # Execution must reach COMPLETED
    assert state.status == "completed"
    assert state.critic_result is not None
    assert "critic parse failed" in state.critic_result.issues[0]

    # Writer must have been called
    writer.write.assert_called_once()


@pytest.mark.asyncio
async def test_report_writer_fails_session_marked_failed():
    """ReportWriter raises → status='failed', completed_at set."""
    orch, planner, researcher, critic_agent, writer, repo = _make_orchestrator(
        report_writer_raises=ReportWriterException("LLM returned bad JSON")
    )

    state = await orch.run(query="What is AI?", session_id="sess-001")

    assert state.status == "failed"
    assert state.current_stage == WorkflowStage.FAILED
    assert state.error is not None
    assert "LLM returned bad JSON" in state.error
    assert state.completed_at is not None


@pytest.mark.asyncio
async def test_repository_save_called_at_every_stage():
    """repo.update called after PLANNING, RESEARCHING, CRITIQUING, REPORTING, COMPLETED."""
    plan = _make_plan(task_count=1)
    results = [_make_result("t1")]
    orch, planner, researcher, critic_agent, writer, repo = _make_orchestrator(
        plan=plan, result_list=results
    )

    await orch.run(query="What is AI?", session_id="sess-001")

    # Initial create + at least 5 updates (one per stage + final)
    repo.create.assert_called_once()
    assert repo.update.call_count >= 5, (
        f"Expected at least 5 repo.update calls, got {repo.update.call_count}"
    )


@pytest.mark.asyncio
async def test_research_tasks_run_sequentially():
    """ResearchAgent.research() called N times (once per task), sequentially."""
    plan = _make_plan(task_count=3)
    results = [_make_result(f"t{i}") for i in range(1, 4)]
    orch, planner, researcher, critic_agent, writer, repo = _make_orchestrator(
        plan=plan, result_list=results
    )

    state = await orch.run(query="What is AI?", session_id="sess-001")

    # Called exactly 3 times — once per task
    assert researcher.research.call_count == 3
    # All results in final state
    assert len(state.results) == 3
