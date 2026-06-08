"""
tests/phase4/test_orchestrator_loop.py
───────────────────────────────────────
Tests for the Phase 4 evaluation loop in ResearchOrchestrator.

Coverage:
  - evaluator_enabled=False → loop runs once, no EvaluatorAgent call
  - max_iterations=1 → loop runs once regardless of evaluation
  - sufficient=True on iteration 1 → loop exits after iteration 1
  - sufficient=False on iteration 1, max_iterations=2 → loop runs iteration 2 with gap_tasks
  - sufficient=False, max_iterations reached → session still completes
  - PLANNING called exactly once across all iterations
  - state.evaluations has one entry per completed iteration
  - state.results accumulates across iterations (not reset)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.evaluator_agent import EvaluatorAgent
from app.core.config import Settings
from app.core.exceptions import PlannerException, ReportWriterException
from app.orchestration.orchestrator import ResearchOrchestrator
from app.schemas.research import (
    CriticResult,
    EvaluationResult,
    ResearchPlan,
    ResearchReport,
    ResearchResult,
    ResearchTask,
    WorkflowStage,
    WorkflowState,
)


# ── Fixtures / helpers ────────────────────────────────────────────────────────


def _make_settings(
    max_iterations: int = 2,
    threshold: float = 0.5,
    evaluator_enabled: bool = True,
) -> Settings:
    """Build a Settings instance with Phase 4 fields overridden."""
    return Settings(
        APP_ENV="testing",
        MAX_RESEARCH_ITERATIONS=max_iterations,
        CONFIDENCE_THRESHOLD=threshold,
        EVALUATOR_ENABLED=evaluator_enabled,
    )


def _make_plan(
    session_id: str = "sess-001",
    task_count: int = 2,
    complexity: str = "medium",
) -> ResearchPlan:
    return ResearchPlan(
        session_id=session_id,
        original_query="What is AI?",
        tasks=[
            ResearchTask(task_id=f"t{i}", query=f"Sub-query {i}", priority=i)
            for i in range(1, task_count + 1)
        ],
        estimated_complexity=complexity,  # type: ignore[arg-type]
    )


def _make_result(task_id: str = "t1", confidence: float = 0.75) -> ResearchResult:
    return ResearchResult(
        task_id=task_id,
        task_query="Sub-query",
        findings="Some findings",
        sources=["doc-001"],
        confidence=confidence,
    )


def _make_critic() -> CriticResult:
    return CriticResult(
        approved=True,
        issues=[],
        suggestions=[],
        overall_quality="good",
        reviewed_task_ids=["t1"],
    )


def _make_report(session_id: str = "sess-001") -> ResearchReport:
    return ResearchReport(
        session_id=session_id,
        query="What is AI?",
        summary="Summary here.",
        sections=[],
        citations=["doc-001"],
        confidence_score=0.75,
        critic_quality="good",
    )


def _make_eval_result(
    session_id: str = "sess-001",
    iteration: int = 1,
    sufficient: bool = True,
    confidence: float = 0.75,
    threshold: float = 0.5,
    gaps: list[str] | None = None,
    gap_tasks: list[ResearchTask] | None = None,
) -> EvaluationResult:
    return EvaluationResult(
        session_id=session_id,
        iteration=iteration,
        sufficient=sufficient,
        confidence_score=confidence,
        confidence_threshold=threshold,
        gaps=gaps or [],
        gap_tasks=gap_tasks or [],
        reasoning="Test reasoning.",
    )


def _make_orchestrator(
    plan: ResearchPlan | None = None,
    result_list: list[ResearchResult] | None = None,
    critic: CriticResult | None = None,
    report: ResearchReport | None = None,
    eval_results: list[EvaluationResult] | None = None,
    planner_raises: Exception | None = None,
    report_writer_raises: Exception | None = None,
    settings: Settings | None = None,
    evaluator_enabled: bool = True,
) -> tuple[ResearchOrchestrator, MagicMock, MagicMock, MagicMock, MagicMock, MagicMock, MagicMock]:
    """
    Build a fully mocked orchestrator with Phase 4 support.

    Returns (orchestrator, planner, researcher, critic_agent, writer, evaluator, repo).
    """
    cfg = settings or _make_settings(evaluator_enabled=evaluator_enabled)
    plan = plan or _make_plan()
    result_list = result_list or [_make_result("t1"), _make_result("t2")]
    critic = critic or _make_critic()
    report = report or _make_report()
    eval_results = eval_results or [_make_eval_result(sufficient=True)]

    planner_mock = MagicMock()
    if planner_raises:
        planner_mock.plan = AsyncMock(side_effect=planner_raises)
    else:
        planner_mock.plan = AsyncMock(return_value=plan)

    researcher_mock = MagicMock()
    researcher_mock.research = AsyncMock(side_effect=result_list)

    critic_mock = MagicMock()
    critic_mock.critique = AsyncMock(return_value=critic)

    writer_mock = MagicMock()
    if report_writer_raises:
        writer_mock.write = AsyncMock(side_effect=report_writer_raises)
    else:
        writer_mock.write = AsyncMock(return_value=report)

    evaluator_mock = MagicMock(spec=EvaluatorAgent)
    evaluator_mock.evaluate = AsyncMock(side_effect=eval_results)

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
        evaluator=evaluator_mock,
        settings=cfg,
    )

    return (
        orchestrator,
        planner_mock,
        researcher_mock,
        critic_mock,
        writer_mock,
        evaluator_mock,
        repo_mock,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_evaluator_disabled_loop_runs_once_no_evaluator_call():
    """evaluator_enabled=False → one iteration, EvaluatorAgent.evaluate NOT called."""
    cfg = _make_settings(evaluator_enabled=False, max_iterations=2)
    orch, planner, researcher, critic_agent, writer, evaluator, repo = _make_orchestrator(
        settings=cfg,
        eval_results=[_make_eval_result(sufficient=True)],
    )

    state = await orch.run(query="What is AI?", session_id="sess-001")

    assert state.status == "completed"
    evaluator.evaluate.assert_not_called()
    planner.plan.assert_called_once()


@pytest.mark.asyncio
async def test_max_iterations_one_runs_once_regardless_of_evaluation():
    """max_iterations=1 → loop runs exactly once, evaluator called once, session completes."""
    cfg = _make_settings(max_iterations=1, threshold=0.9, evaluator_enabled=True)
    # evaluator returns sufficient=False but max_iterations prevents a second pass
    eval_result = _make_eval_result(sufficient=False, confidence=0.1, threshold=0.9)
    orch, planner, researcher, critic_agent, writer, evaluator, repo = _make_orchestrator(
        settings=cfg,
        eval_results=[eval_result],
    )

    state = await orch.run(query="What is AI?", session_id="sess-001")

    assert state.status == "completed"
    evaluator.evaluate.assert_called_once()
    planner.plan.assert_called_once()
    assert len(state.evaluations) == 1


@pytest.mark.asyncio
async def test_sufficient_on_iteration_1_loop_exits():
    """sufficient=True on first evaluation → loop exits, session completes."""
    cfg = _make_settings(max_iterations=3, threshold=0.5, evaluator_enabled=True)
    eval_result = _make_eval_result(sufficient=True, confidence=0.8, threshold=0.5)
    orch, planner, researcher, critic_agent, writer, evaluator, repo = _make_orchestrator(
        settings=cfg,
        eval_results=[eval_result],
    )

    state = await orch.run(query="What is AI?", session_id="sess-001")

    assert state.status == "completed"
    evaluator.evaluate.assert_called_once()
    planner.plan.assert_called_once()
    assert len(state.evaluations) == 1
    assert state.evaluations[0].sufficient is True


@pytest.mark.asyncio
async def test_insufficient_on_iteration_1_runs_iteration_2_with_gap_tasks():
    """sufficient=False on iteration 1, max_iterations=2 → loop runs iteration 2 with gap_tasks."""
    gap_task = ResearchTask(task_id="gap-t1", query="Missing topic A", priority=2)
    eval_result_1 = _make_eval_result(
        sufficient=False,
        confidence=0.1,
        threshold=0.5,
        gaps=["Missing topic A"],
        gap_tasks=[gap_task],
    )
    eval_result_2 = _make_eval_result(
        iteration=2,
        sufficient=True,
        confidence=0.8,
        threshold=0.5,
    )

    cfg = _make_settings(max_iterations=2, threshold=0.5, evaluator_enabled=True)
    plan = _make_plan(task_count=2)

    # researcher returns a result for each task called
    results_sequence = [
        _make_result("t1", 0.1),
        _make_result("t2", 0.1),
        _make_result("gap-t1", 0.9),  # iteration 2 gap task
    ]

    orch, planner, researcher, critic_agent, writer, evaluator, repo = _make_orchestrator(
        plan=plan,
        result_list=results_sequence,
        settings=cfg,
        eval_results=[eval_result_1, eval_result_2],
    )

    state = await orch.run(query="What is AI?", session_id="sess-001")

    assert state.status == "completed"
    assert planner.plan.call_count == 1  # PLANNING exactly once
    assert evaluator.evaluate.call_count == 2  # once per iteration
    assert len(state.evaluations) == 2
    assert researcher.research.call_count == 3  # 2 original + 1 gap


@pytest.mark.asyncio
async def test_results_accumulate_across_iterations():
    """state.results accumulates — previous findings never discarded on iteration 2."""
    gap_task = ResearchTask(task_id="gap-t1", query="Gap topic", priority=2)
    eval_result_1 = _make_eval_result(
        sufficient=False,
        confidence=0.1,
        threshold=0.5,
        gaps=["Gap topic"],
        gap_tasks=[gap_task],
    )
    eval_result_2 = _make_eval_result(iteration=2, sufficient=True, confidence=0.8)

    cfg = _make_settings(max_iterations=2, threshold=0.5, evaluator_enabled=True)
    plan = _make_plan(task_count=1)

    results_sequence = [
        _make_result("t1", 0.1),       # iteration 1
        _make_result("gap-t1", 0.9),   # iteration 2 gap
    ]

    orch, _, researcher, _, _, evaluator, _ = _make_orchestrator(
        plan=plan,
        result_list=results_sequence,
        settings=cfg,
        eval_results=[eval_result_1, eval_result_2],
    )

    state = await orch.run(query="What is AI?", session_id="sess-001")

    # Both iteration 1 and iteration 2 results must be present
    assert len(state.results) == 2
    task_ids = {r.task_id for r in state.results}
    assert "t1" in task_ids
    assert "gap-t1" in task_ids


@pytest.mark.asyncio
async def test_planning_called_exactly_once():
    """PLANNING is called exactly once, regardless of how many loop iterations run."""
    gap_task = ResearchTask(task_id="gap-t1", query="Gap topic", priority=2)
    eval_result_1 = _make_eval_result(
        sufficient=False, confidence=0.1, threshold=0.5,
        gaps=["Gap topic"], gap_tasks=[gap_task],
    )
    eval_result_2 = _make_eval_result(iteration=2, sufficient=True, confidence=0.9)

    cfg = _make_settings(max_iterations=2, threshold=0.5, evaluator_enabled=True)
    plan = _make_plan(task_count=1)
    results_sequence = [_make_result("t1", 0.1), _make_result("gap-t1", 0.9)]

    orch, planner, _, _, _, evaluator, _ = _make_orchestrator(
        plan=plan,
        result_list=results_sequence,
        settings=cfg,
        eval_results=[eval_result_1, eval_result_2],
    )

    await orch.run(query="What is AI?", session_id="sess-001")

    planner.plan.assert_called_once()


@pytest.mark.asyncio
async def test_max_iterations_reached_session_still_completes():
    """sufficient=False across all iterations → loop ends at max, session still completes."""
    gap_task = ResearchTask(task_id="gap-t1", query="Gap topic", priority=2)
    # Both evaluations return sufficient=False
    eval_result_1 = _make_eval_result(
        sufficient=False, confidence=0.1, threshold=0.9,
        gaps=["Gap topic"], gap_tasks=[gap_task],
    )
    eval_result_2 = _make_eval_result(
        iteration=2, sufficient=False, confidence=0.2, threshold=0.9,
    )

    cfg = _make_settings(max_iterations=2, threshold=0.9, evaluator_enabled=True)
    plan = _make_plan(task_count=1)
    results_sequence = [_make_result("t1", 0.1), _make_result("gap-t1", 0.2)]

    orch, _, _, _, _, evaluator, _ = _make_orchestrator(
        plan=plan,
        result_list=results_sequence,
        settings=cfg,
        eval_results=[eval_result_1, eval_result_2],
    )

    state = await orch.run(query="What is AI?", session_id="sess-001")

    # Session must still complete — never stuck
    assert state.status == "completed"
    assert evaluator.evaluate.call_count == 2
    assert len(state.evaluations) == 2


@pytest.mark.asyncio
async def test_evaluations_list_has_one_entry_per_completed_iteration():
    """state.evaluations has exactly one EvaluationResult per iteration."""
    gap_task = ResearchTask(task_id="gap-t1", query="Gap", priority=2)
    eval_results = [
        _make_eval_result(iteration=1, sufficient=False, confidence=0.1,
                          gaps=["Gap"], gap_tasks=[gap_task]),
        _make_eval_result(iteration=2, sufficient=True, confidence=0.8),
    ]

    cfg = _make_settings(max_iterations=2, threshold=0.5, evaluator_enabled=True)
    plan = _make_plan(task_count=1)
    results_sequence = [_make_result("t1", 0.1), _make_result("gap-t1", 0.8)]

    orch, _, _, _, _, _, _ = _make_orchestrator(
        plan=plan,
        result_list=results_sequence,
        settings=cfg,
        eval_results=eval_results,
    )

    state = await orch.run(query="What is AI?", session_id="sess-001")

    assert len(state.evaluations) == 2
    assert state.evaluations[0].iteration == 1
    assert state.evaluations[1].iteration == 2


@pytest.mark.asyncio
async def test_phase3_behaviour_preserved_no_evaluator():
    """
    Orchestrator constructed without evaluator/settings (Phase 3 style)
    still completes successfully.
    """
    plan = _make_plan(task_count=1)
    result_list = [_make_result("t1")]
    critic = _make_critic()
    report = _make_report()

    planner_mock = MagicMock()
    planner_mock.plan = AsyncMock(return_value=plan)
    researcher_mock = MagicMock()
    researcher_mock.research = AsyncMock(side_effect=result_list)
    critic_mock = MagicMock()
    critic_mock.critique = AsyncMock(return_value=critic)
    writer_mock = MagicMock()
    writer_mock.write = AsyncMock(return_value=report)
    repo_mock = MagicMock()
    repo_mock.create = AsyncMock(return_value=None)
    repo_mock.update = AsyncMock(return_value=None)
    repo_mock.record_agent_metric = AsyncMock(return_value=None)

    # Phase 3 constructor style — no evaluator, no settings
    orch = ResearchOrchestrator(
        planner=planner_mock,
        researcher=researcher_mock,
        critic=critic_mock,
        report_writer=writer_mock,
        memory_service=MagicMock(),
        research_repository=repo_mock,
        execution_manager=MagicMock(),
    )

    state = await orch.run(query="What is AI?", session_id="sess-p3")

    assert state.status == "completed"
    assert len(state.evaluations) == 0


@pytest.mark.asyncio
async def test_planner_fails_still_works_in_phase4():
    """PlannerAgent failure → session=failed, loop never entered."""
    cfg = _make_settings(max_iterations=2, evaluator_enabled=True)
    orch, planner, researcher, critic_agent, writer, evaluator, repo = _make_orchestrator(
        settings=cfg,
        planner_raises=PlannerException("bad JSON"),
    )

    state = await orch.run(query="What is AI?", session_id="sess-001")

    assert state.status == "failed"
    evaluator.evaluate.assert_not_called()
    researcher.research.assert_not_called()
