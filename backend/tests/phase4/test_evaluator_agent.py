"""
tests/phase4/test_evaluator_agent.py
──────────────────────────────────────
Tests for EvaluatorAgent.

Coverage:
  - confidence >= threshold → sufficient=True, no LLM call made
  - confidence < threshold, valid LLM response → sufficient=False, gap_tasks populated
  - confidence < threshold, LLM parse fails → sufficient=True (failsafe)
  - confidence < threshold, LLM raises → sufficient=True (failsafe)
  - gap_tasks have new uuid4 task_ids (not reusing existing ones)
  - empty results list → confidence=0.0
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.evaluator_agent import EvaluatorAgent
from app.schemas.research import (
    EvaluationResult,
    ResearchReport,
    ResearchResult,
    ResearchTask,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_result(confidence: float = 0.8, task_id: str = "t1") -> ResearchResult:
    return ResearchResult(
        task_id=task_id,
        task_query="What is AI?",
        findings="AI is artificial intelligence.",
        sources=["doc-001"],
        confidence=confidence,
    )


def _make_report(session_id: str = "sess-001") -> ResearchReport:
    return ResearchReport(
        session_id=session_id,
        query="What is AI?",
        summary="AI is a broad field.",
        sections=[],
        citations=[],
        confidence_score=0.8,
        critic_quality="good",
    )


def _make_agent(
    llm_response: str | None = None,
    llm_raises: Exception | None = None,
) -> tuple[EvaluatorAgent, MagicMock]:
    """Build an EvaluatorAgent with a mocked LLM provider + execution manager."""
    provider_mock = MagicMock()

    manager_mock = MagicMock()
    if llm_raises:
        manager_mock.execute = AsyncMock(side_effect=llm_raises)
    else:
        manager_mock.execute = AsyncMock(return_value=llm_response or "")

    # provider.generate() returns a coroutine — manager.execute() awaits it
    provider_mock.generate = MagicMock(return_value=AsyncMock())

    agent = EvaluatorAgent(
        provider=provider_mock,
        execution_manager=manager_mock,
    )
    return agent, manager_mock


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_confidence_meets_threshold_returns_sufficient_no_llm_call():
    """confidence >= threshold → sufficient=True, manager.execute NOT called."""
    agent, manager_mock = _make_agent()
    results = [_make_result(0.8), _make_result(0.9, "t2")]
    report = _make_report()

    result = await agent.evaluate(
        query="What is AI?",
        results=results,
        report=report,
        iteration=1,
        confidence_threshold=0.5,  # 0.85 mean >= 0.5
        session_id="sess-001",
    )

    assert result.sufficient is True
    assert result.confidence_score == pytest.approx(0.85)
    assert result.gaps == []
    assert result.gap_tasks == []
    assert result.reasoning == "Confidence threshold met."
    manager_mock.execute.assert_not_called()


@pytest.mark.asyncio
async def test_empty_results_returns_zero_confidence():
    """Empty results list → confidence_score=0.0."""
    agent, manager_mock = _make_agent(
        llm_response=json.dumps({
            "gaps": ["What are the main applications of AI?"],
            "reasoning": "Report has no research findings.",
        })
    )
    report = _make_report()

    result = await agent.evaluate(
        query="What is AI?",
        results=[],
        report=report,
        iteration=1,
        confidence_threshold=0.5,
        session_id="sess-001",
    )

    assert result.confidence_score == 0.0
    assert result.sufficient is False  # 0.0 < 0.5


@pytest.mark.asyncio
async def test_confidence_below_threshold_calls_llm_returns_gap_tasks():
    """confidence < threshold, valid LLM response → sufficient=False, gap_tasks populated."""
    llm_payload = json.dumps({
        "gaps": [
            "What are the computational requirements of transformer models?",
            "How does reinforcement learning differ from supervised learning?",
        ],
        "reasoning": "The report lacks depth on model architecture and learning paradigms.",
    })
    agent, manager_mock = _make_agent(llm_response=llm_payload)
    results = [_make_result(0.2)]  # low confidence: 0.2 < 0.5
    report = _make_report()

    result = await agent.evaluate(
        query="What is AI?",
        results=results,
        report=report,
        iteration=1,
        confidence_threshold=0.5,
        session_id="sess-001",
    )

    assert result.sufficient is False
    assert result.confidence_score == pytest.approx(0.2)
    assert len(result.gaps) == 2
    assert len(result.gap_tasks) == 2
    assert all(isinstance(t, ResearchTask) for t in result.gap_tasks)
    assert "transformer" in result.gap_tasks[0].query.lower()
    manager_mock.execute.assert_called_once()


@pytest.mark.asyncio
async def test_gap_tasks_have_fresh_uuid4_task_ids():
    """Each gap_task must have a new uuid4 task_id, not reusing existing ones."""
    existing_task_id = "existing-task-id-1234"
    llm_payload = json.dumps({
        "gaps": ["Gap topic 1", "Gap topic 2", "Gap topic 3"],
        "reasoning": "Several gaps found.",
    })
    agent, _ = _make_agent(llm_response=llm_payload)
    results = [_make_result(0.1, task_id=existing_task_id)]
    report = _make_report()

    result = await agent.evaluate(
        query="What is AI?",
        results=results,
        report=report,
        iteration=1,
        confidence_threshold=0.5,
        session_id="sess-001",
    )

    assert len(result.gap_tasks) == 3
    task_ids = [t.task_id for t in result.gap_tasks]

    # No task_id should match the existing task id
    assert existing_task_id not in task_ids

    # All task_ids must be unique
    assert len(set(task_ids)) == 3

    # All task_ids should be valid UUIDs (36-char format)
    for tid in task_ids:
        assert len(tid) == 36, f"task_id is not a uuid4: {tid}"


@pytest.mark.asyncio
async def test_llm_parse_failure_returns_failsafe_sufficient_true():
    """LLM returns non-JSON → sufficient=True (fail-safe, never raises)."""
    agent, manager_mock = _make_agent(llm_response="This is not JSON at all!")
    results = [_make_result(0.1)]
    report = _make_report()

    result = await agent.evaluate(
        query="What is AI?",
        results=results,
        report=report,
        iteration=1,
        confidence_threshold=0.5,
        session_id="sess-001",
    )

    assert result.sufficient is True
    assert "parse failed" in result.reasoning.lower() or "delivering" in result.reasoning.lower()
    assert result.gap_tasks == []
    assert result.gaps == []


@pytest.mark.asyncio
async def test_llm_raises_exception_returns_failsafe_sufficient_true():
    """LLM call raises → sufficient=True (fail-safe, never raises)."""
    agent, manager_mock = _make_agent(llm_raises=RuntimeError("LLM timeout"))
    results = [_make_result(0.1)]
    report = _make_report()

    result = await agent.evaluate(
        query="What is AI?",
        results=results,
        report=report,
        iteration=1,
        confidence_threshold=0.5,
        session_id="sess-001",
    )

    assert result.sufficient is True
    assert result.gap_tasks == []
    assert result.gaps == []
    # The method must NOT re-raise
    assert isinstance(result, EvaluationResult)


@pytest.mark.asyncio
async def test_exactly_at_threshold_is_sufficient():
    """confidence_score == threshold exactly → sufficient=True (>= comparison)."""
    agent, manager_mock = _make_agent()
    results = [_make_result(0.5)]
    report = _make_report()

    result = await agent.evaluate(
        query="What is AI?",
        results=results,
        report=report,
        iteration=1,
        confidence_threshold=0.5,
        session_id="sess-001",
    )

    assert result.sufficient is True
    manager_mock.execute.assert_not_called()


@pytest.mark.asyncio
async def test_gap_tasks_have_priority_2():
    """All gap_tasks created by EvaluatorAgent must have priority=2."""
    llm_payload = json.dumps({
        "gaps": ["Missing topic A", "Missing topic B"],
        "reasoning": "Coverage gaps identified.",
    })
    agent, _ = _make_agent(llm_response=llm_payload)
    results = [_make_result(0.1)]
    report = _make_report()

    result = await agent.evaluate(
        query="What is AI?",
        results=results,
        report=report,
        iteration=1,
        confidence_threshold=0.5,
        session_id="sess-001",
    )

    assert all(t.priority == 2 for t in result.gap_tasks)


@pytest.mark.asyncio
async def test_evaluation_result_fields_set_correctly():
    """Verify all EvaluationResult fields are set on a successful evaluation."""
    llm_payload = json.dumps({
        "gaps": ["Explain neural networks"],
        "reasoning": "Report lacks technical depth.",
    })
    agent, _ = _make_agent(llm_response=llm_payload)
    results = [_make_result(0.3)]
    report = _make_report()

    result = await agent.evaluate(
        query="What is AI?",
        results=results,
        report=report,
        iteration=2,
        confidence_threshold=0.5,
        session_id="sess-test",
    )

    assert result.session_id == "sess-test"
    assert result.iteration == 2
    assert result.confidence_threshold == 0.5
    assert result.confidence_score == pytest.approx(0.3)
    assert result.sufficient is False
    assert result.evaluated_at is not None
