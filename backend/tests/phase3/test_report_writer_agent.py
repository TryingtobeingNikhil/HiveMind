"""
tests/phase3/test_report_writer_agent.py
──────────────────────────────────────────
Tests for ReportWriterAgent.

Coverage:
  - Valid inputs → ResearchReport with correct confidence_score
  - confidence_score = mean of all ResearchResult.confidence values
  - Citations aggregated from all results.sources (deduplicated)
  - critic_quality mirrored from CriticResult.overall_quality
  - LLM returns invalid JSON → ReportWriterException raised
  - LLM call fails → ReportWriterException raised
  - Sections parsed correctly from LLM response
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.report_writer_agent import ReportWriterAgent
from app.core.exceptions import ReportWriterException
from app.schemas.research import (
    CriticResult,
    ResearchPlan,
    ResearchReport,
    ResearchResult,
    ResearchTask,
)


def _make_plan(session_id: str = "sess-001") -> ResearchPlan:
    return ResearchPlan(
        session_id=session_id,
        original_query="What is AI?",
        tasks=[ResearchTask(task_id="t1", query="Define AI", priority=1)],
        estimated_complexity="low",
    )


def _make_results(confidences: list[float] | None = None) -> list[ResearchResult]:
    confidences = confidences or [0.8, 0.6]
    return [
        ResearchResult(
            task_id=f"t{i+1}",
            task_query=f"Query {i+1}",
            findings=f"Findings for task {i+1}",
            sources=[f"doc-{i+1:03d}"],
            confidence=c,
        )
        for i, c in enumerate(confidences)
    ]


def _make_critic(quality: str = "good") -> CriticResult:
    return CriticResult(
        approved=quality != "poor",
        issues=[],
        suggestions=[],
        overall_quality=quality,  # type: ignore[arg-type]
        reviewed_task_ids=["t1"],
    )


def _make_agent(llm_response: str) -> ReportWriterAgent:
    provider = MagicMock()
    provider.generate = MagicMock(return_value=None)
    manager = MagicMock()
    manager.execute = AsyncMock(return_value=llm_response)
    return ReportWriterAgent(provider=provider, execution_manager=manager)


def _valid_response() -> str:
    return json.dumps({
        "summary": "This is the executive summary.",
        "sections": [
            {
                "title": "Introduction",
                "content": "Background on the topic.",
                "supporting_task_ids": ["t1"],
            },
            {
                "title": "Findings",
                "content": "Detailed findings section.",
                "supporting_task_ids": ["t1", "t2"],
            },
        ],
    })


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_write_returns_research_report():
    """Valid inputs → ResearchReport returned with correct structure."""
    agent = _make_agent(_valid_response())
    plan = _make_plan()
    results = _make_results([0.8, 0.6])
    critic = _make_critic("good")

    report = await agent.write(
        plan=plan,
        results=results,
        critic=critic,
        original_query="What is AI?",
        session_id="s1",
    )

    assert isinstance(report, ResearchReport)
    assert report.session_id == "sess-001"
    assert report.query == "What is AI?"
    assert report.summary == "This is the executive summary."
    assert len(report.sections) == 2


@pytest.mark.asyncio
async def test_write_confidence_score_is_mean_of_results():
    """confidence_score = mean of all ResearchResult.confidence values."""
    agent = _make_agent(_valid_response())
    plan = _make_plan()
    results = _make_results([0.8, 0.6])  # mean = 0.7
    critic = _make_critic("good")

    report = await agent.write(
        plan=plan, results=results, critic=critic,
        original_query="test", session_id="s2",
    )

    expected = (0.8 + 0.6) / 2
    assert abs(report.confidence_score - expected) < 0.001


@pytest.mark.asyncio
async def test_write_confidence_score_single_result():
    """Single result → confidence_score equals that result's confidence."""
    agent = _make_agent(_valid_response())
    plan = _make_plan()
    results = _make_results([0.55])
    critic = _make_critic()

    report = await agent.write(
        plan=plan, results=results, critic=critic,
        original_query="test", session_id="s3",
    )

    assert abs(report.confidence_score - 0.55) < 0.001


@pytest.mark.asyncio
async def test_write_citations_aggregated_and_deduplicated():
    """Citations from all results.sources are merged and deduplicated."""
    agent = _make_agent(_valid_response())
    plan = _make_plan()
    results = [
        ResearchResult(task_id="t1", task_query="q", findings="f",
                       sources=["doc-001", "doc-002"], confidence=0.8),
        ResearchResult(task_id="t2", task_query="q", findings="f",
                       sources=["doc-002", "doc-003"], confidence=0.6),  # doc-002 duplicate
    ]
    critic = _make_critic()

    report = await agent.write(
        plan=plan, results=results, critic=critic,
        original_query="test", session_id="s4",
    )

    assert "doc-001" in report.citations
    assert "doc-002" in report.citations
    assert "doc-003" in report.citations
    assert report.citations.count("doc-002") == 1  # not duplicated


@pytest.mark.asyncio
async def test_write_critic_quality_mirrored():
    """critic_quality in report mirrors CriticResult.overall_quality."""
    agent = _make_agent(_valid_response())
    plan = _make_plan()
    results = _make_results()
    critic = _make_critic("acceptable")

    report = await agent.write(
        plan=plan, results=results, critic=critic,
        original_query="test", session_id="s5",
    )

    assert report.critic_quality == "acceptable"


@pytest.mark.asyncio
async def test_write_raises_report_writer_exception_on_invalid_json():
    """LLM returns invalid JSON → ReportWriterException raised."""
    agent = _make_agent("Not JSON at all!!!")
    plan = _make_plan()
    results = _make_results()
    critic = _make_critic()

    with pytest.raises(ReportWriterException, match="invalid JSON"):
        await agent.write(
            plan=plan, results=results, critic=critic,
            original_query="test", session_id="s6",
        )


@pytest.mark.asyncio
async def test_write_raises_report_writer_exception_on_missing_summary():
    """JSON missing 'summary' key → ReportWriterException raised."""
    response = json.dumps({"sections": []})
    agent = _make_agent(response)
    plan = _make_plan()
    results = _make_results()
    critic = _make_critic()

    with pytest.raises(ReportWriterException):
        await agent.write(
            plan=plan, results=results, critic=critic,
            original_query="test", session_id="s7",
        )


@pytest.mark.asyncio
async def test_write_raises_report_writer_exception_on_llm_failure():
    """LLM call raises → ReportWriterException raised."""
    provider = MagicMock()
    provider.generate = MagicMock(return_value=None)
    manager = MagicMock()
    manager.execute = AsyncMock(side_effect=RuntimeError("LLM down"))
    agent = ReportWriterAgent(provider=provider, execution_manager=manager)

    with pytest.raises(ReportWriterException):
        await agent.write(
            plan=_make_plan(), results=_make_results(), critic=_make_critic(),
            original_query="test", session_id="s8",
        )


@pytest.mark.asyncio
async def test_write_execution_manager_called_with_correct_caller():
    """LLMExecutionManager.execute() called with caller='report_writer_agent'."""
    provider = MagicMock()
    provider.generate = MagicMock(return_value=None)
    manager = MagicMock()
    manager.execute = AsyncMock(return_value=_valid_response())
    agent = ReportWriterAgent(provider=provider, execution_manager=manager)

    await agent.write(
        plan=_make_plan(), results=_make_results(), critic=_make_critic(),
        original_query="test", session_id="s9",
    )

    manager.execute.assert_called_once()
    call_args = manager.execute.call_args
    caller_value = (
        call_args.kwargs.get("caller")
        or (call_args.args[1] if len(call_args.args) >= 2 else None)
    )
    assert caller_value == "report_writer_agent"
