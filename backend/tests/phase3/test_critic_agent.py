"""
tests/phase3/test_critic_agent.py
───────────────────────────────────
Tests for CriticAgent.

Coverage:
  - Valid findings → CriticResult returned correctly
  - overall_quality='poor' → approved=False
  - overall_quality='acceptable' → approved=True
  - overall_quality='good' → approved=True
  - LLM parse fails → fallback CriticResult returned, no exception
  - LLM call fails → fallback CriticResult returned, no exception
  - reviewed_task_ids populated from input results
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.critic_agent import CriticAgent
from app.schemas.research import CriticResult, ResearchResult


def _make_result(task_id: str = "t1", quality: str = "good") -> ResearchResult:
    return ResearchResult(
        task_id=task_id,
        task_query="What is X?",
        findings="Some detailed findings about X.",
        sources=["doc-001"],
        confidence=0.75,
    )


def _make_agent(llm_response: str) -> CriticAgent:
    provider = MagicMock()
    provider.generate = MagicMock(return_value=None)
    manager = MagicMock()
    manager.execute = AsyncMock(return_value=llm_response)
    return CriticAgent(provider=provider, execution_manager=manager)


def _valid_response(quality: str = "good") -> str:
    return json.dumps({
        "issues": [],
        "suggestions": ["Add more citations"],
        "overall_quality": quality,
    })


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_critique_returns_critic_result_on_valid_response():
    """Valid LLM JSON → CriticResult with correct fields."""
    agent = _make_agent(_valid_response("good"))
    results = [_make_result("t1"), _make_result("t2")]

    critic = await agent.critique(
        results=results, original_query="What is X?", session_id="s1"
    )

    assert isinstance(critic, CriticResult)
    assert critic.overall_quality == "good"
    assert critic.approved is True
    assert "t1" in critic.reviewed_task_ids
    assert "t2" in critic.reviewed_task_ids


@pytest.mark.asyncio
async def test_critique_poor_quality_sets_approved_false():
    """overall_quality='poor' → approved=False."""
    agent = _make_agent(_valid_response("poor"))
    results = [_make_result()]

    critic = await agent.critique(
        results=results, original_query="test", session_id="s2"
    )

    assert critic.overall_quality == "poor"
    assert critic.approved is False


@pytest.mark.asyncio
async def test_critique_acceptable_quality_sets_approved_true():
    """overall_quality='acceptable' → approved=True."""
    agent = _make_agent(_valid_response("acceptable"))
    results = [_make_result()]

    critic = await agent.critique(
        results=results, original_query="test", session_id="s3"
    )

    assert critic.overall_quality == "acceptable"
    assert critic.approved is True


@pytest.mark.asyncio
async def test_critique_good_quality_sets_approved_true():
    """overall_quality='good' → approved=True."""
    agent = _make_agent(_valid_response("good"))
    results = [_make_result()]

    critic = await agent.critique(
        results=results, original_query="test", session_id="s4"
    )

    assert critic.overall_quality == "good"
    assert critic.approved is True


@pytest.mark.asyncio
async def test_critique_returns_fallback_on_invalid_json():
    """LLM returns invalid JSON → fallback CriticResult, no exception."""
    agent = _make_agent("This is not JSON at all!!!")
    results = [_make_result("t1")]

    critic = await agent.critique(
        results=results, original_query="test", session_id="s5"
    )

    # Must not raise
    assert isinstance(critic, CriticResult)
    assert critic.approved is True  # fallback approved=True
    assert len(critic.issues) > 0
    assert "critic parse failed" in critic.issues[0]
    assert critic.overall_quality == "acceptable"


@pytest.mark.asyncio
async def test_critique_returns_fallback_on_llm_failure():
    """LLM call raises → fallback CriticResult returned, no exception."""
    provider = MagicMock()
    provider.generate = MagicMock(return_value=None)
    manager = MagicMock()
    manager.execute = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
    agent = CriticAgent(provider=provider, execution_manager=manager)

    results = [_make_result("t1"), _make_result("t2")]

    critic = await agent.critique(
        results=results, original_query="test", session_id="s6"
    )

    assert isinstance(critic, CriticResult)
    assert critic.approved is True
    assert len(critic.issues) > 0


@pytest.mark.asyncio
async def test_critique_reviewed_task_ids_from_results():
    """reviewed_task_ids populated from input results, regardless of LLM."""
    agent = _make_agent(_valid_response("good"))
    results = [_make_result("task-aaa"), _make_result("task-bbb")]

    critic = await agent.critique(
        results=results, original_query="test", session_id="s7"
    )

    assert "task-aaa" in critic.reviewed_task_ids
    assert "task-bbb" in critic.reviewed_task_ids


@pytest.mark.asyncio
async def test_critique_issues_and_suggestions_parsed():
    """Issues and suggestions from LLM response are correctly parsed."""
    response = json.dumps({
        "issues": ["Missing citation", "Vague conclusion"],
        "suggestions": ["Add sources", "Be more specific"],
        "overall_quality": "acceptable",
    })
    agent = _make_agent(response)
    results = [_make_result()]

    critic = await agent.critique(
        results=results, original_query="test", session_id="s8"
    )

    assert "Missing citation" in critic.issues
    assert "Add sources" in critic.suggestions
