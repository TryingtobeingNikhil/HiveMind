"""
tests/phase3/test_planner_agent.py
────────────────────────────────────
Tests for PlannerAgent.

Coverage:
  - Valid query returns ResearchPlan with correct fields
  - LLM returns invalid JSON → PlannerException raised
  - Task count 1 → complexity="low"
  - Task count 3 → complexity="medium"
  - Task count 5 → complexity="high"
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.planner_agent import PlannerAgent
from app.core.exceptions import PlannerException
from app.schemas.research import ResearchPlan


def _make_agent(llm_response: str) -> PlannerAgent:
    """Helper: construct a PlannerAgent with mocked dependencies."""
    provider = MagicMock()
    provider.generate = MagicMock(return_value=None)  # coroutine created per call

    execution_manager = MagicMock()
    # execute() is async and returns the llm_response string
    execution_manager.execute = AsyncMock(return_value=llm_response)

    return PlannerAgent(provider=provider, execution_manager=execution_manager)


def _make_task_json(count: int) -> str:
    """Build a valid LLM JSON response with `count` tasks."""
    tasks = [{"query": f"Sub-question {i}", "priority": i} for i in range(1, count + 1)]
    return json.dumps({"tasks": tasks})


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_plan_returns_research_plan_with_correct_fields():
    """Valid JSON response → ResearchPlan with correct session_id and tasks."""
    agent = _make_agent(_make_task_json(2))
    plan = await agent.plan(query="What is quantum computing?", session_id="sess-001")

    assert isinstance(plan, ResearchPlan)
    assert plan.session_id == "sess-001"
    assert plan.original_query == "What is quantum computing?"
    assert len(plan.tasks) == 2
    assert all(t.task_id for t in plan.tasks)  # uuid4 assigned
    assert plan.tasks[0].query == "Sub-question 1"
    assert plan.tasks[0].priority == 1


@pytest.mark.asyncio
async def test_plan_raises_planner_exception_on_invalid_json():
    """LLM returns non-JSON → PlannerException raised."""
    agent = _make_agent("This is definitely not JSON at all!!!")
    with pytest.raises(PlannerException, match="invalid JSON"):
        await agent.plan(query="test", session_id="sess-002")


@pytest.mark.asyncio
async def test_plan_raises_planner_exception_on_missing_tasks_key():
    """LLM returns valid JSON but missing 'tasks' key → PlannerException."""
    agent = _make_agent(json.dumps({"wrong_key": []}))
    with pytest.raises(PlannerException):
        await agent.plan(query="test", session_id="sess-003")


@pytest.mark.asyncio
async def test_plan_raises_planner_exception_on_empty_tasks():
    """LLM returns tasks as empty list → PlannerException."""
    agent = _make_agent(json.dumps({"tasks": []}))
    with pytest.raises(PlannerException):
        await agent.plan(query="test", session_id="sess-004")


@pytest.mark.asyncio
async def test_complexity_low_for_one_task():
    """1 task → complexity='low'."""
    agent = _make_agent(_make_task_json(1))
    plan = await agent.plan(query="test", session_id="s1")
    assert plan.estimated_complexity == "low"


@pytest.mark.asyncio
async def test_complexity_low_for_two_tasks():
    """2 tasks → complexity='low'."""
    agent = _make_agent(_make_task_json(2))
    plan = await agent.plan(query="test", session_id="s2")
    assert plan.estimated_complexity == "low"


@pytest.mark.asyncio
async def test_complexity_medium_for_three_tasks():
    """3 tasks → complexity='medium'."""
    agent = _make_agent(_make_task_json(3))
    plan = await agent.plan(query="test", session_id="s3")
    assert plan.estimated_complexity == "medium"


@pytest.mark.asyncio
async def test_complexity_medium_for_four_tasks():
    """4 tasks → complexity='medium'."""
    agent = _make_agent(_make_task_json(4))
    plan = await agent.plan(query="test", session_id="s4")
    assert plan.estimated_complexity == "medium"


@pytest.mark.asyncio
async def test_complexity_high_for_five_tasks():
    """5 tasks → complexity='high'."""
    agent = _make_agent(_make_task_json(5))
    plan = await agent.plan(query="test", session_id="s5")
    assert plan.estimated_complexity == "high"


@pytest.mark.asyncio
async def test_tasks_have_unique_uuid4_ids():
    """Each task gets a unique uuid4 task_id (not from LLM)."""
    agent = _make_agent(_make_task_json(3))
    plan = await agent.plan(query="test", session_id="s6")
    ids = [t.task_id for t in plan.tasks]
    assert len(ids) == len(set(ids)), "All task_ids must be unique"


@pytest.mark.asyncio
async def test_plan_strips_markdown_code_fences():
    """LLM response wrapped in ```json fences → still parsed correctly."""
    raw = "```json\n" + _make_task_json(2) + "\n```"
    agent = _make_agent(raw)
    plan = await agent.plan(query="test", session_id="s7")
    assert len(plan.tasks) == 2


@pytest.mark.asyncio
async def test_execution_manager_called_with_correct_caller():
    """LLMExecutionManager.execute() must be called with caller='planner_agent'."""
    provider = MagicMock()
    provider.generate = MagicMock(return_value=None)
    manager = MagicMock()
    manager.execute = AsyncMock(return_value=_make_task_json(1))

    agent = PlannerAgent(provider=provider, execution_manager=manager)
    await agent.plan(query="test", session_id="s8")

    manager.execute.assert_called_once()
    call_kwargs = manager.execute.call_args
    assert call_kwargs.kwargs.get("caller") == "planner_agent" or \
           (len(call_kwargs.args) >= 2 and call_kwargs.args[1] == "planner_agent")
