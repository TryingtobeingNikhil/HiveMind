"""
tests/phase3/test_research_agent.py
─────────────────────────────────────
Tests for ResearchAgent.

Coverage:
  - Retrieval returns chunks → confidence = avg(chunk.retrieval_scores)
  - Retrieval returns empty → confidence = 0.2 (default)
  - memory_service.retrieve raises → degraded result returned, no exception
  - Sources de-duplicated correctly
  - LLM failure → degraded result, no exception
  - execution_manager called with caller='research_agent'
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.research_agent import ResearchAgent, _DEFAULT_CONFIDENCE
from app.schemas.research import ResearchResult, ResearchTask
from app.schemas.retrieval import RetrievedChunk


def _make_chunk(
    chunk_id: str = "c1",
    document_id: str = "doc-001",
    score: float = 0.8,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=0,
        content="Sample document content about the topic.",
        token_count=20,
        filename="test.pdf",
        retrieval_score=score,
    )


def _make_task(query: str = "What is X?") -> ResearchTask:
    return ResearchTask(task_id="task-001", query=query, priority=1)


def _make_agent(llm_response: str = "Synthesised findings.") -> tuple[ResearchAgent, MagicMock]:
    provider = MagicMock()
    provider.generate = MagicMock(return_value=None)
    manager = MagicMock()
    manager.execute = AsyncMock(return_value=llm_response)
    return ResearchAgent(provider=provider, execution_manager=manager), manager


def _make_memory(chunks: list[RetrievedChunk] | None = None, raises: bool = False) -> MagicMock:
    memory = MagicMock()
    if raises:
        memory.retrieve = AsyncMock(side_effect=RuntimeError("vector store down"))
    else:
        memory.retrieve = AsyncMock(return_value=chunks or [])
    return memory


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_research_returns_result_with_chunks():
    """Retrieval returns 2 chunks → confidence = avg(scores)."""
    chunks = [_make_chunk("c1", "doc-001", 0.8), _make_chunk("c2", "doc-001", 0.6)]
    agent, _ = _make_agent()
    memory = _make_memory(chunks)

    result = await agent.research(
        task=_make_task(), memory_service=memory, session_id="s1"
    )

    assert isinstance(result, ResearchResult)
    expected_confidence = (0.8 + 0.6) / 2
    assert abs(result.confidence - expected_confidence) < 0.001
    assert result.findings == "Synthesised findings."
    assert "doc-001" in result.sources


@pytest.mark.asyncio
async def test_research_confidence_default_when_no_chunks():
    """Empty retrieval → confidence = _DEFAULT_CONFIDENCE (0.2)."""
    agent, _ = _make_agent()
    memory = _make_memory(chunks=[])

    result = await agent.research(
        task=_make_task(), memory_service=memory, session_id="s2"
    )

    assert result.confidence == _DEFAULT_CONFIDENCE
    assert result.sources == []


@pytest.mark.asyncio
async def test_research_degraded_result_on_retrieval_failure():
    """memory_service.retrieve raises → degraded result returned, no exception."""
    agent, _ = _make_agent()
    memory = _make_memory(raises=True)

    result = await agent.research(
        task=_make_task("What is X?"), memory_service=memory, session_id="s3"
    )

    # Must not raise
    assert isinstance(result, ResearchResult)
    assert result.confidence == 0.0
    assert result.findings == "retrieval failed"
    assert result.sources == []
    assert result.task_id == "task-001"


@pytest.mark.asyncio
async def test_research_sources_deduplicated():
    """Multiple chunks from the same document → source appears once."""
    chunks = [
        _make_chunk("c1", "doc-001", 0.9),
        _make_chunk("c2", "doc-001", 0.7),  # same doc_id
        _make_chunk("c3", "doc-002", 0.6),
    ]
    agent, _ = _make_agent()
    memory = _make_memory(chunks)

    result = await agent.research(
        task=_make_task(), memory_service=memory, session_id="s4"
    )

    assert result.sources == ["doc-001", "doc-002"]
    assert len(result.sources) == 2


@pytest.mark.asyncio
async def test_research_degraded_result_on_llm_failure():
    """LLM call raises → degraded result returned with confidence=0.0."""
    provider = MagicMock()
    provider.generate = MagicMock(return_value=None)
    manager = MagicMock()
    manager.execute = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
    agent = ResearchAgent(provider=provider, execution_manager=manager)

    chunks = [_make_chunk("c1", "doc-001", 0.9)]
    memory = _make_memory(chunks)

    result = await agent.research(
        task=_make_task(), memory_service=memory, session_id="s5"
    )

    assert result.confidence == 0.0
    assert "failed" in result.findings.lower()


@pytest.mark.asyncio
async def test_research_execution_manager_called_with_correct_caller():
    """LLMExecutionManager.execute() called with caller='research_agent'."""
    agent, manager = _make_agent()
    memory = _make_memory([_make_chunk()])

    await agent.research(task=_make_task(), memory_service=memory, session_id="s6")

    manager.execute.assert_called_once()
    call_args = manager.execute.call_args
    caller_value = (
        call_args.kwargs.get("caller")
        or (call_args.args[1] if len(call_args.args) >= 2 else None)
    )
    assert caller_value == "research_agent"


@pytest.mark.asyncio
async def test_research_confidence_clamped_to_one():
    """Confidence never exceeds 1.0 even if scores are artificially high."""
    chunks = [_make_chunk(score=1.0), _make_chunk("c2", score=1.0)]
    agent, _ = _make_agent()
    memory = _make_memory(chunks)

    result = await agent.research(
        task=_make_task(), memory_service=memory, session_id="s7"
    )

    assert result.confidence <= 1.0
