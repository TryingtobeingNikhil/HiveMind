"""
app/agents/research_agent.py
─────────────────────────────
ResearchAgent — executes a single ResearchTask against the memory layer.

Responsibilities:
  1. Call MemoryService.retrieve() to fetch relevant document chunks.
  2. Build a context string from retrieved chunks.
  3. Route the generation call through LLMExecutionManager (caller="research_agent").
  4. Compute confidence from retrieval scores (not self-reported by LLM).
  5. Extract source document_ids from RetrievedChunk metadata.

Failure contract (graceful degradation):
  - If memory_service.retrieve() raises: return a degraded ResearchResult
    with confidence=0.0, sources=[], findings="retrieval failed". Do NOT raise.
  - If the LLM call itself fails: the exception propagates to the orchestrator,
    which stores a degraded result and continues (does not fail the session).

This class does NOT extend BaseAgent.
Rationale: BaseAgent accepts a flat AgentTask string input. ResearchAgent
requires a typed ResearchTask + a MemoryService instance — neither can be
safely serialised into AgentTask.input / AgentTask.parameters.
"""

from __future__ import annotations

import logging

from app.llm.base_provider import BaseLLMProvider
from app.llm.execution_manager import LLMExecutionManager
from app.memory.memory_service import MemoryService
from app.schemas.research import ResearchResult, ResearchTask
from app.schemas.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

# ── Retrieval parameters ──────────────────────────────────────────────────────

_TOP_K = 5
_SCORE_THRESHOLD = 0.3
_DEFAULT_CONFIDENCE = 0.2   # used when no chunks are retrieved

# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a research analyst. Your job is to synthesise retrieved document \
excerpts into clear, factual findings for a specific research sub-question.

RULES:
1. Base your findings ONLY on the provided document excerpts.
2. Be specific and cite information from the excerpts.
3. If the excerpts do not contain relevant information, say so clearly.
4. Write in clear, concise prose. No bullet lists unless appropriate.
5. Do NOT make up information not present in the excerpts.
"""

_USER_TEMPLATE = """\
Research sub-question: {query}

Retrieved document excerpts:
{context}

Synthesise the above excerpts into findings for the research sub-question. \
Write 2-4 paragraphs.
"""

_NO_CONTEXT_TEMPLATE = """\
Research sub-question: {query}

No relevant documents were found in the knowledge base for this query.
Provide a brief explanation of what information would be needed to answer \
this question and note that the knowledge base did not contain relevant data.
"""


def _build_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a numbered context string."""
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Excerpt {i} — {chunk.filename}, score={chunk.retrieval_score:.3f}]\n"
            f"{chunk.content.strip()}"
        )
    return "\n\n".join(parts)


def _compute_confidence(chunks: list[RetrievedChunk]) -> float:
    """
    Compute confidence as the mean of retrieval scores.

    Returns _DEFAULT_CONFIDENCE if no chunks were retrieved.
    Clamped to [0.0, 1.0].
    """
    if not chunks:
        return _DEFAULT_CONFIDENCE
    mean_score = sum(c.retrieval_score for c in chunks) / len(chunks)
    return max(0.0, min(1.0, mean_score))


class ResearchAgent:
    """
    Executes a single ResearchTask and returns a ResearchResult.

    Constructor Args:
        provider:          The LLM provider to use for generation.
        execution_manager: The global LLM execution guard.
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        execution_manager: LLMExecutionManager,
    ) -> None:
        self._provider = provider
        self._manager = execution_manager

    async def research(
        self,
        task: ResearchTask,
        memory_service: MemoryService,
        session_id: str,
    ) -> ResearchResult:
        """
        Execute a single research task against the memory layer.

        Steps:
          1. Retrieve relevant chunks from MemoryService.
          2. Build context string from chunks.
          3. Generate findings via LLM (through execution manager).
          4. Compute confidence from retrieval scores.
          5. Extract source document_ids.

        Args:
            task:           The research task to execute.
            memory_service: Shared MemoryService instance.
            session_id:     Parent session ID (for logging).

        Returns:
            A ResearchResult (never raises — returns degraded result on failure).
        """
        logger.info(
            "ResearchAgent starting task",
            extra={
                "session_id": session_id,
                "task_id": task.task_id,
                "query_len": len(task.query),
            },
        )

        # ── Step 1: Retrieve relevant chunks ─────────────────────────────────
        chunks: list[RetrievedChunk] = []
        try:
            chunks = await memory_service.retrieve(
                query=task.query,
                top_k=_TOP_K,
                score_threshold=_SCORE_THRESHOLD,
            )
            logger.info(
                "ResearchAgent retrieval complete",
                extra={
                    "session_id": session_id,
                    "task_id": task.task_id,
                    "chunk_count": len(chunks),
                },
            )
        except Exception as exc:
            logger.warning(
                "ResearchAgent retrieval failed — returning degraded result",
                extra={
                    "session_id": session_id,
                    "task_id": task.task_id,
                    "error": str(exc),
                },
            )
            return ResearchResult(
                task_id=task.task_id,
                task_query=task.query,
                findings="retrieval failed",
                sources=[],
                confidence=0.0,
            )

        # ── Step 2: Build context string ──────────────────────────────────────
        if chunks:
            context_str = _build_context(chunks)
            user_prompt = _USER_TEMPLATE.format(
                query=task.query,
                context=context_str,
            )
        else:
            user_prompt = _NO_CONTEXT_TEMPLATE.format(query=task.query)

        # ── Step 3: Generate findings ─────────────────────────────────────────
        try:
            findings: str = await self._manager.execute(
                self._provider.generate(
                    prompt=user_prompt,
                    system=_SYSTEM_PROMPT,
                ),
                caller="research_agent",
            )
        except Exception as exc:
            logger.warning(
                "ResearchAgent LLM call failed — returning degraded result",
                extra={
                    "session_id": session_id,
                    "task_id": task.task_id,
                    "error": str(exc),
                },
            )
            return ResearchResult(
                task_id=task.task_id,
                task_query=task.query,
                findings="LLM generation failed",
                sources=[c.document_id for c in chunks],
                confidence=0.0,
            )

        # ── Step 4: Compute confidence ────────────────────────────────────────
        confidence = _compute_confidence(chunks)

        # ── Step 5: Extract sources ───────────────────────────────────────────
        # De-duplicate while preserving order
        seen: set[str] = set()
        sources: list[str] = []
        for chunk in chunks:
            if chunk.document_id not in seen:
                seen.add(chunk.document_id)
                sources.append(chunk.document_id)

        logger.info(
            "ResearchAgent task complete",
            extra={
                "session_id": session_id,
                "task_id": task.task_id,
                "confidence": round(confidence, 3),
                "source_count": len(sources),
            },
        )

        return ResearchResult(
            task_id=task.task_id,
            task_query=task.query,
            findings=findings.strip(),
            sources=sources,
            confidence=confidence,
        )
