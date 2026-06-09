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
  - If memory_service.retrieve() raises OR returns empty: skip RAG and fall
    through to a general-knowledge LLM call. confidence is set to
    _GENERAL_KNOWLEDGE_CONFIDENCE (0.3) to honestly reflect no document backing.
    This mirrors the critic-bypass pattern for low-complexity tasks.
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
_DEFAULT_CONFIDENCE = 0.2        # mean retrieval score when chunks exist but score is low
_GENERAL_KNOWLEDGE_CONFIDENCE = 0.3  # fixed score when answering from LLM knowledge (no RAG)

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

# System prompt used when no documents are available — answers from general knowledge.
_GENERAL_KNOWLEDGE_SYSTEM_PROMPT = """\
You are a research analyst with broad general knowledge. Your job is to answer \
a research sub-question accurately and directly when no documents are available.

RULES:
1. Answer directly and factually from general knowledge.
2. Do NOT refuse or say you cannot answer — provide the best answer you can.
3. Write in clear, concise prose. 2-4 paragraphs.
4. End with a single sentence noting the answer is based on general knowledge,
   not retrieved documents.
"""

_USER_TEMPLATE = """\
Research sub-question: {query}

Retrieved document excerpts:
{context}

Synthesise the above excerpts into findings for the research sub-question. \
Write 2-4 paragraphs.
"""

# Used when RAG is skipped (empty knowledge base or retrieval error).
_GENERAL_KNOWLEDGE_TEMPLATE = """\
Research sub-question: {query}

Answer this sub-question using your general knowledge. \
Write 2-4 clear, factual paragraphs.
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

    Returns _DEFAULT_CONFIDENCE if chunks exist but scores are low.
    Returns _GENERAL_KNOWLEDGE_CONFIDENCE (via caller) when chunks=[]
    to distinguish from a real retrieval score of 0.
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
        # If retrieval fails or returns nothing, fall through to a general-knowledge
        # LLM call rather than short-circuiting with a useless "retrieval failed"
        # result. This mirrors the critic-bypass for low-complexity tasks.
        chunks: list[RetrievedChunk] = []
        retrieval_skipped = False
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
                "ResearchAgent retrieval failed — falling back to general knowledge",
                extra={
                    "session_id": session_id,
                    "task_id": task.task_id,
                    "error": str(exc),
                },
            )
            chunks = []
            retrieval_skipped = True

        if not chunks:
            retrieval_skipped = True

        # ── Step 2: Build context string ──────────────────────────────────────
        if chunks:
            # RAG mode: synthesise from retrieved document excerpts
            context_str = _build_context(chunks)
            user_prompt = _USER_TEMPLATE.format(
                query=task.query,
                context=context_str,
            )
            active_system = _SYSTEM_PROMPT
        else:
            # General-knowledge mode: skip RAG, answer directly from LLM knowledge
            logger.info(
                "ResearchAgent using general knowledge (no documents retrieved)",
                extra={"session_id": session_id, "task_id": task.task_id},
            )
            user_prompt = _GENERAL_KNOWLEDGE_TEMPLATE.format(query=task.query)
            active_system = _GENERAL_KNOWLEDGE_SYSTEM_PROMPT

        # ── Step 3: Generate findings ─────────────────────────────────────────
        try:
            findings: str = await self._manager.execute(
                self._provider.generate(
                    prompt=user_prompt,
                    system=active_system,
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
        # When RAG was skipped, use the fixed general-knowledge confidence rather
        # than the retrieval-score mean (which would be 0.0 and misleading).
        confidence = (
            _GENERAL_KNOWLEDGE_CONFIDENCE
            if retrieval_skipped
            else _compute_confidence(chunks)
        )

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
                "rag_used": not retrieval_skipped,
            },
        )

        return ResearchResult(
            task_id=task.task_id,
            task_query=task.query,
            findings=findings.strip(),
            sources=sources,
            confidence=confidence,
        )
