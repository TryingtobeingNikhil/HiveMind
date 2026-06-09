"""
app/agents/evaluator_agent.py
──────────────────────────────
EvaluatorAgent — assesses report quality and controls the research loop.

Phase 4 loop control:
  - Deterministically computes confidence_score from ResearchResult values.
  - If score >= threshold: return sufficient=True immediately (no LLM call).
  - If score < threshold: make ONE LLM call to identify specific research gaps.
  - Converts gap strings into new ResearchTasks for the next iteration.

Failure contract (fully fail-safe):
  - On ANY failure (LLM error, parse error, validation error):
      return EvaluationResult(sufficient=True, ...) — deliver the report.
  - EvaluatorAgent NEVER raises. The session always reaches COMPLETED.

This class does NOT extend BaseAgent.
Rationale: Same as CriticAgent — typed inputs/outputs are incompatible with
BaseAgent.run()'s generic AgentResult contract.
"""

from __future__ import annotations

import json
import logging
from uuid import uuid4

from app.llm.base_provider import BaseLLMProvider
from app.llm.execution_manager import LLMExecutionManager
from app.schemas.research import (
    EvaluationResult,
    ResearchReport,
    ResearchResult,
    ResearchTask,
)

logger = logging.getLogger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a research quality evaluator. Your job is to identify specific \
information gaps in a research report.

STRICT RULES:
1. Respond with VALID JSON ONLY. No markdown, no code fences, no extra text.
2. The JSON must exactly match this schema:
   {
     "gaps": ["<specific missing topic 1>", "<specific missing topic 2>"],
     "reasoning": "<one sentence explanation of why the report is insufficient>"
   }
3. "gaps" must be specific, actionable sub-queries — not vague descriptions.
   Good: "What are the computational requirements of transformer models?"
   Bad:  "More details needed"
4. "reasoning" must be a single sentence.
5. List 1–4 gaps maximum.
"""

_USER_TEMPLATE = """\
Original research query: {query}

Research report summary:
{summary}

Report sections:
{sections_block}

The research confidence score is {confidence_score:.2f} (threshold: {threshold:.2f}).
This score is below the quality threshold.

Identify the most important specific information gaps in this report. \
Return ONLY the JSON object.
"""


def _build_sections_block(report: ResearchReport) -> str:
    """Format report sections into a numbered block for the evaluator."""
    if not report.sections:
        return "(no sections)"
    parts: list[str] = []
    for i, section in enumerate(report.sections, start=1):
        parts.append(f"[Section {i}] {section.title}\n{section.content[:500]}")
    return "\n\n".join(parts)


def _compute_confidence(results: list[ResearchResult]) -> float:
    """
    Compute mean confidence from all ResearchResult objects.

    Returns 0.0 for empty lists.
    Clamped to [0.0, 1.0].
    """
    if not results:
        return 0.0
    mean = sum(r.confidence for r in results) / len(results)
    return max(0.0, min(1.0, mean))


def _make_failsafe(
    session_id: str,
    iteration: int,
    confidence_score: float,
    confidence_threshold: float,
    reason: str,
) -> EvaluationResult:
    """Return a fail-safe EvaluationResult that delivers the current report."""
    logger.warning(
        "evaluator_parse_failed",
        extra={
            "session_id": session_id,
            "iteration": iteration,
            "error": reason,
        },
    )
    return EvaluationResult(
        session_id=session_id,
        iteration=iteration,
        sufficient=True,
        confidence_score=confidence_score,
        confidence_threshold=confidence_threshold,
        gaps=[],
        gap_tasks=[],
        reasoning="Evaluation parse failed — delivering current report.",
    )


class EvaluatorAgent:
    """
    Evaluates report quality and controls the autonomous research loop.

    Constructor Args:
        provider:          The LLM provider to use for gap identification.
        execution_manager: The global LLM execution guard.
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        execution_manager: LLMExecutionManager,
    ) -> None:
        self._provider = provider
        self._manager = execution_manager

    async def evaluate(
        self,
        *,
        query: str,
        results: list[ResearchResult],
        report: ResearchReport,
        iteration: int,
        confidence_threshold: float,
        session_id: str,
    ) -> EvaluationResult:
        """
        Evaluate the current research report and decide whether to loop.

        Steps:
          1. Compute confidence_score deterministically from results.
          2. If score >= threshold: return sufficient=True (no LLM call).
          3. If score < threshold: call LLM to identify gaps.
          4. Build ResearchTasks from gaps for the next iteration.

        This method NEVER raises. Any failure returns a fail-safe result
        with sufficient=True (deliver the current report).

        Args:
            query:               The original user research query.
            results:             All ResearchResult objects accumulated so far.
            report:              The current ResearchReport to evaluate.
            iteration:           Current iteration number (1-based).
            confidence_threshold: Minimum mean confidence to consider sufficient.
            session_id:          Parent session ID (for logging).

        Returns:
            An EvaluationResult (always — never raises).
        """
        # ── Step 1: Compute confidence deterministically ───────────────────────
        confidence_score = _compute_confidence(results)

        logger.info(
            "evaluation_started",
            extra={
                "agent": "evaluator_agent",
                "stage": "evaluating",
                "session_id": session_id,
                "iteration": iteration,
                "confidence_score": round(confidence_score, 3),
                "confidence_threshold": confidence_threshold,
            },
        )

        # ── Step 2: Check threshold without LLM ───────────────────────────────
        if confidence_score >= confidence_threshold:
            logger.info(
                "evaluation_sufficient",
                extra={
                    "session_id": session_id,
                    "iteration": iteration,
                    "confidence_score": round(confidence_score, 3),
                    "threshold": confidence_threshold,
                },
            )
            return EvaluationResult(
                session_id=session_id,
                iteration=iteration,
                sufficient=True,
                confidence_score=confidence_score,
                confidence_threshold=confidence_threshold,
                gaps=[],
                gap_tasks=[],
                reasoning="Confidence threshold met.",
            )

        # ── Step 3: LLM call to identify gaps ────────────────────────────────
        sections_block = _build_sections_block(report)
        user_prompt = _USER_TEMPLATE.format(
            query=query,
            summary=report.summary,
            sections_block=sections_block,
            confidence_score=confidence_score,
            threshold=confidence_threshold,
        )

        try:
            raw_response: str = await self._manager.execute(
                self._provider.generate(
                    prompt=user_prompt,
                    system=_SYSTEM_PROMPT,
                ),
                caller="evaluator_agent",
            )
        except Exception as exc:
            return _make_failsafe(
                session_id=session_id,
                iteration=iteration,
                confidence_score=confidence_score,
                confidence_threshold=confidence_threshold,
                reason=f"LLM call failed: {exc}",
            )

        # ── Step 4: Parse LLM response ────────────────────────────────────────
        try:
            gaps, reasoning = self._parse_response(raw_response)
        except Exception as exc:
            return _make_failsafe(
                session_id=session_id,
                iteration=iteration,
                confidence_score=confidence_score,
                confidence_threshold=confidence_threshold,
                reason=f"parse failed: {exc}",
            )

        # ── Step 5: Build ResearchTasks from gaps ─────────────────────────────
        gap_tasks: list[ResearchTask] = [
            ResearchTask(
                task_id=str(uuid4()),  # always new uuid4 — never reuse existing ids
                query=gap,
                priority=2,
            )
            for gap in gaps
        ]

        logger.info(
            "evaluation_insufficient_continuing",
            extra={
                "session_id": session_id,
                "iteration": iteration,
                "confidence_score": round(confidence_score, 3),
                "threshold": confidence_threshold,
                "gap_count": len(gaps),
                "next_iteration": iteration + 1,
            },
        )

        return EvaluationResult(
            session_id=session_id,
            iteration=iteration,
            sufficient=False,
            confidence_score=confidence_score,
            confidence_threshold=confidence_threshold,
            gaps=gaps,
            gap_tasks=gap_tasks,
            reasoning=reasoning,
        )

    def _parse_response(self, raw: str) -> tuple[list[str], str]:
        """
        Parse the LLM response into (gaps, reasoning).

        Raises:
            Any exception on parse or validation failure (caught by evaluate()).
        """
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]).strip()

        data = json.loads(cleaned)

        gaps_raw = data.get("gaps", [])
        if not isinstance(gaps_raw, list):
            gaps_raw = []

        # Coerce to strings and filter out empty strings
        gaps = [str(g).strip() for g in gaps_raw if str(g).strip()]

        reasoning = str(data.get("reasoning", "Research gaps identified.")).strip()
        if not reasoning:
            reasoning = "Research gaps identified."

        return gaps, reasoning
