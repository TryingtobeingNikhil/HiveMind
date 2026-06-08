"""
app/agents/critic_agent.py
───────────────────────────
CriticAgent — evaluates research results and produces a quality judgement.

Responsibilities:
  1. Build a prompt with all research findings and the original query.
  2. Route the generation call through LLMExecutionManager (caller="critic_agent").
  3. Parse the LLM response into a validated CriticResult.
  4. Derive approved from overall_quality (poor → False, else → True).

Failure contract (fully graceful):
  - On ANY failure (LLM error, parse error, validation error):
      return a fallback CriticResult with approved=True, issues=["critic parse failed"].
  - CriticAgent NEVER raises. Session must continue to REPORTING.

This class does NOT extend BaseAgent.
Rationale: BaseAgent.run() wraps output into a generic AgentResult.
CriticAgent.critique() accepts a typed list[ResearchResult] and returns a
typed CriticResult. The two contracts are incompatible without unsafe hacks.
"""

from __future__ import annotations

import json
import logging

from app.llm.base_provider import BaseLLMProvider
from app.llm.execution_manager import LLMExecutionManager
from app.schemas.research import CriticResult, ResearchResult

logger = logging.getLogger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a senior research critic. Your job is to evaluate the quality and \
completeness of research findings.

STRICT RULES:
1. Respond with VALID JSON ONLY. No markdown, no code fences, no extra text.
2. The JSON must exactly match this schema:
   {
     "issues": ["<specific problem 1>", ...],
     "suggestions": ["<specific improvement 1>", ...],
     "overall_quality": "<poor|acceptable|good>"
   }
3. "issues" lists specific problems: missing information, contradictions, \
vagueness, low confidence.
4. "suggestions" lists actionable improvements.
5. "overall_quality" must be exactly one of: "poor", "acceptable", "good".
6. "poor" = major gaps or contradictions that make the report unreliable.
   "acceptable" = minor issues but usable.
   "good" = thorough, well-supported findings.
"""

_USER_TEMPLATE = """\
Original research query: {query}

Research findings to evaluate:
{findings_block}

Evaluate the completeness and quality of these findings. \
Return ONLY the JSON object.
"""


def _build_findings_block(results: list[ResearchResult]) -> str:
    """Format all research results into a numbered block for the critic."""
    parts: list[str] = []
    for i, result in enumerate(results, start=1):
        parts.append(
            f"[Task {i} — confidence={result.confidence:.2f}]\n"
            f"Sub-query: {result.task_query}\n"
            f"Findings: {result.findings}"
        )
    return "\n\n".join(parts)


def _make_fallback(results: list[ResearchResult], reason: str) -> CriticResult:
    """Return a safe fallback CriticResult when parsing fails."""
    logger.warning(
        "CriticAgent returning fallback result",
        extra={"reason": reason},
    )
    return CriticResult(
        approved=True,
        issues=[f"critic parse failed: {reason}"],
        suggestions=[],
        overall_quality="acceptable",
        reviewed_task_ids=[r.task_id for r in results],
    )


class CriticAgent:
    """
    Evaluates all research results and returns a CriticResult.

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

    async def critique(
        self,
        results: list[ResearchResult],
        original_query: str,
        session_id: str,
    ) -> CriticResult:
        """
        Evaluate research results and return a quality judgement.

        This method NEVER raises. Any failure returns a fallback CriticResult.

        Args:
            results:        All ResearchResult objects from the research stage.
            original_query: The original user query for context.
            session_id:     Parent session ID (for logging).

        Returns:
            A CriticResult (always — never raises).
        """
        logger.info(
            "CriticAgent starting evaluation",
            extra={
                "session_id": session_id,
                "result_count": len(results),
            },
        )

        findings_block = _build_findings_block(results)
        user_prompt = _USER_TEMPLATE.format(
            query=original_query,
            findings_block=findings_block,
        )

        # ── Generate ──────────────────────────────────────────────────────────
        try:
            raw_response: str = await self._manager.execute(
                self._provider.generate(
                    prompt=user_prompt,
                    system=_SYSTEM_PROMPT,
                ),
                caller="critic_agent",
            )
        except Exception as exc:
            return _make_fallback(results, f"LLM call failed: {exc}")

        # ── Parse ─────────────────────────────────────────────────────────────
        try:
            critic_result = self._parse_response(raw_response, results)
        except Exception as exc:
            return _make_fallback(results, f"parse failed: {exc}")

        logger.info(
            "CriticAgent evaluation complete",
            extra={
                "session_id": session_id,
                "overall_quality": critic_result.overall_quality,
                "approved": critic_result.approved,
                "issue_count": len(critic_result.issues),
            },
        )

        return critic_result

    def _parse_response(
        self,
        raw: str,
        results: list[ResearchResult],
    ) -> CriticResult:
        """
        Parse the LLM response into a CriticResult.

        Raises:
            Any exception on parse or validation failure (caught by critique()).
        """
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]).strip()

        data = json.loads(cleaned)

        overall_quality: str = str(data.get("overall_quality", "acceptable")).lower()
        if overall_quality not in ("poor", "acceptable", "good"):
            overall_quality = "acceptable"

        # Derive approved from overall_quality — never set independently
        approved: bool = overall_quality != "poor"

        issues = [str(i) for i in data.get("issues", [])]
        suggestions = [str(s) for s in data.get("suggestions", [])]
        reviewed_task_ids = [r.task_id for r in results]

        return CriticResult(
            approved=approved,
            issues=issues,
            suggestions=suggestions,
            overall_quality=overall_quality,  # type: ignore[arg-type]
            reviewed_task_ids=reviewed_task_ids,
        )
