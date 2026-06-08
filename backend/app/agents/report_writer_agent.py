"""
app/agents/report_writer_agent.py
──────────────────────────────────
ReportWriterAgent — synthesises all research results into a final ResearchReport.

Responsibilities:
  1. Build a comprehensive prompt with the plan, all findings, and critic feedback.
  2. Route the generation call through LLMExecutionManager (caller="report_writer_agent").
  3. Parse the LLM response into a validated ResearchReport.
  4. Compute confidence_score as the mean of all ResearchResult.confidence values.
  5. Aggregate all source citations from ResearchResult.sources.

Failure contract:
  - If JSON parsing fails or the response is malformed: raise ReportWriterException.
  - ReportWriterException is NOT caught here — the orchestrator catches it,
    marks the session failed, and halts execution.

This class does NOT extend BaseAgent.
Rationale: BaseAgent.run() accepts zero arguments and wraps output into a
generic AgentResult. ReportWriterAgent.write() accepts four typed arguments
and returns a typed ResearchReport. The contracts are incompatible.
"""

from __future__ import annotations

import json
import logging

from app.core.exceptions import ReportWriterException
from app.llm.base_provider import BaseLLMProvider
from app.llm.execution_manager import LLMExecutionManager
from app.schemas.research import (
    CriticResult,
    ReportSection,
    ResearchPlan,
    ResearchReport,
    ResearchResult,
)

logger = logging.getLogger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a research report writer. Your job is to synthesise research findings \
into a well-structured, professional research report.

STRICT RULES:
1. Respond with VALID JSON ONLY. No markdown, no code fences, no extra text.
2. The JSON must exactly match this schema:
   {
     "summary": "<executive summary, 2-3 sentences>",
     "sections": [
       {
         "title": "<section heading>",
         "content": "<section body, 2-4 paragraphs>",
         "supporting_task_ids": ["<task_id>", ...]
       }
     ]
   }
3. Create one section per major theme or topic identified across the findings.
4. Each section must reference the task_ids whose findings it draws from.
5. The summary must capture the key answer to the original research query.
6. Write in clear, professional prose.
"""

_USER_TEMPLATE = """\
Original research query: {query}

Research plan complexity: {complexity}

Research findings:
{findings_block}

Critic feedback:
Quality: {quality}
Issues noted: {issues}
Suggestions: {suggestions}

Write a comprehensive research report. Return ONLY the JSON object.
"""


def _build_findings_block(results: list[ResearchResult]) -> str:
    """Format all research results into a numbered block for the writer."""
    parts: list[str] = []
    for i, result in enumerate(results, start=1):
        parts.append(
            f"[Task {i} | ID={result.task_id} | confidence={result.confidence:.2f}]\n"
            f"Sub-query: {result.task_query}\n"
            f"Findings: {result.findings}"
        )
    return "\n\n".join(parts)


def _compute_confidence_score(results: list[ResearchResult]) -> float:
    """Mean confidence across all results, clamped to [0.0, 1.0]."""
    if not results:
        return 0.0
    mean = sum(r.confidence for r in results) / len(results)
    return max(0.0, min(1.0, mean))


def _collect_citations(results: list[ResearchResult]) -> list[str]:
    """Aggregate all source document_ids across results, de-duplicated."""
    seen: set[str] = set()
    citations: list[str] = []
    for result in results:
        for src in result.sources:
            if src not in seen:
                seen.add(src)
                citations.append(src)
    return citations


class ReportWriterAgent:
    """
    Synthesises all research results into a final ResearchReport.

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

    async def write(
        self,
        plan: ResearchPlan,
        results: list[ResearchResult],
        critic: CriticResult,
        original_query: str,
        session_id: str,
    ) -> ResearchReport:
        """
        Write a final ResearchReport from all pipeline outputs.

        Args:
            plan:           The ResearchPlan produced by PlannerAgent.
            results:        All ResearchResult objects from ResearchAgent.
            critic:         The CriticResult from CriticAgent.
            original_query: The original user research query.
            session_id:     Parent session ID (for logging).

        Returns:
            A validated ResearchReport.

        Raises:
            ReportWriterException: If the LLM returns unparseable output.
        """
        logger.info(
            "ReportWriterAgent starting report generation",
            extra={
                "session_id": session_id,
                "result_count": len(results),
                "critic_quality": critic.overall_quality,
            },
        )

        findings_block = _build_findings_block(results)
        issues_str = "; ".join(critic.issues) if critic.issues else "None"
        suggestions_str = "; ".join(critic.suggestions) if critic.suggestions else "None"

        user_prompt = _USER_TEMPLATE.format(
            query=original_query,
            complexity=plan.estimated_complexity,
            findings_block=findings_block,
            quality=critic.overall_quality,
            issues=issues_str,
            suggestions=suggestions_str,
        )

        # ── Generate ──────────────────────────────────────────────────────────
        try:
            raw_response: str = await self._manager.execute(
                self._provider.generate(
                    prompt=user_prompt,
                    system=_SYSTEM_PROMPT,
                ),
                caller="report_writer_agent",
            )
        except Exception as exc:
            raise ReportWriterException(
                f"ReportWriterAgent: LLM call failed: {exc}"
            ) from exc

        # ── Parse ─────────────────────────────────────────────────────────────
        report = self._parse_response(
            raw=raw_response,
            plan=plan,
            results=results,
            critic=critic,
            original_query=original_query,
            session_id=session_id,
        )

        logger.info(
            "ReportWriterAgent report generated",
            extra={
                "session_id": session_id,
                "section_count": len(report.sections),
                "confidence_score": round(report.confidence_score, 3),
            },
        )

        return report

    def _parse_response(
        self,
        raw: str,
        plan: ResearchPlan,
        results: list[ResearchResult],
        critic: CriticResult,
        original_query: str,
        session_id: str,
    ) -> ResearchReport:
        """
        Parse the LLM response into a ResearchReport.

        Raises:
            ReportWriterException: On any parse or validation failure.
        """
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ReportWriterException(
                f"ReportWriterAgent: LLM returned invalid JSON. "
                f"Parse error: {exc}. Raw response: {raw[:500]}"
            ) from exc

        try:
            summary = str(data["summary"])
        except KeyError as exc:
            raise ReportWriterException(
                f"ReportWriterAgent: JSON missing 'summary' field. Data: {str(data)[:300]}"
            ) from exc

        sections: list[ReportSection] = []
        for i, raw_section in enumerate(data.get("sections", [])):
            try:
                sections.append(
                    ReportSection(
                        title=str(raw_section["title"]),
                        content=str(raw_section["content"]),
                        supporting_task_ids=[
                            str(tid)
                            for tid in raw_section.get("supporting_task_ids", [])
                        ],
                    )
                )
            except (KeyError, TypeError) as exc:
                raise ReportWriterException(
                    f"ReportWriterAgent: Section {i} is malformed: {exc}"
                ) from exc

        confidence_score = _compute_confidence_score(results)
        citations = _collect_citations(results)

        return ResearchReport(
            session_id=plan.session_id,
            query=original_query,
            summary=summary,
            sections=sections,
            citations=citations,
            confidence_score=confidence_score,
            critic_quality=critic.overall_quality,  # type: ignore[arg-type]
        )
