"""
app/agents/planner_agent.py
────────────────────────────
PlannerAgent — decomposes a research query into a structured ResearchPlan.

Responsibilities:
  1. Build a JSON-only prompt asking the LLM to decompose the query.
  2. Route the generation call through LLMExecutionManager (caller="planner_agent").
  3. Parse the LLM response into a validated ResearchPlan.
  4. Assign uuid4 task_ids to all tasks.
  5. Compute estimated_complexity from task count.

Failure contract:
  - If JSON parsing fails for any reason: raise PlannerException.
  - PlannerException is NOT caught here — the orchestrator catches it,
    marks the session failed, and halts execution.

This class does NOT extend BaseAgent.
Rationale: BaseAgent.run() accepts zero arguments and wraps output into a
generic AgentResult. PlannerAgent.plan() accepts a typed str and returns a
typed ResearchPlan. The two contracts are incompatible without unsafe
serialisation hacks.
"""

from __future__ import annotations

import json
import logging
from uuid import uuid4

from app.core.exceptions import PlannerException
from app.llm.base_provider import BaseLLMProvider
from app.llm.execution_manager import LLMExecutionManager
from app.schemas.research import ResearchPlan, ResearchTask

logger = logging.getLogger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a research planning assistant. Your ONLY job is to decompose a research \
query into the MINIMUM number of sub-tasks needed to answer it fully.

STRICT RULES:
1. Respond with VALID JSON ONLY. No markdown, no code fences, no extra text.
2. The JSON must exactly match this schema:
   {
     "tasks": [
       {
         "query": "<specific sub-question>",
         "priority": <integer starting at 1>
       }
     ]
   }
3. Choose the task count based on query complexity — use the MINIMUM needed:
   - Simple factual questions ("what is X", "define Y"): 1-3 tasks
   - Moderately complex questions ("how does X work", "compare X and Y"): 3-4 tasks
   - Complex multi-part research (multiple distinct concepts, deep comparisons): 5-7 tasks
   Do NOT pad with redundant tasks. Do NOT generate more tasks than necessary.
4. Priority 1 is the most important task. Assign priorities sequentially.
5. Tasks must be ordered by priority (ascending).
"""

_USER_TEMPLATE = """\
Research query: {query}

Decompose this into the MINIMUM number of specific research sub-tasks needed \
to answer it fully. Return ONLY the JSON object.
"""

# ── Complexity thresholds ─────────────────────────────────────────────────────
# Widened so that "high" is genuinely rare — only deep multi-faceted research.
# Matches the calibrated prompt above (simple = 1-3 tasks, moderate = 3-5, complex = 6-7).

_COMPLEXITY_LOW    = 3  # 1–3 tasks → low
_COMPLEXITY_MEDIUM = 5  # 4–5 tasks → medium
# 6–7 tasks → high


def _estimate_complexity(task_count: int) -> str:
    if task_count <= _COMPLEXITY_LOW:
        return "low"
    if task_count <= _COMPLEXITY_MEDIUM:
        return "medium"
    return "high"


class PlannerAgent:
    """
    Decomposes a research query into a ResearchPlan.

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

    async def plan(self, query: str, session_id: str) -> ResearchPlan:
        """
        Decompose a research query into a ResearchPlan.

        Args:
            query:      The original user research query.
            session_id: Parent session ID (embedded in the plan).

        Returns:
            A validated ResearchPlan with uuid4 task_ids.

        Raises:
            PlannerException: If the LLM returns non-JSON or the JSON
                              does not match the expected schema.
        """
        user_prompt = _USER_TEMPLATE.format(query=query)

        logger.info(
            "PlannerAgent generating plan",
            extra={"session_id": session_id, "query_len": len(query)},
        )

        raw_response: str = await self._manager.execute(
            self._provider.generate(
                prompt=user_prompt,
                system=_SYSTEM_PROMPT,
            ),
            caller="planner_agent",
        )

        logger.info(
            "PlannerAgent received LLM response",
            extra={"session_id": session_id, "response_len": len(raw_response)},
        )

        return self._parse_response(raw_response, query, session_id)

    def _parse_response(
        self,
        raw: str,
        original_query: str,
        session_id: str,
    ) -> ResearchPlan:
        """
        Parse the LLM response into a ResearchPlan.

        Raises:
            PlannerException: On any parse or validation failure.
        """
        # Strip optional markdown code fences that some models add
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first line (```json or ```) and last line (```)
            cleaned = "\n".join(lines[1:-1]).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise PlannerException(
                f"PlannerAgent: LLM returned invalid JSON. "
                f"Parse error: {exc}. Raw response: {raw[:500]}"
            ) from exc

        try:
            raw_tasks = data["tasks"]
            if not isinstance(raw_tasks, list) or len(raw_tasks) == 0:
                raise ValueError("'tasks' must be a non-empty list")
        except (KeyError, ValueError) as exc:
            raise PlannerException(
                f"PlannerAgent: JSON missing required 'tasks' field. "
                f"Error: {exc}. Data: {str(data)[:500]}"
            ) from exc

        tasks: list[ResearchTask] = []
        for i, raw_task in enumerate(raw_tasks):
            try:
                task = ResearchTask(
                    task_id=str(uuid4()),
                    query=str(raw_task["query"]),
                    priority=int(raw_task.get("priority", i + 1)),
                )
                tasks.append(task)
            except (KeyError, TypeError, ValueError) as exc:
                raise PlannerException(
                    f"PlannerAgent: Task {i} is malformed. "
                    f"Error: {exc}. Task data: {raw_task}"
                ) from exc

        complexity = _estimate_complexity(len(tasks))

        logger.info(
            "PlannerAgent plan parsed",
            extra={
                "session_id": session_id,
                "task_count": len(tasks),
                "complexity": complexity,
            },
        )

        return ResearchPlan(
            session_id=session_id,
            original_query=original_query,
            tasks=tasks,
            estimated_complexity=complexity,  # type: ignore[arg-type]
        )
