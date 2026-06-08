"""
app/schemas/research.py
────────────────────────
Pydantic models for Phase 3 + Phase 4 — Sequential Multi-Agent Orchestration
with Autonomous Research Loop.

These schemas form the complete data contract across all agents and the
orchestrator. Every field is typed and validated at runtime.

Model hierarchy:
    ResearchTask        → atomic unit of work for ResearchAgent
    ResearchPlan        → PlannerAgent output (list of ResearchTasks)
    ResearchResult      → ResearchAgent output (findings per task)
    CriticResult        → CriticAgent output (quality judgement)
    ReportSection       → one section of the final report
    ResearchReport      → ReportWriterAgent output (final deliverable)
    EvaluationResult    → EvaluatorAgent output (Phase 4 loop control)
    WorkflowStage       → enum of all orchestration stages
    WorkflowState       → full session state (persisted to SQLite)
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


# ── Atomic task ───────────────────────────────────────────────────────────────


class ResearchTask(BaseModel):
    """
    An atomic unit of work issued to the ResearchAgent.

    Produced by PlannerAgent as part of a ResearchPlan.
    task_id is a uuid4 string assigned by PlannerAgent.
    """

    task_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique task identifier (uuid4 string)",
    )
    query: str = Field(
        ...,
        description="Specific sub-query this task investigates",
    )
    priority: int = Field(
        default=1,
        description="Execution priority — 1 is highest",
        ge=1,
    )


# ── Planner output ────────────────────────────────────────────────────────────


class ResearchPlan(BaseModel):
    """
    Output of PlannerAgent.

    Contains an ordered list of ResearchTasks and an estimated
    complexity level based on task count.

    Complexity rules (enforced by PlannerAgent):
        1–2 tasks → "low"
        3–4 tasks → "medium"
        5+ tasks  → "high"
    """

    session_id: str = Field(..., description="Parent session identifier")
    original_query: str = Field(..., description="The user's original research query")
    tasks: list[ResearchTask] = Field(
        ...,
        description="Ordered list of research tasks (executed sequentially)",
    )
    estimated_complexity: Literal["low", "medium", "high"] = Field(
        ...,
        description="Complexity estimate based on number of tasks",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the plan was created",
    )


# ── Research output ───────────────────────────────────────────────────────────


class ResearchResult(BaseModel):
    """
    Output of ResearchAgent for a single ResearchTask.

    confidence is computed from retrieval scores — never self-reported.
    sources are document_ids extracted from RetrievedChunk metadata.
    """

    task_id: str = Field(
        ...,
        description="Matches the originating ResearchTask.task_id",
    )
    task_query: str = Field(
        ...,
        description="The sub-query this result addresses",
    )
    findings: str = Field(
        ...,
        description="LLM-synthesised findings for this task",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="document_ids or URLs that informed these findings",
    )
    confidence: float = Field(
        ...,
        description="Computed confidence score [0.0, 1.0] based on retrieval scores",
        ge=0.0,
        le=1.0,
    )


# ── Critic output ─────────────────────────────────────────────────────────────


class CriticResult(BaseModel):
    """
    Output of CriticAgent.

    approved is derived from overall_quality (never set independently):
        "poor"       → approved=False
        "acceptable" → approved=True
        "good"       → approved=True
    """

    approved: bool = Field(
        ...,
        description="True if quality is acceptable or good",
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Specific problems found in the research results",
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Specific improvements recommended",
    )
    overall_quality: Literal["poor", "acceptable", "good"] = Field(
        ...,
        description="Overall quality assessment",
    )
    reviewed_task_ids: list[str] = Field(
        default_factory=list,
        description="task_ids that were reviewed",
    )


# ── Report building blocks ────────────────────────────────────────────────────


class ReportSection(BaseModel):
    """A single section within the final ResearchReport."""

    title: str = Field(..., description="Section heading")
    content: str = Field(..., description="Section body text")
    supporting_task_ids: list[str] = Field(
        default_factory=list,
        description="task_ids that contributed to this section",
    )


# ── Final report ──────────────────────────────────────────────────────────────


class ResearchReport(BaseModel):
    """
    The final output of the orchestration pipeline.

    Produced by ReportWriterAgent.
    confidence_score is the mean of all ResearchResult.confidence values.
    critic_quality mirrors CriticResult.overall_quality.
    """

    session_id: str = Field(..., description="Parent session identifier")
    query: str = Field(..., description="Original research query")
    summary: str = Field(..., description="Executive summary of all findings")
    sections: list[ReportSection] = Field(
        default_factory=list,
        description="Detailed sections of the report",
    )
    citations: list[str] = Field(
        default_factory=list,
        description="All document_ids or URLs cited across all results",
    )
    confidence_score: float = Field(
        ...,
        description="Mean confidence across all ResearchResults [0.0, 1.0]",
        ge=0.0,
        le=1.0,
    )
    critic_quality: Literal["poor", "acceptable", "good"] = Field(
        ...,
        description="Quality level from CriticAgent",
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the report was generated",
    )


# ── Workflow state ────────────────────────────────────────────────────────────


class WorkflowStage(str, Enum):
    """All possible stages in the research orchestration pipeline."""

    PLANNING = "planning"
    RESEARCHING = "researching"
    CRITIQUING = "critiquing"
    REPORTING = "reporting"
    EVALUATING = "evaluating"  # Phase 4: autonomous research loop
    COMPLETED = "completed"
    FAILED = "failed"


class WorkflowState(BaseModel):
    """
    Full state of a research session.

    Persisted to SQLite at every stage transition.
    JSON fields (plan, results, critic_result, report, evaluations) are
    serialised as strings by ResearchRepository and deserialised on read.
    """

    session_id: str = Field(..., description="Unique session identifier (uuid4 string)")
    query: str = Field(..., description="Original research query")
    status: Literal["running", "completed", "failed"] = Field(
        default="running",
        description="High-level session status",
    )
    current_stage: WorkflowStage = Field(
        default=WorkflowStage.PLANNING,
        description="Current pipeline stage",
    )
    plan: ResearchPlan | None = Field(
        default=None,
        description="Research plan produced by PlannerAgent",
    )
    results: list[ResearchResult] = Field(
        default_factory=list,
        description="Research results accumulated across all tasks",
    )
    critic_result: CriticResult | None = Field(
        default=None,
        description="Critic evaluation of all research results",
    )
    report: ResearchReport | None = Field(
        default=None,
        description="Final report produced by ReportWriterAgent",
    )
    error: str | None = Field(
        default=None,
        description="Error message if session failed",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the session was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the last state update",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="UTC timestamp when the session reached COMPLETED or FAILED",
    )
    # ── Phase 4: Autonomous research loop fields ───────────────────────────────
    iteration: int = Field(
        default=1,
        description="Current research loop iteration number (1-based)",
    )
    evaluations: list["EvaluationResult"] = Field(
        default_factory=list,
        description="Evaluation results from each completed iteration",
    )


# ── Phase 4: Evaluation result ────────────────────────────────────────────────


class EvaluationResult(BaseModel):
    """Output of EvaluatorAgent — controls the autonomous research loop."""

    session_id: str = Field(..., description="Parent session identifier")
    iteration: int = Field(..., description="Iteration number this evaluation belongs to")
    sufficient: bool = Field(
        description="True = deliver report as-is. False = research more gaps."
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Computed from ResearchResult.confidence values. "
            "Not self-reported by LLM."
        ),
    )
    confidence_threshold: float = Field(
        description="The threshold this score was compared against."
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="Specific missing topics identified by the LLM evaluator.",
    )
    gap_tasks: list[ResearchTask] = Field(
        default_factory=list,
        description=(
            "New ResearchTasks targeting identified gaps. "
            "Empty if sufficient=True."
        ),
    )
    reasoning: str = Field(
        description="One-sentence explanation of the evaluation decision."
    )
    evaluated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the evaluation was performed.",
    )


# Rebuild WorkflowState so the forward reference to EvaluationResult resolves
WorkflowState.model_rebuild()
