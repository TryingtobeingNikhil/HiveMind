"""
tests/phase3/test_research_routes.py
──────────────────────────────────────
Tests for the Phase 3 research API routes.

Coverage:
  - POST /start → 202 with session_id
  - GET /{session_id} → 200 with WorkflowState
  - GET /{session_id}/status → 200 with status fields
  - GET /{session_id}/report → 200 when complete, 404/422 edge cases
  - GET /history → 200 with list
  - 404 when session not found

All dependencies are overridden — no live DB, no live LLM.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.exceptions import SessionNotFoundException
from app.main import create_app
from app.schemas.research import (
    CriticResult,
    ResearchPlan,
    ResearchReport,
    ResearchTask,
    WorkflowStage,
    WorkflowState,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_state(
    session_id: str = "sess-001",
    status: str = "completed",
    stage: WorkflowStage = WorkflowStage.COMPLETED,
    report: ResearchReport | None = None,
) -> WorkflowState:
    now = datetime.now(timezone.utc)
    return WorkflowState(
        session_id=session_id,
        query="What is AI?",
        status=status,
        current_stage=stage,
        plan=ResearchPlan(
            session_id=session_id,
            original_query="What is AI?",
            tasks=[ResearchTask(task_id="t1", query="Define AI", priority=1)],
            estimated_complexity="low",
        ),
        results=[],
        critic_result=CriticResult(
            approved=True,
            issues=[],
            suggestions=[],
            overall_quality="good",
            reviewed_task_ids=["t1"],
        ),
        report=report or ResearchReport(
            session_id=session_id,
            query="What is AI?",
            summary="AI stands for Artificial Intelligence.",
            sections=[],
            citations=[],
            confidence_score=0.8,
            critic_quality="good",
        ),
        error=None,
        created_at=now,
        updated_at=now,
        completed_at=now if status == "completed" else None,
    )


@pytest.fixture
def client():
    """
    Provide a TestClient with all Phase 3 dependencies overridden.

    The orchestrator mock runs the full pipeline and returns a mocked state.
    The repository mock returns a mocked state or raises SessionNotFoundException.
    """
    from app.dependencies.providers import (
        get_research_orchestrator,
        get_research_repository,
    )

    completed_state = _make_state("sess-001", status="completed")

    mock_orchestrator = MagicMock()
    mock_orchestrator.run = AsyncMock(return_value=completed_state)

    mock_repo = MagicMock()
    mock_repo.get = AsyncMock(return_value=completed_state)
    mock_repo.list_all = AsyncMock(return_value=[completed_state, _make_state("sess-002")])

    app = create_app()
    app.dependency_overrides[get_research_orchestrator] = lambda: mock_orchestrator
    app.dependency_overrides[get_research_repository] = lambda: mock_repo

    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def client_with_repo() -> tuple[TestClient, MagicMock, MagicMock]:
    """
    Provide a TestClient plus direct access to the mock repo and orchestrator.
    """
    from app.dependencies.providers import (
        get_research_orchestrator,
        get_research_repository,
    )

    mock_orchestrator = MagicMock()
    mock_repo = MagicMock()

    app = create_app()
    app.dependency_overrides[get_research_orchestrator] = lambda: mock_orchestrator
    app.dependency_overrides[get_research_repository] = lambda: mock_repo

    return TestClient(app, raise_server_exceptions=False), mock_repo, mock_orchestrator


# ── POST /start ───────────────────────────────────────────────────────────────


def test_post_start_returns_202(client):
    """POST /start with valid query → 202 with session_id and status."""
    resp = client.post("/api/v1/research/start", json={"query": "What is AI?"})

    assert resp.status_code == 202
    data = resp.json()
    assert "session_id" in data
    assert data["status"] == "completed"
    assert "created_at" in data


def test_post_start_missing_query_returns_422(client):
    """POST /start without query body → 422 validation error."""
    resp = client.post("/api/v1/research/start", json={})
    assert resp.status_code == 422


def test_post_start_empty_query_returns_422(client):
    """POST /start with empty string query → 422 validation error."""
    resp = client.post("/api/v1/research/start", json={"query": ""})
    assert resp.status_code == 422


# ── GET /history ──────────────────────────────────────────────────────────────


def test_get_history_returns_200(client):
    """GET /history → 200 with list of session summaries."""
    resp = client.get("/api/v1/research/history")

    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert "session_id" in data[0]
    assert "query" in data[0]
    assert "status" in data[0]
    assert "current_stage" in data[0]
    assert "created_at" in data[0]


# ── GET /{session_id} ─────────────────────────────────────────────────────────


def test_get_session_returns_200(client):
    """GET /{session_id} for existing session → 200 with full WorkflowState."""
    resp = client.get("/api/v1/research/sess-001")

    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "sess-001"
    assert data["status"] == "completed"
    assert "plan" in data
    assert "results" in data


def test_get_session_not_found_returns_404(client_with_repo):
    """GET /{session_id} for unknown session → 404."""
    test_client, mock_repo, _ = client_with_repo
    mock_repo.get = AsyncMock(side_effect=SessionNotFoundException("not found"))

    resp = test_client.get("/api/v1/research/unknown-session")
    assert resp.status_code == 404


# ── GET /{session_id}/status ──────────────────────────────────────────────────


def test_get_status_returns_200(client):
    """GET /{session_id}/status → 200 with lightweight status payload."""
    resp = client.get("/api/v1/research/sess-001/status")

    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert "status" in data
    assert "current_stage" in data
    assert "error" in data
    assert data["session_id"] == "sess-001"


def test_get_status_not_found_returns_404(client_with_repo):
    """GET /{session_id}/status for unknown session → 404."""
    test_client, mock_repo, _ = client_with_repo
    mock_repo.get = AsyncMock(side_effect=SessionNotFoundException("not found"))

    resp = test_client.get("/api/v1/research/unknown/status")
    assert resp.status_code == 404


# ── GET /{session_id}/report ──────────────────────────────────────────────────


def test_get_report_returns_200_when_complete(client):
    """GET /{session_id}/report for completed session → 200 with ResearchReport."""
    resp = client.get("/api/v1/research/sess-001/report")

    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert "sections" in data
    assert "confidence_score" in data
    assert data["session_id"] == "sess-001"


def test_get_report_returns_404_when_not_found(client_with_repo):
    """GET /{session_id}/report for unknown session → 404."""
    test_client, mock_repo, _ = client_with_repo
    mock_repo.get = AsyncMock(side_effect=SessionNotFoundException("not found"))

    resp = test_client.get("/api/v1/research/missing/report")
    assert resp.status_code == 404


def test_get_report_returns_422_when_session_failed(client_with_repo):
    """GET /{session_id}/report for failed session without report → 422."""
    test_client, mock_repo, _ = client_with_repo
    failed_state = _make_state(
        "sess-fail",
        status="failed",
        stage=WorkflowStage.FAILED,
        report=None,
    )
    failed_state = failed_state.model_copy(update={"report": None, "error": "Planner failed"})
    mock_repo.get = AsyncMock(return_value=failed_state)

    resp = test_client.get("/api/v1/research/sess-fail/report")
    assert resp.status_code == 422
