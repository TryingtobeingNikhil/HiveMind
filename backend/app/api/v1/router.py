"""
app/api/v1/router.py
─────────────────────
API version 1 router — aggregates all v1 route modules.

All routes registered here are accessible under the prefix
configured in :data:`Settings.api_v1_prefix` (default: ``/api/v1``).
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.routes import agents, documents, health, memory, retrieval, research, research_stream

router = APIRouter()

# ── Phase 1 routes ────────────────────────────────────────────────────────────
router.include_router(health.router)
router.include_router(agents.router)

# ── Phase 2 routes ────────────────────────────────────────────────────────────
router.include_router(documents.router)
router.include_router(retrieval.router)
router.include_router(memory.router)

# ── Phase 3 routes ────────────────────────────────────────────────────────────
router.include_router(research.router)

# ── Phase 5 routes ────────────────────────────────────────────────────────────
router.include_router(research_stream.router)
