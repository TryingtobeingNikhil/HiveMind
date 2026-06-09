"""
app/api/router.py
──────────────────
Root API router — mounts all versioned sub-routers.

New API versions are added here without touching application startup code.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import router as v1_router
from app.core.config import get_settings

settings = get_settings()

api_router = APIRouter()

# ── Mount versioned routers ───────────────────────────────────────────────────
api_router.include_router(
    v1_router.router,
    prefix=settings.api_v1_prefix,
)
