"""
app/api/v1/routes/retrieval.py
────────────────────────────────
Retrieval and context building endpoints.

POST /api/v1/retrieval/search   — Semantic search
POST /api/v1/retrieval/context  — Build context package for LLM injection
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, status

from app.dependencies.providers import ContextBuilderDep, MemoryServiceDep, SettingsDep
from app.schemas.common import APIResponse
from app.schemas.retrieval import (
    ContextPackage,
    ContextRequest,
    RetrievedChunk,
    SearchRequest,
)

router = APIRouter(prefix="/retrieval", tags=["Retrieval"])
logger = logging.getLogger(__name__)


# ── Semantic search ───────────────────────────────────────────────────────────


@router.post(
    "/search",
    response_model=APIResponse[list[RetrievedChunk]],
    summary="Semantic search across all documents",
)
async def search(
    request: SearchRequest,
    memory: MemoryServiceDep,
) -> APIResponse[list[RetrievedChunk]]:
    """
    Perform semantic search using query embeddings.

    Pipeline:
        embed query → vector search → rerank → return top_k results
    """
    logger.info(
        "Retrieval search request",
        extra={
            "query_len": len(request.query),
            "top_k": request.top_k,
            "score_threshold": request.score_threshold,
        },
    )

    results = await memory.retrieve(
        query=request.query,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        metadata_filters=request.metadata_filters or None,
    )

    return APIResponse(
        data=results,
        meta={"count": len(results), "query": request.query},
    )


# ── Context building ──────────────────────────────────────────────────────────


@router.post(
    "/context",
    response_model=APIResponse[ContextPackage],
    summary="Build a context package for LLM injection",
)
async def build_context(
    request: ContextRequest,
    memory: MemoryServiceDep,
    context_builder: ContextBuilderDep,
) -> APIResponse[ContextPackage]:
    """
    Retrieve relevant chunks and build a complete ContextPackage.

    Accounts for token budget (system prompt + instructions + history + query
    + reserved output) and returns a formatted context string with full
    source attribution.
    """
    logger.info(
        "Context build request",
        extra={"query_len": len(request.query), "top_k": request.top_k},
    )

    # Retrieve candidate chunks
    chunks = await memory.retrieve(
        query=request.query,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        metadata_filters=request.metadata_filters or None,
    )

    # Build context package with token budget management
    package = context_builder.build(
        query=request.query,
        chunks=chunks,
        system_prompt=request.system_prompt,
        instructions=request.instructions,
        history=request.history,
    )

    logger.info(
        "Context package built",
        extra={
            "chunk_count": package.chunk_count,
            "context_tokens": package.token_usage.retrieved_context,
            "budget_remaining": package.token_usage.budget_remaining,
        },
    )

    return APIResponse(
        data=package,
        meta={
            "query": request.query,
            "chunks_retrieved": len(chunks),
            "chunks_in_context": package.chunk_count,
        },
    )
