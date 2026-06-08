"""
app/schemas/retrieval.py
────────────────────────
Pydantic schemas for the retrieval, reranking, and context building pipeline.

All types preserve full source attribution so future agents can generate
citations, fact-check claims, and trace answers to specific document chunks.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


# ── Retrieved chunk ───────────────────────────────────────────────────────────


class RetrievedChunk(BaseModel):
    """
    A document chunk returned by the retrieval pipeline.

    Includes full source attribution and both retrieval and rerank scores.
    The rerank_score field is None when reranking is disabled.
    """

    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Source document identifier")
    chunk_index: int = Field(..., description="Position of this chunk in the source document")
    content: str = Field(..., description="Text content of the chunk")
    token_count: int = Field(..., description="Token count for this chunk")
    filename: str = Field(..., description="Source filename for citation")
    retrieval_score: float = Field(
        ...,
        description="Cosine similarity score from vector search [0, 1]",
    )
    rerank_score: float | None = Field(
        default=None,
        description="Score assigned by the reranker (higher = better)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional chunk metadata from the vector store",
    )


# ── Search request ────────────────────────────────────────────────────────────


class SearchRequest(BaseModel):
    """
    Request body for POST /api/v1/retrieval/search.
    """

    query: str = Field(..., min_length=1, max_length=2000, description="Search query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return")
    score_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum retrieval score (0.0 = no filter)",
    )
    metadata_filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional ChromaDB metadata filters (e.g. {\"filename\": \"report.pdf\"})",
    )


# ── Context request ───────────────────────────────────────────────────────────


class ContextRequest(BaseModel):
    """
    Request body for POST /api/v1/retrieval/context.

    Combines retrieval + token budget management + context building.
    """

    query: str = Field(..., min_length=1, max_length=2000)
    system_prompt: str = Field(default="", description="System prompt text (for budget calc)")
    instructions: str = Field(default="", description="Agent instructions (for budget calc)")
    history: str = Field(default="", description="Conversation history (for budget calc)")
    top_k: int = Field(default=5, ge=1, le=50)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata_filters: dict[str, Any] = Field(default_factory=dict)


# ── Context package ───────────────────────────────────────────────────────────


class TokenUsage(BaseModel):
    """Token budget breakdown for a context package."""

    system_prompt: int = Field(default=0)
    instructions: int = Field(default=0)
    history: int = Field(default=0)
    query: int = Field(default=0)
    retrieved_context: int = Field(default=0)
    reserved_output: int = Field(default=0)
    total_used: int = Field(default=0)
    budget_available: int = Field(default=0)
    budget_remaining: int = Field(default=0)


class ContextSource(BaseModel):
    """Attribution source for a single chunk in the context package."""

    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    retrieval_score: float
    rerank_score: float | None = None


class ContextPackage(BaseModel):
    """
    Output of the ContextBuilder.

    Contains the formatted context string ready for injection into an LLM
    prompt, plus full source attribution and token usage statistics.

    Future agents will consume this directly.
    """

    query: str = Field(..., description="Original query")
    formatted_context: str = Field(
        ..., description="Context text ready for LLM injection"
    )
    sources: list[ContextSource] = Field(
        default_factory=list,
        description="Source attribution for each included chunk",
    )
    token_usage: TokenUsage = Field(
        default_factory=TokenUsage,
        description="Token budget breakdown",
    )
    chunk_count: int = Field(default=0, description="Number of chunks included")
    deduplication_count: int = Field(
        default=0, description="Number of duplicate chunks removed"
    )


# ── Retrieval metrics ─────────────────────────────────────────────────────────


class RetrievalMetrics(BaseModel):
    """
    Persisted record of a single retrieval operation.

    Stored in the SQLite retrieval_metrics table for observability,
    debugging, and future benchmark/evaluation systems.
    """

    metrics_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique metrics record identifier",
    )
    query: str = Field(..., description="The search query")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the retrieval occurred (UTC)",
    )
    retrieved_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Ordered list of retrieved chunk IDs",
    )
    retrieval_scores: list[float] = Field(
        default_factory=list,
        description="Retrieval similarity scores (aligned with chunk_ids)",
    )
    rerank_scores: list[float] | None = Field(
        default=None,
        description="Rerank scores if reranking was applied",
    )
    retrieval_latency_ms: float = Field(
        ..., description="Total retrieval latency in milliseconds"
    )
    total_results: int = Field(..., description="Number of results returned")
