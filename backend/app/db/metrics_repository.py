"""
app/db/metrics_repository.py
─────────────────────────────
Repository for retrieval metrics persistence.

Tracks every retrieval operation for observability, debugging, and
future evaluation/benchmark systems. All data is stored in SQLite.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import aiosqlite

from app.core.exceptions import DatabaseException
from app.schemas.retrieval import RetrievalMetrics

logger = logging.getLogger(__name__)


class MetricsRepository:
    """
    Async repository for :class:`RetrievalMetrics` records.
    """

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._db = conn

    # ── Write ─────────────────────────────────────────────────────────────────

    async def record(self, metrics: RetrievalMetrics) -> None:
        """
        Persist a retrieval metrics record.

        Args:
            metrics: The metrics to persist.
        """
        try:
            await self._db.execute(
                """
                INSERT INTO retrieval_metrics
                    (metrics_id, query, timestamp, retrieved_chunk_ids,
                     retrieval_scores, rerank_scores, retrieval_latency_ms,
                     total_results)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metrics.metrics_id,
                    metrics.query,
                    metrics.timestamp.isoformat(),
                    json.dumps(metrics.retrieved_chunk_ids),
                    json.dumps(metrics.retrieval_scores),
                    json.dumps(metrics.rerank_scores) if metrics.rerank_scores else None,
                    metrics.retrieval_latency_ms,
                    metrics.total_results,
                ),
            )
            await self._db.commit()
            logger.debug(
                "Retrieval metrics recorded",
                extra={
                    "metrics_id": metrics.metrics_id,
                    "query_len": len(metrics.query),
                    "results": metrics.total_results,
                    "latency_ms": metrics.retrieval_latency_ms,
                },
            )
        except Exception as exc:
            raise DatabaseException(
                f"Failed to record retrieval metrics: {exc}",
                context={"metrics_id": metrics.metrics_id},
            ) from exc

    # ── Read ──────────────────────────────────────────────────────────────────

    async def list_recent(self, limit: int = 50) -> list[RetrievalMetrics]:
        """
        Return the most recent retrieval metrics records.

        Args:
            limit: Maximum records to return (default 50).
        """
        try:
            cursor = await self._db.execute(
                "SELECT * FROM retrieval_metrics ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
            rows = await cursor.fetchall()
        except Exception as exc:
            raise DatabaseException(f"Failed to list metrics: {exc}") from exc

        return [self._row_to_metrics(dict(r)) for r in rows]

    async def get(self, metrics_id: str) -> RetrievalMetrics | None:
        """Return a single metrics record by ID, or None if not found."""
        try:
            cursor = await self._db.execute(
                "SELECT * FROM retrieval_metrics WHERE metrics_id = ?",
                (metrics_id,),
            )
            row = await cursor.fetchone()
        except Exception as exc:
            raise DatabaseException(
                f"Failed to get metrics record: {exc}",
                context={"metrics_id": metrics_id},
            ) from exc

        if row is None:
            return None
        return self._row_to_metrics(dict(row))

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _row_to_metrics(row: dict[str, Any]) -> RetrievalMetrics:
        rerank_scores_raw = row.get("rerank_scores")
        return RetrievalMetrics(
            metrics_id=row["metrics_id"],
            query=row["query"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            retrieved_chunk_ids=json.loads(row["retrieved_chunk_ids"]),
            retrieval_scores=json.loads(row["retrieval_scores"]),
            rerank_scores=json.loads(rerank_scores_raw) if rerank_scores_raw else None,
            retrieval_latency_ms=row["retrieval_latency_ms"],
            total_results=row["total_results"],
        )
