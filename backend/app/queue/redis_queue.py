"""
app/queue/redis_queue.py
────────────────────────
Redis task queue for persistent session queuing.
Used strictly for enqueueing tasks (producer).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from redis import asyncio as aioredis
from redis.exceptions import RedisError

from app.core.config import Settings

logger = logging.getLogger(__name__)


class ResearchTaskQueue:
    """
    Task queue for research sessions.

    Usage:
        queue = ResearchTaskQueue(settings)
        await queue.connect()
        await queue.enqueue_session("sess-123", "Some query")
    """

    def __init__(self, settings: Settings) -> None:
        self._url = settings.redis_url
        self._redis: aioredis.Redis | None = None
        self._queue_key = "queue:research_tasks"

    async def connect(self) -> None:
        """Connect to Redis."""
        if self._redis is not None:
            return
        
        try:
            self._redis = aioredis.from_url(self._url, decode_responses=True)
            await self._redis.ping()
            logger.info("Connected to Redis", extra={"url": self._url})
        except RedisError as exc:
            logger.warning("Failed to connect to Redis", extra={"url": self._url, "error": str(exc)})
            self._redis = None

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
            logger.info("Disconnected from Redis")

    async def enqueue_session(self, session_id: str, query: str) -> bool:
        """
        Enqueue a session for research execution.
        
        Returns:
            True if enqueued successfully, False if Redis is unavailable.
        """
        if self._redis is None:
            logger.warning("Redis not available, cannot enqueue session", extra={"session_id": session_id})
            return False

        payload = {
            "session_id": session_id,
            "query": query,
        }
        try:
            await self._redis.lpush(self._queue_key, json.dumps(payload))
            logger.info("Session enqueued", extra={"session_id": session_id})
            return True
        except RedisError as exc:
            logger.error("Failed to enqueue session", extra={"session_id": session_id, "error": str(exc)})
            return False

    async def health(self) -> bool:
        """Check if Redis is reachable."""
        if self._redis is None:
            return False
        try:
            return await self._redis.ping()
        except RedisError:
            return False
