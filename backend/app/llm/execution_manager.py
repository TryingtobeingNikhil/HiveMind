"""
app/llm/execution_manager.py
──────────────────────────────
Global LLM Execution Manager.

Guarantees that only ONE LLM call is active across the entire system
at any given moment. All provider calls MUST pass through this manager.

DESIGN: Queue, not rejection.
  - Callers acquire the lock and wait for it to become available.
  - No request is ever rejected — they queue sequentially.
  - The lock is a standard asyncio.Lock (single-event-loop safe).

Expected Phase 3 log pattern:
    [ORCHESTRATOR] Starting pipeline
    [LLM] Planner executing (acquired lock)
    [LLM] Planner complete (released lock)
    [LLM] Research executing (acquired lock)
    [LLM] Research complete (released lock)
    [ORCHESTRATOR] Pipeline complete

EXECUTION MODEL:
  Maximum concurrent LLM calls: 1 (hard constraint)
  Callers: agents, services, future orchestrator
  Lock scope: single asyncio event loop (single process)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Coroutine
from typing import Any, TypeVar

from app.core.telemetry import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


T = TypeVar("T")


class LLMExecutionManager:
    """
    Global execution guard ensuring at most one active LLM call.

    Usage (direct):
        manager = LLMExecutionManager()
        async with manager:
            result = await provider.generate(...)

    Usage (execute helper):
        result = await manager.execute(provider.generate(...))

    The manager is a singleton stored on ``app.state.llm_manager``
    and injected via FastAPI's dependency system.
    """

    def __init__(self, delay_seconds: float = 0.0) -> None:
        self._lock: asyncio.Lock = asyncio.Lock()
        self._current_caller: str | None = None
        self._total_calls: int = 0
        self._total_wait_ms: float = 0.0
        self._delay_seconds: float = delay_seconds
        logger.info(
            "[LLM] LLMExecutionManager initialised",
            extra={"max_concurrent": 1, "delay_seconds": delay_seconds},
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_busy(self) -> bool:
        """True when an LLM call is currently active."""
        return self._lock.locked()

    @property
    def stats(self) -> dict[str, Any]:
        """Runtime statistics for observability."""
        return {
            "is_busy": self.is_busy,
            "current_caller": self._current_caller,
            "total_calls": self._total_calls,
            "total_wait_ms": round(self._total_wait_ms, 2),
        }

    # ── Async context manager ─────────────────────────────────────────────────

    async def __aenter__(self) -> "LLMExecutionManager":
        await self.acquire()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.release()

    # ── Public API ────────────────────────────────────────────────────────────

    async def acquire(self, caller: str = "unknown") -> None:
        """
        Acquire the global LLM execution lock.

        If another call is active, this method WAITS (queues) until
        the lock is released. No exception is raised on contention.

        Args:
            caller: Human-readable name of the caller (for logging).
        """
        wait_start = time.monotonic()

        if self._lock.locked():
            logger.info(
                "[LLM] Waiting for lock (queue)",
                extra={"caller": caller, "current": self._current_caller},
            )

        await self._lock.acquire()

        wait_ms = (time.monotonic() - wait_start) * 1000
        self._total_wait_ms += wait_ms
        self._total_calls += 1
        self._current_caller = caller

        logger.info(
            "[LLM] Lock acquired",
            extra={
                "caller": caller,
                "wait_ms": round(wait_ms, 2),
                "call_number": self._total_calls,
            },
        )

    async def release(self) -> None:
        """
        Release the global LLM execution lock.

        Safe to call even if the lock is not held (no-op in that case).
        """
        caller = self._current_caller
        if self._lock.locked():
            self._current_caller = None
            self._lock.release()
            logger.info("[LLM] Lock released", extra={"caller": caller})

    async def execute(
        self,
        coro: Coroutine[Any, Any, T],
        caller: str = "unknown",
    ) -> T:
        """
        Acquire lock → run coroutine → release lock.

        This is the preferred entry point. Callers never need to
        manage acquire/release manually.

        Args:
            coro:   An awaitable coroutine (e.g. provider.generate(...)).
            caller: Name used in logs to identify the calling agent/service.

        Returns:
            The result of the coroutine.

        Raises:
            Any exception raised by the coroutine (lock is released on error).
        """
        with tracer.start_as_current_span("llm.execute") as span:
            span.set_attribute("caller", caller)
            start = time.monotonic()
            await self.acquire(caller=caller)
            try:
                result: T = await coro
                elapsed_ms = (time.monotonic() - start) * 1000
                logger.info(
                    "[LLM] Execution complete",
                    extra={"caller": caller, "elapsed_ms": round(elapsed_ms, 2)},
                )
                return result
            except Exception:
                logger.warning(
                    "[LLM] Execution failed — releasing lock",
                    extra={"caller": caller},
                )
                raise
            finally:
                if self._delay_seconds > 0.0:
                    await asyncio.sleep(self._delay_seconds)
                await self.release()

