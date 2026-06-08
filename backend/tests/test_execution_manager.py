"""
tests/test_execution_manager.py
────────────────────────────────
Tests for LLMExecutionManager.

Verifies sequential queuing behaviour, stats, and context manager usage.
"""

from __future__ import annotations

import asyncio

import pytest

from app.llm.execution_manager import LLMExecutionManager


class TestLLMExecutionManager:
    async def test_initially_not_busy(self) -> None:
        manager = LLMExecutionManager()
        assert manager.is_busy is False

    async def test_acquire_and_release(self) -> None:
        manager = LLMExecutionManager()
        await manager.acquire(caller="test")
        assert manager.is_busy is True
        await manager.release()
        assert manager.is_busy is False

    async def test_execute_runs_coroutine(self) -> None:
        manager = LLMExecutionManager()

        async def fake_llm_call() -> str:
            return "hello from llm"

        result = await manager.execute(fake_llm_call(), caller="test-agent")
        assert result == "hello from llm"

    async def test_execute_releases_lock_after_completion(self) -> None:
        manager = LLMExecutionManager()

        async def fake_call() -> str:
            return "done"

        await manager.execute(fake_call())
        assert manager.is_busy is False

    async def test_execute_releases_lock_on_exception(self) -> None:
        manager = LLMExecutionManager()

        async def failing_call() -> None:
            raise ValueError("LLM error")

        with pytest.raises(ValueError, match="LLM error"):
            await manager.execute(failing_call())

        assert manager.is_busy is False

    async def test_sequential_calls_complete_in_order(self) -> None:
        """
        Verify that two concurrent execute() calls complete sequentially.
        """
        manager = LLMExecutionManager()
        order: list[str] = []

        async def first_call() -> str:
            order.append("first_start")
            await asyncio.sleep(0.05)
            order.append("first_end")
            return "first"

        async def second_call() -> str:
            order.append("second_start")
            order.append("second_end")
            return "second"

        # Run both concurrently — the lock enforces sequential execution
        results = await asyncio.gather(
            manager.execute(first_call(), caller="first"),
            manager.execute(second_call(), caller="second"),
        )

        # Both should complete
        assert "first" in results
        assert "second" in results

        # first_start must come before second_start (or second must wait)
        assert order.index("first_end") < order.index("second_start")

    async def test_context_manager_usage(self) -> None:
        manager = LLMExecutionManager()

        async with manager:
            assert manager.is_busy is True

        assert manager.is_busy is False

    async def test_stats_track_calls(self) -> None:
        manager = LLMExecutionManager()

        async def noop() -> None:
            pass

        await manager.execute(noop(), caller="agent-1")
        await manager.execute(noop(), caller="agent-2")

        stats = manager.stats
        assert stats["total_calls"] == 2
        assert stats["is_busy"] is False

    async def test_release_is_safe_when_not_locked(self) -> None:
        """Release should be a no-op when the lock is not held."""
        manager = LLMExecutionManager()
        # Should not raise
        await manager.release()

    async def test_is_busy_false_after_release(self) -> None:
        manager = LLMExecutionManager()
        await manager.acquire()
        await manager.release()
        assert manager.is_busy is False
