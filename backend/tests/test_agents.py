"""
tests/test_agents.py
─────────────────────
Tests for BaseAgent contract and SimpleEchoAgent behaviour.
"""

from __future__ import annotations

import pytest

from app.agents.base.agent import BaseAgent
from app.agents.base.echo_agent import SimpleEchoAgent
from app.core.exceptions import AgentExecutionError, AgentNotReadyError
from app.schemas.agent import AgentResult, AgentState, AgentTask


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_task(input_text: str = "test input", **kwargs) -> AgentTask:
    return AgentTask(input=input_text, **kwargs)


# ── BaseAgent contract tests ──────────────────────────────────────────────────


class TestBaseAgentContract:
    """Verify the abstract BaseAgent enforces its contract correctly."""

    def test_cannot_instantiate_base_agent_directly(self) -> None:
        """BaseAgent is abstract — direct instantiation must fail."""
        with pytest.raises(TypeError):
            BaseAgent()  # type: ignore[abstract]

    def test_subclass_without_agent_id_raises(self) -> None:
        """Subclasses missing agent_id must raise TypeError at init."""

        class MissingIdAgent(BaseAgent):
            agent_name = "Missing ID Agent"

            async def execute(self) -> str:
                return "ok"

        with pytest.raises(TypeError, match="agent_id"):
            MissingIdAgent()

    def test_subclass_without_agent_name_raises(self) -> None:
        """Subclasses missing agent_name must raise TypeError at init."""

        class MissingNameAgent(BaseAgent):
            agent_id = "missing-name-v1"

            async def execute(self) -> str:
                return "ok"

        with pytest.raises(TypeError, match="agent_name"):
            MissingNameAgent()

    def test_valid_subclass_initialises(self) -> None:
        """A properly defined subclass must initialise without error."""
        agent = SimpleEchoAgent()
        assert agent.agent_id == "simple-echo-agent-v1"
        assert agent.state == AgentState.IDLE

    def test_initial_state_is_idle(self) -> None:
        agent = SimpleEchoAgent()
        assert agent.state == AgentState.IDLE

    def test_current_task_is_none_initially(self) -> None:
        agent = SimpleEchoAgent()
        assert agent.current_task is None

    def test_last_result_is_none_initially(self) -> None:
        agent = SimpleEchoAgent()
        assert agent.last_result is None


# ── receive_task() ────────────────────────────────────────────────────────────


class TestReceiveTask:

    def test_receive_task_stores_task(self) -> None:
        agent = SimpleEchoAgent()
        task = make_task("hello")
        agent.receive_task(task)
        assert agent.current_task is task

    def test_receive_task_sets_state_to_idle(self) -> None:
        agent = SimpleEchoAgent()
        task = make_task("hello")
        agent.receive_task(task)
        assert agent.state == AgentState.IDLE

    def test_receive_task_overwrites_previous_task(self) -> None:
        agent = SimpleEchoAgent()
        task1 = make_task("first")
        task2 = make_task("second")
        agent.receive_task(task1)
        agent.receive_task(task2)
        assert agent.current_task is task2


# ── run() / execute() ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestAgentRun:

    async def test_run_without_task_raises_not_ready(self) -> None:
        agent = SimpleEchoAgent()
        with pytest.raises(AgentNotReadyError):
            await agent.run()

    async def test_run_returns_agent_result(self) -> None:
        agent = SimpleEchoAgent()
        agent.receive_task(make_task("hello world"))
        result = await agent.run()
        assert isinstance(result, AgentResult)

    async def test_run_sets_state_to_done(self) -> None:
        agent = SimpleEchoAgent()
        agent.receive_task(make_task("hello"))
        await agent.run()
        assert agent.state == AgentState.DONE

    async def test_run_stores_last_result(self) -> None:
        agent = SimpleEchoAgent()
        agent.receive_task(make_task("hello"))
        result = await agent.run()
        assert agent.last_result is result

    async def test_run_on_failing_agent_raises_execution_error(self) -> None:
        """A failing execute() must be wrapped in AgentExecutionError."""

        class BrokenAgent(BaseAgent):
            agent_id = "broken-v1"
            agent_name = "Broken Agent"

            async def execute(self) -> None:
                raise RuntimeError("something broke")

        agent = BrokenAgent()
        agent.receive_task(make_task("trigger error"))

        with pytest.raises(AgentExecutionError):
            await agent.run()

    async def test_run_on_failing_agent_sets_state_to_error(self) -> None:
        class BrokenAgent(BaseAgent):
            agent_id = "broken-v1"
            agent_name = "Broken Agent"

            async def execute(self) -> None:
                raise RuntimeError("something broke")

        agent = BrokenAgent()
        agent.receive_task(make_task("trigger error"))

        with pytest.raises(AgentExecutionError):
            await agent.run()

        assert agent.state == AgentState.ERROR


# ── SimpleEchoAgent ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestSimpleEchoAgent:

    async def test_echo_agent_returns_prefixed_input(self) -> None:
        agent = SimpleEchoAgent()
        agent.receive_task(make_task("Hello, world!"))
        result = await agent.run()
        assert result.output == "[ECHO] Hello, world!"

    async def test_echo_agent_custom_prefix(self) -> None:
        agent = SimpleEchoAgent(prefix="[TEST]")
        agent.receive_task(make_task("custom prefix"))
        result = await agent.run()
        assert result.output == "[TEST] custom prefix"

    async def test_echo_agent_result_has_correct_task_id(self) -> None:
        agent = SimpleEchoAgent()
        task = make_task("id check")
        agent.receive_task(task)
        result = await agent.run()
        assert result.task_id == task.task_id

    async def test_echo_agent_result_has_correct_agent_id(self) -> None:
        agent = SimpleEchoAgent()
        agent.receive_task(make_task("agent id check"))
        result = await agent.run()
        assert result.agent_id == "simple-echo-agent-v1"

    async def test_echo_agent_result_state_is_done(self) -> None:
        agent = SimpleEchoAgent()
        agent.receive_task(make_task("state check"))
        result = await agent.run()
        assert result.state == AgentState.DONE

    async def test_echo_agent_result_no_error(self) -> None:
        agent = SimpleEchoAgent()
        agent.receive_task(make_task("no error check"))
        result = await agent.run()
        assert result.error is None

    async def test_echo_agent_result_has_completed_at(self) -> None:
        agent = SimpleEchoAgent()
        agent.receive_task(make_task("timestamp check"))
        result = await agent.run()
        assert result.completed_at is not None


# ── reset() ───────────────────────────────────────────────────────────────────


class TestAgentReset:

    def test_reset_clears_current_task(self) -> None:
        agent = SimpleEchoAgent()
        agent.receive_task(make_task("reset me"))
        agent.reset()
        assert agent.current_task is None

    def test_reset_sets_state_to_idle(self) -> None:
        agent = SimpleEchoAgent()
        agent.receive_task(make_task("reset me"))
        agent.reset()
        assert agent.state == AgentState.IDLE


# ── API echo endpoint ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestEchoAgentEndpoint:
    """Integration tests for POST /api/v1/agents/echo."""

    async def test_echo_endpoint_returns_200(self, async_client) -> None:
        response = await async_client.post(
            "/api/v1/agents/echo",
            json={"input": "test message"},
        )
        assert response.status_code == 200

    async def test_echo_endpoint_returns_correct_output(self, async_client) -> None:
        response = await async_client.post(
            "/api/v1/agents/echo",
            json={"input": "Hello from test"},
        )
        data = response.json()
        assert data["output"] == "[ECHO] Hello from test"
        assert data["state"] == "done"
        assert data["agent_id"] == "simple-echo-agent-v1"

    async def test_echo_endpoint_rejects_empty_input(self, async_client) -> None:
        response = await async_client.post(
            "/api/v1/agents/echo",
            json={"input": ""},
        )
        assert response.status_code == 422

    async def test_echo_endpoint_rejects_missing_input(self, async_client) -> None:
        response = await async_client.post(
            "/api/v1/agents/echo",
            json={},
        )
        assert response.status_code == 422

    async def test_echo_endpoint_with_parameters(self, async_client) -> None:
        response = await async_client.post(
            "/api/v1/agents/echo",
            json={"input": "test", "parameters": {"key": "value"}},
        )
        assert response.status_code == 200
