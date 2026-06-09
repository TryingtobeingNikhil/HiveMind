"""
app/agents/base/echo_agent.py
──────────────────────────────
SimpleEchoAgent — a minimal agent implementation for framework validation.

Purpose:
  Verify that the BaseAgent contract is correct and the agent framework
  is wired properly end-to-end (API → agent → result). This agent is
  NOT used in production research workflows.

Behaviour:
  - Receives any text input.
  - Returns the same text, optionally prefixed with a tag.
  - Logs the round-trip for observability verification.
"""

from __future__ import annotations

import logging

from app.agents.base.agent import BaseAgent

logger = logging.getLogger(__name__)


class SimpleEchoAgent(BaseAgent):
    """
    Echo agent — returns the input unchanged.

    Used exclusively for framework smoke-testing and integration validation.

    Example::

        agent = SimpleEchoAgent()
        agent.receive_task(AgentTask(input="Hello, world!"))
        result = await agent.run()
        assert result.output == "[ECHO] Hello, world!"
    """

    agent_id: str = "simple-echo-agent-v1"
    agent_name: str = "Simple Echo Agent"

    def __init__(self, prefix: str = "[ECHO]") -> None:
        """
        Args:
            prefix: String prepended to the echoed output (default ``[ECHO]``).
        """
        super().__init__()
        self._prefix = prefix

    async def execute(self) -> str:
        """
        Echo the task input back with the configured prefix.

        Returns:
            The input string, prefixed.
        """
        assert self.current_task is not None  # guaranteed by BaseAgent.run()

        input_text = self.current_task.input
        output = f"{self._prefix} {input_text}"

        logger.info(
            "Echo agent executed",
            extra={
                "agent_id": self.agent_id,
                "task_id": str(self.current_task.task_id),
                "input_length": len(input_text),
                "output_length": len(output),
            },
        )
        return output
