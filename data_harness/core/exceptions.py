from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_harness.llm.providers.base import NormalizedResponse


class MaxTurnsExceeded(RuntimeError):
    """Raised when the ReAct loop reaches ``max_turns`` without an end-turn stop.

    Attributes:
        turns: The number of turns that were executed before the limit was hit.
        last_response: The final provider response, if available.
    """

    def __init__(self, turns: int, last_response: "NormalizedResponse | None" = None):
        self.turns = turns
        self.last_response = last_response
        super().__init__(f"Max turns exceeded: {turns}")


class ToolNotFoundError(KeyError):
    """Raised when a tool invocation names a tool that is not registered."""


class SubagentRecursionError(RuntimeError):
    """Raised when a subagent attempts to spawn another subagent."""
