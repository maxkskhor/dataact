"""The exception taxonomy.

Every failure this library raises carries a stable ``code``. That is the point
of the taxonomy: a caller deciding between retrying, showing the user an
error, and billing for the attempt needs to tell a rate-limited provider from
a typo in the model's pandas apart from a sandbox timeout. Matching on message
text is how that decision silently rots.

Codes are part of the API. Messages are not, and may be reworded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_harness.llm.providers.base import NormalizedResponse


class DataHarnessError(Exception):
    """Base for everything this library raises deliberately.

    ``except DataHarnessError`` catches the library's own failures without
    also swallowing a bug in the caller's tool handler.
    """

    #: Stable, machine-readable identifier. Subclasses set it.
    code: str = "unknown"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class MaxTurnsExceeded(DataHarnessError, RuntimeError):
    """The loop reached ``max_turns`` without the model finishing.

    Still a `RuntimeError` because it was one before the taxonomy existed and
    callers catch it that way.

    Attributes:
        turns: How many turns ran before the limit was hit.
        last_response: The final provider response, if available.
    """

    code = "max_turns_exceeded"

    def __init__(self, turns: int, last_response: NormalizedResponse | None = None):
        self.turns = turns
        self.last_response = last_response
        super().__init__(f"Max turns exceeded: {turns}")


class ToolNotFoundError(DataHarnessError, KeyError):
    """A tool invocation named a tool that is not registered."""

    code = "tool_not_found"


class SubagentRecursionError(DataHarnessError, RuntimeError):
    """A subagent tried to spawn another subagent."""

    code = "subagent_recursion"


class ProviderError(DataHarnessError, RuntimeError):
    """The model provider failed: rate limit, auth, timeout, malformed reply.

    Distinct from a tool failing, which is reported to the model as a tool
    result rather than raised. A provider failure ends the run.

    Also a `RuntimeError`, because that is what a failed run raised before the
    taxonomy existed and callers catch it that way.
    """

    code = "provider_error"


class ExecutionError(DataHarnessError, RuntimeError):
    """Sandboxed code could not be run: timeout, killed, environment missing.

    Distinct from code that ran and raised, which is the model's problem and
    is handed back to it as a tool result to fix.
    """

    code = "execution_error"


class ConfigurationError(DataHarnessError, ValueError):
    """The harness was wired up in a way that cannot work.

    Raised at construction where possible, so a misconfiguration fails before
    any tokens are spent rather than on turn nine. Also a `ValueError`, which
    is what these checks raised before the taxonomy.
    """

    code = "configuration_error"
