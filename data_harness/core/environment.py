"""What the loop needs from a domain, and nothing more.

The loop has to do two things it cannot decide by itself: turn a tool's return
value into text for the transcript, and describe the run's final state for the
`RunResult`. For a data harness both answers come from the session cache: a
DataFrame becomes a handle plus a snapshot, and the final state is the set of
handles, the recorded answer, and any charts.

None of that is true of an agent in another domain, so the loop asks a
`RunEnvironment` instead of reaching for a `SessionCache`. That is the whole
reason `data_harness.core` can be read, tested, and reused without pandas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from data_harness.core.artifacts import ChartArtifact
from data_harness.core.result import CacheStorageInfo


@dataclass(frozen=True)
class RunState:
    """The domain's contribution to a `RunResult`, captured when a run ends.

    Attributes:
        snapshots: Handle name -> compact, payload-free description.
        storage: Handle name -> where the value physically lives.
        value: The structured final answer the run recorded, if any.
        artifacts: Charts or other artefacts produced during the run.
    """

    snapshots: dict[str, str] = field(default_factory=dict)
    storage: dict[str, CacheStorageInfo] = field(default_factory=dict)
    value: Any = None
    artifacts: list[ChartArtifact] = field(default_factory=list)


@runtime_checkable
class RunEnvironment(Protocol):
    """The domain services the loop depends on.

    Implementations must not raise from these methods: they run while the loop
    is assembling a turn or a result, where an exception would lose the run
    rather than report it.
    """

    def render_tool_output(self, value: Any) -> str:
        """Render a tool's return value as the text the model will read.

        Deliberately not named after `data.format.format_tool_output`, which
        is only one possible implementation of it.
        """
        ...

    def capture(self) -> RunState:
        """Describe the current domain state for a `RunResult`."""
        ...

    def storage_metadata(self) -> dict[str, dict[str, str]]:
        """Per-handle storage detail for the turn log. Empty when there is none."""
        ...


class NullEnvironment:
    """A domain-free environment: no cache, no handles, no artefacts.

    The default, so `data_harness.core` stands alone. Tool output is rendered
    with `str`, which is right for text-returning tools and lossy for anything
    large: that is exactly the problem the data layer's cache solves.
    """

    def render_tool_output(self, value: Any) -> str:
        if isinstance(value, Exception):
            return f"Error: {type(value).__name__}: {value}"
        return str(value)

    def capture(self) -> RunState:
        return RunState()

    def storage_metadata(self) -> dict[str, dict[str, str]]:
        return {}
