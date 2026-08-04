"""The data domain's answer to `RunEnvironment`, backed by a `SessionCache`.

This is the seam where the handle/snapshot discipline plugs into the loop. A
tool that returns a DataFrame gets it stored under a handle and reports a
compact snapshot; the model works against the handle name and the raw frame
never enters the transcript.
"""

from __future__ import annotations

from typing import Any, Literal, cast

from data_harness.core.environment import RunState
from data_harness.core.result import CacheStorageInfo
from data_harness.data.cache import SessionCache
from data_harness.data.format import format_tool_output


class CacheEnvironment:
    """A `RunEnvironment` whose state is a `SessionCache`."""

    def __init__(self, cache: SessionCache | None = None) -> None:
        self.cache = cache if cache is not None else SessionCache()

    def render_tool_output(self, value: Any) -> str:
        """Inline small values; spill large ones into the cache as handles."""
        return format_tool_output(value, cache=self.cache)

    def capture(self) -> RunState:
        return RunState(
            snapshots=self.cache.list_handles(),
            storage={
                name: CacheStorageInfo(
                    location=cast(Literal["memory", "disk"], meta["location"]),
                    storage_type=meta["storage_type"],
                )
                for name, meta in self.cache.storage_metadata().items()
            },
            value=self.cache.get_answer(),
            artifacts=self.cache.list_charts(),
        )

    def storage_metadata(self) -> dict[str, dict[str, str]]:
        return self.cache.storage_metadata()
