"""Code-replay cache.

Caches the *sequence of interpreter / SQL code* an agent ran for a given
question + data *schema* (not the data itself) + system prompt. On a later hit
the recorded code is replayed deterministically against the current cache —
re-executing against fresh data — without calling the model at all. This cuts
latency and tokens to zero on repeat questions while staying correct when the
underlying data changes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from data_harness.llm.types import Message, ToolResultBlock, ToolUseBlock

_REPLAYABLE_TOOLS = ("python_interpreter", "sql_query")


@dataclass
class CachedRun:
    """A recorded run: the replayable code steps and the final text."""

    steps: list[dict] = field(default_factory=list)
    text: str = ""

    def to_dict(self) -> dict:
        return {"steps": self.steps, "text": self.text}

    @classmethod
    def from_dict(cls, data: dict) -> CachedRun:
        return cls(steps=data.get("steps", []), text=data.get("text", ""))


class ExecutionCache:
    """A keyed store of `CachedRun` records, optionally persisted to JSON."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._data: dict[str, dict] = {}
        if self._path is not None and self._path.exists():
            self._data = json.loads(self._path.read_text())

    def get(self, key: str) -> CachedRun | None:
        record = self._data.get(key)
        return CachedRun.from_dict(record) if record is not None else None

    def put(self, key: str, run: CachedRun) -> None:
        self._data[key] = run.to_dict()
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._data, indent=2))

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data


def _structural(snapshot: str) -> str:
    """Reduce a snapshot to its schema (type/columns/dtype), dropping data."""
    try:
        obj = json.loads(snapshot)
    except (ValueError, TypeError):
        return snapshot[:50]
    keep: dict = {"type": obj.get("type")}
    if "columns" in obj:
        keep["columns"] = obj["columns"]
    if "dtype" in obj:
        keep["dtype"] = obj["dtype"]
    return json.dumps(keep, sort_keys=True)


def schema_fingerprint(cache) -> str:
    """A stable fingerprint of the cache's schema, independent of row values."""
    parts = [
        f"{name}={_structural(cache.snapshot(name))}"
        for name in sorted(cache.handle_names())
    ]
    return "|".join(parts)


def make_key(question: str, cache, system: str) -> str:
    """Build the cache key from question, data schema, and system prompt."""
    raw = json.dumps(
        {"q": question, "schema": schema_fingerprint(cache), "sys": system},
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def extract_steps(messages: list[Message]) -> list[dict]:
    """Pull *successful* replayable tool calls (interpreter / SQL) from history.

    Tool calls whose result was an error are skipped: the model recovered from
    them with a later call, so replaying them would just re-raise.
    """
    errored: set[str] = set()
    for msg in messages:
        for block in msg.content:
            if isinstance(block, ToolResultBlock) and block.is_error:
                errored.add(block.tool_use_id)

    steps: list[dict] = []
    for msg in messages:
        if msg.role != "assistant":
            continue
        for block in msg.content:
            if (
                isinstance(block, ToolUseBlock)
                and block.tool_name in _REPLAYABLE_TOOLS
                and block.tool_use_id not in errored
            ):
                steps.append({"tool": block.tool_name, "input": block.tool_input})
    return steps
