"""The typed entries a session is made of.

A session is an append-only log of these, each naming its parent. That makes
the log a *tree*, not a list, and the tree is the session's actual state: the
conversation the model sees is derived by walking root to leaf, never stored.

Deriving rather than storing is what makes the rest work. Compaction becomes
an entry that changes where the walk starts, not a destructive edit. Retrying
a turn with a different model is a second child of the same parent, and both
survive. Nothing is ever overwritten, so "what did the agent actually see at
turn 7" is a question with an answer.

The previous design was a write-only JSONL log that re-serialised the entire
message history on every turn: quadratic to write, and impossible to read back
into a runnable state.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from data_harness.llm.types import Message


def new_entry_id() -> str:
    """A short, sortable-enough id. Collisions are checked by the store."""
    return uuid.uuid4().hex[:12]


def utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


@dataclass(frozen=True)
class BaseEntry:
    """Common shape. ``parent_id`` is ``None`` only for the first entry.

    Frozen: an entry is a fact about something that happened. Correcting the
    record means appending, not editing.
    """

    id: str = field(default_factory=new_entry_id)
    parent_id: str | None = None
    timestamp: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class MessageEntry(BaseEntry):
    """One message in the conversation."""

    type: Literal["message"] = "message"
    message: Message | None = None


@dataclass(frozen=True)
class TurnEntry(BaseEntry):
    """What one provider turn cost, for accounting and replay analysis."""

    type: Literal["turn"] = "turn"
    turn: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_ms: float = 0.0
    stop_reason: str | None = None
    tool_error_count: int = 0
    visible_tools: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CompactionEntry(BaseEntry):
    """History before ``first_kept_entry_id`` is replaced by ``summary``.

    The compacted entries stay in the tree. Only the derived context shrinks,
    so a compaction is auditable and reversible: move the leaf back before it
    and the full history returns.
    """

    type: Literal["compaction"] = "compaction"
    summary: str = ""
    first_kept_entry_id: str | None = None
    tokens_before: int = 0


@dataclass(frozen=True)
class LabelEntry(BaseEntry):
    """A human-readable name for another entry. ``None`` clears it."""

    type: Literal["label"] = "label"
    target_id: str = ""
    label: str | None = None


@dataclass(frozen=True)
class LeafEntry(BaseEntry):
    """Records that the active leaf moved, so branching is itself in the log."""

    type: Literal["leaf"] = "leaf"
    target_id: str | None = None


@dataclass(frozen=True)
class CustomEntry(BaseEntry):
    """An entry type the core does not know about.

    The extension point that keeps `core` domain-free while letting the data
    layer record what matters to it: a cache handle being written, a chart
    being produced. `data_harness.data.session` defines those.
    """

    type: Literal["custom"] = "custom"
    custom_type: str = ""
    data: dict[str, Any] = field(default_factory=dict)


Entry = (
    MessageEntry | TurnEntry | CompactionEntry | LabelEntry | LeafEntry | CustomEntry
)

#: Discriminator value -> class, for decoding a stored entry.
ENTRY_TYPES: dict[str, type] = {
    "message": MessageEntry,
    "turn": TurnEntry,
    "compaction": CompactionEntry,
    "label": LabelEntry,
    "leaf": LeafEntry,
    "custom": CustomEntry,
}


def leaf_after(entry: Entry) -> str | None:
    """Where the leaf sits once ``entry`` is appended.

    Every entry advances the leaf to itself, except a `LeafEntry`, whose whole
    purpose is to move it somewhere else.
    """
    return entry.target_id if isinstance(entry, LeafEntry) else entry.id
