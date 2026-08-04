"""A session persisted as one JSONL file: a header, then one line per entry.

Appending is one line, so writing a session is linear in the number of
entries. The older `runs/*.jsonl` turn log re-serialises the whole message
history every turn, which is quadratic and cannot be read back into anything
runnable. This store is what replaces it; that log is still written alongside
for now, so per-turn write cost is unchanged until it is removed.

The format is deliberately boring. A session file is greppable, diffable, and
recoverable by hand, which matters more for a debugging artefact than
compactness does.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from data_harness.core.session.entries import (
    ENTRY_TYPES,
    Entry,
    LeafEntry,
    MessageEntry,
    leaf_after,
    new_entry_id,
    utc_now,
)
from data_harness.core.session.store import SessionStoreError
from data_harness.llm.types import (
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

FORMAT_VERSION = 1


def _encode_block(block: Any) -> dict[str, Any]:
    """Encode one content block so it decodes back to an equal object.

    Deliberately not `to_jsonable`: that produces the *log* shape, which
    renames fields for readability and snapshots large values away. Lossy is
    right for a debugging log and wrong for a store whose whole purpose is to
    reconstruct a runnable conversation.
    """
    if isinstance(block, ToolUseBlock):
        return {
            "type": "tool_use",
            "tool_use_id": block.tool_use_id,
            "tool_name": block.tool_name,
            "tool_input": block.tool_input,
        }
    if isinstance(block, ToolResultBlock):
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": block.content,
            "is_error": block.is_error,
        }
    return {"type": "text", "text": getattr(block, "text", str(block))}


def _encode_message(message: Message) -> dict[str, Any]:
    return {
        "role": message.role,
        "content": [_encode_block(b) for b in message.content],
    }


def _decode_message(raw: dict[str, Any]) -> Message:
    blocks: list[Any] = []
    for block in raw.get("content", []):
        kind = block.get("type")
        if kind == "tool_use":
            blocks.append(
                ToolUseBlock(
                    tool_use_id=block["tool_use_id"],
                    tool_name=block["tool_name"],
                    tool_input=block.get("tool_input", {}),
                )
            )
        elif kind == "tool_result":
            blocks.append(
                ToolResultBlock(
                    tool_use_id=block["tool_use_id"],
                    content=block.get("content", ""),
                    is_error=block.get("is_error", False),
                )
            )
        else:
            blocks.append(TextBlock(text=block.get("text", "")))
    return Message(role=raw["role"], content=blocks)


def encode_entry(entry: Entry) -> dict[str, Any]:
    raw = dataclasses.asdict(entry)
    if isinstance(entry, MessageEntry) and entry.message is not None:
        raw["message"] = _encode_message(entry.message)
    return raw


def decode_entry(raw: dict[str, Any], *, source: str, line: int) -> Entry:
    kind = raw.get("type")
    cls = ENTRY_TYPES.get(kind or "")
    if cls is None:
        raise SessionStoreError(
            "invalid_entry", f"{source}:{line} unknown entry type {kind!r}"
        )
    fields = {f.name for f in dataclasses.fields(cls)}
    kwargs = {k: v for k, v in raw.items() if k in fields and k != "type"}
    if cls is MessageEntry and isinstance(kwargs.get("message"), dict):
        kwargs["message"] = _decode_message(kwargs["message"])
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise SessionStoreError(
            "invalid_entry", f"{source}:{line} could not be decoded: {exc}"
        ) from exc


class JsonlSessionStore:
    """A `SessionStore` backed by an append-only JSONL file."""

    def __init__(self, path: str | Path, session_id: str) -> None:
        self._path = Path(path)
        self._session_id = session_id
        self._entries: list[Entry] = []
        self._by_id: dict[str, Entry] = {}
        self._leaf_id: str | None = None
        #: Set when opening had to drop a torn final entry.
        self._truncated = False

    # ── lifecycle ───────────────────────────────────────────────────────────

    @classmethod
    def create(cls, path: str | Path, session_id: str) -> JsonlSessionStore:
        """Start a new session file, overwriting any file already there."""
        store = cls(path, session_id)
        store._path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "type": "session",
            "version": FORMAT_VERSION,
            "id": session_id,
            "timestamp": utc_now(),
        }
        store._path.write_text(json.dumps(header) + "\n")
        return store

    @classmethod
    def open(cls, path: str | Path) -> JsonlSessionStore:
        """Load an existing session file, replaying its entries.

        Raises:
            SessionStoreError: If the file is missing, has no header, or
                contains an entry that cannot be decoded.
        """
        p = Path(path)
        if not p.exists():
            raise SessionStoreError("not_found", f"No session file at {p}")
        lines = [line for line in p.read_text().splitlines() if line.strip()]
        if not lines:
            raise SessionStoreError("invalid_session", f"{p} is empty")

        try:
            header = json.loads(lines[0])
        except json.JSONDecodeError as exc:
            raise SessionStoreError(
                "invalid_session", f"{p}:1 header is not valid JSON"
            ) from exc
        if header.get("type") != "session":
            raise SessionStoreError("invalid_session", f"{p}:1 is not a session header")
        if header.get("version") != FORMAT_VERSION:
            raise SessionStoreError(
                "invalid_session",
                f"{p}:1 unsupported session version {header.get('version')!r}",
            )

        store = cls(p, header["id"])
        last_number = len(lines)
        for number, line in enumerate(lines[1:], start=2):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                if number == last_number:
                    # A crash mid-append leaves a torn final line. This is an
                    # append-only recovery log: losing the last entry is the
                    # cost of the crash, but refusing the file would lose every
                    # entry before it, which is the opposite of the point.
                    store._truncated = True
                    break
                raise SessionStoreError(
                    "invalid_entry", f"{p}:{number} is not valid JSON"
                ) from exc
            entry = decode_entry(raw, source=str(p), line=number)
            store._entries.append(entry)
            store._by_id[entry.id] = entry
            store._leaf_id = leaf_after(entry)
        return store

    @classmethod
    def open_or_create(cls, path: str | Path, session_id: str) -> JsonlSessionStore:
        """Open an existing session, or start one if the file is not there.

        What a restarting process almost always wants. `create` truncates,
        which on a restart path destroys the very session this store exists to
        preserve.
        """
        if Path(path).exists():
            return cls.open(path)
        return cls.create(path, session_id)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def truncated(self) -> bool:
        """Whether opening the file had to drop a torn final entry."""
        return self._truncated

    # ── SessionStore ────────────────────────────────────────────────────────

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def leaf_id(self) -> str | None:
        return self._leaf_id

    def append(self, entry: Entry) -> None:
        if entry.id in self._by_id:
            raise SessionStoreError(
                "duplicate_entry", f"Entry {entry.id} already exists"
            )
        if entry.parent_id is not None and entry.parent_id not in self._by_id:
            raise SessionStoreError(
                "missing_parent",
                f"Entry {entry.id} names unknown parent {entry.parent_id}",
            )
        try:
            line = json.dumps(encode_entry(entry))
        except (TypeError, ValueError) as exc:
            raise SessionStoreError(
                "unserializable_entry",
                f"Entry {entry.id} cannot be written as JSON: {exc}",
            ) from exc
        with self._path.open("a") as handle:
            handle.write(line + "\n")
        self._entries.append(entry)
        self._by_id[entry.id] = entry
        self._leaf_id = leaf_after(entry)

    def get(self, entry_id: str) -> Entry | None:
        return self._by_id.get(entry_id)

    def entries(self) -> list[Entry]:
        return list(self._entries)

    def set_leaf(self, entry_id: str | None) -> None:
        if entry_id is not None and entry_id not in self._by_id:
            raise SessionStoreError("not_found", f"Entry {entry_id} not found")
        self.append(
            LeafEntry(
                id=new_entry_id(),
                parent_id=self._leaf_id,
                timestamp=utc_now(),
                target_id=entry_id,
            )
        )

    def path_to_root(self, entry_id: str | None) -> list[Entry]:
        if entry_id is None:
            return []
        path: list[Entry] = []
        seen: set[str] = set()
        current = self._by_id.get(entry_id)
        if current is None:
            raise SessionStoreError("not_found", f"Entry {entry_id} not found")
        while current is not None:
            if current.id in seen:
                raise SessionStoreError(
                    "cycle", f"Entry {current.id} is its own ancestor"
                )
            seen.add(current.id)
            path.append(current)
            if current.parent_id is None:
                break
            parent = self._by_id.get(current.parent_id)
            if parent is None:
                raise SessionStoreError(
                    "missing_parent", f"Entry {current.parent_id} not found"
                )
            current = parent
        path.reverse()
        return path
