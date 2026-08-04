"""Where a session's entries live.

The store is deliberately dumb: append, look up, walk parents. All the meaning
is in `Session`. That split is what lets an application swap Postgres in
without the session logic knowing, and it is why the loop no longer opens
files itself.
"""

from __future__ import annotations

import copy
from typing import Protocol, runtime_checkable

from data_harness.core.session.entries import Entry, leaf_after


class SessionStoreError(Exception):
    """Raised for a session that cannot be read or is internally inconsistent.

    Carries a stable ``code`` so a caller can tell a missing entry from a
    corrupt file without matching on message text.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@runtime_checkable
class SessionStore(Protocol):
    """Append-only storage for one session's entries."""

    @property
    def session_id(self) -> str: ...

    @property
    def leaf_id(self) -> str | None:
        """The entry the next append will hang from. ``None`` for an empty session."""
        ...

    def append(self, entry: Entry) -> None:
        """Persist ``entry`` and advance the leaf. Ids must be unique."""
        ...

    def get(self, entry_id: str) -> Entry | None: ...

    def entries(self) -> list[Entry]:
        """Every entry, in the order appended. Includes abandoned branches."""
        ...

    def set_leaf(self, entry_id: str | None) -> None:
        """Move the active leaf, recording the move as an entry."""
        ...

    def path_to_root(self, entry_id: str | None) -> list[Entry]:
        """Root-first path to ``entry_id``, following ``parent_id``."""
        ...


class MemorySessionStore:
    """In-process store. The default, and what tests use.

    Also the honest default for a server that keeps sessions in memory: it
    makes the durability boundary explicit rather than implying a session
    survives a restart when it does not.
    """

    def __init__(self, session_id: str = "session") -> None:
        self._session_id = session_id
        self._entries: list[Entry] = []
        self._by_id: dict[str, Entry] = {}
        self._leaf_id: str | None = None

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def leaf_id(self) -> str | None:
        return self._leaf_id

    def append(self, entry: Entry) -> None:
        """Store a snapshot of ``entry``.

        Copied, not referenced. A caller holding the same `Message` object can
        otherwise keep editing it and silently rewrite history, and the JSONL
        store (which serialises on write) would then disagree with this one
        about what was sent.
        """
        entry = copy.deepcopy(entry)
        if entry.id in self._by_id:
            raise SessionStoreError(
                "duplicate_entry", f"Entry {entry.id} already exists"
            )
        if entry.parent_id is not None and entry.parent_id not in self._by_id:
            raise SessionStoreError(
                "missing_parent",
                f"Entry {entry.id} names unknown parent {entry.parent_id}",
            )
        self._entries.append(entry)
        self._by_id[entry.id] = entry
        self._leaf_id = leaf_after(entry)

    def get(self, entry_id: str) -> Entry | None:
        return self._by_id.get(entry_id)

    def entries(self) -> list[Entry]:
        return list(self._entries)

    def set_leaf(self, entry_id: str | None) -> None:
        from data_harness.core.session.entries import LeafEntry, new_entry_id, utc_now

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
