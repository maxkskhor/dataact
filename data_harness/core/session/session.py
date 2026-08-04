"""The session tree, and the context derived from it.

`Session` is the only thing that gives entries meaning. It appends, it moves
the leaf, and it answers the one question the loop actually needs: what
messages should the model see right now.

That answer is computed on every call from the path root-to-leaf. Nothing
caches it, because a cache is what turns "the history" into a second source of
truth that can disagree with the log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from data_harness.core.session.entries import (
    CompactionEntry,
    CustomEntry,
    Entry,
    LabelEntry,
    LeafEntry,
    MessageEntry,
    TurnEntry,
    new_entry_id,
    utc_now,
)
from data_harness.core.session.store import MemorySessionStore, SessionStore
from data_harness.llm.types import (
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

#: Turns a custom entry into messages for the model, or nothing.
#:
#: Custom entries are invisible to the model by default: the data layer
#: records a cache write so a human can trace it, not so the model re-reads it.
#: A projector opts one in.
CustomProjector = Callable[[CustomEntry], list[Message]]


@dataclass
class SessionStats:
    """Totals across every entry, including branches no longer on the path."""

    entries: int = 0
    messages: int = 0
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class Session:
    """An append-only tree of entries, plus the context it derives.

    Args:
        store: Where entries live. Defaults to memory.
        projectors: Custom entry type -> how to render it for the model.
    """

    store: SessionStore = field(default_factory=MemorySessionStore)
    projectors: dict[str, CustomProjector] = field(default_factory=dict)

    # ── appending ───────────────────────────────────────────────────────────

    def _append(self, make: Callable[[str, str | None, str], Entry]) -> str:
        entry = make(new_entry_id(), self.store.leaf_id, utc_now())
        self.store.append(entry)
        return entry.id

    def append_message(self, message: Message) -> str:
        return self._append(
            lambda i, p, t: MessageEntry(
                id=i, parent_id=p, timestamp=t, message=message
            )
        )

    def append_turn(self, **fields: Any) -> str:
        return self._append(
            lambda i, p, t: TurnEntry(id=i, parent_id=p, timestamp=t, **fields)
        )

    def append_custom(self, custom_type: str, data: dict[str, Any]) -> str:
        return self._append(
            lambda i, p, t: CustomEntry(
                id=i, parent_id=p, timestamp=t, custom_type=custom_type, data=data
            )
        )

    def append_compaction(
        self, summary: str, first_kept_entry_id: str | None, tokens_before: int
    ) -> str:
        """Summarise history before ``first_kept_entry_id``.

        The kept id must be on the current path. Anything else is a bug in the
        caller: an id from an abandoned branch, or one recorded before a fork,
        would silently amnesia the agent rather than fail.

        Raises:
            SessionStoreError: If the kept id is not on the current path.
        """
        if first_kept_entry_id is not None:
            from data_harness.core.session.store import SessionStoreError

            if first_kept_entry_id not in {e.id for e in self.branch()}:
                raise SessionStoreError(
                    "invalid_cut",
                    f"Compaction keeps from {first_kept_entry_id}, which is not "
                    "on the current path",
                )
        return self._append(
            lambda i, p, t: CompactionEntry(
                id=i,
                parent_id=p,
                timestamp=t,
                summary=summary,
                first_kept_entry_id=first_kept_entry_id,
                tokens_before=tokens_before,
            )
        )

    def label(self, target_id: str, label: str | None) -> str:
        """Name an entry, so a branch point can be found again by a human."""
        if self.store.get(target_id) is None:
            from data_harness.core.session.store import SessionStoreError

            raise SessionStoreError("not_found", f"Entry {target_id} not found")
        return self._append(
            lambda i, p, t: LabelEntry(
                id=i, parent_id=p, timestamp=t, target_id=target_id, label=label
            )
        )

    # ── navigating ──────────────────────────────────────────────────────────

    @property
    def leaf_id(self) -> str | None:
        return self.store.leaf_id

    def move_to(self, entry_id: str | None) -> None:
        """Make ``entry_id`` the active leaf. The next append branches from it.

        The old branch is untouched and still reachable: this is how a turn is
        retried with a different model without losing what the first attempt
        did.
        """
        self.store.set_leaf(entry_id)

    def branch(self, from_id: str | None = None) -> list[Entry]:
        """Root-first path to ``from_id``, defaulting to the active leaf."""
        return self.store.path_to_root(
            from_id if from_id is not None else self.store.leaf_id
        )

    def labels(self) -> dict[str, str]:
        """Entry id -> current label. Later labels win; ``None`` clears."""
        found: dict[str, str] = {}
        for entry in self.store.entries():
            if isinstance(entry, LabelEntry):
                if entry.label:
                    found[entry.target_id] = entry.label
                else:
                    found.pop(entry.target_id, None)
        return found

    # ── deriving the context ────────────────────────────────────────────────

    def context_entries(self, from_id: str | None = None) -> list[Entry]:
        """The entries a run should replay, after applying compaction.

        Everything before the newest compaction on the path is dropped, except
        the tail it chose to keep. The dropped entries stay in the tree.
        """
        path = self.branch(from_id)

        newest_compaction = None
        compaction_index = -1
        for index, entry in enumerate(path):
            if isinstance(entry, CompactionEntry):
                newest_compaction = entry
                compaction_index = index
        if newest_compaction is None:
            return path

        kept: list[Entry] = [newest_compaction]
        if newest_compaction.first_kept_entry_id is not None:
            keeping = False
            for entry in path[:compaction_index]:
                if entry.id == newest_compaction.first_kept_entry_id:
                    keeping = True
                if not keeping:
                    continue
                # An older compaction inside the kept tail has already been
                # subsumed by this one. Replaying it would emit a second,
                # older summary *after* the newer one and resurrect the very
                # entries it had dropped.
                if isinstance(entry, CompactionEntry):
                    continue
                kept.append(entry)
        kept.extend(path[compaction_index + 1 :])
        return kept

    def build_context(self, from_id: str | None = None) -> list[Message]:
        """The messages the model should see. Derived, never stored.

        Always something a provider will accept: a tool call with no result,
        or a result with no call, is dropped. Both are reachable without any
        bug here. A run killed between issuing a tool call and recording its
        result leaves the call orphaned on disk, and resuming would otherwise
        send a transcript the provider rejects outright, breaking resume for
        exactly the sessions most worth resuming.
        """
        messages: list[Message] = []
        for entry in self.context_entries(from_id):
            if isinstance(entry, MessageEntry) and entry.message is not None:
                messages.append(entry.message)
            elif isinstance(entry, CompactionEntry):
                messages.append(
                    Message(
                        role="user",
                        content=[
                            TextBlock(
                                text=(
                                    "Summary of the earlier conversation:\n"
                                    f"{entry.summary}"
                                )
                            )
                        ],
                    )
                )
            elif isinstance(entry, CustomEntry):
                projector = self.projectors.get(entry.custom_type)
                if projector is not None:
                    messages.extend(projector(entry))
        return _drop_unpaired_tool_blocks(messages)

    # ── reporting ───────────────────────────────────────────────────────────

    def stats(self) -> SessionStats:
        stats = SessionStats()
        for entry in self.store.entries():
            stats.entries += 1
            if isinstance(entry, MessageEntry):
                stats.messages += 1
            elif isinstance(entry, TurnEntry):
                stats.turns += 1
                stats.input_tokens += entry.input_tokens
                stats.output_tokens += entry.output_tokens
                stats.cache_read_tokens += entry.cache_read_tokens
                stats.cache_write_tokens += entry.cache_write_tokens
        return stats

    def custom_entries(self, custom_type: str) -> list[CustomEntry]:
        """Every custom entry of one type, across all branches."""
        return [
            entry
            for entry in self.store.entries()
            if isinstance(entry, CustomEntry) and entry.custom_type == custom_type
        ]


def _drop_unpaired_tool_blocks(messages: list[Message]) -> list[Message]:
    """Remove tool calls with no result, and results with no call.

    Providers reject either. Both arise legitimately: a run killed between
    issuing a call and recording its result orphans the call, and a compaction
    cut between the two orphans the result. A message left with no content is
    dropped too, since an empty message is itself invalid.
    """
    answered = {
        block.tool_use_id
        for message in messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
    }
    requested = {
        block.tool_use_id
        for message in messages
        for block in message.content
        if isinstance(block, ToolUseBlock)
    }

    repaired: list[Message] = []
    for message in messages:
        content = [
            block
            for block in message.content
            if not (
                isinstance(block, ToolUseBlock) and block.tool_use_id not in answered
            )
            and not (
                isinstance(block, ToolResultBlock)
                and block.tool_use_id not in requested
            )
        ]
        if content:
            repaired.append(Message(role=message.role, content=content))
    return repaired


def leaf_entries(session: Session) -> list[LeafEntry]:
    """Every recorded leaf move, oldest first. The session's branch history."""
    return [e for e in session.store.entries() if isinstance(e, LeafEntry)]
