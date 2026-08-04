"""Compaction: keeping a long run inside the context window.

Before this the only context management was `max_turns`, and hitting it raised
`MaxTurnsExceeded`. That is a wall, not a strategy: a run that needs thirty
turns fails at twenty-five having spent the money for all of them.

Compaction summarises the older part of the conversation and replays the
summary in its place. Two things make it safe rather than lossy:

**The cut lands on a turn boundary.** An assistant message asking for a tool
is never separated from the result answering it. Split them and the provider
rejects the request outright, which is the first bug everyone writes here.

**Nothing is deleted.** The summary is a session entry, so the compacted turns
stay in the tree and moving the leaf back restores them. Compaction is a view,
not an edit.

There is also a reason compaction costs this library less than it costs a
coding agent: the data lives in the session cache under handles, not in the
transcript. Compacting away the turn that loaded a DataFrame does not lose the
DataFrame, only the discussion of it. What is being summarised is reasoning,
not state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from data_harness.core.session.entries import Entry, MessageEntry
from data_harness.llm.types import (
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

#: Rough characters per token. Deliberately crude: the decision this feeds is
#: "are we near the limit", and being wrong by 20% moves the trigger point
#: slightly rather than breaking anything. A real tokeniser would tie the core
#: to a provider's vocabulary for no benefit at this precision.
CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class CompactionSettings:
    """When to compact and how much to keep.

    Attributes:
        context_window: The model's limit, in tokens.
        reserve_tokens: Headroom left for the next response and its tools.
            Compaction triggers once the context exceeds the window minus
            this.
        keep_recent_tokens: Roughly how much recent conversation to carry
            through uncompacted, so the model keeps its immediate footing.
    """

    context_window: int = 128_000
    reserve_tokens: int = 24_000
    keep_recent_tokens: int = 16_000

    def __post_init__(self) -> None:
        if self.keep_recent_tokens >= self.context_window - self.reserve_tokens:
            raise ValueError(
                "keep_recent_tokens must leave room under the trigger point, "
                f"got keep={self.keep_recent_tokens} with window="
                f"{self.context_window} and reserve={self.reserve_tokens}"
            )


def estimate_tokens(messages: list[Message]) -> int:
    """A crude token estimate for a conversation. See `CHARS_PER_TOKEN`."""
    characters = 0
    for message in messages:
        for block in message.content:
            if isinstance(block, TextBlock):
                characters += len(block.text)
            elif isinstance(block, ToolUseBlock):
                characters += len(block.tool_name) + len(str(block.tool_input))
            elif isinstance(block, ToolResultBlock):
                characters += len(block.content)
    return characters // CHARS_PER_TOKEN


def should_compact(context_tokens: int, settings: CompactionSettings) -> bool:
    """Whether the conversation has grown past its safe size."""
    return context_tokens > settings.context_window - settings.reserve_tokens


def starts_a_turn(entries: list[Entry], index: int) -> bool:
    """Whether ``index`` is a safe place to cut.

    Safe means a user message that is not carrying tool results. Cutting
    before a tool-result message would orphan the assistant tool call that
    asked for it, and cutting before an assistant message would open the
    conversation with one, which providers reject.
    """
    entry = entries[index]
    if not isinstance(entry, MessageEntry) or entry.message is None:
        return False
    if entry.message.role != "user":
        return False
    return not any(
        isinstance(block, ToolResultBlock) for block in entry.message.content
    )


def find_cut_point(entries: list[Entry], settings: CompactionSettings) -> str | None:
    """The id of the first entry to keep, or ``None`` if there is nowhere safe.

    Walks back from the newest entry accumulating tokens, then keeps walking
    until it reaches a turn boundary. Returning ``None`` means the whole
    conversation is one indivisible turn, which is not compactable: better to
    do nothing than to produce a transcript the provider will refuse.
    """
    kept = 0
    boundary: int | None = None
    for index in range(len(entries) - 1, -1, -1):
        entry = entries[index]
        if isinstance(entry, MessageEntry) and entry.message is not None:
            kept += estimate_tokens([entry.message])
        if kept >= settings.keep_recent_tokens and starts_a_turn(entries, index):
            boundary = index
            break

    if boundary is None or boundary == 0:
        # Nothing before the boundary to compact.
        return None
    return entries[boundary].id


def messages_before(entries: list[Entry], cut_id: str) -> list[Message]:
    """The messages a compaction would summarise: everything before the cut."""
    summarised: list[Message] = []
    for entry in entries:
        if entry.id == cut_id:
            break
        if isinstance(entry, MessageEntry) and entry.message is not None:
            summarised.append(entry.message)
    return summarised


SUMMARY_INSTRUCTIONS = """\
Summarise the conversation so far for an agent that will continue the work \
without seeing it.

Keep: the user's goal, decisions taken and why, findings that later steps \
depend on, and the names of any cached data handles that were created.

Do not restate the data itself. It is still available under its handles; only \
the discussion of it is being replaced."""


#: Given the messages being replaced, return prose standing in for them.
Summarizer = Callable[[list[Message]], str]

#: Called before each turn with the run's session. Returns the id of the
#: compaction entry it appended, or ``None`` if it did nothing.
Compactor = Callable[[object], "str | None"]


def make_compactor(
    summarize: Summarizer,
    settings: CompactionSettings | None = None,
) -> Compactor:
    """Build a `Compactor` the harness consults before each turn.

    ``summarize`` is injected rather than built in because summarising means
    calling a model, and which model at what cost is the application's
    decision, not the loop's.
    """
    resolved = settings if settings is not None else CompactionSettings()

    def compact(session: object) -> str | None:
        return maybe_compact(session, summarize, resolved)

    return compact


def maybe_compact(
    session, summarize: Summarizer, settings: CompactionSettings
) -> str | None:
    """Compact ``session`` if it has grown too large. Returns the new entry id.

    Does nothing, and says so by returning ``None``, when the context is small
    enough or when there is no safe place to cut.
    """
    context = session.build_context()
    tokens = estimate_tokens(context)
    if not should_compact(tokens, settings):
        return None

    entries = session.context_entries()
    cut_id = find_cut_point(entries, settings)
    if cut_id is None:
        return None

    replaced = messages_before(entries, cut_id)
    if not replaced:
        return None

    return session.append_compaction(
        summary=summarize(replaced),
        first_kept_entry_id=cut_id,
        tokens_before=tokens,
    )
