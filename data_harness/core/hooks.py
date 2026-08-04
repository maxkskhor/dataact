"""Hooks: the loop's extension points.

Before this, the only way to influence a run without forking the loop was
`register_reminder`, which could append text to the prompt suffix and nothing
else. Everything more interesting was hardcoded. The interpreter approval gate
lived inside the dispatcher and was keyed on the literal tool name
``"python_interpreter"``; a budget check had to be reimplemented by every
application on top.

A hook observes an event and may return a decision. Returning ``None`` means
"no opinion", which is what most hooks do most of the time.

Contract, and it matters: **a hook must not raise.** It runs while the loop is
assembling a turn, where an exception loses the run instead of reporting it.
Hooks that raise anyway are caught and reported through `HookError` rather
than corrupting the run, but a hook that needs to signal a problem should
return a decision saying so.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from data_harness.llm.types import Message, ToolResultBlock

# ── events ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BeforeTurn:
    """A provider turn is about to start.

    Return a `Reminder` to append text to the conversation suffix, or a `Stop`
    to end the run before spending anything.
    """

    turn: int
    max_turns: int
    messages: list[Message]


@dataclass(frozen=True)
class BeforeToolCall:
    """A tool is about to run, after the model chose it.

    Return a `Block` to refuse the call. The model is told what it was told,
    so the reason is part of the prompt, not just a log line.
    """

    turn: int
    tool_name: str
    tool_input: dict


@dataclass(frozen=True)
class AfterToolCall:
    """A tool returned. Return a `Replace` to rewrite what the model sees."""

    turn: int
    tool_name: str
    tool_input: dict
    result: ToolResultBlock


@dataclass(frozen=True)
class AfterTurn:
    """A turn finished and has been accounted for.

    Return a `Stop` to end the run. This is where a spend cap belongs: the
    tokens are already counted, so the decision is made on real numbers.
    """

    turn: int
    max_turns: int
    input_tokens: int
    output_tokens: int
    tool_results: list[ToolResultBlock]


Event = BeforeTurn | BeforeToolCall | AfterToolCall | AfterTurn


# ── decisions ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Reminder:
    """Append ``text`` to the conversation suffix before the turn.

    Suffix-only on purpose: the system prompt stays byte-identical across
    turns so the provider's cache of it is not invalidated.
    """

    text: str


@dataclass(frozen=True)
class Block:
    """Refuse the tool call and return ``reason`` to the model instead.

    ``is_error`` defaults to False because a refusal is a decision, not a
    malfunction: telling the model its code was broken makes it rewrite and
    retry rather than stop.
    """

    reason: str
    is_error: bool = False


@dataclass(frozen=True)
class Replace:
    """Rewrite a tool result before the model sees it."""

    content: str
    is_error: bool = False


@dataclass(frozen=True)
class Stop:
    """End the run cleanly. ``reason`` is recorded on the `RunResult`."""

    reason: str


Decision = Reminder | Block | Replace | Stop
Hook = Callable[[Any], Decision | None]

_E = TypeVar("_E", bound=Event)
_D = TypeVar("_D", bound=Decision)


class HookError(Exception):
    """A hook raised. Names the hook and the event so the culprit is obvious."""

    def __init__(self, event: Event, hook: Hook, cause: BaseException) -> None:
        name = getattr(hook, "__qualname__", repr(hook))
        super().__init__(
            f"Hook {name} raised on {type(event).__name__}: {cause!r}. "
            "Hooks must return a decision rather than raise."
        )
        self.event = event
        self.hook = hook
        self.__cause__ = cause


@dataclass
class HookRegistry:
    """Hooks grouped by the event type they care about, in registration order."""

    hooks: dict[type, list[Hook]] = field(default_factory=dict)

    def add(self, event_type: type[_E], hook: Callable[[_E], Decision | None]) -> None:
        self.hooks.setdefault(event_type, []).append(hook)

    def emit(self, event: Event) -> list[Decision]:
        """Run every hook for ``event`` and collect the decisions.

        Every hook sees the event, even after one has decided: a hook that
        records spend should not be skipped because an earlier one asked to
        stop. Precedence between conflicting decisions is the caller's to
        resolve.

        Raises:
            HookError: If a hook raises. The loop turns this into a failed
                run rather than letting it escape mid-turn.
        """
        decisions: list[Decision] = []
        for hook in self.hooks.get(type(event), ()):
            try:
                decision = hook(event)
            except Exception as exc:  # noqa: BLE001 - re-raised as HookError
                raise HookError(event, hook, exc) from exc
            if decision is not None:
                decisions.append(decision)
        return decisions

    def first(self, event: Event, kind: type[_D]) -> _D | None:
        """The first decision of type ``kind``, or ``None``.

        Later decisions of the same kind are discarded. Conflicting policies
        are the caller's problem to avoid, not this class's to arbitrate.
        """
        for decision in self.emit(event):
            if isinstance(decision, kind):
                return decision
        return None

    def copy(self) -> HookRegistry:
        """An independent registry with the same hooks.

        Taken whenever a registry is handed to a harness. The harness adds its
        own hooks (the approval gate, for one), and writing those into the
        caller's object would leak one harness's policy into every other
        harness sharing it, which is exactly the reuse the parameter exists
        to support.
        """
        return HookRegistry({k: list(v) for k, v in self.hooks.items()})
