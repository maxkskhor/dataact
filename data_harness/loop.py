"""The ReAct loop.

There is exactly one loop implementation: `AsyncHarness._turns`. Everything
else in this module is a facade over it.

- `AsyncHarness.run_result` / `ask_result` drain the loop and return a
  `RunResult`.
- `AsyncHarness.run_stream` / `ask_stream` yield the loop's `StreamEvent`s and
  leave the `RunResult` on `last_result`.
- `Harness` is the synchronous facade: it bridges a sync `ProviderAdapter` onto
  the async loop and drives it to completion.

Before this collapse there were four near-copies of the loop (sync/async x
result/stream) that had already drifted apart: the streaming path silently
dropped token usage on provider errors, and `AsyncAgent` was missing features
`Agent` had. One implementation means one place for behaviour to live.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import functools
from collections.abc import AsyncGenerator, Callable, Coroutine
from typing import Any, Literal, TypeVar, cast

from data_harness.cache import SessionCache
from data_harness.exceptions import MaxTurnsExceeded
from data_harness.format import format_tool_output
from data_harness.logger import log_error_turn, log_turn, setup_logger
from data_harness.observe import time_block
from data_harness.providers.base import (
    AsyncProviderAdapter,
    NormalizedResponse,
    ProviderAdapter,
    StopReason,
)
from data_harness.result import CacheStorageInfo, RunResult, Usage
from data_harness.streaming import (
    StreamEvent,
    ToolResultEvent,
    accumulate_stream_events,
)
from data_harness.types import (
    Message,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)

_MAX_TURN_REMINDER = (
    "This is the final turn. You MUST produce your complete final output now. "
    "Do not use any more tools. Respond with your answer directly."
)

_T = TypeVar("_T")


def _evaluate_code_gate(
    on_code: Callable[[str], object] | None,
    code_only: bool,
    code: str,
) -> str | None:
    """Decide whether interpreter ``code`` may run.

    Returns a string to short-circuit execution (a dry-run echo or a denial
    message returned to the model), or ``None`` to proceed.
    """
    if code_only:
        return f"DRY RUN — code not executed:\n{code}"
    if on_code is not None:
        decision = on_code(code)
        if decision is False:
            return "Execution blocked by the approval gate."
        if isinstance(decision, str):
            return decision
    return None


def run_coroutine_blocking(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run ``coro`` to completion from synchronous code.

    Uses `asyncio.run` when no event loop is running. When one already is (a
    Jupyter kernel, an async web handler calling into sync code), `asyncio.run`
    would raise, so the coroutine is handed to a dedicated worker thread with
    its own loop instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class _SyncAdapterBridge(AsyncProviderAdapter):
    """Presents a synchronous `ProviderAdapter` as an `AsyncProviderAdapter`.

    Used only by the synchronous `Harness` facade, which owns its own event
    loop. The provider call is made inline rather than on a worker thread:
    nothing else is scheduled on that loop, so blocking it is harmless and it
    keeps provider calls on the caller's thread exactly as before the collapse.
    """

    def __init__(self, adapter: ProviderAdapter) -> None:
        self._adapter = adapter

    async def chat(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> NormalizedResponse:
        return self._adapter.chat(system=system, messages=messages, tools=tools)

    def format_cache_control(self, obj: dict) -> dict:
        return self._adapter.format_cache_control(obj)


def as_async_adapter(
    adapter: ProviderAdapter | AsyncProviderAdapter,
) -> AsyncProviderAdapter:
    """Return ``adapter`` as an async adapter, bridging a synchronous one.

    Lets callers accept either adapter kind without branching. The bridge is a
    pass-through for adapters that are already async.
    """
    if isinstance(adapter, AsyncProviderAdapter):
        return adapter
    return _SyncAdapterBridge(adapter)


class AsyncHarness:
    """The core ReAct loop.

    `AsyncHarness` owns the message list, dispatches tools, applies suffix-only
    reminder hooks, and logs every turn to a JSONL file. It is the central
    implementation boundary in data-harness: everything above it (`Agent`,
    `AgentSession`) is a convenience layer; everything below it
    (`AsyncProviderAdapter`, `SessionCache`, `ToolSpec`) is a pure dependency.

    The system prompt is never mutated between turns. Reminders, nags, and
    dynamic state are always appended to the conversation suffix so the
    provider's KV cache is not invalidated.

    For most use cases, prefer `AsyncAgent` over constructing `AsyncHarness`
    directly. Use `AsyncHarness` when you need full control over tool wiring,
    as shown in ``examples/advanced_wiring.py``.

    Args:
        adapter: Async provider adapter that translates provider SDK objects
            into harness types.
        system: System prompt. Kept byte-identical across all turns.
        tools: Full tool list. Invisible tools (``visible=False``) are excluded
            from the provider call but can still be dispatched.
        max_turns: Hard cap on provider turns before the loop stops and returns
            a ``"max_turns_exceeded"`` result.
        run_dir: Directory where JSONL logs are written. Created on first run.
        cache: Shared `SessionCache`. A fresh cache is created if ``None``.
        on_code: Optional approval gate called with interpreter code before it
            runs. Return ``False`` to block, or a string to return that string
            to the model instead of executing.
        code_only: When ``True``, interpreter code is echoed back as a dry run
            and never executed.
    """

    def __init__(
        self,
        adapter: AsyncProviderAdapter,
        system: str,
        tools: list[ToolSpec],
        max_turns: int = 25,
        run_dir: str = "./runs",
        cache: SessionCache | None = None,
        on_code: Callable[[str], object] | None = None,
        code_only: bool = False,
    ) -> None:
        if max_turns < 1:
            raise ValueError(f"max_turns must be at least 1, got {max_turns!r}")
        self._adapter = adapter
        self._system = system
        self._tools = list(tools)
        self._max_turns = max_turns
        self._run_dir = run_dir
        self._cache = cache if cache is not None else SessionCache()
        self._messages: list[Message] = []
        self._reminders: list[Callable[[int, int], str | None]] = []
        self._run_file: str | None = None
        self._on_code = on_code
        self._code_only = code_only
        self._last_result: RunResult | None = None

    def register_reminder(self, hook: Callable[[int, int], str | None]) -> None:
        """Register a suffix reminder hook called before each provider turn.

        The hook receives ``(current_turn, max_turns)`` and returns a reminder
        string to append to the conversation suffix, or ``None`` to skip.

        Args:
            hook: Callable with signature ``(turn: int, max_turns: int) -> str | None``.
        """
        self._reminders.append(hook)

    @property
    def run_file(self) -> str | None:
        """Path to the JSONL log for this run, or ``None`` before the first run."""
        return self._run_file

    @property
    def last_result(self) -> RunResult | None:
        """`RunResult` for the most recently completed loop, including streamed ones.

        Streaming callers get their token usage, cache snapshots, charts, and
        error status from here: the event protocol itself carries no run-level
        summary.
        """
        return self._last_result

    # ── inspection ──────────────────────────────────────────────────────────
    #
    # The loop's state is readable so callers can assert on what the model was
    # actually shown. `tools`, `messages`, and `reminders` return the live
    # lists: mutating them mutates the harness, which is how tools are added
    # to an already-constructed session.

    @property
    def system(self) -> str:
        """The system prompt. Byte-identical across every turn of a run."""
        return self._system

    @property
    def tools(self) -> list[ToolSpec]:
        """The full tool list, including invisible tools. Live and mutable."""
        return self._tools

    @property
    def max_turns(self) -> int:
        """Hard cap on provider turns per run."""
        return self._max_turns

    @property
    def cache(self) -> SessionCache:
        """The `SessionCache` backing tool results and handles."""
        return self._cache

    @property
    def messages(self) -> list[Message]:
        """The conversation history the model sees. Live and mutable."""
        return self._messages

    @property
    def reminders(self) -> list[Callable[[int, int], str | None]]:
        """Registered suffix reminder hooks, in registration order."""
        return self._reminders

    # ── entry points ────────────────────────────────────────────────────────

    async def run_result(
        self,
        user_message: str,
        *,
        run_id: str | None = None,
        session_id: str | None = None,
    ) -> RunResult:
        """Start a fresh run and return the full `RunResult`.

        Resets message history. Use `ask_result` for follow-up turns on the
        same history.

        Args:
            user_message: The initial user prompt.
            run_id: Optional identifier stamped into the `RunResult`.
            session_id: Optional session identifier stamped into the `RunResult`.

        Returns:
            A `RunResult` describing the outcome, token usage, and cache state.
        """
        self._begin_run(user_message)
        result = await self._drain(stream=False)
        return self._stamp(result, run_id, session_id)

    async def ask_result(
        self,
        user_message: str,
        *,
        run_id: str | None = None,
        session_id: str | None = None,
    ) -> RunResult:
        """Append a follow-up message and continue the existing run.

        Appends ``user_message`` to the current history without resetting it.

        Args:
            user_message: The follow-up user prompt.
            run_id: Optional identifier stamped into the `RunResult`.
            session_id: Optional session identifier stamped into the `RunResult`.

        Returns:
            A `RunResult` describing the outcome of this turn sequence.
        """
        self._begin_ask(user_message)
        result = await self._drain(stream=False)
        return self._stamp(result, run_id, session_id)

    async def run(self, user_message: str) -> str:
        """Start a fresh run and return the final text response.

        Args:
            user_message: The initial user prompt.

        Returns:
            The model's final text response.

        Raises:
            MaxTurnsExceeded: If the loop reaches ``max_turns`` without stopping.
            RuntimeError: If the provider raises an exception during the run.
        """
        return _unwrap(await self.run_result(user_message))

    async def ask(self, user_message: str) -> str:
        """Append a follow-up message and return the final text response.

        Args:
            user_message: The follow-up user prompt.

        Returns:
            The model's final text response.

        Raises:
            MaxTurnsExceeded: If the loop reaches ``max_turns`` without stopping.
            RuntimeError: If the provider raises an exception during the run.
        """
        return _unwrap(await self.ask_result(user_message))

    async def run_stream(self, user_message: str) -> AsyncGenerator[StreamEvent, None]:
        """Stream events for a one-shot run.

        Yields StreamEvent objects following the same protocol as the Claude
        Agent SDK. Each provider turn emits message_start,
        content_block_start/delta/stop, message_delta, and message_stop events.
        After the harness dispatches a tool call a ToolResultEvent is emitted.
        The JSONL logger records fully assembled messages, not individual events.

        When the generator is exhausted, `last_result` holds the `RunResult`
        for the run, including token usage and any error.
        """
        self._begin_run(user_message)
        async for event in self._turns(stream=True):
            yield event

    async def ask_stream(self, user_message: str) -> AsyncGenerator[StreamEvent, None]:
        """Stream events for a follow-up turn in a session.

        When the generator is exhausted, `last_result` holds the `RunResult`.
        """
        self._begin_ask(user_message)
        async for event in self._turns(stream=True):
            yield event

    # ── loop plumbing ───────────────────────────────────────────────────────

    def _begin_run(self, user_message: str) -> None:
        self._run_file = setup_logger(self._run_dir)
        self._messages = [Message(role="user", content=[TextBlock(text=user_message)])]

    def _begin_ask(self, user_message: str) -> None:
        if self._run_file is None:
            self._run_file = setup_logger(self._run_dir)
        self._messages.append(
            Message(role="user", content=[TextBlock(text=user_message)])
        )

    async def _drain(self, *, stream: bool) -> RunResult:
        async for _ in self._turns(stream=stream):
            pass
        result = self._last_result
        if result is None:  # pragma: no cover - _turns always sets a result
            raise RuntimeError("loop finished without producing a result")
        return result

    def _stamp(
        self, result: RunResult, run_id: str | None, session_id: str | None
    ) -> RunResult:
        stamped = dataclasses.replace(result, run_id=run_id, session_id=session_id)
        self._last_result = stamped
        return stamped

    async def _turns(self, *, stream: bool) -> AsyncGenerator[StreamEvent, None]:
        """The loop. The only one.

        ``stream`` selects how a turn's response is obtained: real provider
        stream events, or a single assembled `chat` call. Every other concern
        (reminders, tool dispatch, logging, result construction, error
        handling) is shared.
        """
        if self._run_file is None:
            raise RuntimeError("run_file must be initialised before running the loop")

        self._last_result = None
        total_usage = Usage()

        for turn in range(1, self._max_turns + 1):
            self._apply_reminders(turn)
            visible_tools = [t for t in self._tools if t.visible]

            events_this_turn: list[StreamEvent] = []
            with time_block() as tb:
                try:
                    if stream:
                        async for evt in self._adapter.stream_events(
                            system=self._system,
                            messages=self._messages,
                            tools=visible_tools,
                        ):
                            events_this_turn.append(evt)
                            yield evt
                        response = accumulate_stream_events(events_this_turn)
                    else:
                        response = await self._adapter.chat(
                            system=self._system,
                            messages=self._messages,
                            tools=visible_tools,
                        )
                except Exception as exc:
                    log_error_turn(
                        turn=turn,
                        system=self._system,
                        messages=self._messages,
                        error=repr(exc),
                        run_file=self._run_file,
                    )
                    self._last_result = self._build_result(
                        text="",
                        status="error",
                        turns=turn,
                        stop_reason=None,
                        usage=total_usage,
                        error=repr(exc),
                    )
                    return

            latency = tb.elapsed_ms

            total_usage = total_usage + Usage(
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cache_read_tokens=response.cache_read_tokens,
                cache_write_tokens=response.cache_write_tokens,
            )

            self._messages.append(Message(role="assistant", content=response.content))

            tool_results: list[ToolResultBlock] = []

            if response.stop_reason == StopReason.TOOL_USE:
                tool_results = await self._dispatch_tools(response.content)

                if stream:
                    tool_name_map = {
                        b.tool_use_id: b.tool_name
                        for b in response.content
                        if isinstance(b, ToolUseBlock)
                    }
                    for result_block in tool_results:
                        yield ToolResultEvent(
                            tool_use_id=result_block.tool_use_id,
                            tool_name=tool_name_map.get(result_block.tool_use_id, ""),
                            content=result_block.content,
                            is_error=result_block.is_error,
                        )

                self._messages.append(Message(role="user", content=list(tool_results)))

            log_turn(
                turn=turn,
                system=self._system,
                messages=self._messages,
                response=response,
                tool_results=tool_results,
                latency_ms=latency,
                run_file=self._run_file,
                cache_storage=self._cache.storage_metadata(),
                visible_tools=[t.name for t in visible_tools],
                tool_error_count=sum(1 for r in tool_results if r.is_error),
                all_tools=self._tools,
            )

            if response.stop_reason != StopReason.TOOL_USE:
                self._last_result = self._build_result(
                    text=_extract_text(response),
                    status="success",
                    turns=turn,
                    stop_reason=response.stop_reason,
                    usage=total_usage,
                )
                return

            if turn == self._max_turns:
                self._last_result = self._build_result(
                    text=_extract_text(response),
                    status="max_turns_exceeded",
                    turns=turn,
                    stop_reason=None,
                    usage=total_usage,
                )
                return

    def _build_result(
        self,
        *,
        text: str,
        status: Literal["success", "max_turns_exceeded", "error"],
        turns: int,
        stop_reason: StopReason | None,
        usage: Usage,
        error: str | None = None,
    ) -> RunResult:
        return RunResult(
            text=text,
            status=status,
            turns=turns,
            run_file=self._run_file,
            stop_reason=stop_reason,
            usage=usage,
            cache_snapshots=self._cache.list_handles(),
            cache_storage=self._build_cache_storage(),
            value=self._cache.get_answer(),
            charts=self._cache.list_charts(),
            error=error,
        )

    def _build_cache_storage(self) -> dict[str, CacheStorageInfo]:
        raw = self._cache.storage_metadata()
        return {
            name: CacheStorageInfo(
                location=cast(Literal["memory", "disk"], meta["location"]),
                storage_type=meta["storage_type"],
            )
            for name, meta in raw.items()
        }

    def _apply_reminders(self, turn: int) -> None:
        reminder_texts: list[str] = []

        for hook in self._reminders:
            text = hook(turn, self._max_turns)
            if text:
                reminder_texts.append(text)

        # Built-in max-turn reminder
        if turn == self._max_turns - 1:
            reminder_texts.append(_MAX_TURN_REMINDER)

        if not reminder_texts:
            return

        reminder_block = TextBlock(text="\n\n".join(reminder_texts))

        # Append to existing user message or create a new one
        if self._messages and self._messages[-1].role == "user":
            self._messages[-1].content.append(reminder_block)
        else:
            self._messages.append(Message(role="user", content=[reminder_block]))

    async def _dispatch_tools(self, content: list) -> list[ToolResultBlock]:
        tool_uses = [b for b in content if isinstance(b, ToolUseBlock)]
        results: list[ToolResultBlock] = []
        tool_map = {t.name: t for t in self._tools}

        for tub in tool_uses:
            spec = tool_map.get(tub.tool_name)
            if spec is None or spec.handler is None:
                results.append(
                    ToolResultBlock(
                        tool_use_id=tub.tool_use_id,
                        content=f"Tool not found: {tub.tool_name!r}",
                        is_error=True,
                    )
                )
                continue
            if tub.tool_name == "python_interpreter":
                gate = _evaluate_code_gate(
                    self._on_code, self._code_only, tub.tool_input.get("code", "")
                )
                if gate is not None:
                    results.append(
                        ToolResultBlock(
                            tool_use_id=tub.tool_use_id,
                            content=gate,
                            is_error=False,
                        )
                    )
                    continue
            try:
                if asyncio.iscoroutinefunction(spec.handler):
                    raw = await spec.handler(**tub.tool_input)
                else:
                    raw = await asyncio.to_thread(
                        functools.partial(spec.handler, **tub.tool_input)
                    )
                output = format_tool_output(raw, cache=self._cache)
            except Exception as exc:
                results.append(
                    ToolResultBlock(
                        tool_use_id=tub.tool_use_id,
                        content=repr(exc),
                        is_error=True,
                    )
                )
                continue
            results.append(
                ToolResultBlock(
                    tool_use_id=tub.tool_use_id,
                    content=output,
                    is_error=False,
                )
            )

        return results


class Harness:
    """Synchronous facade over `AsyncHarness`.

    Holds no loop logic of its own: it wraps the supplied synchronous
    `ProviderAdapter` in an async bridge and drives `AsyncHarness` to
    completion. Behaviour matches `AsyncHarness` exactly, because it *is*
    `AsyncHarness`.

    Safe to call from inside a running event loop (a Jupyter kernel, for
    instance); the loop is then driven on a dedicated worker thread.

    Args:
        adapter: Synchronous provider adapter.
        system: System prompt. Kept byte-identical across all turns.
        tools: Full tool list.
        max_turns: Hard cap on provider turns.
        run_dir: Directory where JSONL logs are written.
        cache: Shared `SessionCache`. A fresh cache is created if ``None``.
        on_code: Optional approval gate for interpreter code.
        code_only: When ``True``, interpreter code is never executed.
    """

    def __init__(
        self,
        adapter: ProviderAdapter,
        system: str,
        tools: list[ToolSpec],
        max_turns: int = 25,
        run_dir: str = "./runs",
        cache: SessionCache | None = None,
        on_code: Callable[[str], object] | None = None,
        code_only: bool = False,
    ) -> None:
        self._inner = AsyncHarness(
            adapter=_SyncAdapterBridge(adapter),
            system=system,
            tools=tools,
            max_turns=max_turns,
            run_dir=run_dir,
            cache=cache,
            on_code=on_code,
            code_only=code_only,
        )
        self._adapter = adapter

    def register_reminder(self, hook: Callable[[int, int], str | None]) -> None:
        """Register a suffix reminder hook called before each provider turn."""
        self._inner.register_reminder(hook)

    @property
    def run_file(self) -> str | None:
        """Path to the JSONL log for this run, or ``None`` before the first run."""
        return self._inner.run_file

    @property
    def last_result(self) -> RunResult | None:
        """`RunResult` for the most recently completed run."""
        return self._inner.last_result

    @property
    def system(self) -> str:
        """The system prompt. Byte-identical across every turn of a run."""
        return self._inner.system

    @property
    def tools(self) -> list[ToolSpec]:
        """The full tool list, including invisible tools. Live and mutable."""
        return self._inner.tools

    @property
    def max_turns(self) -> int:
        """Hard cap on provider turns per run."""
        return self._inner.max_turns

    @property
    def cache(self) -> SessionCache:
        """The `SessionCache` backing tool results and handles."""
        return self._inner.cache

    @property
    def messages(self) -> list[Message]:
        """The conversation history the model sees. Live and mutable."""
        return self._inner.messages

    @property
    def reminders(self) -> list[Callable[[int, int], str | None]]:
        """Registered suffix reminder hooks, in registration order."""
        return self._inner.reminders

    def run_result(
        self,
        user_message: str,
        *,
        run_id: str | None = None,
        session_id: str | None = None,
    ) -> RunResult:
        """Start a fresh run and return the full `RunResult`."""
        return run_coroutine_blocking(
            self._inner.run_result(user_message, run_id=run_id, session_id=session_id)
        )

    def ask_result(
        self,
        user_message: str,
        *,
        run_id: str | None = None,
        session_id: str | None = None,
    ) -> RunResult:
        """Append a follow-up message and continue the existing run."""
        return run_coroutine_blocking(
            self._inner.ask_result(user_message, run_id=run_id, session_id=session_id)
        )

    def run(self, user_message: str) -> str:
        """Start a fresh run and return the final text response.

        Raises:
            MaxTurnsExceeded: If the loop reaches ``max_turns`` without stopping.
            RuntimeError: If the provider raises an exception during the run.
        """
        return _unwrap(self.run_result(user_message))

    def ask(self, user_message: str) -> str:
        """Append a follow-up message and return the final text response.

        Raises:
            MaxTurnsExceeded: If the loop reaches ``max_turns`` without stopping.
            RuntimeError: If the provider raises an exception during the run.
        """
        return _unwrap(self.ask_result(user_message))


def _unwrap(result: RunResult) -> str:
    """Return the text of a successful run, or raise the matching exception."""
    if result.status == "max_turns_exceeded":
        raise MaxTurnsExceeded(result.turns)
    if result.status == "error":
        raise RuntimeError(result.error or "unknown error")
    return result.text


def _extract_text(response: NormalizedResponse) -> str:
    return "\n".join(b.text for b in response.content if isinstance(b, TextBlock))
