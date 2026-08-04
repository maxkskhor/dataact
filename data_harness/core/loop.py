"""The ReAct loop.

There is exactly one loop implementation: `_HarnessBase._plan`. It is a
generator that owns every decision in a turn (reminders, message assembly,
tool gating, when to stop, what the `RunResult` says) and does no *network*
I/O itself. Instead it asks for it by yielding an effect:

- `CallProvider` — run one provider turn
- `CallTool` — invoke one tool handler
- `ToolFinished` — a tool result block is ready, emit an event if you want one

The driver answers `CallProvider` and `CallTool` with `Ok(value)` or
`Failed(exc)`; the loop, not the driver, decides what a failure means. The
wrapper matters: without it a handler that legitimately returned a `Failed`
would be mistaken for one that raised.

Turn logging is the one effect not modelled as an effect: `_plan` writes JSONL
to disk directly. That is a deliberate leftover, and it is why the "no I/O"
claim above is scoped to the network. Phase 3 replaces the log with a session
store and the write moves behind an interface.

Two drivers perform the effects:

- `Harness` runs everything inline on the calling thread. No event loop is
  created, so `KeyboardInterrupt` interrupts promptly, handlers keep their
  thread affinity (a `sqlite3` connection built at setup still works), and the
  caller's ambient event loop is untouched.
- `AsyncHarness` awaits the provider, offloads blocking handlers to
  `asyncio.to_thread` so they cannot stall the event loop, and can stream.

This replaced four near-copies of the loop (sync/async x result/stream) that
had already drifted: the streaming copy swallowed provider errors without
recording the tokens it had already spent.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import enum
import functools
import sys
from collections.abc import AsyncGenerator, Callable, Coroutine, Generator
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from data_harness.core.environment import NullEnvironment, RunEnvironment
from data_harness.core.logger import log_error_turn, log_turn, setup_logger
from data_harness.core.observe import time_block
from data_harness.core.result import RunResult, Usage, unwrap_text
from data_harness.llm.providers.base import (
    AsyncProviderAdapter,
    NormalizedResponse,
    ProviderAdapter,
    StopReason,
)
from data_harness.llm.streaming import (
    StreamEvent,
    ToolResultEvent,
    accumulate_stream_events,
)
from data_harness.llm.types import (
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


# ── effects ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CallProvider:
    """Run one provider turn. Answer with `Ok(NormalizedResponse)` or `Failed`."""

    system: str
    messages: list[Message]
    tools: list[ToolSpec]


@dataclass(frozen=True)
class CallTool:
    """Invoke one tool handler. Answer with `Ok(return_value)` or `Failed`."""

    tool_use_id: str
    tool_name: str
    handler: Callable[..., Any]
    tool_input: dict


@dataclass(frozen=True)
class ToolFinished:
    """A tool result block is ready. Answer with ``None``.

    Emitted for every result, including tool-not-found and gate-blocked calls,
    so a streaming driver can surface them all.
    """

    tool_name: str
    block: ToolResultBlock


@dataclass(frozen=True)
class Ok:
    """An effect succeeded, carrying whatever it produced.

    Successes are wrapped rather than sent raw so that no possible return
    value can be confused with a `Failed`.
    """

    value: Any


@dataclass(frozen=True)
class Failed:
    """An effect raised. The loop, not the driver, decides what that means."""

    error: BaseException


Effect = CallProvider | CallTool | ToolFinished

#: What a driver may send back. `CallProvider` and `CallTool` require an
#: `Ok` or a `Failed`; `ToolFinished` is an announcement and takes ``None``.
Answer = Ok | Failed | None


def _require_answer(effect: Effect, answer: Answer) -> Ok | Failed:
    """Reject a driver that answered a request with nothing.

    Without this the loop would do ``answer.value`` and raise
    ``AttributeError: 'NoneType' object has no attribute 'value'`` several
    frames from the driver that actually got it wrong.
    """
    if isinstance(answer, (Ok, Failed)):
        return answer
    raise TypeError(
        f"{type(effect).__name__} must be answered with Ok(...) or Failed(...), "
        f"got {answer!r}"
    )


# ── helpers ─────────────────────────────────────────────────────────────────


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


class _Unreadable(enum.Enum):
    """Sentinel type, so `_ambient_event_loop` keeps a real return union."""

    TOKEN = "unreadable"


#: Returned by `_ambient_event_loop` when the thread's loop cannot be read
#: without risking a side effect. Distinct from ``None``, which means "read
#: successfully, and there is no loop".
_UNREADABLE = _Unreadable.TOKEN


def _ambient_event_loop() -> asyncio.AbstractEventLoop | None | _Unreadable:
    """The loop already installed on this thread, ``None``, or `_UNREADABLE`.

    Must not install one as a side effect. Reading it needs two strategies,
    because the safe way to ask changed:

    - 3.14 and later: `asyncio.get_event_loop` raises when no loop is set
      instead of creating one, so asking is safe. The policy API it used to
      need is deprecated there and goes away in 3.16.
    - 3.10 to 3.13: `get_event_loop` *creates* a loop when none is set, which
      would leave a never-run, never-closed loop and its selector fd behind on
      a thread that only ever used the synchronous API. There is no public
      read-only accessor, so the policy's private slot is what we read.

    Under an exotic policy that does not expose that slot we decline to guess:
    creating a loop to find out whether one exists is the exact bug this
    guards against.
    """
    if sys.version_info >= (3, 14):
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            return None

    policy = asyncio.get_event_loop_policy()
    local = getattr(policy, "_local", None)
    if local is None:
        return _UNREADABLE
    return getattr(local, "_loop", None)


def run_coroutine_blocking(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run ``coro`` to completion from synchronous code.

    Only needed when synchronous code must drive something inherently async
    (an `AsyncProviderAdapter`, an async tool handler). The synchronous
    `Harness` does not use this for its own loop.

    Restores whatever event loop was installed on the calling thread, because
    `asyncio.run` otherwise clears it. When a loop is already *running*, the
    coroutine goes to a worker thread with its own loop, since `asyncio.run`
    cannot be nested.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        previous = _ambient_event_loop()
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                loop.close()
                # `set_event_loop(loop)` already ran, so simply skipping the
                # restore would leave the thread pointing at a *closed* loop,
                # which is worse than the leak this branch exists to avoid.
                # When the previous loop was unreadable, `None` is the only
                # honest answer.
                asyncio.set_event_loop(None if previous is _UNREADABLE else previous)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


# ── the loop ────────────────────────────────────────────────────────────────


class _HarnessBase:
    """Loop state, pure turn logic, and the effect generator.

    Both `Harness` and `AsyncHarness` inherit this, so the loop exists once and
    subclasses of either see the same state and the same overridable seams.
    """

    def __init__(
        self,
        system: str,
        tools: list[ToolSpec],
        max_turns: int = 25,
        run_dir: str = "./runs",
        environment: RunEnvironment | None = None,
        on_code: Callable[[str], object] | None = None,
        code_only: bool = False,
    ) -> None:
        if max_turns < 1:
            raise ValueError(f"max_turns must be at least 1, got {max_turns!r}")
        self._system = system
        self._tools = list(tools)
        self._max_turns = max_turns
        self._run_dir = run_dir
        self._environment = (
            environment if environment is not None else NullEnvironment()
        )
        self._messages: list[Message] = []
        self._reminders: list[Callable[[int, int], str | None]] = []
        self._run_file: str | None = None
        self._on_code = on_code
        self._code_only = code_only
        self._last_result: RunResult | None = None
        self._turns_completed = 0
        self._usage_so_far = Usage()

    def register_reminder(self, hook: Callable[[int, int], str | None]) -> None:
        """Register a suffix reminder hook called before each provider turn.

        The hook receives ``(current_turn, max_turns)`` and returns a reminder
        string to append to the conversation suffix, or ``None`` to skip.

        Args:
            hook: Callable with signature ``(turn: int, max_turns: int) -> str | None``.
        """
        self._reminders.append(hook)

    # ── inspection ──────────────────────────────────────────────────────────
    #
    # `tools`, `messages`, and `reminders` return the live lists: mutating them
    # mutates the harness, which is how tools get added to a live session.

    @property
    def run_file(self) -> str | None:
        """Path to the JSONL log for this run, or ``None`` before the first run."""
        return self._run_file

    @property
    def last_result(self) -> RunResult | None:
        """`RunResult` for the most recently completed loop, including streamed ones.

        Streaming callers get their token usage, cache snapshots, charts, and
        error status from here: the event protocol carries no run-level summary.
        Stays ``None`` if a stream is abandoned before it is exhausted.
        """
        return self._last_result

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
    def environment(self) -> RunEnvironment:
        """The domain services this run uses. See `RunEnvironment`."""
        return self._environment

    @property
    def messages(self) -> list[Message]:
        """The conversation history the model sees. Live and mutable."""
        return self._messages

    @property
    def reminders(self) -> list[Callable[[int, int], str | None]]:
        """Registered suffix reminder hooks, in registration order."""
        return self._reminders

    # ── run setup ───────────────────────────────────────────────────────────

    def _begin_run(self, user_message: str) -> None:
        self._run_file = setup_logger(self._run_dir)
        self._messages = [Message(role="user", content=[TextBlock(text=user_message)])]

    def _begin_ask(self, user_message: str) -> None:
        if self._run_file is None:
            self._run_file = setup_logger(self._run_dir)
        self._messages.append(
            Message(role="user", content=[TextBlock(text=user_message)])
        )

    def _stamp(
        self, result: RunResult, run_id: str | None, session_id: str | None
    ) -> RunResult:
        """Attach run/session ids and make the result the harness's latest.

        Internal, but reached from `data_harness.app.agent` so that a streamed run
        is identified exactly like a non-streamed one. Mutates
        `self._last_result`; anything reading it afterwards sees the stamped
        object.
        """
        stamped = dataclasses.replace(result, run_id=run_id, session_id=session_id)
        self._last_result = stamped
        return stamped

    # ── the loop ────────────────────────────────────────────────────────────

    def _plan(self) -> Generator[Effect, Any, None]:
        """Drive one run, yielding the I/O it needs. The only loop in the library.

        Sets `last_result` before returning, on every exit path, including
        abandonment.
        """
        if self._run_file is None:
            raise RuntimeError("run_file must be initialised before running the loop")

        self._last_result = None
        total_usage = Usage()

        try:
            yield from self._turns(total_usage)
        except GeneratorExit:
            # A consumer that stops mid-stream still spent whatever the
            # completed turns cost. Recording it here is the same rule that
            # made provider errors keep their usage: never drop tokens the
            # provider has already billed just because the caller walked away.
            if self._last_result is None:
                self._last_result = self._build_result(
                    text="",
                    status="error",
                    turns=self._turns_completed,
                    stop_reason=None,
                    usage=self._usage_so_far,
                    error="RunAbandoned('stream closed before the run finished')",
                )
            raise

    def _turns(self, total_usage: Usage) -> Generator[Effect, Any, None]:
        self._turns_completed = 0
        self._usage_so_far = total_usage

        for turn in range(1, self._max_turns + 1):
            self._turns_completed = turn
            self._apply_reminders(turn)
            visible_tools = [t for t in self._tools if t.visible]

            with time_block() as tb:
                request = CallProvider(
                    system=self._system,
                    messages=self._messages,
                    tools=visible_tools,
                )
                outcome = _require_answer(request, (yield request))

            if isinstance(outcome, Failed):
                log_error_turn(
                    turn=turn,
                    system=self._system,
                    messages=self._messages,
                    error=repr(outcome.error),
                    run_file=self._run_file,
                )
                self._last_result = self._build_result(
                    text="",
                    status="error",
                    turns=turn,
                    stop_reason=None,
                    usage=total_usage,
                    error=repr(outcome.error),
                )
                return

            response: NormalizedResponse = outcome.value
            latency = tb.elapsed_ms

            total_usage = total_usage + Usage(
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cache_read_tokens=response.cache_read_tokens,
                cache_write_tokens=response.cache_write_tokens,
            )
            # Mirrored onto the instance so an abandoned run can still report
            # what it spent; the generator's local is gone by then.
            self._usage_so_far = total_usage

            self._messages.append(Message(role="assistant", content=response.content))

            tool_results: list[ToolResultBlock] = []

            if response.stop_reason == StopReason.TOOL_USE:
                # Every tool is dispatched before any result is announced, so a
                # streaming consumer sees the same ordering it always has.
                finished = yield from self._dispatch(response.content)
                tool_results = [block for _, block in finished]
                for tool_name, block in finished:
                    yield ToolFinished(tool_name=tool_name, block=block)
                self._messages.append(Message(role="user", content=list(tool_results)))

            log_turn(
                turn=turn,
                system=self._system,
                messages=self._messages,
                response=response,
                tool_results=tool_results,
                latency_ms=latency,
                run_file=self._run_file,
                cache_storage=self._environment.storage_metadata(),
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

    def _dispatch(
        self, content: list
    ) -> Generator[Effect, Any, list[tuple[str, ToolResultBlock]]]:
        """Turn the assistant's tool-use blocks into result blocks, in order."""
        tool_map = {t.name: t for t in self._tools}
        finished: list[tuple[str, ToolResultBlock]] = []

        for tub in (b for b in content if isinstance(b, ToolUseBlock)):
            spec = tool_map.get(tub.tool_name)
            if spec is None or spec.handler is None:
                finished.append(
                    (
                        tub.tool_name,
                        ToolResultBlock(
                            tool_use_id=tub.tool_use_id,
                            content=f"Tool not found: {tub.tool_name!r}",
                            is_error=True,
                        ),
                    )
                )
                continue

            if tub.tool_name == "python_interpreter":
                gate = _evaluate_code_gate(
                    self._on_code, self._code_only, tub.tool_input.get("code", "")
                )
                if gate is not None:
                    finished.append(
                        (
                            tub.tool_name,
                            ToolResultBlock(
                                tool_use_id=tub.tool_use_id,
                                content=gate,
                                is_error=False,
                            ),
                        )
                    )
                    continue

            request = CallTool(
                tool_use_id=tub.tool_use_id,
                tool_name=tub.tool_name,
                handler=spec.handler,
                tool_input=tub.tool_input,
            )
            answer = _require_answer(request, (yield request))

            if isinstance(answer, Failed):
                finished.append(
                    (
                        tub.tool_name,
                        ToolResultBlock(
                            tool_use_id=tub.tool_use_id,
                            content=repr(answer.error),
                            is_error=True,
                        ),
                    )
                )
                continue

            try:
                output = self._environment.render_tool_output(answer.value)
            except Exception as exc:  # noqa: BLE001 - surfaced to the model
                finished.append(
                    (
                        tub.tool_name,
                        ToolResultBlock(
                            tool_use_id=tub.tool_use_id,
                            content=repr(exc),
                            is_error=True,
                        ),
                    )
                )
                continue

            finished.append(
                (
                    tub.tool_name,
                    ToolResultBlock(
                        tool_use_id=tub.tool_use_id,
                        content=output,
                        is_error=False,
                    ),
                )
            )

        return finished

    # ── pure helpers ────────────────────────────────────────────────────────

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
        state = self._environment.capture()
        return RunResult(
            text=text,
            status=status,
            turns=turns,
            run_file=self._run_file,
            stop_reason=stop_reason,
            usage=usage,
            cache_snapshots=state.snapshots,
            cache_storage=state.storage,
            value=state.value,
            charts=state.artifacts,
            error=error,
        )

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


class Harness(_HarnessBase):
    """The synchronous driver for the loop.

    Runs the provider call and every tool handler inline on the calling
    thread. No event loop is created, so `KeyboardInterrupt` lands promptly and
    handlers keep whatever thread affinity they were built with (a `sqlite3`
    connection opened at setup keeps working). The only exception is an
    ``async def`` tool handler, which necessarily needs a loop.

    `Harness` owns the message list, dispatches tools, applies suffix-only
    reminder hooks, and logs every turn to a JSONL file. It is the central
    implementation boundary in data-harness: everything above it (`Agent`,
    `AgentSession`) is a convenience layer; everything below it
    (`ProviderAdapter`, `RunEnvironment`, `ToolSpec`) is a pure dependency.

    The system prompt is never mutated between turns. Reminders, nags, and
    dynamic state are always appended to the conversation suffix so the
    provider's KV cache is not invalidated.

    For most use cases, prefer `Agent` over constructing `Harness` directly.
    Use `Harness` when you need full control over tool wiring, as shown in
    ``examples/advanced_wiring.py``.

    Args:
        adapter: Synchronous provider adapter that translates provider SDK
            objects into harness types.
        system: System prompt. Kept byte-identical across all turns.
        tools: Full tool list. Invisible tools (``visible=False``) are excluded
            from the provider call but can still be dispatched.
        max_turns: Hard cap on provider turns before the loop stops and returns
            a ``"max_turns_exceeded"`` result.
        run_dir: Directory where JSONL logs are written. Created on first run.
        environment: Domain services (see `RunEnvironment`). Defaults to
            `NullEnvironment`, which renders tool output with `str` and
            contributes no run state.
        on_code: Approval gate called with interpreter code before it runs.
        code_only: When ``True``, interpreter code is echoed, never executed.
    """

    def __init__(
        self,
        adapter: ProviderAdapter,
        system: str,
        tools: list[ToolSpec],
        max_turns: int = 25,
        run_dir: str = "./runs",
        environment: RunEnvironment | None = None,
        on_code: Callable[[str], object] | None = None,
        code_only: bool = False,
    ) -> None:
        super().__init__(
            system=system,
            tools=tools,
            max_turns=max_turns,
            run_dir=run_dir,
            environment=environment,
            on_code=on_code,
            code_only=code_only,
        )
        self._adapter = adapter

    def run_result(
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
        return self._stamp(self._drive(), run_id, session_id)

    def ask_result(
        self,
        user_message: str,
        *,
        run_id: str | None = None,
        session_id: str | None = None,
    ) -> RunResult:
        """Append a follow-up message and continue the existing run.

        Args:
            user_message: The follow-up user prompt.
            run_id: Optional identifier stamped into the `RunResult`.
            session_id: Optional session identifier stamped into the `RunResult`.

        Returns:
            A `RunResult` describing the outcome of this turn sequence.
        """
        self._begin_ask(user_message)
        return self._stamp(self._drive(), run_id, session_id)

    def run(self, user_message: str) -> str:
        """Start a fresh run and return the final text response.

        Args:
            user_message: The initial user prompt.

        Returns:
            The model's final text response.

        Raises:
            MaxTurnsExceeded: If the loop reaches ``max_turns`` without stopping.
            RuntimeError: If the provider raises an exception during the run.
        """
        return unwrap_text(self.run_result(user_message))

    def ask(self, user_message: str) -> str:
        """Append a follow-up message and return the final text response.

        Args:
            user_message: The follow-up user prompt.

        Returns:
            The model's final text response.

        Raises:
            MaxTurnsExceeded: If the loop reaches ``max_turns`` without stopping.
            RuntimeError: If the provider raises an exception during the run.
        """
        return unwrap_text(self.ask_result(user_message))

    # ── driver ──────────────────────────────────────────────────────────────

    def _drive(self) -> RunResult:
        plan = self._plan()
        sent: Answer = None
        while True:
            try:
                effect = plan.send(sent)
            except StopIteration:
                break
            sent = self._perform(effect)
        result = self._last_result
        if result is None:  # pragma: no cover - _plan always sets a result
            raise RuntimeError("loop finished without producing a result")
        return result

    def _perform(self, effect: Effect) -> Answer:
        if isinstance(effect, CallProvider):
            try:
                return Ok(
                    self._adapter.chat(
                        system=effect.system,
                        messages=effect.messages,
                        tools=effect.tools,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - reported as a RunResult
                return Failed(exc)
        if isinstance(effect, CallTool):
            try:
                return Ok(self._call_tool(effect))
            except Exception as exc:  # noqa: BLE001 - surfaced to the model
                return Failed(exc)
        return None

    def _call_tool(self, call: CallTool) -> Any:
        """Invoke one tool handler on the calling thread.

        The overridable seam for changing dispatch (sandboxing, timeouts,
        instrumentation). An ``async def`` handler is the one case that needs
        an event loop, and gets a short-lived one.
        """
        if asyncio.iscoroutinefunction(call.handler):
            return run_coroutine_blocking(call.handler(**call.tool_input))
        return call.handler(**call.tool_input)


class AsyncHarness(_HarnessBase):
    """The asynchronous driver for the loop.

    Awaits the provider and offloads blocking tool handlers to
    `asyncio.to_thread`, so a long pandas operation cannot stall the event
    loop it shares with a web server. Adds token-level streaming.

    Same arguments as `Harness`, but takes an `AsyncProviderAdapter`.
    """

    def __init__(
        self,
        adapter: AsyncProviderAdapter,
        system: str,
        tools: list[ToolSpec],
        max_turns: int = 25,
        run_dir: str = "./runs",
        environment: RunEnvironment | None = None,
        on_code: Callable[[str], object] | None = None,
        code_only: bool = False,
    ) -> None:
        super().__init__(
            system=system,
            tools=tools,
            max_turns=max_turns,
            run_dir=run_dir,
            environment=environment,
            on_code=on_code,
            code_only=code_only,
        )
        self._adapter = adapter

    async def run_result(
        self,
        user_message: str,
        *,
        run_id: str | None = None,
        session_id: str | None = None,
    ) -> RunResult:
        """Start a fresh run and return the full `RunResult`."""
        self._begin_run(user_message)
        return self._stamp(await self._drain(stream=False), run_id, session_id)

    async def ask_result(
        self,
        user_message: str,
        *,
        run_id: str | None = None,
        session_id: str | None = None,
    ) -> RunResult:
        """Append a follow-up message and continue the existing run."""
        self._begin_ask(user_message)
        return self._stamp(await self._drain(stream=False), run_id, session_id)

    async def run(self, user_message: str) -> str:
        """Start a fresh run and return the final text response.

        Raises:
            MaxTurnsExceeded: If the loop reaches ``max_turns`` without stopping.
            RuntimeError: If the provider raises an exception during the run.
        """
        return unwrap_text(await self.run_result(user_message))

    async def ask(self, user_message: str) -> str:
        """Append a follow-up message and return the final text response.

        Raises:
            MaxTurnsExceeded: If the loop reaches ``max_turns`` without stopping.
            RuntimeError: If the provider raises an exception during the run.
        """
        return unwrap_text(await self.ask_result(user_message))

    async def run_stream(self, user_message: str) -> AsyncGenerator[StreamEvent, None]:
        """Stream events for a one-shot run.

        Yields StreamEvent objects following the same protocol as the Claude
        Agent SDK. Each provider turn emits message_start,
        content_block_start/delta/stop, message_delta, and message_stop events.
        After the harness dispatches a tool call a ToolResultEvent is emitted.
        The JSONL logger records fully assembled messages, not individual events.

        Once the generator is exhausted, `last_result` holds the `RunResult`
        for the run, including token usage and any error. Abandoning the
        generator early leaves `last_result` unset.
        """
        self._begin_run(user_message)
        # `aclosing` matters: a bare `async for` does not close its iterator,
        # so a consumer that stops early would leave the driver (and through
        # it the provider's HTTP stream) suspended until the GC got to it.
        async with aclosing(self._drive(stream=True)) as events:
            async for event in events:
                yield event

    async def ask_stream(self, user_message: str) -> AsyncGenerator[StreamEvent, None]:
        """Stream events for a follow-up turn in a session.

        Once the generator is exhausted, `last_result` holds the `RunResult`.
        """
        self._begin_ask(user_message)
        async with aclosing(self._drive(stream=True)) as events:
            async for event in events:
                yield event

    # ── driver ──────────────────────────────────────────────────────────────

    async def _drain(self, *, stream: bool) -> RunResult:
        async with aclosing(self._drive(stream=stream)) as events:
            async for _ in events:
                pass
        result = self._last_result
        if result is None:  # pragma: no cover - _plan always sets a result
            raise RuntimeError("loop finished without producing a result")
        return result

    async def _drive(self, *, stream: bool) -> AsyncGenerator[StreamEvent, None]:
        """Perform the loop's effects.

        Deliberately one generator rather than a stack of them: a generator can
        only clean up the resources held in its *own* frame when
        `GeneratorExit` arrives, and an intermediate layer would have to be
        closed explicitly by the layer above it. Keeping the provider stream,
        the plan, and the yields together means one unwind closes everything.
        """
        plan = self._plan()
        sent: Answer = None
        try:
            while True:
                try:
                    effect = plan.send(sent)
                except StopIteration:
                    return
                sent = None

                if isinstance(effect, CallProvider):
                    if stream:
                        events: list[StreamEvent] = []
                        provider = self._adapter.stream_events(
                            system=effect.system,
                            messages=effect.messages,
                            tools=effect.tools,
                        )
                        try:
                            async for evt in provider:
                                events.append(evt)
                                yield evt
                            # Inside the try on purpose: a failure while
                            # assembling the turn is a provider failure, and
                            # should land in the RunResult rather than escape
                            # into the caller's loop.
                            sent = Ok(accumulate_stream_events(events))
                        except Exception as exc:  # noqa: BLE001 - reported as a RunResult
                            sent = Failed(exc)
                        finally:
                            # A consumer that stops early unwinds through here
                            # via GeneratorExit. Close the provider's generator
                            # now so a real HTTP stream releases its connection
                            # rather than waiting for the GC.
                            aclose = getattr(provider, "aclose", None)
                            if aclose is not None:
                                await aclose()
                    else:
                        try:
                            sent = Ok(
                                await self._adapter.chat(
                                    system=effect.system,
                                    messages=effect.messages,
                                    tools=effect.tools,
                                )
                            )
                        except Exception as exc:  # noqa: BLE001 - reported as a RunResult
                            sent = Failed(exc)

                elif isinstance(effect, CallTool):
                    try:
                        sent = Ok(await self._call_tool(effect))
                    except Exception as exc:  # noqa: BLE001 - surfaced to the model
                        sent = Failed(exc)

                elif isinstance(effect, ToolFinished) and stream:
                    yield ToolResultEvent(
                        tool_use_id=effect.block.tool_use_id,
                        tool_name=effect.tool_name,
                        content=effect.block.content,
                        is_error=effect.block.is_error,
                    )
        finally:
            plan.close()

    async def _call_tool(self, call: CallTool) -> Any:
        """Invoke one tool handler without stalling the event loop.

        The overridable seam for changing dispatch. Blocking handlers go to a
        worker thread, so a handler that needs thread affinity must either be
        ``async def`` or manage its own resources per call.
        """
        if asyncio.iscoroutinefunction(call.handler):
            return await call.handler(**call.tool_input)
        return await asyncio.to_thread(
            functools.partial(call.handler, **call.tool_input)
        )


def _extract_text(response: NormalizedResponse) -> str:
    return "\n".join(b.text for b in response.content if isinstance(b, TextBlock))


__all__ = [
    "Answer",
    "AsyncHarness",
    "CallProvider",
    "CallTool",
    "Effect",
    "Failed",
    "Harness",
    "Ok",
    "ToolFinished",
    "run_coroutine_blocking",
]
