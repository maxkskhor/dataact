"""Phase 1: one loop, one agent base.

These tests exist because the sync/async and stream/non-stream copies had
already drifted apart before they were collapsed. Each test pins one of the
specific defects that drift produced, so a future split would fail loudly.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading

import pytest

from data_harness.app.agent import Agent, AgentSession, AsyncAgent, AsyncAgentSession
from data_harness.app.quickstart import resolve_adapter, resolve_async_adapter
from data_harness.data.harness import AsyncHarness, Harness, run_coroutine_blocking
from data_harness.llm.providers.base import (
    AsyncProviderAdapter,
    NormalizedResponse,
    ProviderAdapter,
    StopReason,
)
from data_harness.llm.streaming import MessageDeltaEvent, ToolResultEvent
from data_harness.llm.testing import FakeAdapter, FakeAsyncAdapter
from data_harness.llm.types import Message, ToolSpec, ToolUseBlock

# ── helpers ─────────────────────────────────────────────────────────────────


class ExplodingAsyncAdapter(FakeAsyncAdapter):
    """Serves scripted responses, then raises. Models a provider dying mid-run."""

    def __init__(self, responses, error: Exception) -> None:
        super().__init__(responses)
        self._error = error

    async def chat(self, system, messages, tools) -> NormalizedResponse:
        if not self._responses:
            raise self._error
        return await super().chat(system, messages, tools)


def echo_spec() -> ToolSpec:
    return ToolSpec(
        name="echo",
        description="Echo the input back.",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        handler=lambda value: value,
    )


# ── API parity: the two agents must not drift again ─────────────────────────


def test_agent_and_async_agent_expose_the_same_features():
    """Both agents share one base, so their public surfaces differ only by design.

    `AsyncAgent` previously lacked from_dataframe, from_csv, subagents, MCP,
    the replay cache, close(), and explain(). Nothing flagged it.
    """
    sync_api = {n for n in dir(Agent) if not n.startswith("_")}
    async_api = {n for n in dir(AsyncAgent) if not n.startswith("_")}

    assert sync_api - async_api == {"session"}
    assert async_api - sync_api == {"async_session", "run_stream"}


@pytest.mark.parametrize(
    "feature",
    [
        "from_dataframe",
        "from_csv",
        "enable_subagents",
        "enable_planner",
        "enable_sql",
        "enable_cache",
        "add_mcp_server",
        "connector",
        "close",
        "explain",
        "exec_cache",
    ],
)
def test_async_agent_has_feature(feature):
    assert hasattr(AsyncAgent, feature)


def test_async_agent_accepts_the_sandbox_and_gate_options(tmp_path):
    """`on_code`, `code_only`, `execution` used to be Agent-only constructor args."""
    agent = AsyncAgent(
        adapter=FakeAsyncAdapter([]),
        system="sys",
        run_dir=str(tmp_path),
        execution="inprocess",
        on_code=lambda code: True,
        code_only=True,
    )
    assert agent._code_only is True


# ── streaming keeps its accounting ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_records_run_result_on_success(tmp_path):
    """A streamed run exposes usage and status via `last_result`.

    The event protocol carries no run-level summary, so before this the only
    way to get usage out of a stream was to reassemble it from raw events.
    """
    harness = AsyncHarness(
        adapter=FakeAsyncAdapter([FakeAsyncAdapter.text("done")]),
        system="sys",
        tools=[],
    )

    events = [evt async for evt in harness.run_stream("go")]

    assert any(isinstance(e, MessageDeltaEvent) for e in events)
    result = harness.last_result
    assert result is not None
    assert result.status == "success"
    assert result.text == "done"
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5


@pytest.mark.asyncio
async def test_stream_keeps_usage_when_the_provider_fails_mid_run(tmp_path):
    """Tokens already billed before a provider error must still be accounted for.

    The old streaming loop caught the exception, logged, and returned without
    building a RunResult, so usage from every completed turn was lost. A
    metered deployment silently under-billed itself.
    """
    adapter = ExplodingAsyncAdapter(
        [FakeAsyncAdapter.tool_use("tu_1", "echo", {"value": "hi"})],
        RuntimeError("provider down"),
    )
    harness = AsyncHarness(
        adapter=adapter,
        system="sys",
        tools=[echo_spec()],
    )

    events = [evt async for evt in harness.ask_stream("go")]

    assert any(isinstance(e, ToolResultEvent) for e in events)
    result = harness.last_result
    assert result is not None
    assert result.status == "error"
    assert "provider down" in (result.error or "")
    # Turn 1 completed and was billed before turn 2 failed.
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5


@pytest.mark.asyncio
async def test_non_streaming_error_also_reports_usage(tmp_path):
    """The non-streaming path always behaved correctly here; it must keep doing so."""
    adapter = ExplodingAsyncAdapter(
        [FakeAsyncAdapter.tool_use("tu_1", "echo", {"value": "hi"})],
        RuntimeError("provider down"),
    )
    harness = AsyncHarness(adapter=adapter, system="sys", tools=[echo_spec()])

    result = await harness.run_result("go")

    assert result.status == "error"
    assert result.usage.input_tokens == 10


@pytest.mark.asyncio
async def test_session_ask_stream_updates_session_totals(tmp_path):
    """A streamed session turn counts toward `turns` and `last_result`."""
    agent = AsyncAgent(
        adapter=FakeAsyncAdapter(
            [FakeAsyncAdapter.text("one"), FakeAsyncAdapter.text("two")]
        ),
        system="sys",
        run_dir=str(tmp_path),
    )
    session = agent.async_session()

    async for _ in session.ask_stream("first"):
        pass

    assert session.turns == 1
    assert session.last_result is not None
    assert session.last_result.text == "one"
    assert session.last_result.usage.output_tokens == 5

    await session.ask_result("second")
    assert session.turns == 2


# ── streaming and non-streaming agree ───────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_and_result_paths_agree(tmp_path):
    """Same script, same outcome, whichever entry point is used.

    They are one loop now, so this asserts the parameterisation did not change
    behaviour rather than that two implementations happen to match.
    """

    def script():
        return [
            FakeAsyncAdapter.tool_use("tu_1", "echo", {"value": "hi"}),
            FakeAsyncAdapter.text("final answer"),
        ]

    streamed = AsyncHarness(
        adapter=FakeAsyncAdapter(script()),
        system="sys",
        tools=[echo_spec()],
    )
    async for _ in streamed.run_stream("go"):
        pass

    direct = AsyncHarness(
        adapter=FakeAsyncAdapter(script()),
        system="sys",
        tools=[echo_spec()],
    )
    direct_result = await direct.run_result("go")

    stream_result = streamed.last_result
    assert stream_result is not None
    assert stream_result.text == direct_result.text == "final answer"
    assert stream_result.turns == direct_result.turns == 2
    assert stream_result.usage == direct_result.usage
    assert [m.role for m in streamed.messages] == [m.role for m in direct.messages]


# ── the two drivers agree ───────────────────────────────────────────────────


def test_sync_and_async_harness_produce_identical_results(tmp_path):
    """Both drivers run the same `_plan` generator, so they must agree."""

    def script(cls):
        return [
            cls.tool_use("tu_1", "echo", {"value": "hi"}),
            cls.text("final answer"),
        ]

    sync_result = Harness(
        adapter=FakeAdapter(script(FakeAdapter)),
        system="sys",
        tools=[echo_spec()],
    ).run_result("go")

    async_result = asyncio.run(
        AsyncHarness(
            adapter=FakeAsyncAdapter(script(FakeAsyncAdapter)),
            system="sys",
            tools=[echo_spec()],
        ).run_result("go")
    )

    assert sync_result.text == async_result.text
    assert sync_result.turns == async_result.turns
    assert sync_result.status == async_result.status


def test_sync_harness_still_calls_the_sync_adapter(tmp_path):
    adapter = FakeAdapter([FakeAdapter.text("hi")])
    harness = Harness(adapter=adapter, system="sys", tools=[])

    harness.run("go")

    assert len(adapter.calls) == 1
    assert adapter.calls[0]["system"] == "sys"


def test_sync_harness_works_inside_a_running_event_loop(tmp_path):
    """A notebook kernel or async web handler already has a loop running."""

    async def outer():
        harness = Harness(
            adapter=FakeAdapter([FakeAdapter.text("from inside a loop")]),
            system="sys",
            tools=[],
        )
        return harness.run("go")

    assert asyncio.run(outer()) == "from inside a loop"


def test_run_coroutine_blocking_without_a_running_loop():
    async def coro():
        return 7

    assert run_coroutine_blocking(coro()) == 7


# ── the sync driver must stay genuinely synchronous ─────────────────────────


def test_sync_driver_runs_tool_handlers_on_the_calling_thread(tmp_path):
    """Handlers must keep thread affinity.

    A connector holding a `sqlite3` connection built at setup time is the most
    ordinary pattern this library has, and sqlite3 refuses cross-thread use.
    Driving the sync path through an event loop moved handlers onto an
    `asyncio.to_thread` worker and turned that into a tool error string.
    """
    seen: list[str] = []

    spec = ToolSpec(
        name="whereami",
        description="Report the executing thread.",
        input_schema={"type": "object", "properties": {}},
        handler=lambda: seen.append(threading.current_thread().name) or "ok",
    )

    Harness(
        adapter=FakeAdapter(
            [
                FakeAdapter.tool_use("tu_1", "whereami", {}),
                FakeAdapter.text("done"),
            ]
        ),
        system="sys",
        tools=[spec],
    ).run("go")

    assert seen == [threading.current_thread().name]


def test_sync_driver_keeps_a_thread_bound_resource_usable(tmp_path):
    """The concrete failure the thread-affinity rule protects against."""
    connection = sqlite3.connect(":memory:")
    connection.execute("create table t (v integer)")
    connection.execute("insert into t values (42)")

    spec = ToolSpec(
        name="query",
        description="Read the fixed row.",
        input_schema={"type": "object", "properties": {}},
        handler=lambda: str(connection.execute("select v from t").fetchone()),
    )

    harness = Harness(
        adapter=FakeAdapter(
            [FakeAdapter.tool_use("tu_1", "query", {}), FakeAdapter.text("done")]
        ),
        system="sys",
        tools=[spec],
    )
    harness.run("go")

    tool_result = harness.messages[2].content[0]
    assert tool_result.is_error is False
    assert "42" in tool_result.content


def test_sync_driver_leaves_the_ambient_event_loop_alone(tmp_path):
    """`asyncio.run` clears the thread's loop; the sync driver must not.

    A program that installs its own loop and then calls the *synchronous* API
    used to find the loop gone afterwards, failing much later and elsewhere.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        Harness(
            adapter=FakeAdapter([FakeAdapter.text("hi")]),
            system="sys",
            tools=[],
        ).run("go")
        assert asyncio.get_event_loop() is loop
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def test_run_coroutine_blocking_restores_the_ambient_event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def coro():
        return 1

    try:
        assert run_coroutine_blocking(coro()) == 1
        assert asyncio.get_event_loop() is loop
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def test_keyboard_interrupt_escapes_the_sync_driver(tmp_path):
    """Ctrl-C must reach the caller, not be absorbed as a tool error.

    Tool failures are caught as `Exception` and reported to the model.
    `KeyboardInterrupt` is a `BaseException` and must pass straight through,
    and with inline dispatch it does so without waiting on a worker thread.
    """

    def interrupt():
        raise KeyboardInterrupt

    spec = ToolSpec(
        name="boom",
        description="Interrupt.",
        input_schema={"type": "object", "properties": {}},
        handler=interrupt,
    )

    with pytest.raises(KeyboardInterrupt):
        Harness(
            adapter=FakeAdapter(
                [FakeAdapter.tool_use("tu_1", "boom", {}), FakeAdapter.text("done")]
            ),
            system="sys",
            tools=[spec],
        ).run("go")


@pytest.mark.asyncio
async def test_async_driver_offloads_blocking_handlers(tmp_path):
    """The async driver must NOT run blocking handlers on the event loop.

    The mirror image of the sync rule: a long pandas call on the loop thread
    would stall every other task sharing it.
    """
    seen: list[str] = []

    spec = ToolSpec(
        name="whereami",
        description="Report the executing thread.",
        input_schema={"type": "object", "properties": {}},
        handler=lambda: seen.append(threading.current_thread().name) or "ok",
    )

    await AsyncHarness(
        adapter=FakeAsyncAdapter(
            [
                FakeAsyncAdapter.tool_use("tu_1", "whereami", {}),
                FakeAsyncAdapter.text("done"),
            ]
        ),
        system="sys",
        tools=[spec],
    ).run("go")

    assert seen and seen != [threading.current_thread().name]


# ── the drivers stay overridable ────────────────────────────────────────────


def test_sync_driver_dispatch_is_overridable(tmp_path):
    """`_call_tool` is the documented seam for sandboxing or instrumentation.

    A facade that forwarded only public methods would accept the subclass and
    silently never call its override.
    """

    class Instrumented(Harness):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.dispatched: list[str] = []

        def _call_tool(self, call):
            self.dispatched.append(call.tool_name)
            return super()._call_tool(call)

    harness = Instrumented(
        adapter=FakeAdapter(
            [
                FakeAdapter.tool_use("tu_1", "echo", {"value": "hi"}),
                FakeAdapter.text("d"),
            ]
        ),
        system="sys",
        tools=[echo_spec()],
    )
    harness.run("go")

    assert harness.dispatched == ["echo"]


@pytest.mark.asyncio
async def test_async_driver_dispatch_is_overridable(tmp_path):
    class Instrumented(AsyncHarness):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.dispatched: list[str] = []

        async def _call_tool(self, call):
            self.dispatched.append(call.tool_name)
            return await super()._call_tool(call)

    harness = Instrumented(
        adapter=FakeAsyncAdapter(
            [
                FakeAsyncAdapter.tool_use("tu_1", "echo", {"value": "hi"}),
                FakeAsyncAdapter.text("d"),
            ]
        ),
        system="sys",
        tools=[echo_spec()],
    )
    await harness.run("go")

    assert harness.dispatched == ["echo"]


@pytest.mark.parametrize(
    "attribute",
    [
        "_messages",
        "_tools",
        "_system",
        "_max_turns",
        "_environment",
        "_on_code",
        "_code_only",
    ],
)
def test_both_drivers_carry_the_loop_state(attribute, tmp_path):
    """Loop state lives on the shared base, so neither driver is a hollow shell."""
    sync = Harness(adapter=FakeAdapter([]), system="sys", tools=[])
    asynchronous = AsyncHarness(adapter=FakeAsyncAdapter([]), system="sys", tools=[])
    assert hasattr(sync, attribute)
    assert hasattr(asynchronous, attribute)


# ── features that only Agent used to have, now on AsyncAgent ────────────────


@pytest.mark.asyncio
async def test_async_agent_from_dataframe_preloads_handles(tmp_path):
    pd = pytest.importorskip("pandas")
    agent = AsyncAgent.from_dataframe(
        pd.DataFrame({"a": [1, 2, 3]}),
        adapter=FakeAsyncAdapter([FakeAsyncAdapter.text("ok")]),
        run_dir=str(tmp_path),
    )
    assert agent.cache.list_handles()
    assert await agent.run("summarise") == "ok"


@pytest.mark.asyncio
async def test_async_agent_code_only_blocks_execution(tmp_path):
    """The interpreter approval gate used to be unreachable from AsyncAgent."""
    agent = AsyncAgent(
        adapter=FakeAsyncAdapter(
            [
                FakeAsyncAdapter.tool_use(
                    "tu_1", "python_interpreter", {"code": "save('x', 1)"}
                ),
                FakeAsyncAdapter.text("done"),
            ]
        ),
        system="sys",
        run_dir=str(tmp_path),
        code_only=True,
    )

    await agent.run_result("compute")

    assert "x" not in agent.cache.handle_names()
    harness = agent.last_harness
    assert harness is not None
    tool_result = harness.messages[2].content[0]
    assert "DRY RUN" in tool_result.content


@pytest.mark.asyncio
async def test_async_agent_subagent_accepts_a_sync_adapter_factory(tmp_path):
    """A sync adapter factory must still work under an async parent."""
    agent = AsyncAgent(
        adapter=FakeAsyncAdapter(
            [
                FakeAsyncAdapter.tool_use("tu_1", "subagent", {"task": "work"}),
                FakeAsyncAdapter.text("parent done"),
            ]
        ),
        system="sys",
        run_dir=str(tmp_path),
    )
    agent.enable_subagents(
        adapter_factory=lambda: FakeAdapter([FakeAdapter.text("sub done")])
    )

    await agent.run_result("delegate")

    harness = agent.last_harness
    assert harness is not None
    tool_result = harness.messages[2].content[0]
    assert "sub done" in tool_result.content
    assert tool_result.is_error is False


@pytest.mark.asyncio
async def test_async_agent_subagent_accepts_an_async_adapter_factory(tmp_path):
    agent = AsyncAgent(
        adapter=FakeAsyncAdapter(
            [
                FakeAsyncAdapter.tool_use("tu_1", "subagent", {"task": "work"}),
                FakeAsyncAdapter.text("parent done"),
            ]
        ),
        system="sys",
        run_dir=str(tmp_path),
    )
    agent.enable_subagents(
        adapter_factory=lambda: FakeAsyncAdapter([FakeAsyncAdapter.text("async sub")])
    )

    await agent.run_result("delegate")

    harness = agent.last_harness
    assert harness is not None
    assert "async sub" in harness.messages[2].content[0].content


@pytest.mark.asyncio
async def test_async_agent_replay_cache_skips_the_model(tmp_path):
    """`enable_cache` used to be Agent-only, so async callers paid for every repeat."""
    pd = pytest.importorskip("pandas")

    def build(responses):
        agent = AsyncAgent.from_dataframe(
            pd.DataFrame({"a": [1, 2, 3]}),
            adapter=FakeAsyncAdapter(responses),
            run_dir=str(tmp_path),
        )
        return agent

    shared_path = tmp_path / "replay.json"

    first = build(
        [
            FakeAsyncAdapter.tool_use(
                "tu_1", "python_interpreter", {"code": "answer(6)"}
            ),
            FakeAsyncAdapter.text("The answer is 6"),
        ]
    )
    first.enable_cache(str(shared_path))
    first_result = await first.run_result("what is the total")
    assert first_result.text == "The answer is 6"

    second = build([])
    second.enable_cache(str(shared_path))
    second_result = await second.run_result("what is the total")

    assert second_result.text == "The answer is 6"
    assert second_result.turns == 0
    assert second_result.value == 6


# ── inspection surface ──────────────────────────────────────────────────────


def test_harness_inspection_properties_are_live(tmp_path):
    """`tools` and `messages` return the live lists, so sessions can add tools."""
    harness = Harness(
        adapter=FakeAdapter([FakeAdapter.text("hi")]),
        system="the system prompt",
        tools=[],
        max_turns=9,
    )

    assert harness.system == "the system prompt"
    assert harness.max_turns == 9
    assert harness.tools == []

    harness.tools.append(echo_spec())
    assert [t.name for t in harness.tools] == ["echo"]

    harness.run("go")
    assert isinstance(harness.messages[0], Message)
    assert harness.last_result is not None


# ── async adapter resolution ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-sonnet-4-6", "AsyncAnthropicAdapter"),
        ("gpt-4o-mini", "AsyncOpenAIAdapter"),
        ("o3-mini", "AsyncOpenAIAdapter"),
        ("deepseek-chat", "AsyncDeepSeekAdapter"),
        ("openai/gpt-4o-mini", "AsyncOpenRouterAdapter"),
    ],
)
def test_resolve_async_adapter_routes_like_the_sync_one(model, expected, monkeypatch):
    """The two resolvers share `_route`, so they cannot disagree about a model."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")

    assert type(resolve_async_adapter(model)).__name__ == expected
    assert type(resolve_adapter(model)).__name__ == expected.replace("Async", "", 1)


def test_resolve_async_adapter_reads_the_environment(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")

    assert type(resolve_async_adapter()).__name__ == "AsyncDeepSeekAdapter"


def test_resolve_async_adapter_without_any_provider(monkeypatch):
    for var in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "DEEPSEEK_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)

    with pytest.raises(RuntimeError, match="No provider configured"):
        resolve_async_adapter()


def test_async_agent_from_dataframe_resolves_an_async_adapter(monkeypatch):
    """Without an explicit adapter, `AsyncAgent` must not pick a sync one."""
    pd = pytest.importorskip("pandas")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    agent = AsyncAgent.from_dataframe(pd.DataFrame({"a": [1]}))

    assert isinstance(agent._adapter, AsyncProviderAdapter)
    assert isinstance(
        Agent.from_dataframe(pd.DataFrame({"a": [1]}))._adapter, ProviderAdapter
    )


# ── session parity ──────────────────────────────────────────────────────────


def test_sessions_expose_the_same_surface():
    """The sessions are the remaining hand-written pair; guard them too."""
    sync_api = {n for n in dir(AgentSession) if not n.startswith("_")}
    async_api = {n for n in dir(AsyncAgentSession) if not n.startswith("_")}

    assert sync_api - async_api == set()
    assert async_api - sync_api == {"ask_stream"}


def test_sessions_seed_from_a_copy_of_the_agent_cache(tmp_path):
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"a": [1, 2, 3]})

    sync_session = Agent.from_dataframe(
        frame, adapter=FakeAdapter([]), run_dir=str(tmp_path)
    ).session()
    async_session = AsyncAgent.from_dataframe(
        frame, adapter=FakeAsyncAdapter([]), run_dir=str(tmp_path)
    ).async_session()

    assert sync_session.list_handles().keys() == async_session.list_handles().keys()
    assert sync_session.turns == async_session.turns == 0


# ── streaming reaches every terminal state ──────────────────────────────────


@pytest.mark.asyncio
async def test_stream_reports_max_turns_exceeded(tmp_path):
    """The streaming path used to produce no result at all for this outcome."""
    harness = AsyncHarness(
        adapter=FakeAsyncAdapter(
            [FakeAsyncAdapter.tool_use("tu_1", "echo", {"value": "hi"})]
        ),
        system="sys",
        tools=[echo_spec()],
        max_turns=1,
    )

    async for _ in harness.run_stream("go"):
        pass

    result = harness.last_result
    assert result is not None
    assert result.status == "max_turns_exceeded"
    assert result.turns == 1
    assert result.usage.input_tokens == 10


@pytest.mark.asyncio
async def test_abandoning_a_stream_still_accounts_for_it(tmp_path):
    """An abandoned run reports what it spent, rather than nothing at all.

    This used to assert `last_result is None`. That was the same defect the
    phase set out to fix, one case over: a consumer that reads the final
    event and breaks has paid for that turn in full, and reporting nothing
    loses those tokens exactly the way the old streaming error path did.
    """
    harness = AsyncHarness(
        adapter=FakeAsyncAdapter([FakeAsyncAdapter.text("done")]),
        system="sys",
        tools=[],
    )

    stream = harness.run_stream("go")
    async for _ in stream:
        break  # consume one event, then walk away
    await stream.aclose()

    result = harness.last_result
    assert result is not None
    assert result.status == "error"
    assert "RunAbandoned" in (result.error or "")


@pytest.mark.asyncio
async def test_an_abandoned_stream_keeps_the_usage_of_finished_turns(tmp_path):
    """Turns that completed before the caller walked away are still billed."""
    harness = AsyncHarness(
        adapter=FakeAsyncAdapter(
            [
                FakeAsyncAdapter.tool_use("tu_1", "echo", {"value": "hi"}),
                FakeAsyncAdapter.text("done"),
            ]
        ),
        system="sys",
        tools=[echo_spec()],
    )

    stream = harness.run_stream("go")
    seen = 0
    async for event in stream:
        seen += 1
        if isinstance(event, ToolResultEvent):
            break
    await stream.aclose()

    result = harness.last_result
    assert result is not None
    assert result.status == "error"
    # Turn 1 completed and was billed before the caller stopped reading.
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5


@pytest.mark.asyncio
async def test_tool_result_events_cover_failures_and_missing_tools(tmp_path):
    """Every result block gets an event, not only the ones that succeeded."""

    def raises():
        raise ValueError("nope")

    boom = ToolSpec(
        name="boom",
        description="Fail.",
        input_schema={"type": "object", "properties": {}},
        handler=raises,
    )
    harness = AsyncHarness(
        adapter=FakeAsyncAdapter(
            [
                NormalizedResponse(
                    stop_reason=StopReason.TOOL_USE,
                    content=[
                        ToolUseBlock(
                            tool_use_id="tu_1", tool_name="boom", tool_input={}
                        ),
                        ToolUseBlock(
                            tool_use_id="tu_2", tool_name="ghost", tool_input={}
                        ),
                    ],
                    input_tokens=1,
                    output_tokens=1,
                    cache_read_tokens=0,
                    cache_write_tokens=0,
                ),
                FakeAsyncAdapter.text("done"),
            ]
        ),
        system="sys",
        tools=[boom],
    )

    events = [e async for e in harness.run_stream("go")]
    tool_events = [e for e in events if isinstance(e, ToolResultEvent)]

    assert [e.tool_name for e in tool_events] == ["boom", "ghost"]
    assert all(e.is_error for e in tool_events)


# ── the deepest path the drivers support ────────────────────────────────────


def test_subagent_inside_a_sync_harness_inside_a_running_loop(tmp_path):
    """Async caller -> sync Agent -> sync subagent. Must not deadlock."""

    async def outer():
        agent = Agent(
            adapter=FakeAdapter(
                [
                    FakeAdapter.tool_use("tu_1", "subagent", {"task": "work"}),
                    FakeAdapter.text("parent done"),
                ]
            ),
            system="sys",
            run_dir=str(tmp_path),
        )
        agent.enable_subagents(
            adapter_factory=lambda: FakeAdapter([FakeAdapter.text("sub done")])
        )
        return agent.run_result("delegate")

    result = asyncio.run(outer())

    assert result.status == "success"
    assert result.text == "parent done"
