"""Phase 1: one loop, one agent base.

These tests exist because the sync/async and stream/non-stream copies had
already drifted apart before they were collapsed. Each test pins one of the
specific defects that drift produced, so a future split would fail loudly.
"""

from __future__ import annotations

import asyncio

import pytest

from data_harness.agent import Agent, AsyncAgent
from data_harness.loop import (
    AsyncHarness,
    Harness,
    as_async_adapter,
    run_coroutine_blocking,
)
from data_harness.providers.base import (
    AsyncProviderAdapter,
    NormalizedResponse,
    ProviderAdapter,
)
from data_harness.streaming import MessageDeltaEvent, ToolResultEvent
from data_harness.testing import FakeAdapter, FakeAsyncAdapter
from data_harness.types import Message, ToolSpec

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
        run_dir=str(tmp_path),
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
        run_dir=str(tmp_path),
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
    harness = AsyncHarness(
        adapter=adapter, system="sys", tools=[echo_spec()], run_dir=str(tmp_path)
    )

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
        run_dir=str(tmp_path),
    )
    async for _ in streamed.run_stream("go"):
        pass

    direct = AsyncHarness(
        adapter=FakeAsyncAdapter(script()),
        system="sys",
        tools=[echo_spec()],
        run_dir=str(tmp_path),
    )
    direct_result = await direct.run_result("go")

    stream_result = streamed.last_result
    assert stream_result is not None
    assert stream_result.text == direct_result.text == "final answer"
    assert stream_result.turns == direct_result.turns == 2
    assert stream_result.usage == direct_result.usage
    assert [m.role for m in streamed.messages] == [m.role for m in direct.messages]


# ── the sync facade really is the async loop ────────────────────────────────


def test_sync_and_async_harness_produce_identical_results(tmp_path):
    """`Harness` holds no loop logic; it drives `AsyncHarness`."""

    def script(cls):
        return [
            cls.tool_use("tu_1", "echo", {"value": "hi"}),
            cls.text("final answer"),
        ]

    sync_result = Harness(
        adapter=FakeAdapter(script(FakeAdapter)),
        system="sys",
        tools=[echo_spec()],
        run_dir=str(tmp_path),
    ).run_result("go")

    async_result = asyncio.run(
        AsyncHarness(
            adapter=FakeAsyncAdapter(script(FakeAsyncAdapter)),
            system="sys",
            tools=[echo_spec()],
            run_dir=str(tmp_path),
        ).run_result("go")
    )

    assert sync_result.text == async_result.text
    assert sync_result.turns == async_result.turns
    assert sync_result.status == async_result.status


def test_sync_harness_still_calls_the_sync_adapter(tmp_path):
    """The bridge must forward to the wrapped adapter, not around it."""
    adapter = FakeAdapter([FakeAdapter.text("hi")])
    harness = Harness(adapter=adapter, system="sys", tools=[], run_dir=str(tmp_path))

    harness.run("go")

    assert len(adapter.calls) == 1
    assert adapter.calls[0]["system"] == "sys"


def test_sync_harness_works_inside_a_running_event_loop(tmp_path):
    """Notebook kernels and async web handlers already have a loop running.

    `asyncio.run` would raise there, so the facade falls back to a worker
    thread with its own loop.
    """

    async def outer():
        harness = Harness(
            adapter=FakeAdapter([FakeAdapter.text("from inside a loop")]),
            system="sys",
            tools=[],
            run_dir=str(tmp_path),
        )
        return harness.run("go")

    assert asyncio.run(outer()) == "from inside a loop"


def test_run_coroutine_blocking_without_a_running_loop():
    async def coro():
        return 7

    assert run_coroutine_blocking(coro()) == 7


# ── adapter bridging ────────────────────────────────────────────────────────


def test_as_async_adapter_passes_async_adapters_through():
    adapter = FakeAsyncAdapter([])
    assert as_async_adapter(adapter) is adapter


def test_as_async_adapter_wraps_sync_adapters():
    bridged = as_async_adapter(FakeAdapter([]))
    assert isinstance(bridged, AsyncProviderAdapter)
    assert not isinstance(bridged, ProviderAdapter)


def test_bridge_forwards_cache_control():
    bridged = as_async_adapter(FakeAdapter([]))
    assert bridged.format_cache_control({"type": "text"}) == {
        "type": "text",
        "cache_control": {"type": "ephemeral"},
    }


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
async def test_async_agent_subagent_bridges_a_sync_adapter_factory(tmp_path):
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
        run_dir=str(tmp_path),
        max_turns=9,
    )

    assert harness.system == "the system prompt"
    assert harness.max_turns == 9
    assert harness.tools == []
    assert harness.reminders == []

    harness.tools.append(echo_spec())
    assert [t.name for t in harness.tools] == ["echo"]

    harness.run("go")
    assert isinstance(harness.messages[0], Message)
    assert harness.last_result is not None
