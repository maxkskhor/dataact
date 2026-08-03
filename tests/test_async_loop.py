"""Tests for AsyncHarness, AsyncAgent, and AsyncAgentSession."""

from __future__ import annotations

import pytest

from data_harness.agent import AsyncAgent
from data_harness.exceptions import MaxTurnsExceeded
from data_harness.loop import AsyncHarness
from data_harness.streaming import (
    ContentBlockDeltaEvent,
    MessageStopEvent,
    StreamEvent,
    TextDelta,
    ToolResultEvent,
)
from data_harness.testing import FakeAsyncAdapter
from data_harness.types import ToolSpec

# ---------------------------------------------------------------------------
# AsyncHarness — basic run_result / run
# ---------------------------------------------------------------------------


async def test_async_harness_run_result_success(tmp_path):
    adapter = FakeAsyncAdapter([FakeAsyncAdapter.text("hello async")])
    harness = AsyncHarness(
        adapter=adapter,
        system="sys",
        tools=[],
        max_turns=5,
        run_dir=str(tmp_path),
    )
    result = await harness.run_result("test")
    assert result.text == "hello async"
    assert result.status == "success"
    assert result.turns == 1
    assert result.run_file is not None


async def test_async_harness_run_returns_text(tmp_path):
    adapter = FakeAsyncAdapter([FakeAsyncAdapter.text("world")])
    harness = AsyncHarness(
        adapter=adapter, system="sys", tools=[], max_turns=5, run_dir=str(tmp_path)
    )
    text = await harness.run("hi")
    assert text == "world"


async def test_async_harness_max_turns_exceeded(tmp_path):
    # Every response requests a non-existent tool so the loop keeps going.
    responses = [
        FakeAsyncAdapter.tool_use("id1", "noop", {}),
        FakeAsyncAdapter.tool_use("id2", "noop", {}),
    ]
    adapter = FakeAsyncAdapter(responses)
    harness = AsyncHarness(
        adapter=adapter, system="sys", tools=[], max_turns=2, run_dir=str(tmp_path)
    )
    result = await harness.run_result("go")
    assert result.status == "max_turns_exceeded"
    assert result.turns == 2


async def test_async_harness_run_raises_max_turns(tmp_path):
    responses = [
        FakeAsyncAdapter.tool_use("id1", "noop", {}),
        FakeAsyncAdapter.tool_use("id2", "noop", {}),
    ]
    adapter = FakeAsyncAdapter(responses)
    harness = AsyncHarness(
        adapter=adapter, system="sys", tools=[], max_turns=2, run_dir=str(tmp_path)
    )
    with pytest.raises(MaxTurnsExceeded):
        await harness.run("go")


# ---------------------------------------------------------------------------
# AsyncHarness — ask_result (multi-turn session)
# ---------------------------------------------------------------------------


async def test_async_harness_ask_result(tmp_path):
    adapter = FakeAsyncAdapter(
        [FakeAsyncAdapter.text("first"), FakeAsyncAdapter.text("second")]
    )
    harness = AsyncHarness(
        adapter=adapter, system="sys", tools=[], max_turns=5, run_dir=str(tmp_path)
    )
    r1 = await harness.run_result("q1")
    r2 = await harness.ask_result("q2")
    assert r1.text == "first"
    assert r2.text == "second"
    # Both run_files point to the same file (ask reuses the session logger)
    assert r1.run_file == r2.run_file


# ---------------------------------------------------------------------------
# AsyncHarness — tool dispatch (sync handler via asyncio.to_thread)
# ---------------------------------------------------------------------------


async def test_async_harness_dispatches_sync_tool(tmp_path):
    called_with: list[dict] = []

    def my_tool(x: int) -> str:
        called_with.append({"x": x})
        return f"result:{x}"

    tool_spec = ToolSpec(
        name="my_tool",
        description="test",
        input_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
        handler=my_tool,
        visible=True,
    )

    responses = [
        FakeAsyncAdapter.tool_use("tu1", "my_tool", {"x": 42}),
        FakeAsyncAdapter.text("done"),
    ]
    adapter = FakeAsyncAdapter(responses)
    harness = AsyncHarness(
        adapter=adapter,
        system="sys",
        tools=[tool_spec],
        max_turns=5,
        run_dir=str(tmp_path),
    )
    result = await harness.run_result("go")
    assert result.status == "success"
    assert result.text == "done"
    assert called_with == [{"x": 42}]


async def test_async_harness_dispatches_async_tool(tmp_path):
    results: list[str] = []

    async def async_tool(msg: str) -> str:
        results.append(msg)
        return f"async:{msg}"

    tool_spec = ToolSpec(
        name="async_tool",
        description="async test tool",
        input_schema={"type": "object", "properties": {"msg": {"type": "string"}}},
        handler=async_tool,
        visible=True,
    )

    responses = [
        FakeAsyncAdapter.tool_use("tu1", "async_tool", {"msg": "hi"}),
        FakeAsyncAdapter.text("done"),
    ]
    adapter = FakeAsyncAdapter(responses)
    harness = AsyncHarness(
        adapter=adapter,
        system="sys",
        tools=[tool_spec],
        max_turns=5,
        run_dir=str(tmp_path),
    )
    result = await harness.run_result("go")
    assert result.status == "success"
    assert results == ["hi"]


async def test_async_harness_raising_handler_is_error(tmp_path):
    """A raising tool handler must produce is_error=True in the non-streaming path.

    Regression: tools that catch and return error strings instead of raising
    will silently produce is_error=False, hiding failures from the model.
    """

    def exploding_tool() -> str:
        raise ValueError("async boom")

    tool_spec = ToolSpec(
        name="exploding",
        description="always raises",
        input_schema={"type": "object", "properties": {}},
        handler=exploding_tool,
        visible=True,
    )
    responses = [
        FakeAsyncAdapter.tool_use("tu1", "exploding", {}),
        FakeAsyncAdapter.text("done"),
    ]
    adapter = FakeAsyncAdapter(responses)
    harness = AsyncHarness(
        adapter=adapter,
        system="sys",
        tools=[tool_spec],
        max_turns=5,
        run_dir=str(tmp_path),
    )
    result = await harness.run_result("go")
    assert result.status == "success"
    # Inspect the tool result in the message history
    from data_harness.types import ToolResultBlock

    for msg in reversed(harness.messages):
        if msg.role == "user":
            tool_results = [b for b in msg.content if isinstance(b, ToolResultBlock)]
            if tool_results:
                assert tool_results[0].is_error is True
                assert "async boom" in tool_results[0].content
                return
    raise AssertionError("no ToolResultBlock found")


# ---------------------------------------------------------------------------
# AsyncHarness — streaming
# ---------------------------------------------------------------------------


def _text_from_events(events: list[StreamEvent]) -> str:
    return "".join(
        e.delta.text
        for e in events
        if isinstance(e, ContentBlockDeltaEvent) and isinstance(e.delta, TextDelta)
    )


async def test_async_harness_run_stream_yields_chunks(tmp_path):
    adapter = FakeAsyncAdapter([FakeAsyncAdapter.text("hello world")])
    harness = AsyncHarness(
        adapter=adapter, system="sys", tools=[], max_turns=5, run_dir=str(tmp_path)
    )
    events: list[StreamEvent] = []
    async for event in harness.run_stream("test"):
        events.append(event)
    assert _text_from_events(events) == "hello world"
    assert any(isinstance(e, MessageStopEvent) for e in events)


async def test_async_harness_run_stream_tool_dispatch(tmp_path):
    """Tool turns emit ToolResultEvent; final text is in ContentBlockDeltaEvent."""
    side_effects: list[str] = []

    def side_tool(v: str) -> str:
        side_effects.append(v)
        return "ok"

    tool_spec = ToolSpec(
        name="side_tool",
        description="side effect tool",
        input_schema={"type": "object", "properties": {"v": {"type": "string"}}},
        handler=side_tool,
        visible=True,
    )
    responses = [
        FakeAsyncAdapter.tool_use("tu1", "side_tool", {"v": "x"}),
        FakeAsyncAdapter.text("final answer"),
    ]
    adapter = FakeAsyncAdapter(responses)
    harness = AsyncHarness(
        adapter=adapter,
        system="sys",
        tools=[tool_spec],
        max_turns=5,
        run_dir=str(tmp_path),
    )
    events: list[StreamEvent] = []
    async for event in harness.run_stream("go"):
        events.append(event)
    assert _text_from_events(events) == "final answer"
    assert side_effects == ["x"]
    tool_result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_result_events) == 1
    assert tool_result_events[0].tool_name == "side_tool"
    assert not tool_result_events[0].is_error


# ---------------------------------------------------------------------------
# AsyncHarness — run_id / session_id propagation
# ---------------------------------------------------------------------------


async def test_async_harness_run_result_ids(tmp_path):
    adapter = FakeAsyncAdapter([FakeAsyncAdapter.text("ok")])
    harness = AsyncHarness(
        adapter=adapter, system="sys", tools=[], max_turns=5, run_dir=str(tmp_path)
    )
    result = await harness.run_result("hi", run_id="r1", session_id="s1")
    assert result.run_id == "r1"
    assert result.session_id == "s1"


# ---------------------------------------------------------------------------
# AsyncAgent
# ---------------------------------------------------------------------------


async def test_async_agent_run(tmp_path):
    adapter = FakeAsyncAdapter([FakeAsyncAdapter.text("agent answer")])
    agent = AsyncAgent(adapter=adapter, system="sys", max_turns=5, run_dir=tmp_path)
    text = await agent.run("q")
    assert text == "agent answer"


async def test_async_agent_run_result(tmp_path):
    adapter = FakeAsyncAdapter([FakeAsyncAdapter.text("42")])
    agent = AsyncAgent(adapter=adapter, system="sys", run_dir=tmp_path)
    result = await agent.run_result("q")
    assert result.text == "42"
    assert result.status == "success"
    assert result.run_id is not None


async def test_async_agent_run_stream(tmp_path):
    adapter = FakeAsyncAdapter([FakeAsyncAdapter.text("streamed")])
    agent = AsyncAgent(adapter=adapter, system="sys", run_dir=tmp_path)
    events: list[StreamEvent] = []
    async for event in agent.run_stream("q"):
        events.append(event)
    assert _text_from_events(events) == "streamed"


# ---------------------------------------------------------------------------
# AsyncAgentSession
# ---------------------------------------------------------------------------


async def test_async_agent_session_ask(tmp_path):
    adapter = FakeAsyncAdapter(
        [FakeAsyncAdapter.text("turn1"), FakeAsyncAdapter.text("turn2")]
    )
    agent = AsyncAgent(adapter=adapter, system="sys", run_dir=tmp_path)
    session = agent.async_session()
    r1 = await session.ask_result("first")
    r2 = await session.ask_result("second")
    assert r1.text == "turn1"
    assert r2.text == "turn2"
    assert session.last_result is r2
    # Session ID is stable
    assert r1.session_id == session.id
    assert r2.session_id == session.id


async def test_async_agent_session_ask_stream(tmp_path):
    adapter = FakeAsyncAdapter([FakeAsyncAdapter.text("stream turn")])
    agent = AsyncAgent(adapter=adapter, system="sys", run_dir=tmp_path)
    session = agent.async_session()
    events: list[StreamEvent] = []
    async for event in session.ask_stream("hi"):
        events.append(event)
    assert _text_from_events(events) == "stream turn"
