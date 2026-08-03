"""The effect protocol, and the invariants mutation testing showed were loose.

An independent review mutated `loop.py` and `agent.py` and found seven changes
the suite did not notice: the approval gate's scope, whether a blocked call is
an error, `_stamp` writing through to `last_result`, the `ToolFinished`
send-`None` contract, latency actually being measured, the async replay's
offload, and an errored run's text. Each of those has a test here.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from data_harness.agent import Agent, AsyncAgent
from data_harness.loop import (
    AsyncHarness,
    CallProvider,
    CallTool,
    Failed,
    Harness,
    Ok,
    ToolFinished,
    run_coroutine_blocking,
)
from data_harness.streaming import ContentBlockDeltaEvent, TextDelta
from data_harness.testing import FakeAdapter, FakeAsyncAdapter
from data_harness.types import ToolSpec


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


def _stream_text(events) -> str:
    return "".join(
        e.delta.text
        for e in events
        if isinstance(e, ContentBlockDeltaEvent) and isinstance(e.delta, TextDelta)
    )


# ── the approval gate ───────────────────────────────────────────────────────


def test_the_approval_gate_only_covers_the_interpreter(tmp_path):
    """The gate is scoped to `python_interpreter`, not every tool.

    Widening it would silently turn `code_only` into "disable all tools",
    which no caller asks for and nothing else would notice.
    """
    calls: list[str] = []
    spec = ToolSpec(
        name="not_the_interpreter",
        description="An ordinary tool.",
        input_schema={"type": "object", "properties": {}},
        handler=lambda: calls.append("ran") or "ran",
    )

    harness = Harness(
        adapter=FakeAdapter(
            [
                FakeAdapter.tool_use("tu_1", "not_the_interpreter", {}),
                FakeAdapter.text("done"),
            ]
        ),
        system="sys",
        tools=[spec],
        run_dir=str(tmp_path),
        code_only=True,
    )
    harness.run("go")

    assert calls == ["ran"]
    assert "DRY RUN" not in harness.messages[2].content[0].content


def test_a_blocked_interpreter_call_is_not_an_error(tmp_path):
    """A refused call is a normal result, not an error.

    Flagging it as an error tells the model its code was broken rather than
    declined, so it rewrites and retries instead of stopping.
    """
    agent = Agent(adapter=FakeAdapter([]), system="s", run_dir=str(tmp_path))
    harness = Harness(
        adapter=FakeAdapter(
            [
                FakeAdapter.tool_use("tu_1", "python_interpreter", {"code": "x = 1"}),
                FakeAdapter.text("done"),
            ]
        ),
        system="sys",
        tools=agent._build_tools(),
        run_dir=str(tmp_path),
        on_code=lambda code: False,
    )
    harness.run("go")

    block = harness.messages[2].content[0]
    assert block.is_error is False
    assert "blocked" in block.content.lower()


def test_code_only_echoes_without_executing(tmp_path):
    agent = Agent(adapter=FakeAdapter([]), system="s", run_dir=str(tmp_path))
    harness = Harness(
        adapter=FakeAdapter(
            [
                FakeAdapter.tool_use(
                    "tu_1", "python_interpreter", {"code": "save('x', 1)"}
                ),
                FakeAdapter.text("done"),
            ]
        ),
        system="sys",
        tools=agent._build_tools(cache=agent.cache),
        run_dir=str(tmp_path),
        cache=agent.cache,
        code_only=True,
    )
    harness.run("go")

    block = harness.messages[2].content[0]
    assert block.is_error is False
    assert "DRY RUN" in block.content
    assert "x" not in agent.cache.handle_names()


# ── run identity ────────────────────────────────────────────────────────────


def test_run_ids_reach_last_result(tmp_path):
    """`_stamp` must write through to `last_result`, not only the return value."""
    harness = Harness(
        adapter=FakeAdapter([FakeAdapter.text("hi")]),
        system="sys",
        tools=[],
        run_dir=str(tmp_path),
    )

    returned = harness.run_result("go", run_id="run-7", session_id="sess-9")

    assert returned.run_id == "run-7"
    assert harness.last_result is not None
    assert harness.last_result.run_id == "run-7"
    assert harness.last_result.session_id == "sess-9"


@pytest.mark.asyncio
async def test_streamed_session_turns_are_stamped(tmp_path):
    """A streamed turn carries the same ids a non-streamed one does.

    Without it, anything correlating turns by `session_id` silently drops
    every streamed turn.
    """
    agent = AsyncAgent(
        adapter=FakeAsyncAdapter([FakeAsyncAdapter.text("one")]),
        system="sys",
        run_dir=str(tmp_path),
    )
    session = agent.async_session()

    async for _ in session.ask_stream("first"):
        pass

    assert session.last_result is not None
    assert session.last_result.session_id == session.id
    assert session.last_result.run_id is not None


# ── what gets logged ────────────────────────────────────────────────────────


def test_latency_is_measured_not_hardcoded(tmp_path):
    """The JSONL log's latency must reflect the real provider call."""

    class SlowAdapter(FakeAdapter):
        def chat(self, system, messages, tools):
            time.sleep(0.05)
            return super().chat(system, messages, tools)

    harness = Harness(
        adapter=SlowAdapter([FakeAdapter.text("hi")]),
        system="sys",
        tools=[],
        run_dir=str(tmp_path),
    )
    harness.run("go")

    assert harness.run_file is not None
    records = [
        json.loads(line)
        for line in open(harness.run_file).read().splitlines()
        if line.strip()
    ]
    assert records[0]["metrics"]["latency_ms"] >= 50


def test_a_provider_error_yields_no_partial_text(tmp_path):
    """An errored run reports empty text; the detail lives in `error`."""

    class Broken(FakeAdapter):
        def chat(self, system, messages, tools):
            raise RuntimeError("down")

    result = Harness(
        adapter=Broken([]), system="sys", tools=[], run_dir=str(tmp_path)
    ).run_result("go")

    assert result.status == "error"
    assert result.text == ""
    assert "down" in (result.error or "")


@pytest.mark.asyncio
async def test_provider_error_on_the_first_turn(tmp_path):
    """Turn-1 failure: zero usage, still a well-formed result."""

    class Broken(FakeAsyncAdapter):
        async def chat(self, system, messages, tools):
            raise RuntimeError("down")

    result = await AsyncHarness(
        adapter=Broken([]), system="sys", tools=[], run_dir=str(tmp_path)
    ).run_result("go")

    assert result.status == "error"
    assert result.turns == 1
    assert result.usage.input_tokens == 0
    assert result.run_file is not None


# ── the effect protocol ─────────────────────────────────────────────────────


def test_a_handler_returning_failed_is_not_mistaken_for_a_failure(tmp_path):
    """Successes are wrapped in `Ok`, so no return value can impersonate one.

    Sending raw handler values back into the loop meant a handler that
    legitimately returned a `Failed` was reported to the model as raising.
    """
    sentinel = Failed(ValueError("I am a legitimate return value"))
    spec = ToolSpec(
        name="returns_failed",
        description="Return a Failed instance.",
        input_schema={"type": "object", "properties": {}},
        handler=lambda: sentinel,
    )

    harness = Harness(
        adapter=FakeAdapter(
            [FakeAdapter.tool_use("tu_1", "returns_failed", {}), FakeAdapter.text("d")]
        ),
        system="sys",
        tools=[spec],
        run_dir=str(tmp_path),
    )
    harness.run("go")

    block = harness.messages[2].content[0]
    assert block.is_error is False
    assert "legitimate return value" in block.content


def test_the_effect_and_answer_sequence_lines_up(tmp_path):
    """Drive `_plan` by hand and pin the protocol.

    `ToolFinished` is an announcement answered with ``None``. Answering it
    with a value would shift every later send by one and feed a tool result
    into the wrong `yield`, which no end-to-end test would localise.
    """
    harness = Harness(
        adapter=FakeAdapter([]),
        system="sys",
        tools=[echo_spec()],
        run_dir=str(tmp_path),
    )
    harness._begin_run("go")
    plan = harness._plan()

    effect = plan.send(None)
    assert isinstance(effect, CallProvider)
    assert [t.name for t in effect.tools] == ["echo"]

    effect = plan.send(Ok(FakeAdapter.tool_use("tu_1", "echo", {"value": "hi"})))
    assert isinstance(effect, CallTool)
    assert effect.tool_name == "echo"
    assert effect.tool_input == {"value": "hi"}

    effect = plan.send(Ok("echoed"))
    assert isinstance(effect, ToolFinished)
    assert effect.tool_name == "echo"
    assert effect.block.content == "echoed"
    assert effect.block.is_error is False

    effect = plan.send(None)
    assert isinstance(effect, CallProvider)

    with pytest.raises(StopIteration):
        plan.send(Ok(FakeAdapter.text("final")))

    assert harness.last_result is not None
    assert harness.last_result.text == "final"
    assert harness.last_result.turns == 2


def test_every_tool_is_dispatched_before_any_result_is_announced(tmp_path):
    """Dispatch-then-announce ordering, which streaming consumers observe."""
    harness = Harness(
        adapter=FakeAdapter([]),
        system="sys",
        tools=[echo_spec()],
        run_dir=str(tmp_path),
    )
    harness._begin_run("go")
    plan = harness._plan()
    plan.send(None)

    two_calls = FakeAdapter.tool_use("tu_1", "echo", {"value": "a"})
    two_calls.content.append(
        type(two_calls.content[0])(
            tool_use_id="tu_2", tool_name="echo", tool_input={"value": "b"}
        )
    )

    kinds = []
    answer = Ok(two_calls)
    while True:
        effect = plan.send(answer)
        kinds.append(type(effect).__name__)
        if isinstance(effect, CallTool):
            answer = Ok(effect.tool_input["value"])
        elif isinstance(effect, ToolFinished):
            answer = None
        else:
            break

    assert kinds == [
        "CallTool",
        "CallTool",
        "ToolFinished",
        "ToolFinished",
        "CallProvider",
    ]
    plan.close()


def test_sync_driver_awaits_async_tool_handlers(tmp_path):
    """An `async def` handler under the sync driver must actually be awaited.

    It used to be called and its coroutine object stringified into the
    transcript, so the model was shown '<coroutine object ...>'.
    """

    async def handler(value: str) -> str:
        await asyncio.sleep(0)
        return f"awaited:{value}"

    spec = ToolSpec(
        name="async_echo",
        description="Async echo.",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        handler=handler,
    )

    harness = Harness(
        adapter=FakeAdapter(
            [
                FakeAdapter.tool_use("tu_1", "async_echo", {"value": "v"}),
                FakeAdapter.text("done"),
            ]
        ),
        system="sys",
        tools=[spec],
        run_dir=str(tmp_path),
    )
    harness.run("go")

    block = harness.messages[2].content[0]
    assert block.content == "awaited:v"
    assert block.is_error is False


# ── event-loop hygiene ──────────────────────────────────────────────────────


def test_looking_up_the_ambient_loop_does_not_create_one():
    """The helper protecting the ambient loop must not install one itself.

    On Python 3.10/3.11 `get_event_loop()` creates a loop when none is set,
    which would leave a never-run, never-closed loop and its selector fd
    behind on a thread that only ever used the synchronous API.
    """
    from data_harness.loop import _ambient_event_loop

    asyncio.set_event_loop(None)
    assert _ambient_event_loop() is None

    async def coro():
        return 1

    assert run_coroutine_blocking(coro()) == 1
    assert _ambient_event_loop() is None


@pytest.mark.asyncio
async def test_abandoning_a_stream_closes_the_provider_stream(tmp_path):
    """A consumer that stops early must not leave the provider's stream open.

    With a real HTTP adapter that generator is holding the connection.
    """
    closed: list[bool] = []

    class TrackingAdapter(FakeAsyncAdapter):
        async def stream_events(self, system, messages, tools):
            try:
                async for evt in super().stream_events(system, messages, tools):
                    yield evt
            finally:
                closed.append(True)

    harness = AsyncHarness(
        adapter=TrackingAdapter([FakeAsyncAdapter.text("hello")]),
        system="sys",
        tools=[],
        run_dir=str(tmp_path),
    )

    stream = harness.run_stream("go")
    async for _ in stream:
        break
    await stream.aclose()

    assert closed == [True]


# ── replay parity between streamed and non-streamed runs ────────────────────


@pytest.mark.asyncio
async def test_streamed_runs_use_and_fill_the_replay_cache(tmp_path):
    """`run_stream` must not quietly opt out of the replay cache.

    It previously neither served a hit nor recorded a success, so whether a
    caller paid for a repeated question depended on the entry point it chose.
    """
    pd = pytest.importorskip("pandas")
    shared = tmp_path / "replay.json"

    def build(responses):
        agent = AsyncAgent.from_dataframe(
            pd.DataFrame({"a": [1, 2, 3]}),
            adapter=FakeAsyncAdapter(responses),
            run_dir=str(tmp_path),
        )
        return agent.enable_cache(str(shared))

    warming = build(
        [
            FakeAsyncAdapter.tool_use(
                "tu_1", "python_interpreter", {"code": "answer(6)"}
            ),
            FakeAsyncAdapter.text("The answer is 6"),
        ]
    )
    async for _ in warming.run_stream("what is the total"):
        pass
    assert warming.last_result is not None
    assert warming.last_result.text == "The answer is 6"

    # The streamed run recorded the answer, so an empty adapter is enough.
    replayed = build([])
    events = [e async for e in replayed.run_stream("what is the total")]

    assert _stream_text(events) == "The answer is 6"
    assert replayed.last_result is not None
    assert replayed.last_result.turns == 0
    assert replayed.last_result.value == 6


@pytest.mark.asyncio
async def test_async_replay_does_not_block_the_event_loop(tmp_path):
    """Replayed steps are offloaded, matching how the async driver dispatches."""
    import threading

    seen: list[str] = []
    agent = AsyncAgent(
        adapter=FakeAsyncAdapter([]), system="sys", run_dir=str(tmp_path)
    )
    agent.enable_cache()

    spec = ToolSpec(
        name="whereami",
        description="Report the executing thread.",
        input_schema={"type": "object", "properties": {}},
        handler=lambda: seen.append(threading.current_thread().name) or "ok",
    )

    class Cached:
        steps = [{"tool": "whereami", "input": {}}]
        text = "cached answer"

    agent._build_tools = lambda **kwargs: [spec]  # type: ignore[method-assign]
    result = await agent._replay(Cached())

    assert result.text == "cached answer"
    assert seen and seen != [threading.current_thread().name]
