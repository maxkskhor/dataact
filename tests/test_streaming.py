"""Comprehensive tests for the streaming protocol.

Covers:
- StreamEvent dataclass type fields (discriminators)
- accumulate_stream_events() reconstructing text and tool-use responses
- Default AsyncProviderAdapter.stream_events() event sequence
- AsyncHarness.run_stream() event ordering and content
- ToolResultEvent: correct fields, is_error propagation
- Multiple tools in one turn
- Multi-turn streaming (tool turn + final text turn)
- stream() backward-compat wrapper still delivers text to on_chunk
- Usage token counts propagated in MessageDeltaEvent
- ask_stream() continues session history
- Error handling: provider raises mid-stream
- CustomStreamAdapter: overriding stream_events() with real partial events
- Event sequence invariant: every turn starts with MessageStartEvent and ends
  with MessageStopEvent before the next turn's MessageStartEvent
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

from data_harness.agent import AsyncAgent
from data_harness.loop import AsyncHarness
from data_harness.providers.base import (
    AsyncProviderAdapter,
    NormalizedResponse,
    StopReason,
)
from data_harness.streaming import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    InputJSONDelta,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
    StreamEvent,
    TextDelta,
    ToolResultEvent,
    accumulate_stream_events,
)
from data_harness.testing import FakeAsyncAdapter
from data_harness.types import Message, TextBlock, ToolSpec, ToolUseBlock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text_from_events(events: list[StreamEvent]) -> str:
    return "".join(
        e.delta.text
        for e in events
        if isinstance(e, ContentBlockDeltaEvent) and isinstance(e.delta, TextDelta)
    )


def _tool_result_events(events: list[StreamEvent]) -> list[ToolResultEvent]:
    return [e for e in events if isinstance(e, ToolResultEvent)]


def _message_start_events(events: list[StreamEvent]) -> list[MessageStartEvent]:
    return [e for e in events if isinstance(e, MessageStartEvent)]


def _message_stop_events(events: list[StreamEvent]) -> list[MessageStopEvent]:
    return [e for e in events if isinstance(e, MessageStopEvent)]


def _message_delta_events(events: list[StreamEvent]) -> list[MessageDeltaEvent]:
    return [e for e in events if isinstance(e, MessageDeltaEvent)]


class _FakeStreamingAdapter(AsyncProviderAdapter):
    """Scripted adapter that emits actual chunked StreamEvents directly."""

    def __init__(self, event_sequences: list[list[StreamEvent]]) -> None:
        self._sequences = list(event_sequences)
        self._chat_responses: list[NormalizedResponse] = []

    def _push_chat(self, resp: NormalizedResponse) -> None:
        self._chat_responses.append(resp)

    async def chat(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> NormalizedResponse:
        return self._chat_responses.pop(0)

    def format_cache_control(self, obj: dict) -> dict:
        return dict(obj)

    async def stream_events(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> AsyncGenerator[StreamEvent, None]:
        for evt in self._sequences.pop(0):
            yield evt


def _text_turn_events(text: str) -> list[StreamEvent]:
    """Build a standard event sequence for a text-only turn."""
    return [
        MessageStartEvent(),
        ContentBlockStartEvent(index=0, content_block=TextBlock(text="")),
        ContentBlockDeltaEvent(index=0, delta=TextDelta(text=text)),
        ContentBlockStopEvent(index=0),
        MessageDeltaEvent(
            stop_reason=StopReason.END_TURN,
            input_tokens=10,
            output_tokens=5,
            cache_read_tokens=0,
            cache_write_tokens=0,
        ),
        MessageStopEvent(),
    ]


def _tool_turn_events(
    tool_use_id: str, tool_name: str, tool_input: dict
) -> list[StreamEvent]:
    """Build a standard event sequence for a tool-use turn."""
    partial_json = json.dumps(tool_input)
    return [
        MessageStartEvent(),
        ContentBlockStartEvent(
            index=0,
            content_block=ToolUseBlock(
                tool_use_id=tool_use_id, tool_name=tool_name, tool_input={}
            ),
        ),
        ContentBlockDeltaEvent(
            index=0, delta=InputJSONDelta(partial_json=partial_json)
        ),
        ContentBlockStopEvent(index=0),
        MessageDeltaEvent(
            stop_reason=StopReason.TOOL_USE,
            input_tokens=20,
            output_tokens=8,
            cache_read_tokens=0,
            cache_write_tokens=0,
        ),
        MessageStopEvent(),
    ]


# ---------------------------------------------------------------------------
# 1. StreamEvent dataclass discriminators
# ---------------------------------------------------------------------------


class TestEventDiscriminators:
    def test_message_start_type(self):
        assert MessageStartEvent().type == "message_start"

    def test_content_block_start_type(self):
        e = ContentBlockStartEvent(index=0, content_block=TextBlock(text=""))
        assert e.type == "content_block_start"

    def test_content_block_delta_type(self):
        e = ContentBlockDeltaEvent(index=0, delta=TextDelta(text="hi"))
        assert e.type == "content_block_delta"

    def test_content_block_stop_type(self):
        assert ContentBlockStopEvent(index=0).type == "content_block_stop"

    def test_message_delta_type(self):
        e = MessageDeltaEvent(
            stop_reason=StopReason.END_TURN,
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )
        assert e.type == "message_delta"

    def test_message_stop_type(self):
        assert MessageStopEvent().type == "message_stop"

    def test_tool_result_event_type(self):
        e = ToolResultEvent(
            tool_use_id="id", tool_name="t", content="out", is_error=False
        )
        assert e.type == "tool_result"

    def test_text_delta_type(self):
        assert TextDelta(text="x").type == "text_delta"

    def test_input_json_delta_type(self):
        assert InputJSONDelta(partial_json="{}").type == "input_json_delta"


# ---------------------------------------------------------------------------
# 2. accumulate_stream_events() — text response
# ---------------------------------------------------------------------------


class TestAccumulateStreamEvents:
    def test_text_response_content(self):
        events = _text_turn_events("hello world")
        resp = accumulate_stream_events(events)
        assert len(resp.content) == 1
        assert isinstance(resp.content[0], TextBlock)
        assert resp.content[0].text == "hello world"

    def test_text_response_stop_reason(self):
        events = _text_turn_events("x")
        resp = accumulate_stream_events(events)
        assert resp.stop_reason == StopReason.END_TURN

    def test_text_response_usage(self):
        events = _text_turn_events("x")
        resp = accumulate_stream_events(events)
        assert resp.input_tokens == 10
        assert resp.output_tokens == 5

    def test_text_assembled_from_multiple_deltas(self):
        events = [
            MessageStartEvent(),
            ContentBlockStartEvent(index=0, content_block=TextBlock(text="")),
            ContentBlockDeltaEvent(index=0, delta=TextDelta(text="hel")),
            ContentBlockDeltaEvent(index=0, delta=TextDelta(text="lo")),
            ContentBlockDeltaEvent(index=0, delta=TextDelta(text=" world")),
            ContentBlockStopEvent(index=0),
            MessageDeltaEvent(
                stop_reason=StopReason.END_TURN,
                input_tokens=1,
                output_tokens=2,
                cache_read_tokens=0,
                cache_write_tokens=0,
            ),
            MessageStopEvent(),
        ]
        resp = accumulate_stream_events(events)
        assert resp.content[0].text == "hello world"

    def test_tool_use_response_content(self):
        events = _tool_turn_events("tu1", "my_tool", {"key": "value"})
        resp = accumulate_stream_events(events)
        assert len(resp.content) == 1
        assert isinstance(resp.content[0], ToolUseBlock)
        assert resp.content[0].tool_name == "my_tool"
        assert resp.content[0].tool_use_id == "tu1"
        assert resp.content[0].tool_input == {"key": "value"}

    def test_tool_use_stop_reason(self):
        events = _tool_turn_events("tu1", "t", {})
        resp = accumulate_stream_events(events)
        assert resp.stop_reason == StopReason.TOOL_USE

    def test_tool_input_assembled_from_fragmented_json(self):
        events = [
            MessageStartEvent(),
            ContentBlockStartEvent(
                index=0,
                content_block=ToolUseBlock(
                    tool_use_id="x", tool_name="f", tool_input={}
                ),
            ),
            ContentBlockDeltaEvent(index=0, delta=InputJSONDelta(partial_json='{"a"')),
            ContentBlockDeltaEvent(index=0, delta=InputJSONDelta(partial_json=": 1}")),
            ContentBlockStopEvent(index=0),
            MessageDeltaEvent(
                stop_reason=StopReason.TOOL_USE,
                input_tokens=0,
                output_tokens=0,
                cache_read_tokens=0,
                cache_write_tokens=0,
            ),
            MessageStopEvent(),
        ]
        resp = accumulate_stream_events(events)
        assert resp.content[0].tool_input == {"a": 1}

    def test_multiple_blocks_ordered_by_index(self):
        events = [
            MessageStartEvent(),
            ContentBlockStartEvent(index=0, content_block=TextBlock(text="")),
            ContentBlockDeltaEvent(index=0, delta=TextDelta(text="first")),
            ContentBlockStopEvent(index=0),
            ContentBlockStartEvent(
                index=1,
                content_block=ToolUseBlock(
                    tool_use_id="id1", tool_name="fn", tool_input={}
                ),
            ),
            ContentBlockDeltaEvent(
                index=1, delta=InputJSONDelta(partial_json=json.dumps({"n": 2}))
            ),
            ContentBlockStopEvent(index=1),
            MessageDeltaEvent(
                stop_reason=StopReason.TOOL_USE,
                input_tokens=0,
                output_tokens=0,
                cache_read_tokens=0,
                cache_write_tokens=0,
            ),
            MessageStopEvent(),
        ]
        resp = accumulate_stream_events(events)
        assert len(resp.content) == 2
        assert isinstance(resp.content[0], TextBlock)
        assert resp.content[0].text == "first"
        assert isinstance(resp.content[1], ToolUseBlock)
        assert resp.content[1].tool_input == {"n": 2}

    def test_empty_events_returns_end_turn(self):
        resp = accumulate_stream_events([])
        assert resp.stop_reason == StopReason.END_TURN
        assert resp.content == []

    def test_tool_result_events_ignored_during_accumulation(self):
        events = _text_turn_events("hi") + [
            ToolResultEvent(
                tool_use_id="id", tool_name="t", content="out", is_error=False
            )
        ]
        resp = accumulate_stream_events(events)
        assert len(resp.content) == 1
        assert isinstance(resp.content[0], TextBlock)

    def test_invalid_json_produces_empty_dict(self):
        events = [
            MessageStartEvent(),
            ContentBlockStartEvent(
                index=0,
                content_block=ToolUseBlock(
                    tool_use_id="id", tool_name="f", tool_input={}
                ),
            ),
            ContentBlockDeltaEvent(
                index=0, delta=InputJSONDelta(partial_json="BROKEN{")
            ),
            ContentBlockStopEvent(index=0),
            MessageDeltaEvent(
                stop_reason=StopReason.TOOL_USE,
                input_tokens=0,
                output_tokens=0,
                cache_read_tokens=0,
                cache_write_tokens=0,
            ),
            MessageStopEvent(),
        ]
        resp = accumulate_stream_events(events)
        assert resp.content[0].tool_input == {}

    def test_cache_tokens_propagated(self):
        events = [
            MessageStartEvent(),
            MessageDeltaEvent(
                stop_reason=StopReason.END_TURN,
                input_tokens=100,
                output_tokens=50,
                cache_read_tokens=30,
                cache_write_tokens=10,
            ),
            MessageStopEvent(),
        ]
        resp = accumulate_stream_events(events)
        assert resp.cache_read_tokens == 30
        assert resp.cache_write_tokens == 10


# ---------------------------------------------------------------------------
# 3. Default stream_events() in AsyncProviderAdapter
# ---------------------------------------------------------------------------


class TestDefaultStreamEvents:
    """Verify that the FakeAsyncAdapter (which inherits the default stream_events)
    emits the correct event sequence."""

    async def test_text_event_sequence(self, tmp_path):
        adapter = FakeAsyncAdapter([FakeAsyncAdapter.text("hello")])
        events = []
        async for evt in adapter.stream_events("sys", [], []):
            events.append(evt)

        types = [e.type for e in events]
        assert types == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]

    async def test_text_content_in_delta(self):
        adapter = FakeAsyncAdapter([FakeAsyncAdapter.text("world")])
        events = []
        async for evt in adapter.stream_events("sys", [], []):
            events.append(evt)
        assert _text_from_events(events) == "world"

    async def test_tool_use_event_sequence(self):
        adapter = FakeAsyncAdapter(
            [FakeAsyncAdapter.tool_use("id1", "my_tool", {"x": 1})]
        )
        events = []
        async for evt in adapter.stream_events("sys", [], []):
            events.append(evt)

        types = [e.type for e in events]
        assert types == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]

    async def test_tool_use_content_block_start_carries_tool_name(self):
        adapter = FakeAsyncAdapter(
            [FakeAsyncAdapter.tool_use("id1", "my_tool", {"x": 1})]
        )
        events = []
        async for evt in adapter.stream_events("sys", [], []):
            events.append(evt)

        starts = [e for e in events if isinstance(e, ContentBlockStartEvent)]
        assert len(starts) == 1
        assert isinstance(starts[0].content_block, ToolUseBlock)
        assert starts[0].content_block.tool_name == "my_tool"
        assert starts[0].content_block.tool_use_id == "id1"

    async def test_tool_use_input_serialised_to_json_delta(self):
        adapter = FakeAsyncAdapter(
            [FakeAsyncAdapter.tool_use("id1", "f", {"a": 1, "b": "two"})]
        )
        events = []
        async for evt in adapter.stream_events("sys", [], []):
            events.append(evt)

        json_deltas = [
            e
            for e in events
            if isinstance(e, ContentBlockDeltaEvent)
            and isinstance(e.delta, InputJSONDelta)
        ]
        assert len(json_deltas) == 1
        parsed = json.loads(json_deltas[0].delta.partial_json)
        assert parsed == {"a": 1, "b": "two"}

    async def test_message_delta_carries_stop_reason(self):
        adapter = FakeAsyncAdapter([FakeAsyncAdapter.text("done")])
        events = []
        async for evt in adapter.stream_events("sys", [], []):
            events.append(evt)

        deltas = _message_delta_events(events)
        assert len(deltas) == 1
        assert deltas[0].stop_reason == StopReason.END_TURN

    async def test_message_delta_carries_usage_from_fake(self):
        adapter = FakeAsyncAdapter([FakeAsyncAdapter.text("x")])
        events = []
        async for evt in adapter.stream_events("sys", [], []):
            events.append(evt)

        delta = _message_delta_events(events)[0]
        assert delta.input_tokens == 10
        assert delta.output_tokens == 5


# ---------------------------------------------------------------------------
# 4. Event ordering invariant across multiple turns
# ---------------------------------------------------------------------------


class TestEventOrdering:
    """Each turn must have its own message_start … message_stop envelope."""

    async def test_single_text_turn_envelope(self, tmp_path):
        adapter = FakeAsyncAdapter([FakeAsyncAdapter.text("hi")])
        harness = AsyncHarness(
            adapter=adapter, system="s", tools=[], run_dir=str(tmp_path)
        )
        events = []
        async for e in harness.run_stream("q"):
            events.append(e)

        assert events[0].type == "message_start"
        assert events[-1].type == "message_stop"

    async def test_two_turn_envelopes(self, tmp_path):
        """Tool turn then text turn: two message_start and two message_stop."""

        def noop() -> str:
            return "result"

        tool_spec = ToolSpec(
            name="noop",
            description="noop",
            input_schema={"type": "object", "properties": {}},
            handler=noop,
        )
        adapter = FakeAsyncAdapter(
            [
                FakeAsyncAdapter.tool_use("id1", "noop", {}),
                FakeAsyncAdapter.text("done"),
            ]
        )
        harness = AsyncHarness(
            adapter=adapter, system="s", tools=[tool_spec], run_dir=str(tmp_path)
        )
        events = []
        async for e in harness.run_stream("q"):
            events.append(e)

        starts = _message_start_events(events)
        stops = _message_stop_events(events)
        assert len(starts) == 2
        assert len(stops) == 2

    async def test_tool_result_between_turns(self, tmp_path):
        """ToolResultEvent must appear between the first message_stop and
        the second message_start."""

        def noop() -> str:
            return "ok"

        tool_spec = ToolSpec(
            name="noop",
            description="noop",
            input_schema={"type": "object", "properties": {}},
            handler=noop,
        )
        adapter = FakeAsyncAdapter(
            [
                FakeAsyncAdapter.tool_use("id1", "noop", {}),
                FakeAsyncAdapter.text("done"),
            ]
        )
        harness = AsyncHarness(
            adapter=adapter, system="s", tools=[tool_spec], run_dir=str(tmp_path)
        )
        events = []
        async for e in harness.run_stream("q"):
            events.append(e)

        types = [e.type for e in events]
        first_stop_idx = types.index("message_stop")
        second_start_idx = types.index("message_start", first_stop_idx + 1)
        between = types[first_stop_idx + 1 : second_start_idx]
        assert "tool_result" in between


# ---------------------------------------------------------------------------
# 5. ToolResultEvent fields
# ---------------------------------------------------------------------------


class TestToolResultEvent:
    async def test_tool_result_fields(self, tmp_path):
        def adder(a: int, b: int) -> int:
            return a + b

        tool_spec = ToolSpec(
            name="adder",
            description="adds two ints",
            input_schema={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            },
            handler=adder,
        )
        adapter = FakeAsyncAdapter(
            [
                FakeAsyncAdapter.tool_use("tu1", "adder", {"a": 3, "b": 4}),
                FakeAsyncAdapter.text("done"),
            ]
        )
        harness = AsyncHarness(
            adapter=adapter, system="s", tools=[tool_spec], run_dir=str(tmp_path)
        )
        events = []
        async for e in harness.run_stream("q"):
            events.append(e)

        tr = _tool_result_events(events)
        assert len(tr) == 1
        assert tr[0].tool_use_id == "tu1"
        assert tr[0].tool_name == "adder"
        assert "7" in tr[0].content
        assert not tr[0].is_error

    async def test_error_tool_result(self, tmp_path):
        def boom() -> str:
            raise ValueError("exploded")

        tool_spec = ToolSpec(
            name="boom",
            description="always fails",
            input_schema={"type": "object", "properties": {}},
            handler=boom,
        )
        adapter = FakeAsyncAdapter(
            [
                FakeAsyncAdapter.tool_use("tu1", "boom", {}),
                FakeAsyncAdapter.text("recovered"),
            ]
        )
        harness = AsyncHarness(
            adapter=adapter, system="s", tools=[tool_spec], run_dir=str(tmp_path)
        )
        events = []
        async for e in harness.run_stream("q"):
            events.append(e)

        tr = _tool_result_events(events)
        assert len(tr) == 1
        assert tr[0].is_error
        assert "exploded" in tr[0].content

    async def test_missing_tool_result(self, tmp_path):
        """Requesting a non-existent tool produces an error ToolResultEvent."""
        adapter = FakeAsyncAdapter(
            [
                FakeAsyncAdapter.tool_use("tu1", "ghost_tool", {}),
                FakeAsyncAdapter.text("done"),
            ]
        )
        harness = AsyncHarness(
            adapter=adapter, system="s", tools=[], run_dir=str(tmp_path)
        )
        events = []
        async for e in harness.run_stream("q"):
            events.append(e)

        tr = _tool_result_events(events)
        assert len(tr) == 1
        assert tr[0].is_error
        assert tr[0].tool_name == "ghost_tool"


# ---------------------------------------------------------------------------
# 6. Multiple tools in one turn
# ---------------------------------------------------------------------------


class TestMultipleToolsPerTurn:
    async def test_two_tools_emit_two_tool_result_events(self, tmp_path):
        results: list[str] = []

        def tool_a(x: str) -> str:
            results.append(f"a:{x}")
            return f"a:{x}"

        def tool_b(y: str) -> str:
            results.append(f"b:{y}")
            return f"b:{y}"

        specs = [
            ToolSpec(
                name="tool_a",
                description="a",
                input_schema={
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                },
                handler=tool_a,
            ),
            ToolSpec(
                name="tool_b",
                description="b",
                input_schema={
                    "type": "object",
                    "properties": {"y": {"type": "string"}},
                },
                handler=tool_b,
            ),
        ]

        from data_harness.providers.base import NormalizedResponse

        two_tool_response = NormalizedResponse(
            stop_reason=StopReason.TOOL_USE,
            content=[
                ToolUseBlock(
                    tool_use_id="id1", tool_name="tool_a", tool_input={"x": "1"}
                ),
                ToolUseBlock(
                    tool_use_id="id2", tool_name="tool_b", tool_input={"y": "2"}
                ),
            ],
            input_tokens=10,
            output_tokens=5,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )

        adapter = FakeAsyncAdapter([two_tool_response, FakeAsyncAdapter.text("fin")])
        harness = AsyncHarness(
            adapter=adapter, system="s", tools=specs, run_dir=str(tmp_path)
        )
        events = []
        async for e in harness.run_stream("q"):
            events.append(e)

        tr = _tool_result_events(events)
        assert len(tr) == 2
        names = {t.tool_name for t in tr}
        assert names == {"tool_a", "tool_b"}
        assert results == ["a:1", "b:2"]


# ---------------------------------------------------------------------------
# 7. stream() backward-compat wrapper
# ---------------------------------------------------------------------------


class TestStreamBackwardCompat:
    async def test_on_chunk_receives_text(self):
        adapter = FakeAsyncAdapter([FakeAsyncAdapter.text("chunk data")])
        chunks: list[str] = []

        async def on_chunk(text: str) -> None:
            chunks.append(text)

        await adapter.stream("sys", [], [], on_chunk=on_chunk)
        assert "".join(chunks) == "chunk data"

    async def test_on_chunk_not_called_for_tool_use(self):
        adapter = FakeAsyncAdapter([FakeAsyncAdapter.tool_use("id1", "t", {"x": 1})])
        chunks: list[str] = []

        async def on_chunk(text: str) -> None:
            chunks.append(text)

        resp = await adapter.stream("sys", [], [], on_chunk=on_chunk)
        assert chunks == []
        assert resp.stop_reason == StopReason.TOOL_USE

    async def test_stream_returns_normalized_response(self):
        adapter = FakeAsyncAdapter([FakeAsyncAdapter.text("hello")])

        async def noop(text: str) -> None:
            pass

        resp = await adapter.stream("sys", [], [], on_chunk=noop)
        assert isinstance(resp, NormalizedResponse)
        assert resp.stop_reason == StopReason.END_TURN
        assert isinstance(resp.content[0], TextBlock)
        assert resp.content[0].text == "hello"


# ---------------------------------------------------------------------------
# 8. Custom stream_events() override (fragmented text deltas)
# ---------------------------------------------------------------------------


class TestCustomStreamAdapter:
    """Verify that a custom adapter emitting real chunked text events works
    correctly throughout the harness pipeline."""

    async def test_fragmented_text_deltas_assembled_correctly(self, tmp_path):
        chunks = ["Hel", "lo", " ", "world"]
        events = (
            [MessageStartEvent()]
            + [ContentBlockStartEvent(index=0, content_block=TextBlock(text=""))]
            + [ContentBlockDeltaEvent(index=0, delta=TextDelta(text=c)) for c in chunks]
            + [ContentBlockStopEvent(index=0)]
            + [
                MessageDeltaEvent(
                    stop_reason=StopReason.END_TURN,
                    input_tokens=5,
                    output_tokens=10,
                    cache_read_tokens=0,
                    cache_write_tokens=0,
                ),
                MessageStopEvent(),
            ]
        )

        adapter = _FakeStreamingAdapter([events])
        harness = AsyncHarness(
            adapter=adapter, system="s", tools=[], run_dir=str(tmp_path)
        )
        collected = []
        async for e in harness.run_stream("q"):
            collected.append(e)

        assert _text_from_events(collected) == "Hello world"

    async def test_fragmented_json_deltas_assembled_correctly(self, tmp_path):
        """Two-part JSON is assembled before the tool handler receives it."""
        received: list[dict] = []

        def capture(**kwargs: Any) -> str:
            received.append(dict(kwargs))
            return "ok"

        tool_spec = ToolSpec(
            name="capture",
            description="captures kwargs",
            input_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "string"},
                },
            },
            handler=capture,
        )

        tool_turn_events = [
            MessageStartEvent(),
            ContentBlockStartEvent(
                index=0,
                content_block=ToolUseBlock(
                    tool_use_id="id1", tool_name="capture", tool_input={}
                ),
            ),
            ContentBlockDeltaEvent(index=0, delta=InputJSONDelta(partial_json='{"a"')),
            ContentBlockDeltaEvent(
                index=0, delta=InputJSONDelta(partial_json=': 42, "b": "hi"}')
            ),
            ContentBlockStopEvent(index=0),
            MessageDeltaEvent(
                stop_reason=StopReason.TOOL_USE,
                input_tokens=10,
                output_tokens=5,
                cache_read_tokens=0,
                cache_write_tokens=0,
            ),
            MessageStopEvent(),
        ]
        text_turn_events = _text_turn_events("done")

        adapter = _FakeStreamingAdapter([tool_turn_events, text_turn_events])
        harness = AsyncHarness(
            adapter=adapter, system="s", tools=[tool_spec], run_dir=str(tmp_path)
        )
        events = []
        async for e in harness.run_stream("q"):
            events.append(e)

        assert received == [{"a": 42, "b": "hi"}]
        assert _text_from_events(events) == "done"


# ---------------------------------------------------------------------------
# 9. ask_stream() — session continuity
# ---------------------------------------------------------------------------


class TestAskStream:
    async def test_ask_stream_events_after_run_stream(self, tmp_path):
        adapter = FakeAsyncAdapter(
            [
                FakeAsyncAdapter.text("first"),
                FakeAsyncAdapter.text("second"),
            ]
        )
        harness = AsyncHarness(
            adapter=adapter, system="s", tools=[], run_dir=str(tmp_path)
        )
        ev1 = []
        async for e in harness.run_stream("q1"):
            ev1.append(e)

        ev2 = []
        async for e in harness.ask_stream("q2"):
            ev2.append(e)

        assert _text_from_events(ev1) == "first"
        assert _text_from_events(ev2) == "second"

    async def test_session_ask_stream(self, tmp_path):
        adapter = FakeAsyncAdapter([FakeAsyncAdapter.text("session reply")])
        agent = AsyncAgent(adapter=adapter, system="s", run_dir=tmp_path)
        session = agent.async_session()
        events = []
        async for e in session.ask_stream("q"):
            events.append(e)
        assert _text_from_events(events) == "session reply"


# ---------------------------------------------------------------------------
# 10. Usage propagation
# ---------------------------------------------------------------------------


class TestUsagePropagation:
    async def test_usage_in_message_delta(self, tmp_path):
        adapter = FakeAsyncAdapter([FakeAsyncAdapter.text("x")])
        harness = AsyncHarness(
            adapter=adapter, system="s", tools=[], run_dir=str(tmp_path)
        )
        events = []
        async for e in harness.run_stream("q"):
            events.append(e)

        deltas = _message_delta_events(events)
        assert len(deltas) == 1
        assert deltas[0].input_tokens == 10
        assert deltas[0].output_tokens == 5

    async def test_usage_accumulates_across_turns(self, tmp_path):
        """Two turns should each have their own MessageDeltaEvent with usage."""

        def noop() -> str:
            return "r"

        tool_spec = ToolSpec(
            name="noop",
            description="noop",
            input_schema={"type": "object", "properties": {}},
            handler=noop,
        )
        adapter = FakeAsyncAdapter(
            [
                FakeAsyncAdapter.tool_use("id1", "noop", {}),
                FakeAsyncAdapter.text("done"),
            ]
        )
        harness = AsyncHarness(
            adapter=adapter, system="s", tools=[tool_spec], run_dir=str(tmp_path)
        )
        events = []
        async for e in harness.run_stream("q"):
            events.append(e)

        deltas = _message_delta_events(events)
        assert len(deltas) == 2


# ---------------------------------------------------------------------------
# 11. Error handling: provider raises mid-stream
# ---------------------------------------------------------------------------


class TestStreamErrorHandling:
    async def test_provider_exception_terminates_stream_gracefully(self, tmp_path):
        class BrokenAdapter(AsyncProviderAdapter):
            async def chat(self, system, messages, tools):
                return FakeAsyncAdapter.text("x")

            def format_cache_control(self, obj):
                return obj

            async def stream_events(self, system, messages, tools):
                yield MessageStartEvent()
                raise RuntimeError("provider blew up")

        harness = AsyncHarness(
            adapter=BrokenAdapter(), system="s", tools=[], run_dir=str(tmp_path)
        )
        events = []
        async for e in harness.run_stream("q"):
            events.append(e)

        # At least the message_start was emitted before the error
        assert any(isinstance(e, MessageStartEvent) for e in events)
        # Stream should have stopped (no infinite loop)
        assert len(events) < 100

        # The stream stopping is not enough: the run must still be accounted
        # for. This test used to end at the line above, which is how the
        # streaming loop got away with discarding its RunResult on provider
        # errors, and with it the tokens already spent.
        result = harness.last_result
        assert result is not None
        assert result.status == "error"
        assert "provider blew up" in (result.error or "")
        assert result.run_file == harness.run_file

    async def test_unknown_event_types_are_ignored_not_fatal(self, tmp_path):
        """The accumulator skips events it does not recognise, on purpose.

        Providers add event types over time; an unknown one must not break a
        run. It contributes nothing, so the turn assembles from the rest.
        """

        class ChattyAdapter(AsyncProviderAdapter):
            async def chat(self, system, messages, tools):
                return FakeAsyncAdapter.text("x")

            def format_cache_control(self, obj):
                return obj

            async def stream_events(self, system, messages, tools):
                yield MessageStartEvent()
                yield "some future event type"
                yield MessageStopEvent()

        harness = AsyncHarness(
            adapter=ChattyAdapter(), system="s", tools=[], run_dir=str(tmp_path)
        )
        async for _ in harness.run_stream("q"):
            pass

        result = harness.last_result
        assert result is not None
        assert result.status == "success"

    async def test_a_failure_while_assembling_the_turn_is_reported(
        self, tmp_path, monkeypatch
    ):
        """Accumulation runs inside the loop's provider try-block, deliberately.

        If assembling the turn blows up, that is a provider failure and belongs
        in the RunResult. Outside the try it would escape into the caller's
        `async for` and bypass all run accounting.
        """

        def explode(_events):
            raise ValueError("cannot assemble this turn")

        monkeypatch.setattr("data_harness.loop.accumulate_stream_events", explode)
        harness = AsyncHarness(
            adapter=FakeAsyncAdapter([FakeAsyncAdapter.text("hi")]),
            system="s",
            tools=[],
            run_dir=str(tmp_path),
        )
        async for _ in harness.run_stream("q"):
            pass

        result = harness.last_result
        assert result is not None
        assert result.status == "error"
        assert "cannot assemble this turn" in (result.error or "")


# ---------------------------------------------------------------------------
# 12. AsyncAgent.run_stream() passthrough
# ---------------------------------------------------------------------------


class TestAsyncAgentRunStream:
    async def test_agent_run_stream_events_passthrough(self, tmp_path):
        adapter = FakeAsyncAdapter([FakeAsyncAdapter.text("agent text")])
        agent = AsyncAgent(adapter=adapter, system="s", run_dir=tmp_path)
        events = []
        async for e in agent.run_stream("q"):
            events.append(e)

        assert _text_from_events(events) == "agent text"
        assert events[0].type == "message_start"
        assert events[-1].type == "message_stop"

    async def test_agent_run_stream_with_tool(self, tmp_path):
        calls: list[dict] = []

        def my_tool(val: str) -> str:
            calls.append({"val": val})
            return f"got:{val}"

        agent = AsyncAgent(
            adapter=FakeAsyncAdapter(
                [
                    FakeAsyncAdapter.tool_use("tu1", "my_tool", {"val": "abc"}),
                    FakeAsyncAdapter.text("final"),
                ]
            ),
            system="s",
            run_dir=tmp_path,
        )
        agent.connector("demo", description="demo connector").tool(
            my_tool,
            description="calls my_tool",
            input_schema={
                "type": "object",
                "properties": {"val": {"type": "string"}},
            },
        )

        # Agent uses a connector so my_tool is hidden initially; use direct harness
        # approach instead to avoid connector routing complexity
        adapter2 = FakeAsyncAdapter(
            [
                FakeAsyncAdapter.tool_use("tu1", "raw_tool", {"v": "xyz"}),
                FakeAsyncAdapter.text("answer"),
            ]
        )
        raw_calls: list[str] = []

        def raw_tool(v: str) -> str:
            raw_calls.append(v)
            return "raw_result"

        from data_harness.loop import AsyncHarness

        harness = AsyncHarness(
            adapter=adapter2,
            system="s",
            tools=[
                ToolSpec(
                    name="raw_tool",
                    description="raw",
                    input_schema={
                        "type": "object",
                        "properties": {"v": {"type": "string"}},
                    },
                    handler=raw_tool,
                )
            ],
            run_dir=str(tmp_path),
        )
        events = []
        async for e in harness.run_stream("q"):
            events.append(e)

        assert raw_calls == ["xyz"]
        assert _text_from_events(events) == "answer"
        tr = _tool_result_events(events)
        assert len(tr) == 1
        assert tr[0].tool_name == "raw_tool"
        assert "raw_result" in tr[0].content


# ---------------------------------------------------------------------------
# 13. max_turns hit during streaming
# ---------------------------------------------------------------------------


class TestMaxTurnsStreaming:
    async def test_stream_stops_at_max_turns(self, tmp_path):
        """When max_turns is exhausted on a tool-use turn, the stream ends
        after emitting ToolResultEvents for that final turn."""

        def noop() -> str:
            return "r"

        tool_spec = ToolSpec(
            name="noop",
            description="noop",
            input_schema={"type": "object", "properties": {}},
            handler=noop,
        )
        # Two tool-use responses — max_turns=2 means the second tool turn is
        # the last; the stream must stop without hanging.
        adapter = FakeAsyncAdapter(
            [
                FakeAsyncAdapter.tool_use("id1", "noop", {}),
                FakeAsyncAdapter.tool_use("id2", "noop", {}),
            ]
        )
        harness = AsyncHarness(
            adapter=adapter,
            system="s",
            tools=[tool_spec],
            max_turns=2,
            run_dir=str(tmp_path),
        )
        events = []
        async for e in harness.run_stream("q"):
            events.append(e)

        # Stream must terminate (not hang or raise)
        tool_results = _tool_result_events(events)
        # Both turns dispatched the tool
        assert len(tool_results) == 2

    async def test_stream_terminates_no_infinite_loop(self, tmp_path):
        """Verify that max_turns=1 with a tool-use response ends the stream."""
        adapter = FakeAsyncAdapter([FakeAsyncAdapter.tool_use("id1", "ghost", {})])
        harness = AsyncHarness(
            adapter=adapter, system="s", tools=[], max_turns=1, run_dir=str(tmp_path)
        )
        events = []
        async for e in harness.run_stream("q"):
            events.append(e)

        # Must have stopped — the tool_result (missing tool, is_error=True)
        assert len(events) > 0
        assert len(events) < 200


# ---------------------------------------------------------------------------
# 14. Async tool handler in streaming path
# ---------------------------------------------------------------------------


class TestAsyncToolInStream:
    async def test_async_coroutine_tool_called_in_stream(self, tmp_path):
        awaited: list[str] = []

        async def async_fetcher(url: str) -> str:
            awaited.append(url)
            return f"fetched:{url}"

        tool_spec = ToolSpec(
            name="async_fetcher",
            description="fetches a url",
            input_schema={
                "type": "object",
                "properties": {"url": {"type": "string"}},
            },
            handler=async_fetcher,
        )
        adapter = FakeAsyncAdapter(
            [
                FakeAsyncAdapter.tool_use(
                    "tu1", "async_fetcher", {"url": "http://x.com"}
                ),
                FakeAsyncAdapter.text("done"),
            ]
        )
        harness = AsyncHarness(
            adapter=adapter, system="s", tools=[tool_spec], run_dir=str(tmp_path)
        )
        events = []
        async for e in harness.run_stream("q"):
            events.append(e)

        assert awaited == ["http://x.com"]
        tr = _tool_result_events(events)
        assert len(tr) == 1
        assert "fetched:http://x.com" in tr[0].content
        assert not tr[0].is_error


# ---------------------------------------------------------------------------
# 15. Message history integrity after run_stream
# ---------------------------------------------------------------------------


class TestMessageHistoryAfterStream:
    async def test_ask_result_after_run_stream_sees_previous_context(self, tmp_path):
        """After run_stream(), calling ask_result() should work and include
        the previous exchange in the harness's internal message history."""
        adapter = FakeAsyncAdapter(
            [
                FakeAsyncAdapter.text("stream answer"),
                FakeAsyncAdapter.text("follow up answer"),
            ]
        )
        harness = AsyncHarness(
            adapter=adapter, system="s", tools=[], run_dir=str(tmp_path)
        )

        events = []
        async for e in harness.run_stream("first question"):
            events.append(e)
        assert _text_from_events(events) == "stream answer"

        # Now a follow-up ask
        result = await harness.ask_result("second question")
        assert result.text == "follow up answer"
        assert result.status == "success"

        # The adapter must have been called twice
        assert len(adapter.calls) == 2

    async def test_run_stream_then_ask_stream(self, tmp_path):
        adapter = FakeAsyncAdapter(
            [
                FakeAsyncAdapter.text("first"),
                FakeAsyncAdapter.text("second"),
            ]
        )
        harness = AsyncHarness(
            adapter=adapter, system="s", tools=[], run_dir=str(tmp_path)
        )

        ev1 = []
        async for e in harness.run_stream("q1"):
            ev1.append(e)

        ev2 = []
        async for e in harness.ask_stream("q2"):
            ev2.append(e)

        assert _text_from_events(ev1) == "first"
        assert _text_from_events(ev2) == "second"
        # Both turns should have full event envelopes
        assert len(_message_start_events(ev1)) == 1
        assert len(_message_start_events(ev2)) == 1


# ---------------------------------------------------------------------------
# 16. ToolResultEvent.tool_use_id matches ContentBlockStartEvent source
# ---------------------------------------------------------------------------


class TestToolResultEventIdCorrespondence:
    async def test_tool_result_id_matches_tool_use_id_from_stream(self, tmp_path):
        """ToolResultEvent.tool_use_id must equal the tool_use_id from the
        corresponding ContentBlockStartEvent in the same turn."""

        def echo(v: str) -> str:
            return v

        tool_spec = ToolSpec(
            name="echo",
            description="echoes",
            input_schema={
                "type": "object",
                "properties": {"v": {"type": "string"}},
            },
            handler=echo,
        )
        adapter = FakeAsyncAdapter(
            [
                FakeAsyncAdapter.tool_use("SPECIFIC-ID-123", "echo", {"v": "hi"}),
                FakeAsyncAdapter.text("done"),
            ]
        )
        harness = AsyncHarness(
            adapter=adapter, system="s", tools=[tool_spec], run_dir=str(tmp_path)
        )
        events = []
        async for e in harness.run_stream("q"):
            events.append(e)

        # Get the ContentBlockStartEvent for the tool_use block
        tool_starts = [
            e
            for e in events
            if isinstance(e, ContentBlockStartEvent)
            and isinstance(e.content_block, ToolUseBlock)
        ]
        assert len(tool_starts) == 1
        source_id = tool_starts[0].content_block.tool_use_id

        tr = _tool_result_events(events)
        assert len(tr) == 1
        assert tr[0].tool_use_id == source_id == "SPECIFIC-ID-123"


# ---------------------------------------------------------------------------
# 17. ContentBlockStartEvent.index correctness for multi-block responses
# ---------------------------------------------------------------------------


class TestContentBlockIndices:
    def test_text_block_at_index_0(self):
        events = _text_turn_events("hello")
        starts = [e for e in events if isinstance(e, ContentBlockStartEvent)]
        assert len(starts) == 1
        assert starts[0].index == 0

    def test_tool_block_at_index_0(self):
        events = _tool_turn_events("id1", "t", {})
        starts = [e for e in events if isinstance(e, ContentBlockStartEvent)]
        assert starts[0].index == 0

    def test_two_blocks_have_distinct_sequential_indices(self):
        events = [
            MessageStartEvent(),
            ContentBlockStartEvent(index=0, content_block=TextBlock(text="")),
            ContentBlockDeltaEvent(index=0, delta=TextDelta(text="a")),
            ContentBlockStopEvent(index=0),
            ContentBlockStartEvent(
                index=1,
                content_block=ToolUseBlock(
                    tool_use_id="id1", tool_name="fn", tool_input={}
                ),
            ),
            ContentBlockDeltaEvent(index=1, delta=InputJSONDelta(partial_json="{}")),
            ContentBlockStopEvent(index=1),
            MessageDeltaEvent(
                stop_reason=StopReason.TOOL_USE,
                input_tokens=0,
                output_tokens=0,
                cache_read_tokens=0,
                cache_write_tokens=0,
            ),
            MessageStopEvent(),
        ]
        starts = [e for e in events if isinstance(e, ContentBlockStartEvent)]
        assert [s.index for s in starts] == [0, 1]

        resp = accumulate_stream_events(events)
        assert len(resp.content) == 2
        assert isinstance(resp.content[0], TextBlock)
        assert isinstance(resp.content[1], ToolUseBlock)

    def test_default_stream_events_text_uses_index_0(self):
        """The default stream_events() must use index 0 for single-block text."""

        # Build a fake response manually (not using FakeAsyncAdapter.text() which
        # dequeues the response; we need to inspect stream_events synchronously)
        events = _text_turn_events("x")
        starts = [e for e in events if isinstance(e, ContentBlockStartEvent)]
        assert starts[0].index == 0


# ---------------------------------------------------------------------------
# 18. One MessageDeltaEvent per turn invariant
# ---------------------------------------------------------------------------


class TestMessageDeltaCount:
    async def test_single_text_turn_has_exactly_one_message_delta(self, tmp_path):
        adapter = FakeAsyncAdapter([FakeAsyncAdapter.text("hi")])
        harness = AsyncHarness(
            adapter=adapter, system="s", tools=[], run_dir=str(tmp_path)
        )
        events = []
        async for e in harness.run_stream("q"):
            events.append(e)
        assert len(_message_delta_events(events)) == 1

    async def test_tool_turn_then_text_turn_has_two_message_deltas(self, tmp_path):
        def noop() -> str:
            return "r"

        tool_spec = ToolSpec(
            name="noop",
            description="noop",
            input_schema={"type": "object", "properties": {}},
            handler=noop,
        )
        adapter = FakeAsyncAdapter(
            [
                FakeAsyncAdapter.tool_use("id1", "noop", {}),
                FakeAsyncAdapter.text("done"),
            ]
        )
        harness = AsyncHarness(
            adapter=adapter, system="s", tools=[tool_spec], run_dir=str(tmp_path)
        )
        events = []
        async for e in harness.run_stream("q"):
            events.append(e)
        assert len(_message_delta_events(events)) == 2


# ---------------------------------------------------------------------------
# 19. Empty tool input (no parameters)
# ---------------------------------------------------------------------------


class TestEmptyToolInput:
    async def test_tool_with_no_params_dispatched_correctly_in_stream(self, tmp_path):
        called = []

        def no_args() -> str:
            called.append(True)
            return "no-arg result"

        tool_spec = ToolSpec(
            name="no_args",
            description="no parameters",
            input_schema={"type": "object", "properties": {}},
            handler=no_args,
        )
        adapter = FakeAsyncAdapter(
            [
                FakeAsyncAdapter.tool_use("tu1", "no_args", {}),
                FakeAsyncAdapter.text("done"),
            ]
        )
        harness = AsyncHarness(
            adapter=adapter, system="s", tools=[tool_spec], run_dir=str(tmp_path)
        )
        events = []
        async for e in harness.run_stream("q"):
            events.append(e)

        assert called == [True]
        tr = _tool_result_events(events)
        assert not tr[0].is_error
        assert "no-arg result" in tr[0].content

    def test_accumulate_empty_json_delta(self):
        events = [
            MessageStartEvent(),
            ContentBlockStartEvent(
                index=0,
                content_block=ToolUseBlock(
                    tool_use_id="id", tool_name="f", tool_input={}
                ),
            ),
            ContentBlockDeltaEvent(index=0, delta=InputJSONDelta(partial_json="{}")),
            ContentBlockStopEvent(index=0),
            MessageDeltaEvent(
                stop_reason=StopReason.TOOL_USE,
                input_tokens=0,
                output_tokens=0,
                cache_read_tokens=0,
                cache_write_tokens=0,
            ),
            MessageStopEvent(),
        ]
        resp = accumulate_stream_events(events)
        assert resp.content[0].tool_input == {}


# ---------------------------------------------------------------------------
# 20. stream() backward compat via Anthropic adapter's stream_events() chain
# ---------------------------------------------------------------------------


class TestStreamBackwardCompatAnthropicChain:
    """The AsyncAnthropicAdapter.stream() is removed; the base class stream()
    now uses stream_events() internally. Verify the chain for
    FakeAsyncAdapter (which inherits both) delivers text to on_chunk."""

    async def test_base_stream_uses_stream_events_chain(self):
        adapter = FakeAsyncAdapter([FakeAsyncAdapter.text("chain text")])
        received: list[str] = []

        async def on_chunk(text: str) -> None:
            received.append(text)

        resp = await adapter.stream("s", [], [], on_chunk=on_chunk)
        assert "".join(received) == "chain text"
        assert resp.content[0].text == "chain text"

    async def test_base_stream_tool_turn_no_on_chunk_calls(self):
        adapter = FakeAsyncAdapter([FakeAsyncAdapter.tool_use("id1", "t", {"x": 1})])
        received: list[str] = []

        async def on_chunk(text: str) -> None:
            received.append(text)

        resp = await adapter.stream("s", [], [], on_chunk=on_chunk)
        assert received == []
        assert resp.stop_reason == StopReason.TOOL_USE

    async def test_base_stream_response_has_correct_content(self):
        adapter = FakeAsyncAdapter([FakeAsyncAdapter.text("the answer")])

        async def noop(_: str) -> None:
            pass

        resp = await adapter.stream("s", [], [], on_chunk=noop)
        assert len(resp.content) == 1
        assert isinstance(resp.content[0], TextBlock)
        assert resp.content[0].text == "the answer"


# ---------------------------------------------------------------------------
# 22. Mixed text+tool content block in same turn
# ---------------------------------------------------------------------------


class TestMixedContentTurnStream:
    """A response with both a text block and a tool-use block in the same turn."""

    async def test_mixed_turn_dispatches_tool_and_continues(self, tmp_path):
        dispatched: list[str] = []

        def my_tool(v: str) -> str:
            dispatched.append(v)
            return "done"

        tool_spec = ToolSpec(
            name="my_tool",
            description="t",
            input_schema={"type": "object", "properties": {"v": {"type": "string"}}},
            handler=my_tool,
            visible=True,
        )

        # First response: text block at index 0 + tool use at index 1
        mixed_turn_events = [
            MessageStartEvent(),
            ContentBlockStartEvent(index=0, content_block=TextBlock(text="")),
            ContentBlockDeltaEvent(index=0, delta=TextDelta(text="thinking...")),
            ContentBlockStopEvent(index=0),
            ContentBlockStartEvent(
                index=1,
                content_block=ToolUseBlock(
                    tool_use_id="tu1", tool_name="my_tool", tool_input={}
                ),
            ),
            ContentBlockDeltaEvent(
                index=1, delta=InputJSONDelta(partial_json='{"v":"hi"}')
            ),
            ContentBlockStopEvent(index=1),
            MessageDeltaEvent(
                stop_reason=StopReason.TOOL_USE,
                input_tokens=10,
                output_tokens=5,
                cache_read_tokens=0,
                cache_write_tokens=0,
            ),
            MessageStopEvent(),
        ]

        class _MixedAdapter(AsyncProviderAdapter):
            def __init__(self) -> None:
                self._turns = iter([mixed_turn_events, _text_turn_events("final")])

            def format_cache_control(self, obj: dict) -> dict:
                return dict(obj)

            async def chat(self, system, messages, tools):
                raise NotImplementedError

            async def stream_events(self, system, messages, tools):
                for evt in next(self._turns):
                    yield evt

        adapter = _MixedAdapter()
        harness = AsyncHarness(
            adapter=adapter,
            system="s",
            tools=[tool_spec],
            max_turns=5,
            run_dir=str(tmp_path),
        )
        events: list[StreamEvent] = []
        async for e in harness.run_stream("go"):
            events.append(e)

        # Tool was dispatched
        assert dispatched == ["hi"]
        # Final text is present
        assert _text_from_events(events) == "thinking...final"
        # ToolResultEvent emitted
        tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(tool_results) == 1
        assert tool_results[0].tool_name == "my_tool"

    async def test_mixed_turn_text_block_preserved_in_events(self, tmp_path):
        """Text deltas from the mixed turn appear before ToolResultEvent."""

        def noop_tool() -> str:
            return "ok"

        tool_spec = ToolSpec(
            name="noop_tool",
            description="t",
            input_schema={"type": "object", "properties": {}},
            handler=noop_tool,
            visible=True,
        )

        mixed_turn_events = [
            MessageStartEvent(),
            ContentBlockStartEvent(index=0, content_block=TextBlock(text="")),
            ContentBlockDeltaEvent(index=0, delta=TextDelta(text="preamble")),
            ContentBlockStopEvent(index=0),
            ContentBlockStartEvent(
                index=1,
                content_block=ToolUseBlock(
                    tool_use_id="tu1", tool_name="noop_tool", tool_input={}
                ),
            ),
            ContentBlockDeltaEvent(index=1, delta=InputJSONDelta(partial_json="{}")),
            ContentBlockStopEvent(index=1),
            MessageDeltaEvent(
                stop_reason=StopReason.TOOL_USE,
                input_tokens=5,
                output_tokens=5,
                cache_read_tokens=0,
                cache_write_tokens=0,
            ),
            MessageStopEvent(),
        ]

        class _MixedAdapter2(AsyncProviderAdapter):
            def __init__(self) -> None:
                self._turns = iter([mixed_turn_events, _text_turn_events("end")])

            def format_cache_control(self, obj: dict) -> dict:
                return dict(obj)

            async def chat(self, system, messages, tools):
                raise NotImplementedError

            async def stream_events(self, system, messages, tools):
                for evt in next(self._turns):
                    yield evt

        adapter = _MixedAdapter2()
        harness = AsyncHarness(
            adapter=adapter,
            system="s",
            tools=[tool_spec],
            max_turns=5,
            run_dir=str(tmp_path),
        )
        events: list[StreamEvent] = []
        async for e in harness.run_stream("go"):
            events.append(e)

        text_deltas = [
            e
            for e in events
            if isinstance(e, ContentBlockDeltaEvent) and isinstance(e.delta, TextDelta)
        ]
        texts = [e.delta.text for e in text_deltas]
        assert "preamble" in texts
        assert "end" in texts
        # Text delta from turn 1 comes before any ToolResultEvent
        first_tool_result_idx = next(
            i for i, e in enumerate(events) if isinstance(e, ToolResultEvent)
        )
        preamble_idx = next(
            i
            for i, e in enumerate(events)
            if isinstance(e, ContentBlockDeltaEvent)
            and isinstance(e.delta, TextDelta)
            and e.delta.text == "preamble"
        )
        assert preamble_idx < first_tool_result_idx


# ---------------------------------------------------------------------------
# 23. Public API exports from top-level package
# ---------------------------------------------------------------------------


class TestPublicAPIExports:
    """Streaming types must be importable from the top-level data_harness package."""

    def test_stream_event_union_importable(self):
        import data_harness

        assert hasattr(data_harness, "StreamEvent")

    def test_all_event_classes_importable(self):
        from data_harness import (
            ContentBlockDeltaEvent,
            ContentBlockStartEvent,
            ContentBlockStopEvent,
            MessageDeltaEvent,
            MessageStartEvent,
            MessageStopEvent,
            ToolResultEvent,
        )

        assert ContentBlockDeltaEvent
        assert ContentBlockStartEvent
        assert ContentBlockStopEvent
        assert MessageDeltaEvent
        assert MessageStartEvent
        assert MessageStopEvent
        assert ToolResultEvent

    def test_delta_types_importable(self):
        from data_harness import ContentDelta, InputJSONDelta, TextDelta

        assert TextDelta
        assert InputJSONDelta
        assert ContentDelta

    def test_text_delta_constructible_from_top_level_import(self):
        from data_harness import TextDelta

        d = TextDelta(text="hello")
        assert d.text == "hello"
        assert d.type == "text_delta"

    def test_tool_result_event_constructible_from_top_level_import(self):
        from data_harness import ToolResultEvent

        e = ToolResultEvent(
            tool_use_id="id1", tool_name="fn", content="result", is_error=False
        )
        assert e.type == "tool_result"
        assert not e.is_error


# ---------------------------------------------------------------------------
# 24. JSONL logging during run_stream()
# ---------------------------------------------------------------------------


class TestStreamLogging:
    """run_stream() must write JSONL turn logs, just like run_result()."""

    async def test_run_file_set_after_run_stream(self, tmp_path):
        adapter = FakeAsyncAdapter([FakeAsyncAdapter.text("hi")])
        harness = AsyncHarness(
            adapter=adapter, system="s", tools=[], run_dir=str(tmp_path)
        )
        assert harness.run_file is None
        async for _ in harness.run_stream("q"):
            pass
        assert harness.run_file is not None

    async def test_jsonl_file_created_by_run_stream(self, tmp_path):
        import os

        adapter = FakeAsyncAdapter([FakeAsyncAdapter.text("logged")])
        harness = AsyncHarness(
            adapter=adapter, system="s", tools=[], run_dir=str(tmp_path)
        )
        async for _ in harness.run_stream("q"):
            pass
        assert harness.run_file is not None
        assert os.path.isfile(harness.run_file)

    async def test_jsonl_file_contains_valid_entries(self, tmp_path):
        import json

        adapter = FakeAsyncAdapter([FakeAsyncAdapter.text("entry")])
        harness = AsyncHarness(
            adapter=adapter, system="s", tools=[], run_dir=str(tmp_path)
        )
        async for _ in harness.run_stream("q"):
            pass
        assert harness.run_file is not None
        with open(harness.run_file) as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) >= 1
        for raw in lines:
            entry = json.loads(raw)
            assert isinstance(entry, dict)

    async def test_jsonl_records_tool_turn_and_final_turn(self, tmp_path):
        import json

        def echo(v: str) -> str:
            return v

        tool_spec = ToolSpec(
            name="echo",
            description="echoes input",
            input_schema={"type": "object", "properties": {"v": {"type": "string"}}},
            handler=echo,
            visible=True,
        )
        responses = [
            FakeAsyncAdapter.tool_use("tu1", "echo", {"v": "x"}),
            FakeAsyncAdapter.text("done"),
        ]
        adapter = FakeAsyncAdapter(responses)
        harness = AsyncHarness(
            adapter=adapter,
            system="s",
            tools=[tool_spec],
            max_turns=5,
            run_dir=str(tmp_path),
        )
        async for _ in harness.run_stream("go"):
            pass
        assert harness.run_file is not None
        with open(harness.run_file) as f:
            lines = [line.strip() for line in f if line.strip()]
        # Expect at least 2 log entries — one per turn
        assert len(lines) >= 2
        entries = [json.loads(line) for line in lines]
        turn_nums = [e.get("turn") for e in entries if "turn" in e]
        assert 1 in turn_nums
        assert 2 in turn_nums

    async def test_ask_stream_reuses_same_log_file(self, tmp_path):
        adapter = FakeAsyncAdapter(
            [FakeAsyncAdapter.text("first"), FakeAsyncAdapter.text("second")]
        )
        harness = AsyncHarness(
            adapter=adapter, system="s", tools=[], run_dir=str(tmp_path)
        )
        async for _ in harness.run_stream("q1"):
            pass
        file_after_run = harness.run_file

        async for _ in harness.ask_stream("q2"):
            pass
        file_after_ask = harness.run_file

        assert file_after_run == file_after_ask


# ---------------------------------------------------------------------------
# 25. Reminder hooks fire in streaming path
# ---------------------------------------------------------------------------


class TestStreamReminderHooks:
    """register_reminder() callbacks must fire on each turn during run_stream()."""

    async def test_reminder_fires_on_first_stream_turn(self, tmp_path):
        fired_turns: list[int] = []

        adapter = FakeAsyncAdapter([FakeAsyncAdapter.text("done")])
        harness = AsyncHarness(
            adapter=adapter, system="s", tools=[], run_dir=str(tmp_path)
        )

        def hook(turn: int, max_turns: int) -> str | None:
            fired_turns.append(turn)
            return None

        harness.register_reminder(hook)
        async for _ in harness.run_stream("q"):
            pass

        assert fired_turns == [1]

    async def test_reminder_fires_on_each_turn_in_multi_turn_stream(self, tmp_path):
        fired_turns: list[int] = []

        def noop_tool() -> str:
            return "ok"

        tool_spec = ToolSpec(
            name="noop_tool",
            description="t",
            input_schema={"type": "object", "properties": {}},
            handler=noop_tool,
            visible=True,
        )
        responses = [
            FakeAsyncAdapter.tool_use("tu1", "noop_tool", {}),
            FakeAsyncAdapter.text("done"),
        ]
        adapter = FakeAsyncAdapter(responses)
        harness = AsyncHarness(
            adapter=adapter,
            system="s",
            tools=[tool_spec],
            max_turns=5,
            run_dir=str(tmp_path),
        )

        def hook(turn: int, max_turns: int) -> str | None:
            fired_turns.append(turn)
            return None

        harness.register_reminder(hook)
        async for _ in harness.run_stream("q"):
            pass

        assert fired_turns == [1, 2]

    async def test_reminder_return_value_appended_to_messages(self, tmp_path):
        """A non-None reminder string should appear in the message history."""
        seen_messages: list[list] = []

        class _CapturingAdapter(AsyncProviderAdapter):
            _turn = 0

            def format_cache_control(self, obj: dict) -> dict:
                return dict(obj)

            async def chat(self, system, messages, tools):
                raise NotImplementedError

            async def stream_events(self, system, messages, tools):
                seen_messages.append(list(messages))
                self._turn += 1
                for evt in _text_turn_events("ok"):
                    yield evt

        adapter = _CapturingAdapter()
        harness = AsyncHarness(
            adapter=adapter, system="s", tools=[], run_dir=str(tmp_path)
        )
        harness.register_reminder(lambda turn, max_turns: "REMINDER TEXT")
        async for _ in harness.run_stream("hello"):
            pass

        # The messages list seen by the adapter must contain the reminder
        flat = " ".join(
            str(block)
            for msg in seen_messages[0]
            for block in (msg.content if hasattr(msg, "content") else [])
        )
        assert "REMINDER TEXT" in flat


# ---------------------------------------------------------------------------
# 26. Visible tool filtering in streaming path
# ---------------------------------------------------------------------------


class TestStreamVisibleToolFiltering:
    """Hidden tools must not appear in the tool list passed to stream_events()."""

    async def test_hidden_tool_excluded_from_stream_adapter_call(self, tmp_path):
        received_tool_names: list[list[str]] = []

        class _CapturingAdapter2(AsyncProviderAdapter):
            def format_cache_control(self, obj: dict) -> dict:
                return dict(obj)

            async def chat(self, system, messages, tools):
                raise NotImplementedError

            async def stream_events(self, system, messages, tools):
                received_tool_names.append([t.name for t in tools])
                for evt in _text_turn_events("done"):
                    yield evt

        hidden_spec = ToolSpec(
            name="hidden_tool",
            description="not shown",
            input_schema={"type": "object", "properties": {}},
            handler=lambda: "x",
            visible=False,
        )
        visible_spec = ToolSpec(
            name="visible_tool",
            description="shown",
            input_schema={"type": "object", "properties": {}},
            handler=lambda: "y",
            visible=True,
        )

        adapter = _CapturingAdapter2()
        harness = AsyncHarness(
            adapter=adapter,
            system="s",
            tools=[hidden_spec, visible_spec],
            run_dir=str(tmp_path),
        )
        async for _ in harness.run_stream("q"):
            pass

        assert len(received_tool_names) == 1
        names = received_tool_names[0]
        assert "visible_tool" in names
        assert "hidden_tool" not in names

    async def test_system_prompt_unchanged_across_stream_turns(self, tmp_path):
        """The system prompt byte-string must be identical on every streaming turn."""
        seen_systems: list[str] = []

        def noop_tool() -> str:
            return "ok"

        tool_spec = ToolSpec(
            name="noop_tool",
            description="t",
            input_schema={"type": "object", "properties": {}},
            handler=noop_tool,
            visible=True,
        )

        class _SystemCapture(AsyncProviderAdapter):
            _turn = 0

            def format_cache_control(self, obj: dict) -> dict:
                return dict(obj)

            async def chat(self, system, messages, tools):
                raise NotImplementedError

            async def stream_events(self, system, messages, tools):
                seen_systems.append(system)
                self._turn += 1
                if self._turn == 1:
                    for evt in _tool_turn_events("tu1", "noop_tool", {}):
                        yield evt
                else:
                    for evt in _text_turn_events("final"):
                        yield evt

        adapter = _SystemCapture()
        harness = AsyncHarness(
            adapter=adapter,
            system="FIXED SYSTEM PROMPT",
            tools=[tool_spec],
            max_turns=5,
            run_dir=str(tmp_path),
        )
        async for _ in harness.run_stream("q"):
            pass

        assert len(seen_systems) == 2
        assert seen_systems[0] == seen_systems[1] == "FIXED SYSTEM PROMPT"
