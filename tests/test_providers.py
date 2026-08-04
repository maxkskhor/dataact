import copy
import json
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

from data_harness.llm.providers.base import StopReason
from data_harness.llm.streaming import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    InputJSONDelta,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
    TextDelta,
)
from data_harness.llm.types import Message, TextBlock, ToolSpec, ToolUseBlock


def make_anthropic_response(stop_reason="end_turn", content_blocks=None, usage=None):
    """Build a mock Anthropic SDK response object."""
    mock_resp = MagicMock()
    mock_resp.stop_reason = stop_reason
    mock_resp.content = content_blocks or []
    mock_resp.usage = MagicMock()
    if usage:
        mock_resp.usage.input_tokens = usage.get("input_tokens", 10)
        mock_resp.usage.output_tokens = usage.get("output_tokens", 5)
        mock_resp.usage.cache_read_input_tokens = usage.get("cache_read_tokens", 0)
        mock_resp.usage.cache_creation_input_tokens = usage.get("cache_write_tokens", 0)
    else:
        mock_resp.usage.input_tokens = 10
        mock_resp.usage.output_tokens = 5
        mock_resp.usage.cache_read_input_tokens = 0
        mock_resp.usage.cache_creation_input_tokens = 0
    return mock_resp


def make_sdk_text_block(text):
    b = MagicMock()
    b.type = "text"
    b.text = text
    return b


def make_sdk_tool_use_block(id_, name, input_):
    b = MagicMock()
    b.type = "tool_use"
    b.id = id_
    b.name = name
    b.input = input_
    return b


class TestStopReason:
    def test_values(self):
        assert StopReason.END_TURN.value == "end_turn"
        assert StopReason.TOOL_USE.value == "tool_use"
        assert StopReason.MAX_TOKENS.value == "max_tokens"
        assert StopReason.STOP_SEQUENCE.value == "stop_sequence"


class TestAnthropicAdapter:
    def _make_adapter(self):
        with patch("anthropic.Anthropic"):
            from data_harness.llm.providers.anthropic import AnthropicAdapter

            adapter = AnthropicAdapter(model="claude-3-5-sonnet-20241022")
        return adapter

    def test_format_cache_control_returns_copy(self):
        adapter = self._make_adapter()
        original = {"type": "text", "text": "hello"}
        result = adapter.format_cache_control(original)
        assert result is not original
        assert result["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in original

    def test_format_cache_control_has_all_original_fields(self):
        adapter = self._make_adapter()
        original = {"key": "value", "num": 42}
        result = adapter.format_cache_control(original)
        assert result["key"] == "value"
        assert result["num"] == 42

    def test_chat_end_turn(self):
        adapter = self._make_adapter()
        sdk_resp = make_anthropic_response(
            stop_reason="end_turn",
            content_blocks=[make_sdk_text_block("Hello!")],
            usage={
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
            },
        )
        adapter._client.messages.create.return_value = sdk_resp

        msgs = [Message(role="user", content=[TextBlock(text="Hi")])]
        tools = []
        resp = adapter.chat(system="sys", messages=msgs, tools=tools)

        assert resp.stop_reason == StopReason.END_TURN
        assert len(resp.content) == 1
        assert isinstance(resp.content[0], TextBlock)
        assert resp.content[0].text == "Hello!"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 5

    def test_chat_tool_use(self):
        adapter = self._make_adapter()
        sdk_resp = make_anthropic_response(
            stop_reason="tool_use",
            content_blocks=[make_sdk_tool_use_block("tu_1", "my_tool", {"x": 1})],
        )
        adapter._client.messages.create.return_value = sdk_resp

        msgs = [Message(role="user", content=[TextBlock(text="run tool")])]
        resp = adapter.chat(system="sys", messages=msgs, tools=[])

        assert resp.stop_reason == StopReason.TOOL_USE
        assert isinstance(resp.content[0], ToolUseBlock)
        assert resp.content[0].tool_use_id == "tu_1"
        assert resp.content[0].tool_name == "my_tool"
        assert resp.content[0].tool_input == {"x": 1}

    def test_chat_max_tokens(self):
        adapter = self._make_adapter()
        sdk_resp = make_anthropic_response(stop_reason="max_tokens")
        adapter._client.messages.create.return_value = sdk_resp
        resp = adapter.chat(system="s", messages=[], tools=[])
        assert resp.stop_reason == StopReason.MAX_TOKENS

    def test_chat_stop_sequence(self):
        adapter = self._make_adapter()
        sdk_resp = make_anthropic_response(stop_reason="stop_sequence")
        adapter._client.messages.create.return_value = sdk_resp
        resp = adapter.chat(system="s", messages=[], tools=[])
        assert resp.stop_reason == StopReason.STOP_SEQUENCE

    def test_cache_read_write_tokens(self):
        adapter = self._make_adapter()
        sdk_resp = make_anthropic_response(
            usage={
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_tokens": 200,
                "cache_write_tokens": 300,
            },
        )
        adapter._client.messages.create.return_value = sdk_resp
        resp = adapter.chat(system="s", messages=[], tools=[])
        assert resp.cache_read_tokens == 200
        assert resp.cache_write_tokens == 300

    def test_adapter_input_immutability(self):
        adapter = self._make_adapter()
        sdk_resp = make_anthropic_response()
        adapter._client.messages.create.return_value = sdk_resp

        system = "The system prompt."
        messages = [
            Message(role="user", content=[TextBlock(text="hello")]),
            Message(role="assistant", content=[TextBlock(text="hi")]),
        ]
        tools = [
            ToolSpec(
                name="t", description="d", input_schema={"type": "object"}, handler=None
            )
        ]

        system_before = system
        messages_before = copy.deepcopy(messages)
        tools_before = copy.deepcopy(tools)

        adapter.chat(system=system, messages=messages, tools=tools)

        assert system == system_before
        assert len(messages) == len(messages_before)
        for m_orig, m_after in zip(messages_before, messages):
            assert m_orig.role == m_after.role
            assert len(m_orig.content) == len(m_after.content)
        assert len(tools) == len(tools_before)
        assert tools[0].name == tools_before[0].name

    def test_cache_control_only_in_adapter_payload(self):
        adapter = self._make_adapter()
        sdk_resp = make_anthropic_response()
        adapter._client.messages.create.return_value = sdk_resp

        messages = [
            Message(role="user", content=[TextBlock(text="hello")]),
            Message(role="assistant", content=[TextBlock(text="hi")]),
            Message(role="user", content=[TextBlock(text="again")]),
        ]
        tools = [ToolSpec(name="t", description="d", input_schema={}, handler=None)]

        adapter.chat(system="sys", messages=messages, tools=tools)

        call_kwargs = adapter._client.messages.create.call_args
        # The adapter call should have cache_control in system and last user message
        system_arg = call_kwargs.kwargs.get(
            "system", call_kwargs.args[0] if call_kwargs.args else None
        )
        # system might be a list with cache_control or a string
        if isinstance(system_arg, list):
            assert any("cache_control" in str(s) for s in system_arg)

        # Harness objects must not have cache_control
        for m in messages:
            for block in m.content:
                assert not hasattr(block, "cache_control")


# ---------------------------------------------------------------------------
# AsyncAnthropicAdapter.stream_events() — maps raw SSE events to StreamEvent
# ---------------------------------------------------------------------------


def _make_sse_event(type_: str, **kwargs) -> MagicMock:
    """Build a mock raw SSE event with a given type and attributes."""
    ev = MagicMock()
    ev.type = type_
    for k, v in kwargs.items():
        setattr(ev, k, v)
    return ev


def _make_text_sse_sequence(text: str) -> list[MagicMock]:
    """Minimal SSE event sequence for a single text block."""
    cb = MagicMock()
    cb.type = "text"

    block = MagicMock()
    block.index = 0
    block.content_block = cb

    delta_obj = MagicMock()
    delta_obj.type = "text_delta"
    delta_obj.text = text

    delta_event = MagicMock()
    delta_event.type = "content_block_delta"
    delta_event.index = 0
    delta_event.delta = delta_obj

    stop_event = MagicMock()
    stop_event.type = "content_block_stop"
    stop_event.index = 0

    msg_delta_inner = MagicMock()
    msg_delta_inner.stop_reason = "end_turn"

    usage_obj = MagicMock()
    usage_obj.input_tokens = 10
    usage_obj.output_tokens = 5
    usage_obj.cache_read_input_tokens = 0
    usage_obj.cache_creation_input_tokens = 0

    msg_delta = MagicMock()
    msg_delta.type = "message_delta"
    msg_delta.delta = msg_delta_inner
    msg_delta.usage = usage_obj

    msg_stop = MagicMock()
    msg_stop.type = "message_stop"

    msg_start = MagicMock()
    msg_start.type = "message_start"

    start = MagicMock()
    start.type = "content_block_start"
    start.index = 0
    start.content_block = cb

    return [msg_start, start, delta_event, stop_event, msg_delta, msg_stop]


def _make_tool_sse_sequence(
    tool_use_id: str, tool_name: str, tool_input: dict
) -> list[MagicMock]:
    """Minimal SSE event sequence for a tool_use content block."""
    msg_start = MagicMock()
    msg_start.type = "message_start"

    cb = MagicMock()
    cb.type = "tool_use"
    cb.id = tool_use_id
    cb.name = tool_name

    start = MagicMock()
    start.type = "content_block_start"
    start.index = 0
    start.content_block = cb

    delta_obj = MagicMock()
    delta_obj.type = "input_json_delta"
    delta_obj.partial_json = json.dumps(tool_input)

    delta_event = MagicMock()
    delta_event.type = "content_block_delta"
    delta_event.index = 0
    delta_event.delta = delta_obj

    stop_event = MagicMock()
    stop_event.type = "content_block_stop"
    stop_event.index = 0

    msg_delta_inner = MagicMock()
    msg_delta_inner.stop_reason = "tool_use"

    usage_obj = MagicMock()
    usage_obj.input_tokens = 20
    usage_obj.output_tokens = 8
    usage_obj.cache_read_input_tokens = 0
    usage_obj.cache_creation_input_tokens = 0

    msg_delta = MagicMock()
    msg_delta.type = "message_delta"
    msg_delta.delta = msg_delta_inner
    msg_delta.usage = usage_obj

    msg_stop = MagicMock()
    msg_stop.type = "message_stop"

    return [msg_start, start, delta_event, stop_event, msg_delta, msg_stop]


class TestAsyncAnthropicAdapterStreamEvents:
    def _make_async_adapter(self):
        with patch("anthropic.AsyncAnthropic"):
            from data_harness.llm.providers.anthropic import AsyncAnthropicAdapter

            return AsyncAnthropicAdapter(model="claude-sonnet-4-6")

    def _patch_stream(self, adapter, sse_events: list[MagicMock]) -> None:
        """Patch adapter._client.messages.stream to yield sse_events."""

        async def _aiter(self_inner):
            for e in sse_events:
                yield e

        mock_stream = MagicMock()
        mock_stream.__aiter__ = _aiter

        @asynccontextmanager
        async def _ctx(*args, **kwargs):
            yield mock_stream

        adapter._client.messages.stream = _ctx

    async def test_text_stream_events_types(self):
        adapter = self._make_async_adapter()
        self._patch_stream(adapter, _make_text_sse_sequence("hi"))

        events = []
        async for e in adapter.stream_events("sys", [], []):
            events.append(e)

        types = [e.type for e in events]
        assert types == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]

    async def test_text_stream_carries_text_in_delta(self):
        adapter = self._make_async_adapter()
        self._patch_stream(adapter, _make_text_sse_sequence("hello streaming"))

        events = []
        async for e in adapter.stream_events("sys", [], []):
            events.append(e)

        text_deltas = [
            e
            for e in events
            if isinstance(e, ContentBlockDeltaEvent) and isinstance(e.delta, TextDelta)
        ]
        assert len(text_deltas) == 1
        assert text_deltas[0].delta.text == "hello streaming"

    async def test_text_stream_message_start(self):
        adapter = self._make_async_adapter()
        self._patch_stream(adapter, _make_text_sse_sequence("x"))

        events = []
        async for e in adapter.stream_events("sys", [], []):
            events.append(e)

        assert isinstance(events[0], MessageStartEvent)

    async def test_text_stream_message_stop(self):
        adapter = self._make_async_adapter()
        self._patch_stream(adapter, _make_text_sse_sequence("x"))

        events = []
        async for e in adapter.stream_events("sys", [], []):
            events.append(e)

        assert isinstance(events[-1], MessageStopEvent)

    async def test_text_stream_message_delta_stop_reason(self):
        adapter = self._make_async_adapter()
        self._patch_stream(adapter, _make_text_sse_sequence("x"))

        events = []
        async for e in adapter.stream_events("sys", [], []):
            events.append(e)

        deltas = [e for e in events if isinstance(e, MessageDeltaEvent)]
        assert len(deltas) == 1
        assert deltas[0].stop_reason == StopReason.END_TURN

    async def test_text_stream_usage(self):
        adapter = self._make_async_adapter()
        self._patch_stream(adapter, _make_text_sse_sequence("x"))

        events = []
        async for e in adapter.stream_events("sys", [], []):
            events.append(e)

        deltas = [e for e in events if isinstance(e, MessageDeltaEvent)]
        assert deltas[0].input_tokens == 10
        assert deltas[0].output_tokens == 5

    async def test_tool_use_stream_content_block_start(self):
        adapter = self._make_async_adapter()
        self._patch_stream(adapter, _make_tool_sse_sequence("tu1", "my_fn", {"k": "v"}))

        events = []
        async for e in adapter.stream_events("sys", [], []):
            events.append(e)

        starts = [e for e in events if isinstance(e, ContentBlockStartEvent)]
        assert len(starts) == 1
        from data_harness.llm.types import ToolUseBlock

        assert isinstance(starts[0].content_block, ToolUseBlock)
        assert starts[0].content_block.tool_use_id == "tu1"
        assert starts[0].content_block.tool_name == "my_fn"

    async def test_tool_use_stream_json_delta(self):
        adapter = self._make_async_adapter()
        self._patch_stream(adapter, _make_tool_sse_sequence("tu1", "fn", {"x": 99}))

        events = []
        async for e in adapter.stream_events("sys", [], []):
            events.append(e)

        json_deltas = [
            e
            for e in events
            if isinstance(e, ContentBlockDeltaEvent)
            and isinstance(e.delta, InputJSONDelta)
        ]
        assert len(json_deltas) == 1
        assert json.loads(json_deltas[0].delta.partial_json) == {"x": 99}

    async def test_tool_use_stream_stop_reason_tool_use(self):
        adapter = self._make_async_adapter()
        self._patch_stream(adapter, _make_tool_sse_sequence("tu1", "fn", {}))

        events = []
        async for e in adapter.stream_events("sys", [], []):
            events.append(e)

        deltas = [e for e in events if isinstance(e, MessageDeltaEvent)]
        assert deltas[0].stop_reason == StopReason.TOOL_USE

    async def test_unknown_sse_event_type_skipped(self):
        """Unknown event types (e.g. SDK higher-level events) are silently ignored."""
        adapter = self._make_async_adapter()
        unknown = MagicMock()
        unknown.type = "totally_unknown_event_type"
        sse = _make_text_sse_sequence("x")
        sse.insert(1, unknown)
        self._patch_stream(adapter, sse)

        events = []
        async for e in adapter.stream_events("sys", [], []):
            events.append(e)

        types = [e.type for e in events]
        assert "totally_unknown_event_type" not in types
        assert "message_start" in types
        assert "message_stop" in types
