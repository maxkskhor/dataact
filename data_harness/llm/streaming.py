"""Stream event types for AsyncHarness.run_stream() and AsyncAgent.run_stream().

The event protocol mirrors the Claude Agent SDK's raw SSE event shape so that
callers get the same discriminated-union stream whether they use data-harness or
the Anthropic SDK directly.  An additional ToolResultEvent is emitted after the
harness dispatches a tool call — it has no equivalent in the raw provider stream.

Typical iteration pattern::

    async for event in agent.run_stream("..."):
        match event.type:
            case "content_block_delta":
                if isinstance(event.delta, TextDelta):
                    print(event.delta.text, end="", flush=True)
            case "tool_result":
                print(f"[{event.tool_name}] → {event.content[:80]}")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal

from data_harness.llm.providers.base import NormalizedResponse, StopReason
from data_harness.llm.types import TextBlock, ToolUseBlock

# ---------------------------------------------------------------------------
# Delta types (carried by ContentBlockDeltaEvent)
# ---------------------------------------------------------------------------


@dataclass
class TextDelta:
    text: str
    type: Literal["text_delta"] = field(default="text_delta", init=False)


@dataclass
class InputJSONDelta:
    partial_json: str
    type: Literal["input_json_delta"] = field(default="input_json_delta", init=False)


ContentDelta = TextDelta | InputJSONDelta


# ---------------------------------------------------------------------------
# Stream event types (same discriminator strings as the Anthropic SDK)
# ---------------------------------------------------------------------------


@dataclass
class MessageStartEvent:
    type: Literal["message_start"] = field(default="message_start", init=False)


@dataclass
class ContentBlockStartEvent:
    index: int
    content_block: TextBlock | ToolUseBlock
    type: Literal["content_block_start"] = field(
        default="content_block_start", init=False
    )


@dataclass
class ContentBlockDeltaEvent:
    index: int
    delta: TextDelta | InputJSONDelta
    type: Literal["content_block_delta"] = field(
        default="content_block_delta", init=False
    )


@dataclass
class ContentBlockStopEvent:
    index: int
    type: Literal["content_block_stop"] = field(
        default="content_block_stop", init=False
    )


@dataclass
class MessageDeltaEvent:
    stop_reason: StopReason
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    type: Literal["message_delta"] = field(default="message_delta", init=False)


@dataclass
class MessageStopEvent:
    type: Literal["message_stop"] = field(default="message_stop", init=False)


@dataclass
class ToolResultEvent:
    """Emitted by the harness after a tool call is dispatched and returns.

    This event has no equivalent in the raw provider stream.  It signals that
    the harness has finished executing the tool and the next turn is starting.
    """

    tool_use_id: str
    tool_name: str
    content: str
    is_error: bool
    type: Literal["tool_result"] = field(default="tool_result", init=False)


StreamEvent = (
    MessageStartEvent
    | ContentBlockStartEvent
    | ContentBlockDeltaEvent
    | ContentBlockStopEvent
    | MessageDeltaEvent
    | MessageStopEvent
    | ToolResultEvent
)


# ---------------------------------------------------------------------------
# Helper: reconstruct a NormalizedResponse from accumulated stream events
# ---------------------------------------------------------------------------


def accumulate_stream_events(events: list[StreamEvent]) -> NormalizedResponse:
    """Build a NormalizedResponse by replaying a list of StreamEvents."""
    block_by_index: dict[int, TextBlock | ToolUseBlock] = {}
    json_by_index: dict[int, str] = {}
    stop_reason = StopReason.END_TURN
    input_tokens = output_tokens = cache_read = cache_write = 0

    for evt in events:
        if isinstance(evt, ContentBlockStartEvent):
            if isinstance(evt.content_block, TextBlock):
                block_by_index[evt.index] = TextBlock(text="")
            elif isinstance(evt.content_block, ToolUseBlock):
                cb = evt.content_block
                block_by_index[evt.index] = ToolUseBlock(
                    tool_use_id=cb.tool_use_id,
                    tool_name=cb.tool_name,
                    tool_input={},
                )
                json_by_index[evt.index] = ""

        elif isinstance(evt, ContentBlockDeltaEvent):
            if isinstance(evt.delta, TextDelta):
                b = block_by_index.get(evt.index)
                if isinstance(b, TextBlock):
                    b.text += evt.delta.text
            elif isinstance(evt.delta, InputJSONDelta):
                json_by_index[evt.index] = (
                    json_by_index.get(evt.index, "") + evt.delta.partial_json
                )

        elif isinstance(evt, ContentBlockStopEvent):
            if evt.index in json_by_index:
                b = block_by_index.get(evt.index)
                if isinstance(b, ToolUseBlock):
                    raw = json_by_index[evt.index]
                    try:
                        b.tool_input = json.loads(raw) if raw else {}
                    except json.JSONDecodeError:
                        b.tool_input = {}

        elif isinstance(evt, MessageDeltaEvent):
            stop_reason = evt.stop_reason
            input_tokens = evt.input_tokens
            output_tokens = evt.output_tokens
            cache_read = evt.cache_read_tokens
            cache_write = evt.cache_write_tokens

    content = [block_by_index[i] for i in sorted(block_by_index.keys())]

    return NormalizedResponse(
        stop_reason=stop_reason,
        content=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
    )
