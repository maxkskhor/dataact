from __future__ import annotations

import copy
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

import anthropic

from data_harness.llm.providers.base import (
    AsyncProviderAdapter,
    NormalizedResponse,
    ProviderAdapter,
    StopReason,
)
from data_harness.llm.types import (
    Message,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)

if TYPE_CHECKING:
    from data_harness.llm.streaming import StreamEvent

_STOP_REASON_MAP = {
    "end_turn": StopReason.END_TURN,
    "tool_use": StopReason.TOOL_USE,
    "max_tokens": StopReason.MAX_TOKENS,
    "stop_sequence": StopReason.STOP_SEQUENCE,
}


class _AnthropicHelpers:
    """Shared message-building and response-normalisation logic."""

    _model: str
    _max_tokens: int

    def format_cache_control(self, obj: dict) -> dict:
        result = copy.copy(obj)
        result["cache_control"] = {"type": "ephemeral"}
        return result

    def _build_system(self, system: str) -> list[dict]:
        return [self.format_cache_control({"type": "text", "text": system})]

    def _build_messages(self, messages: list[Message]) -> list[dict]:
        result = []
        for i, msg in enumerate(messages):
            blocks = [self._block_to_dict(b) for b in msg.content]
            if i == len(messages) - 1 and msg.role == "user" and blocks:
                blocks[-1] = self.format_cache_control(blocks[-1])
            result.append({"role": msg.role, "content": blocks})
        return result

    def _block_to_dict(self, block) -> dict:
        if isinstance(block, TextBlock):
            return {"type": "text", "text": block.text}
        if isinstance(block, ToolUseBlock):
            return {
                "type": "tool_use",
                "id": block.tool_use_id,
                "name": block.tool_name,
                "input": block.tool_input,
            }
        if isinstance(block, ToolResultBlock):
            return {
                "type": "tool_result",
                "tool_use_id": block.tool_use_id,
                "content": block.content,
                "is_error": block.is_error,
            }
        raise ValueError(f"Unknown block type: {type(block)}")

    def _build_tools(self, tools: list[ToolSpec]) -> list[dict]:
        return [t.to_provider_dict() for t in tools]

    def _normalize_content(self, content) -> list:
        blocks = []
        for block in content:
            if block.type == "text":
                blocks.append(TextBlock(text=block.text))
            elif block.type == "tool_use":
                blocks.append(
                    ToolUseBlock(
                        tool_use_id=block.id,
                        tool_name=block.name,
                        tool_input=dict(block.input),
                    )
                )
        return blocks

    def _normalize_response(self, resp) -> NormalizedResponse:
        stop_reason = _STOP_REASON_MAP.get(resp.stop_reason, StopReason.END_TURN)
        content = self._normalize_content(resp.content)
        return NormalizedResponse(
            stop_reason=stop_reason,
            content=content,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            cache_read_tokens=getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(resp.usage, "cache_creation_input_tokens", 0)
            or 0,
        )


class AnthropicAdapter(_AnthropicHelpers, ProviderAdapter):
    def __init__(
        self, model: str = "claude-sonnet-4-6", max_tokens: int = 8096
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._client = anthropic.Anthropic()

    def chat(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> NormalizedResponse:
        api_system = self._build_system(system)
        api_messages = self._build_messages(messages)
        api_tools = self._build_tools(tools)

        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=api_system,
            messages=api_messages,
            tools=api_tools or anthropic.NOT_GIVEN,
        )
        return self._normalize_response(resp)


class AsyncAnthropicAdapter(_AnthropicHelpers, AsyncProviderAdapter):
    def __init__(
        self, model: str = "claude-sonnet-4-6", max_tokens: int = 8096
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._client = anthropic.AsyncAnthropic()

    async def chat(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> NormalizedResponse:
        api_system = self._build_system(system)
        api_messages = self._build_messages(messages)
        api_tools = self._build_tools(tools)

        resp = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=api_system,
            messages=api_messages,
            tools=api_tools or anthropic.NOT_GIVEN,
        )
        return self._normalize_response(resp)

    async def stream_events(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> AsyncGenerator[StreamEvent, None]:
        """Yield StreamEvents by mapping raw Anthropic SSE events."""
        from data_harness.llm.streaming import (
            ContentBlockDeltaEvent,
            ContentBlockStartEvent,
            ContentBlockStopEvent,
            InputJSONDelta,
            MessageDeltaEvent,
            MessageStartEvent,
            MessageStopEvent,
            TextDelta,
        )

        api_system = self._build_system(system)
        api_messages = self._build_messages(messages)
        api_tools = self._build_tools(tools)

        async with self._client.messages.stream(
            model=self._model,
            max_tokens=self._max_tokens,
            system=api_system,
            messages=api_messages,
            tools=api_tools or anthropic.NOT_GIVEN,
        ) as stream:
            async for event in stream:
                if event.type == "message_start":
                    yield MessageStartEvent()

                elif event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "text":
                        yield ContentBlockStartEvent(
                            index=event.index,
                            content_block=TextBlock(text=""),
                        )
                    elif block.type == "tool_use":
                        yield ContentBlockStartEvent(
                            index=event.index,
                            content_block=ToolUseBlock(
                                tool_use_id=block.id,
                                tool_name=block.name,
                                tool_input={},
                            ),
                        )

                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        yield ContentBlockDeltaEvent(
                            index=event.index,
                            delta=TextDelta(text=delta.text),
                        )
                    elif delta.type == "input_json_delta":
                        yield ContentBlockDeltaEvent(
                            index=event.index,
                            delta=InputJSONDelta(partial_json=delta.partial_json),
                        )

                elif event.type == "content_block_stop":
                    yield ContentBlockStopEvent(index=event.index)

                elif event.type == "message_delta":
                    stop_reason = _STOP_REASON_MAP.get(
                        event.delta.stop_reason, StopReason.END_TURN
                    )
                    yield MessageDeltaEvent(
                        stop_reason=stop_reason,
                        input_tokens=event.usage.input_tokens,
                        output_tokens=event.usage.output_tokens,
                        cache_read_tokens=getattr(
                            event.usage, "cache_read_input_tokens", 0
                        )
                        or 0,
                        cache_write_tokens=getattr(
                            event.usage, "cache_creation_input_tokens", 0
                        )
                        or 0,
                    )

                elif event.type == "message_stop":
                    yield MessageStopEvent()
