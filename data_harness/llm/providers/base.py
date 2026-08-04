from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from data_harness.llm.types import (
    ContentBlock,
    Message,
    TextBlock,
    ToolSpec,
    ToolUseBlock,
)

if TYPE_CHECKING:
    from data_harness.llm.streaming import StreamEvent


class StopReason(Enum):
    """Why the provider ended the current generation turn.

    Attributes:
        END_TURN: The model produced a complete response with no tool calls.
        TOOL_USE: The model emitted one or more tool-use blocks.
        MAX_TOKENS: The response was truncated at the token limit.
        STOP_SEQUENCE: A stop sequence in the prompt was matched.
    """

    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"


@dataclass
class NormalizedResponse:
    """Provider-normalised response from a single chat call.

    Adapters translate provider-specific response objects into this type so
    that the harness never touches provider SDK classes directly.

    Attributes:
        stop_reason: Why generation stopped.
        content: Ordered list of `TextBlock` and `ToolUseBlock` items.
        input_tokens: Prompt tokens billed by the provider.
        output_tokens: Completion tokens billed by the provider.
        cache_read_tokens: Tokens served from the provider's prompt cache.
        cache_write_tokens: Tokens written to the provider's prompt cache.
    """

    stop_reason: StopReason
    content: list[ContentBlock]
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int


class ProviderAdapter(ABC):
    """Synchronous provider adapter interface.

    Implement `chat` and `format_cache_control` to integrate a new model
    provider. The harness calls `chat` once per turn and never touches any
    provider SDK objects directly.
    """

    @abstractmethod
    def chat(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> NormalizedResponse:
        """Send one turn to the provider and return a normalised response.

        Args:
            system: The system prompt (must be prefix-stable across turns).
            messages: Full conversation history up to and including the latest
                user message.
            tools: Only the currently visible `ToolSpec` instances.

        Returns:
            A `NormalizedResponse` with token counts and assembled content.
        """
        ...

    @abstractmethod
    def format_cache_control(self, obj: dict) -> dict:
        """Attach provider-specific cache-control metadata to a content object."""
        ...


class AsyncProviderAdapter(ABC):
    """Asynchronous provider adapter with optional token-level streaming.

    Implement `chat` and `format_cache_control`.  Override `stream_events` to
    emit real token-level `StreamEvent` objects; the default implementation
    synthesises events from the assembled `chat` response.
    """

    @abstractmethod
    async def chat(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> NormalizedResponse:
        """Send one turn to the provider and return a normalised response.

        Args:
            system: The system prompt (must be prefix-stable across turns).
            messages: Full conversation history up to and including the latest
                user message.
            tools: Only the currently visible `ToolSpec` instances.

        Returns:
            A `NormalizedResponse` with token counts and assembled content.
        """
        ...

    @abstractmethod
    def format_cache_control(self, obj: dict) -> dict: ...

    async def stream_events(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> AsyncGenerator[StreamEvent, None]:
        """Yield stream events for one provider turn.

        The default implementation calls chat() and synthesises the six
        standard event types from the assembled response.  Override in
        provider subclasses to emit real token-level events.
        """
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

        response = await self.chat(system, messages, tools)
        yield MessageStartEvent()
        for i, block in enumerate(response.content):
            if isinstance(block, TextBlock):
                yield ContentBlockStartEvent(index=i, content_block=TextBlock(text=""))
                yield ContentBlockDeltaEvent(index=i, delta=TextDelta(text=block.text))
                yield ContentBlockStopEvent(index=i)
            elif isinstance(block, ToolUseBlock):
                yield ContentBlockStartEvent(
                    index=i,
                    content_block=ToolUseBlock(
                        tool_use_id=block.tool_use_id,
                        tool_name=block.tool_name,
                        tool_input={},
                    ),
                )
                yield ContentBlockDeltaEvent(
                    index=i,
                    delta=InputJSONDelta(partial_json=json.dumps(block.tool_input)),
                )
                yield ContentBlockStopEvent(index=i)
        yield MessageDeltaEvent(
            stop_reason=response.stop_reason,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cache_read_tokens=response.cache_read_tokens,
            cache_write_tokens=response.cache_write_tokens,
        )
        yield MessageStopEvent()
