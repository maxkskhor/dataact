"""Public testing helpers.

`FakeAdapter` is a scripted `ProviderAdapter` that returns pre-built responses
in order. `FakeAsyncAdapter` is the async equivalent for `AsyncProviderAdapter`.
Both exist so that unit tests and examples can run without an API key.
"""

from __future__ import annotations

import copy
from typing import Any

from data_harness.llm.providers.base import (
    AsyncProviderAdapter,
    NormalizedResponse,
    ProviderAdapter,
    StopReason,
)
from data_harness.llm.types import Message, TextBlock, ToolSpec, ToolUseBlock


class FakeAdapter(ProviderAdapter):
    def __init__(self, responses: list[NormalizedResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> NormalizedResponse:
        self.calls.append(
            {
                "system": system,
                "messages": copy.deepcopy(messages),
                "tools": copy.deepcopy(tools),
            }
        )
        return self._responses.pop(0)

    def format_cache_control(self, obj: dict) -> dict:
        result = dict(obj)
        result["cache_control"] = {"type": "ephemeral"}
        return result

    @staticmethod
    def text(text: str) -> NormalizedResponse:
        return NormalizedResponse(
            stop_reason=StopReason.END_TURN,
            content=[TextBlock(text=text)],
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )

    @staticmethod
    def tool_use(
        tool_use_id: str, tool_name: str, tool_input: dict
    ) -> NormalizedResponse:
        return NormalizedResponse(
            stop_reason=StopReason.TOOL_USE,
            content=[
                ToolUseBlock(
                    tool_use_id=tool_use_id,
                    tool_name=tool_name,
                    tool_input=tool_input,
                )
            ],
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )


class FakeAsyncAdapter(AsyncProviderAdapter):
    """Scripted async adapter for unit tests. Uses the default `stream_events()`
    from the base class, which synthesises events from the assembled `chat()`
    response."""

    def __init__(self, responses: list[NormalizedResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> NormalizedResponse:
        self.calls.append(
            {
                "system": system,
                "messages": copy.deepcopy(messages),
                "tools": copy.deepcopy(tools),
            }
        )
        return self._responses.pop(0)

    def format_cache_control(self, obj: dict) -> dict:
        result = dict(obj)
        result["cache_control"] = {"type": "ephemeral"}
        return result

    @staticmethod
    def text(text: str) -> NormalizedResponse:
        return NormalizedResponse(
            stop_reason=StopReason.END_TURN,
            content=[TextBlock(text=text)],
            input_tokens=10,
            output_tokens=5,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )

    @staticmethod
    def tool_use(
        tool_use_id: str, tool_name: str, tool_input: dict
    ) -> NormalizedResponse:
        return NormalizedResponse(
            stop_reason=StopReason.TOOL_USE,
            content=[
                ToolUseBlock(
                    tool_use_id=tool_use_id,
                    tool_name=tool_name,
                    tool_input=tool_input,
                )
            ],
            input_tokens=10,
            output_tokens=5,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )
