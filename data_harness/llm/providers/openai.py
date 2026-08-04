from __future__ import annotations

import copy
import json
import os

import openai

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

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

_STOP_REASON_MAP = {
    "stop": StopReason.END_TURN,
    "tool_calls": StopReason.TOOL_USE,
    "length": StopReason.MAX_TOKENS,
    # TODO: expose content-filter handling separately if the core stop model grows.
    "content_filter": StopReason.END_TURN,
    "function_call": StopReason.TOOL_USE,
}


class _OpenAIHelpers:
    """Shared message-building and response-normalisation logic."""

    _model: str
    _max_tokens: int

    def format_cache_control(self, obj: dict) -> dict:
        return copy.copy(obj)

    def _build_messages(self, system: str, messages: list[Message]) -> list[dict]:
        result = [{"role": "system", "content": system}]
        for message in messages:
            text_blocks = [b for b in message.content if isinstance(b, TextBlock)]
            tool_uses = [b for b in message.content if isinstance(b, ToolUseBlock)]
            tool_results = [
                b for b in message.content if isinstance(b, ToolResultBlock)
            ]

            for block in tool_results:
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.tool_use_id,
                        "content": block.content,
                    }
                )

            if text_blocks or tool_uses:
                api_message: dict = {
                    "role": message.role,
                    "content": "\n".join(b.text for b in text_blocks)
                    if text_blocks
                    else None,
                }
                if tool_uses:
                    api_message["tool_calls"] = [
                        {
                            "id": block.tool_use_id,
                            "type": "function",
                            "function": {
                                "name": block.tool_name,
                                "arguments": json.dumps(block.tool_input),
                            },
                        }
                        for block in tool_uses
                    ]
                result.append(api_message)
        return result

    def _build_tools(self, tools: list[ToolSpec]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in tools
        ]

    def _normalize_message(self, message) -> list:
        blocks = []
        content = getattr(message, "content", None)
        if content:
            blocks.append(TextBlock(text=content))

        for call in getattr(message, "tool_calls", None) or []:
            blocks.append(
                ToolUseBlock(
                    tool_use_id=call.id,
                    tool_name=call.function.name,
                    tool_input=json.loads(call.function.arguments or "{}"),
                )
            )
        return blocks

    def _make_normalized(self, response) -> NormalizedResponse:
        choice = response.choices[0]
        stop_reason = _STOP_REASON_MAP.get(choice.finish_reason, StopReason.END_TURN)
        return NormalizedResponse(
            stop_reason=stop_reason,
            content=self._normalize_message(choice.message),
            input_tokens=getattr(response.usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(response.usage, "completion_tokens", 0) or 0,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )


class OpenAIAdapter(_OpenAIHelpers, ProviderAdapter):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_tokens: int = 4096,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key)

    def chat(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> NormalizedResponse:
        api_messages = self._build_messages(system, messages)
        api_tools = self._build_tools(tools)

        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=api_messages,
            tools=api_tools or openai.NOT_GIVEN,
        )
        return self._make_normalized(response)


class AsyncOpenAIAdapter(_OpenAIHelpers, AsyncProviderAdapter):
    """Async OpenAI adapter. Streaming falls back to a full chat call.

    OpenAI tool-call streaming requires accumulating fragmented deltas; a
    future revision can add real streaming for text-only turns.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_tokens: int = 4096,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def chat(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> NormalizedResponse:
        api_messages = self._build_messages(system, messages)
        api_tools = self._build_tools(tools)

        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=api_messages,
            tools=api_tools or openai.NOT_GIVEN,
        )
        return self._make_normalized(response)


class OpenRouterAdapter(OpenAIAdapter):
    """OpenAI-compatible adapter pointed at OpenRouter.

    OpenRouter exposes many providers (OpenAI, Anthropic, Google, Meta, …) behind
    one OpenAI-format endpoint, which makes it ideal for cross-model testing from
    a single key. Models use ``provider/model`` ids, e.g.
    ``anthropic/claude-3.5-sonnet`` or ``openai/gpt-4o-mini``.

    The API key defaults to the ``OPENROUTER_API_KEY`` environment variable.
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        max_tokens: int = 4096,
        *,
        api_key: str | None = None,
        base_url: str = OPENROUTER_BASE_URL,
    ) -> None:
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            base_url=base_url,
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
        )


class AsyncOpenRouterAdapter(AsyncOpenAIAdapter):
    """Async OpenAI-compatible adapter pointed at OpenRouter."""

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        max_tokens: int = 4096,
        *,
        api_key: str | None = None,
        base_url: str = OPENROUTER_BASE_URL,
    ) -> None:
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            base_url=base_url,
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
        )


class DeepSeekAdapter(OpenAIAdapter):
    """OpenAI-compatible adapter for DeepSeek's (very cheap) direct API.

    Models: ``deepseek-chat`` and ``deepseek-reasoner``. The API key defaults to
    the ``DEEPSEEK_API_KEY`` environment variable. DeepSeek is also reachable via
    OpenRouter as ``deepseek/deepseek-chat`` if you prefer a single key.
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        max_tokens: int = 4096,
        *,
        api_key: str | None = None,
        base_url: str = DEEPSEEK_BASE_URL,
    ) -> None:
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            base_url=base_url,
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"),
        )


class AsyncDeepSeekAdapter(AsyncOpenAIAdapter):
    """Async OpenAI-compatible adapter for DeepSeek's direct API."""

    def __init__(
        self,
        model: str = "deepseek-chat",
        max_tokens: int = 4096,
        *,
        api_key: str | None = None,
        base_url: str = DEEPSEEK_BASE_URL,
    ) -> None:
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            base_url=base_url,
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"),
        )
