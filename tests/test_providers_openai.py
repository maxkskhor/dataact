from __future__ import annotations

import copy
import json
import os
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("openai")

from data_harness.llm.providers.base import StopReason
from data_harness.llm.types import (
    Message,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)


def make_openai_response(finish_reason="stop", message=None, usage=None):
    response = MagicMock()
    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message = message if message is not None else make_openai_message()
    response.choices = [choice]
    response.usage = MagicMock()
    response.usage.prompt_tokens = (usage or {}).get("prompt_tokens", 10)
    response.usage.completion_tokens = (usage or {}).get("completion_tokens", 5)
    return response


def make_openai_message(content="Hello!", tool_calls=None):
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    return message


def make_openai_tool_call(id_, name, arguments):
    call = MagicMock()
    call.id = id_
    call.type = "function"
    call.function = MagicMock()
    call.function.name = name
    call.function.arguments = arguments
    return call


class TestOpenAIAdapter:
    def _make_adapter(self):
        with patch("openai.OpenAI"):
            from data_harness.llm.providers.openai import OpenAIAdapter

            adapter = OpenAIAdapter(model="gpt-test")
        return adapter

    def test_format_cache_control_returns_copy_unchanged(self):
        adapter = self._make_adapter()
        original = {"type": "text", "text": "hello"}
        result = adapter.format_cache_control(original)

        assert result is not original
        assert result == original
        assert "cache_control" not in original
        assert "cache_control" not in result

    def test_chat_end_turn_text(self):
        adapter = self._make_adapter()
        adapter._client.chat.completions.create.return_value = make_openai_response(
            finish_reason="stop",
            message=make_openai_message(content="Hello!"),
        )

        response = adapter.chat(
            system="sys",
            messages=[Message(role="user", content=[TextBlock(text="Hi")])],
            tools=[],
        )

        assert response.stop_reason == StopReason.END_TURN
        assert response.content == [TextBlock(text="Hello!")]
        assert response.input_tokens == 10
        assert response.output_tokens == 5
        assert response.cache_read_tokens == 0
        assert response.cache_write_tokens == 0

    @pytest.mark.parametrize(
        ("finish_reason", "expected"),
        [
            ("stop", StopReason.END_TURN),
            ("tool_calls", StopReason.TOOL_USE),
            ("length", StopReason.MAX_TOKENS),
            ("content_filter", StopReason.END_TURN),
            ("function_call", StopReason.TOOL_USE),
        ],
    )
    def test_finish_reason_mapping(self, finish_reason, expected):
        adapter = self._make_adapter()
        adapter._client.chat.completions.create.return_value = make_openai_response(
            finish_reason=finish_reason,
            message=make_openai_message(
                content=None,
                tool_calls=[
                    make_openai_tool_call("call_1", "my_tool", json.dumps({"x": 1}))
                ],
            ),
        )

        response = adapter.chat(system="sys", messages=[], tools=[])

        assert response.stop_reason == expected

    def test_multi_tool_calls_normalize_in_order(self):
        adapter = self._make_adapter()
        adapter._client.chat.completions.create.return_value = make_openai_response(
            finish_reason="tool_calls",
            message=make_openai_message(
                content=None,
                tool_calls=[
                    make_openai_tool_call("call_1", "first", json.dumps({"x": 1})),
                    make_openai_tool_call("call_2", "second", json.dumps({"y": 2})),
                ],
            ),
        )

        response = adapter.chat(system="sys", messages=[], tools=[])

        assert response.stop_reason == StopReason.TOOL_USE
        assert response.content == [
            ToolUseBlock(
                tool_use_id="call_1",
                tool_name="first",
                tool_input={"x": 1},
            ),
            ToolUseBlock(
                tool_use_id="call_2",
                tool_name="second",
                tool_input={"y": 2},
            ),
        ]

    def test_json_arguments_round_trip(self):
        adapter = self._make_adapter()
        adapter._client.chat.completions.create.return_value = make_openai_response(
            finish_reason="tool_calls",
            message=make_openai_message(
                content=None,
                tool_calls=[
                    make_openai_tool_call(
                        "call_1",
                        "my_tool",
                        json.dumps({"symbol": "AAPL", "limit": 5}),
                    )
                ],
            ),
        )
        messages = [
            Message(
                role="assistant",
                content=[
                    ToolUseBlock(
                        tool_use_id="prior_call",
                        tool_name="prior_tool",
                        tool_input={"a": 1},
                    )
                ],
            )
        ]

        response = adapter.chat(system="sys", messages=messages, tools=[])

        assert response.content[0].tool_input == {"symbol": "AAPL", "limit": 5}
        call_kwargs = adapter._client.chat.completions.create.call_args.kwargs
        sent_tool_calls = call_kwargs["messages"][1]["tool_calls"]
        assert sent_tool_calls[0]["function"]["arguments"] == json.dumps({"a": 1})

    def test_tool_result_mapping(self):
        adapter = self._make_adapter()
        adapter._client.chat.completions.create.return_value = make_openai_response()
        messages = [
            Message(
                role="user",
                content=[
                    ToolResultBlock(
                        tool_use_id="call_1",
                        content="tool output",
                        is_error=False,
                    )
                ],
            )
        ]

        adapter.chat(system="sys", messages=messages, tools=[])

        call_kwargs = adapter._client.chat.completions.create.call_args.kwargs
        assert call_kwargs["messages"][1] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "tool output",
        }

    def test_tool_result_precedes_reminder_text_in_same_internal_message(self):
        adapter = self._make_adapter()
        adapter._client.chat.completions.create.return_value = make_openai_response()
        messages = [
            Message(role="user", content=[TextBlock(text="start")]),
            Message(
                role="assistant",
                content=[
                    ToolUseBlock(
                        tool_use_id="call_1",
                        tool_name="python_interpreter",
                        tool_input={"code": "print(1)"},
                    )
                ],
            ),
            Message(
                role="user",
                content=[
                    ToolResultBlock(
                        tool_use_id="call_1",
                        content="1",
                        is_error=False,
                    ),
                    TextBlock(text="This is a reminder appended by the harness."),
                ],
            ),
        ]

        adapter.chat(system="sys", messages=messages, tools=[])

        sent = adapter._client.chat.completions.create.call_args.kwargs["messages"]
        assert sent[3] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "1",
        }
        assert sent[4] == {
            "role": "user",
            "content": "This is a reminder appended by the harness.",
        }

    def test_tool_specs_map_to_openai_functions(self):
        adapter = self._make_adapter()
        adapter._client.chat.completions.create.return_value = make_openai_response()
        tool = ToolSpec(
            name="lookup",
            description="Lookup data.",
            input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        )

        adapter.chat(system="sys", messages=[], tools=[tool])

        call_kwargs = adapter._client.chat.completions.create.call_args.kwargs
        assert call_kwargs["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Lookup data.",
                    "parameters": tool.input_schema,
                },
            }
        ]

    def test_adapter_input_immutability(self):
        adapter = self._make_adapter()
        adapter._client.chat.completions.create.return_value = make_openai_response()
        system = "The system prompt."
        messages = [
            Message(role="user", content=[TextBlock(text="hello")]),
            Message(
                role="assistant",
                content=[
                    ToolUseBlock(
                        tool_use_id="call_1",
                        tool_name="tool",
                        tool_input={"x": 1},
                    )
                ],
            ),
            Message(
                role="user",
                content=[ToolResultBlock(tool_use_id="call_1", content="ok")],
            ),
        ]
        tools = [
            ToolSpec(
                name="tool",
                description="desc",
                input_schema={"type": "object"},
                handler=None,
            )
        ]
        system_before = system
        messages_before = copy.deepcopy(messages)
        tools_before = copy.deepcopy(tools)

        adapter.chat(system=system, messages=messages, tools=tools)

        assert system == system_before
        assert messages == messages_before
        assert tools == tools_before


@pytest.mark.live
def test_openai_live_smoke():
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    from data_harness.llm.providers.openai import OpenAIAdapter

    adapter = OpenAIAdapter(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    response = adapter.chat(
        system="You reply tersely.",
        messages=[Message(role="user", content=[TextBlock(text="Say ok.")])],
        tools=[],
    )

    assert response.stop_reason == StopReason.END_TURN
    assert response.content
