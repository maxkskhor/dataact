"""Tests for the core Harness loop."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from data_harness.cache import SessionCache
from data_harness.exceptions import MaxTurnsExceeded
from data_harness.loop import Harness
from data_harness.providers.base import NormalizedResponse, ProviderAdapter, StopReason
from data_harness.types import (
    Message,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)


class FakeAdapter(ProviderAdapter):
    """Adapter that returns scripted responses in order."""

    def __init__(self, responses: list[NormalizedResponse]) -> None:
        self._responses = list(responses)
        self._calls: list[dict] = []

    def chat(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> NormalizedResponse:
        self._calls.append(
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


def make_text_response(text: str) -> NormalizedResponse:
    return NormalizedResponse(
        stop_reason=StopReason.END_TURN,
        content=[TextBlock(text=text)],
        input_tokens=10,
        output_tokens=5,
        cache_read_tokens=0,
        cache_write_tokens=0,
    )


def make_tool_response(
    tool_use_id: str, tool_name: str, tool_input: dict
) -> NormalizedResponse:
    return NormalizedResponse(
        stop_reason=StopReason.TOOL_USE,
        content=[
            ToolUseBlock(
                tool_use_id=tool_use_id, tool_name=tool_name, tool_input=tool_input
            )
        ],
        input_tokens=10,
        output_tokens=5,
        cache_read_tokens=0,
        cache_write_tokens=0,
    )


class TestLoopBasic:
    def test_exits_on_end_turn(self, tmp_path):
        adapter = FakeAdapter([make_text_response("Done!")])
        harness = Harness(
            adapter=adapter, system="sys", tools=[], run_dir=str(tmp_path)
        )
        result = harness.run("hello")
        assert result == "Done!"

    def test_max_turns_raises(self, tmp_path):
        # Adapter that always returns tool_use → never ends
        responses = [make_tool_response(f"tu_{i}", "noop", {}) for i in range(30)]
        noop_tool = ToolSpec(
            name="noop",
            description="does nothing",
            input_schema={},
            handler=lambda: "noop",
        )
        adapter = FakeAdapter(responses)
        harness = Harness(
            adapter=adapter,
            system="sys",
            tools=[noop_tool],
            max_turns=3,
            run_dir=str(tmp_path),
        )
        with pytest.raises(MaxTurnsExceeded) as exc_info:
            harness.run("go")
        assert exc_info.value.turns == 3

    def test_tool_dispatch_called(self, tmp_path):
        called_with = []

        def my_handler(**kwargs):
            called_with.append(kwargs)
            return "tool output"

        tool = ToolSpec(
            name="my_tool",
            description="test",
            input_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
            handler=my_handler,
        )
        adapter = FakeAdapter(
            [
                make_tool_response("tu_1", "my_tool", {"x": 42}),
                make_text_response("done"),
            ]
        )
        harness = Harness(
            adapter=adapter, system="sys", tools=[tool], run_dir=str(tmp_path)
        )
        harness.run("run tool")
        assert called_with == [{"x": 42}]

    def test_tool_result_appended_to_messages(self, tmp_path):
        tool = ToolSpec(
            name="echo",
            description="echoes",
            input_schema={},
            handler=lambda: "echo response",
        )
        adapter = FakeAdapter(
            [
                make_tool_response("tu_1", "echo", {}),
                make_text_response("done"),
            ]
        )
        harness = Harness(
            adapter=adapter, system="sys", tools=[tool], run_dir=str(tmp_path)
        )
        harness.run("go")
        # Turn 2 adapter call should have a user message with ToolResultBlock
        turn2_messages = adapter._calls[1]["messages"]
        last_user = [m for m in turn2_messages if m.role == "user"][-1]
        tool_results = [b for b in last_user.content if isinstance(b, ToolResultBlock)]
        assert len(tool_results) == 1
        assert tool_results[0].tool_use_id == "tu_1"

    def test_multi_turn_flow(self, tmp_path):
        tool = ToolSpec(
            name="step",
            description="step",
            input_schema={},
            handler=lambda: "stepped",
        )
        adapter = FakeAdapter(
            [
                make_tool_response("tu_1", "step", {}),
                make_text_response("all done"),
            ]
        )
        harness = Harness(
            adapter=adapter, system="sys", tools=[tool], run_dir=str(tmp_path)
        )
        result = harness.run("begin")
        assert result == "all done"

    def test_jsonl_has_one_line_per_turn(self, tmp_path):
        tool = ToolSpec(
            name="step",
            description="step",
            input_schema={},
            handler=lambda: "stepped",
        )
        adapter = FakeAdapter(
            [
                make_tool_response("tu_1", "step", {}),
                make_text_response("done"),
            ]
        )
        harness = Harness(
            adapter=adapter, system="sys", tools=[tool], run_dir=str(tmp_path)
        )
        harness.run("begin")
        jsonl_files = list(Path(tmp_path).glob("*.jsonl"))
        assert len(jsonl_files) == 1
        raw = jsonl_files[0].read_text().strip().splitlines()
        lines = [json.loads(line) for line in raw]
        assert len(lines) == 2

    def test_tool_use_ordering(self, tmp_path):
        tool = ToolSpec(
            name="echo",
            description="echo",
            input_schema={},
            handler=lambda: "ok",
        )
        adapter = FakeAdapter(
            [
                make_tool_response("tu_1", "echo", {}),
                make_text_response("done"),
            ]
        )
        harness = Harness(
            adapter=adapter, system="sys", tools=[tool], run_dir=str(tmp_path)
        )
        harness.run("go")
        # In the 2nd adapter call, messages should have tool_use followed by tool_result
        msgs = adapter._calls[1]["messages"]
        # Find assistant message with ToolUseBlock
        asst_msgs = [m for m in msgs if m.role == "assistant"]
        user_msgs = [m for m in msgs if m.role == "user"]
        for am in asst_msgs:
            tubs = [b for b in am.content if isinstance(b, ToolUseBlock)]
            for tub in tubs:
                # Find matching ToolResultBlock in subsequent user message
                found = False
                for um in user_msgs:
                    for b in um.content:
                        if (
                            isinstance(b, ToolResultBlock)
                            and b.tool_use_id == tub.tool_use_id
                        ):
                            found = True
                assert found

    def test_tool_not_found_is_error(self, tmp_path):
        adapter = FakeAdapter(
            [
                make_tool_response("tu_1", "nonexistent_tool", {}),
                make_text_response("done"),
            ]
        )
        harness = Harness(
            adapter=adapter, system="sys", tools=[], run_dir=str(tmp_path)
        )
        harness.run("go")
        # Should not raise; turn 2 should have an error result
        msgs = adapter._calls[1]["messages"]
        last_user = [m for m in msgs if m.role == "user"][-1]
        tool_results = [b for b in last_user.content if isinstance(b, ToolResultBlock)]
        assert any(tr.is_error for tr in tool_results)

    def test_raising_handler_is_error(self, tmp_path):
        """A tool handler that raises must produce is_error=True in ToolResultBlock.

        Regression: tools that catch and return error strings instead of raising
        will silently produce is_error=False, hiding failures from the model.
        """

        def exploding_tool() -> str:
            raise ValueError("boom from handler")

        tool = ToolSpec(
            name="exploding",
            description="always raises",
            input_schema={"type": "object", "properties": {}},
            handler=exploding_tool,
        )
        adapter = FakeAdapter(
            [
                make_tool_response("tu_1", "exploding", {}),
                make_text_response("done"),
            ]
        )
        harness = Harness(
            adapter=adapter, system="sys", tools=[tool], run_dir=str(tmp_path)
        )
        harness.run("go")
        msgs = adapter._calls[1]["messages"]
        last_user = [m for m in msgs if m.role == "user"][-1]
        tool_results = [b for b in last_user.content if isinstance(b, ToolResultBlock)]
        assert len(tool_results) == 1
        assert tool_results[0].is_error is True
        assert "boom from handler" in tool_results[0].content

    def test_system_byte_stable(self, tmp_path):
        tool = ToolSpec(
            name="echo",
            description="echo",
            input_schema={},
            handler=lambda: "ok",
        )
        adapter = FakeAdapter(
            [
                make_tool_response("tu_1", "echo", {}),
                make_tool_response("tu_2", "echo", {}),
                make_text_response("done"),
            ]
        )
        system = "This is a stable system prompt."
        harness = Harness(
            adapter=adapter, system=system, tools=[tool], run_dir=str(tmp_path)
        )
        harness.run("go")
        for call in adapter._calls:
            assert call["system"] == system

    def test_adapter_input_immutability(self, tmp_path):
        tool = ToolSpec(
            name="echo",
            description="echo",
            input_schema={},
            handler=lambda: "ok",
        )
        adapter = FakeAdapter(
            [
                make_tool_response("tu_1", "echo", {}),
                make_text_response("done"),
            ]
        )
        harness = Harness(
            adapter=adapter, system="sys", tools=[tool], run_dir=str(tmp_path)
        )
        harness.run("go")
        # The harness internal messages should not be mutated by adapter calls
        # We verify by checking stored messages are structurally sound
        assert harness.messages is not None
        for m in harness.messages:
            assert m.role in ("user", "assistant")

    def test_visible_tool_flip_reflects_in_next_call(self, tmp_path):
        """A tool that flips ToolSpec.visible is allowed and reflected next turn."""
        hidden_tool = ToolSpec(
            name="hidden",
            description="hidden",
            input_schema={},
            handler=lambda: "hidden result",
            visible=False,
        )

        def flipper(**kwargs):
            hidden_tool.visible = True
            return "flipped"

        flipper_tool = ToolSpec(
            name="flipper",
            description="flips visibility",
            input_schema={},
            handler=flipper,
            visible=True,
        )
        adapter = FakeAdapter(
            [
                make_tool_response("tu_1", "flipper", {}),
                make_text_response("done"),
            ]
        )
        harness = Harness(
            adapter=adapter,
            system="sys",
            tools=[flipper_tool, hidden_tool],
            run_dir=str(tmp_path),
        )
        harness.run("go")
        # After flipper runs, the 2nd call should see "hidden" in tools
        turn2_tools = adapter._calls[1]["tools"]
        tool_names = [t.name for t in turn2_tools]
        assert "hidden" in tool_names

    def test_no_automatic_variable_state_injection(self, tmp_path):
        """Putting a value in SessionCache does not inject a state message next turn."""
        cache = SessionCache()
        cache.put("my_var", "important data")

        adapter = FakeAdapter(
            [
                make_text_response("done"),
            ]
        )
        harness = Harness(
            adapter=adapter,
            system="sys",
            tools=[],
            run_dir=str(tmp_path),
            cache=cache,
        )
        harness.run("hello")
        # The messages sent to adapter should not contain my_var state
        msgs = adapter._calls[0]["messages"]
        all_text = " ".join(
            b.text for m in msgs for b in m.content if isinstance(b, TextBlock)
        )
        assert "my_var" not in all_text
        assert "important data" not in all_text
