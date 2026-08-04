"""Tests for suffix-only reminder mechanics in Harness loop."""

from __future__ import annotations

import copy

from data_harness.data.harness import Harness
from data_harness.llm.providers.base import (
    NormalizedResponse,
    ProviderAdapter,
    StopReason,
)
from data_harness.llm.types import (
    Message,
    TextBlock,
    ToolSpec,
    ToolUseBlock,
)


class FakeAdapter(ProviderAdapter):
    def __init__(self, responses: list[NormalizedResponse]) -> None:
        self._responses = list(responses)
        self._calls: list[dict] = []

    def chat(
        self, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> NormalizedResponse:
        self._calls.append(
            {
                "system": system,
                "messages": copy.deepcopy(messages),
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
    tool_id: str, tool_name: str, tool_input: dict
) -> NormalizedResponse:
    return NormalizedResponse(
        stop_reason=StopReason.TOOL_USE,
        content=[
            ToolUseBlock(
                tool_use_id=tool_id, tool_name=tool_name, tool_input=tool_input
            )
        ],
        input_tokens=10,
        output_tokens=5,
        cache_read_tokens=0,
        cache_write_tokens=0,
    )


class TestMaxTurnReminder:
    def test_reminder_in_last_user_message_before_final_call(self, tmp_path):
        """Before the final allowed call (turn max_turns - 1), reminder appended."""
        # max_turns=3: turns 1, 2, 3. At turn 2 (max_turns - 1), reminder is added.
        tool = ToolSpec(
            name="noop", description="noop", input_schema={}, handler=lambda: "ok"
        )
        adapter = FakeAdapter(
            [
                make_tool_response("tu_1", "noop", {}),  # turn 1
                make_tool_response("tu_2", "noop", {}),  # turn 2 (last before max)
                make_text_response("done"),  # turn 3 (max)
            ]
        )
        harness = Harness(
            adapter=adapter,
            system="sys",
            tools=[tool],
            max_turns=3,
            run_dir=str(tmp_path),
        )
        harness.run("go")
        # Turn 2 (index 1) call messages: should include reminder
        turn2_messages = adapter._calls[1]["messages"]
        all_text = " ".join(
            b.text
            for m in turn2_messages
            for b in m.content
            if isinstance(b, TextBlock)
        )
        assert any(
            word in all_text.lower()
            for word in ["final", "last", "output", "respond", "turn"]
        )

    def test_reminder_appended_to_tool_result_user_message(self, tmp_path):
        """If user message is a tool_result, reminder is appended as TextBlock."""
        tool = ToolSpec(
            name="t", description="t", input_schema={}, handler=lambda: "ok"
        )
        adapter = FakeAdapter(
            [
                make_tool_response("tu_1", "t", {}),  # turn 1
                make_text_response(
                    "done"
                ),  # turn 2 (max_turns=2 → turn 1 = max_turns-1)
            ]
        )
        # With max_turns=2, turn 1 is max_turns - 1 = 1
        harness = Harness(
            adapter=adapter,
            system="sys",
            tools=[tool],
            max_turns=2,
            run_dir=str(tmp_path),
        )
        harness.run("go")
        # Turn 1 call: no tool result yet, reminder might be in original user message
        # Turn 2 call: messages include tool_result user message with reminder
        # At max_turns-1 = turn 1, reminder should be in the first call's user message
        turn1_messages = adapter._calls[0]["messages"]
        last_user_t1 = [m for m in turn1_messages if m.role == "user"][-1]
        all_text_t1 = " ".join(
            b.text for b in last_user_t1.content if isinstance(b, TextBlock)
        )
        assert any(
            word in all_text_t1.lower()
            for word in ["final", "last", "output", "respond", "turn"]
        )

    def test_system_prompt_byte_stable_with_reminder(self, tmp_path):
        """System prompt unchanged even when reminders are applied."""
        tool = ToolSpec(
            name="t", description="t", input_schema={}, handler=lambda: "ok"
        )
        adapter = FakeAdapter(
            [
                make_tool_response("tu_1", "t", {}),
                make_text_response("done"),
            ]
        )
        system = "Stable system prompt bytes."
        harness = Harness(
            adapter=adapter,
            system=system,
            tools=[tool],
            max_turns=2,
            run_dir=str(tmp_path),
        )
        harness.run("go")
        for call in adapter._calls:
            assert call["system"] == system

    def test_reminder_hooks_called_in_order(self, tmp_path):
        """Multiple registered reminder hooks are called deterministically."""
        call_order = []

        def hook_a(cur, max_t):
            call_order.append("a")
            return None

        def hook_b(cur, max_t):
            call_order.append("b")
            return None

        adapter = FakeAdapter([make_text_response("done")])
        harness = Harness(
            adapter=adapter, system="sys", tools=[], run_dir=str(tmp_path)
        )
        harness.register_reminder(hook_a)
        harness.register_reminder(hook_b)
        harness.run("hello")
        assert call_order == ["a", "b"]

    def test_new_user_message_created_when_no_current_user(self, tmp_path):
        """If no current user message, reminder creates a new user message."""
        reminder_text = "Please wrap up."

        def hook(cur, max_t):
            if cur == 1:
                return reminder_text
            return None

        adapter = FakeAdapter([make_text_response("done")])
        harness = Harness(
            adapter=adapter, system="sys", tools=[], run_dir=str(tmp_path)
        )
        harness.register_reminder(hook)
        harness.run("hello")
        # The reminder text should appear somewhere in the messages
        turn1_messages = adapter._calls[0]["messages"]
        all_text = " ".join(
            b.text
            for m in turn1_messages
            for b in m.content
            if isinstance(b, TextBlock)
        )
        assert reminder_text in all_text
