"""Tests for `data_harness.llm.testing` — the public FakeAdapter for docs and tests."""

from __future__ import annotations

from data_harness.llm.providers.base import (
    NormalizedResponse,
    ProviderAdapter,
    StopReason,
)
from data_harness.llm.testing import FakeAdapter
from data_harness.llm.types import TextBlock, ToolUseBlock


def _text_response(text: str) -> NormalizedResponse:
    return NormalizedResponse(
        stop_reason=StopReason.END_TURN,
        content=[TextBlock(text=text)],
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
    )


def _tool_response(tool_use_id: str, name: str, tool_input: dict) -> NormalizedResponse:
    return NormalizedResponse(
        stop_reason=StopReason.TOOL_USE,
        content=[
            ToolUseBlock(tool_use_id=tool_use_id, tool_name=name, tool_input=tool_input)
        ],
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
    )


class TestFakeAdapter:
    def test_is_a_provider_adapter(self):
        adapter = FakeAdapter([_text_response("hi")])
        assert isinstance(adapter, ProviderAdapter)

    def test_returns_responses_in_order(self):
        adapter = FakeAdapter([_text_response("first"), _text_response("second")])
        r1 = adapter.chat(system="s", messages=[], tools=[])
        r2 = adapter.chat(system="s", messages=[], tools=[])
        assert r1.content[0].text == "first"
        assert r2.content[0].text == "second"

    def test_records_calls(self):
        adapter = FakeAdapter([_text_response("done")])
        adapter.chat(system="sys", messages=[], tools=[])
        assert len(adapter.calls) == 1
        assert adapter.calls[0]["system"] == "sys"

    def test_format_cache_control_sets_ephemeral(self):
        adapter = FakeAdapter([_text_response("done")])
        out = adapter.format_cache_control({"type": "text", "text": "x"})
        assert out["cache_control"] == {"type": "ephemeral"}

    def test_text_helper_classmethod(self):
        # Convenience helper to avoid hand-building NormalizedResponse in docs/tests
        resp = FakeAdapter.text("hello")
        assert resp.stop_reason == StopReason.END_TURN
        assert resp.content[0].text == "hello"

    def test_tool_use_helper_classmethod(self):
        resp = FakeAdapter.tool_use("tu_1", "echo", {"x": 1})
        assert resp.stop_reason == StopReason.TOOL_USE
        block = resp.content[0]
        assert isinstance(block, ToolUseBlock)
        assert block.tool_use_id == "tu_1"
        assert block.tool_name == "echo"
        assert block.tool_input == {"x": 1}
