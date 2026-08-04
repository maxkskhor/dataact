"""Per-turn accounting invariants, now recorded on the session tree.

These used to assert against the JSONL turn log; that log is gone (the
session tree replaced it), but the invariants it protected — one record per
turn, correct token/latency metrics, tool-error counting, visible-tool
tracking, no raw cache payloads leaking into the record — still matter and
are pinned here against `TurnEntry` instead.
"""

from __future__ import annotations

from data_harness.core.session.entries import TurnEntry
from data_harness.data.cache import SessionCache
from data_harness.data.harness import Harness
from data_harness.llm.providers.base import NormalizedResponse, StopReason
from data_harness.llm.testing import FakeAdapter
from data_harness.llm.types import TextBlock, ToolSpec, ToolUseBlock


def make_text_response(
    text: str,
    *,
    input_tokens: int = 10,
    output_tokens: int = 5,
    cache_read: int = 0,
    cache_write: int = 0,
) -> NormalizedResponse:
    return NormalizedResponse(
        stop_reason=StopReason.END_TURN,
        content=[TextBlock(text=text)],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
    )


def turn_entries(harness: Harness) -> list[TurnEntry]:
    return [e for e in harness.session.store.entries() if isinstance(e, TurnEntry)]


# ---------------------------------------------------------------------------
# Session tree invariants preserved from the old JSONL log
# ---------------------------------------------------------------------------


class TestSessionTurnInvariants:
    def test_one_turn_entry_per_turn(self, tmp_path):
        tool_resp = NormalizedResponse(
            stop_reason=StopReason.TOOL_USE,
            content=[
                ToolUseBlock(
                    tool_use_id="t1", tool_name="echo", tool_input={"text": "hi"}
                )
            ],
            input_tokens=5,
            output_tokens=2,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )
        final_resp = make_text_response("done")
        echo_spec = ToolSpec(
            name="echo",
            description="echo",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            handler=lambda text: text,
        )
        adapter = FakeAdapter([tool_resp, final_resp])
        harness = Harness(adapter=adapter, system="s", tools=[echo_spec], max_turns=5)
        harness.run_result("go")
        assert len(turn_entries(harness)) == 2  # one tool-use turn, one end-turn

    def test_turn_entry_has_turn_number(self, tmp_path):
        adapter = FakeAdapter([make_text_response("ok")])
        harness = Harness(adapter=adapter, system="s", tools=[], max_turns=5)
        harness.run_result("x")
        assert turn_entries(harness)[0].turn == 1

    def test_turn_entry_metrics_present(self, tmp_path):
        adapter = FakeAdapter(
            [make_text_response("ok", input_tokens=12, output_tokens=4)]
        )
        harness = Harness(adapter=adapter, system="s", tools=[], max_turns=5)
        harness.run_result("x")
        entry = turn_entries(harness)[0]
        assert entry.input_tokens == 12
        assert entry.output_tokens == 4

    def test_session_holds_no_raw_cache_payloads(self, tmp_path):
        """The session must not contain raw cache values — only turn metadata."""
        cache = SessionCache()
        cache.put("secret", list(range(1000)))
        adapter = FakeAdapter([make_text_response("ok")])
        harness = Harness(
            adapter=adapter, system="s", tools=[], max_turns=5, cache=cache
        )
        harness.run_result("go")
        # The turn entries carry counts and metadata only, never handle values.
        for entry in turn_entries(harness):
            assert not hasattr(entry, "cache_snapshot")
            assert not hasattr(entry, "cache_values")


# ---------------------------------------------------------------------------
# RunResult aggregation equals sum of per-turn session metrics
# ---------------------------------------------------------------------------


class TestUsageAggregation:
    def test_aggregated_usage_equals_sum_of_turn_entries(self, tmp_path):
        """result.usage must equal the sum of per-turn metrics in the session."""
        tool_resp = NormalizedResponse(
            stop_reason=StopReason.TOOL_USE,
            content=[
                ToolUseBlock(
                    tool_use_id="t1", tool_name="echo", tool_input={"text": "x"}
                )
            ],
            input_tokens=10,
            output_tokens=3,
            cache_read_tokens=2,
            cache_write_tokens=1,
        )
        final_resp = NormalizedResponse(
            stop_reason=StopReason.END_TURN,
            content=[TextBlock(text="done")],
            input_tokens=8,
            output_tokens=5,
            cache_read_tokens=0,
            cache_write_tokens=4,
        )
        echo_spec = ToolSpec(
            name="echo",
            description="echo",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            handler=lambda text: text,
        )
        adapter = FakeAdapter([tool_resp, final_resp])
        harness = Harness(adapter=adapter, system="s", tools=[echo_spec], max_turns=5)
        result = harness.run_result("go")
        entries = turn_entries(harness)

        assert result.usage.input_tokens == sum(e.input_tokens for e in entries)
        assert result.usage.output_tokens == sum(e.output_tokens for e in entries)
        assert result.usage.cache_read_tokens == sum(
            e.cache_read_tokens for e in entries
        )
        assert result.usage.cache_write_tokens == sum(
            e.cache_write_tokens for e in entries
        )


# ---------------------------------------------------------------------------
# Visible tools recorded per turn
# ---------------------------------------------------------------------------


class TestVisibleToolsLogged:
    def test_visible_tool_names_on_turn_entry(self, tmp_path):
        """The session should record the names of visible tools for reconstruction."""
        echo_spec = ToolSpec(
            name="echo",
            description="echo",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            handler=lambda text: text,
            visible=True,
        )
        hidden_spec = ToolSpec(
            name="hidden_tool",
            description="hidden",
            input_schema={"type": "object"},
            handler=lambda: None,
            visible=False,
        )
        adapter = FakeAdapter([make_text_response("ok")])
        harness = Harness(
            adapter=adapter, system="s", tools=[echo_spec, hidden_spec], max_turns=5
        )
        harness.run_result("go")
        entry = turn_entries(harness)[0]
        assert "echo" in entry.visible_tools
        assert "hidden_tool" not in entry.visible_tools


# ---------------------------------------------------------------------------
# Tool error counting
# ---------------------------------------------------------------------------


class TestToolErrorCount:
    def test_tool_errors_counted_on_turn_entry(self, tmp_path):
        """The session should record tool_error_count without changing message flow."""

        def boom(**_kwargs):
            raise ValueError("exploded")

        error_spec = ToolSpec(
            name="boom",
            description="boom",
            input_schema={"type": "object"},
            handler=boom,
        )
        tool_resp = NormalizedResponse(
            stop_reason=StopReason.TOOL_USE,
            content=[ToolUseBlock(tool_use_id="t1", tool_name="boom", tool_input={})],
            input_tokens=5,
            output_tokens=2,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )
        final_resp = make_text_response("recovered")
        adapter = FakeAdapter([tool_resp, final_resp])
        harness = Harness(adapter=adapter, system="s", tools=[error_spec], max_turns=5)
        result = harness.run_result("go")
        assert result.status == "success"  # harness continues after tool errors
        tool_turn = turn_entries(harness)[0]
        assert tool_turn.tool_error_count == 1
