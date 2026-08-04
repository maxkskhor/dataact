"""Tests for Phase 2: shared TurnSummary used by both logger and RunResult.

TDD: written before implementation.
"""

from __future__ import annotations

import json
from pathlib import Path

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


def read_jsonl(path: str) -> list[dict]:
    lines = Path(path).read_text().strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# ---------------------------------------------------------------------------
# JSONL format invariants preserved
# ---------------------------------------------------------------------------


class TestJsonlInvariants:
    def test_one_line_per_turn(self, tmp_path):
        """JSONL must still have exactly one line per turn."""
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
        harness = Harness(
            adapter=adapter,
            system="s",
            tools=[echo_spec],
            max_turns=5,
            run_dir=str(tmp_path),
        )
        result = harness.run_result("go")
        records = read_jsonl(result.run_file)
        assert len(records) == 2  # one tool-use turn, one end-turn

    def test_jsonl_has_turn_number(self, tmp_path):
        adapter = FakeAdapter([make_text_response("ok")])
        harness = Harness(
            adapter=adapter, system="s", tools=[], max_turns=5, run_dir=str(tmp_path)
        )
        result = harness.run_result("x")
        records = read_jsonl(result.run_file)
        assert records[0]["turn"] == 1

    def test_jsonl_metrics_present(self, tmp_path):
        adapter = FakeAdapter(
            [make_text_response("ok", input_tokens=12, output_tokens=4)]
        )
        harness = Harness(
            adapter=adapter, system="s", tools=[], max_turns=5, run_dir=str(tmp_path)
        )
        result = harness.run_result("x")
        records = read_jsonl(result.run_file)
        metrics = records[0]["metrics"]
        assert metrics["input_tokens"] == 12
        assert metrics["output_tokens"] == 4

    def test_jsonl_no_raw_cache_payloads(self, tmp_path):
        """JSONL must not contain raw cache values — only storage metadata."""
        cache = SessionCache()
        cache.put("secret", list(range(1000)))
        adapter = FakeAdapter([make_text_response("ok")])
        harness = Harness(
            adapter=adapter,
            system="s",
            tools=[],
            max_turns=5,
            run_dir=str(tmp_path),
            cache=cache,
        )
        result = harness.run_result("go")
        raw = Path(result.run_file).read_text()
        # The raw list should not appear in the log
        assert "999" not in raw or "storage_type" in raw  # metadata may mention type

    def test_jsonl_system_hash_stable(self, tmp_path):
        """System hash must be identical across turns."""
        tool_resp = NormalizedResponse(
            stop_reason=StopReason.TOOL_USE,
            content=[
                ToolUseBlock(
                    tool_use_id="t1", tool_name="echo", tool_input={"text": "x"}
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
        harness = Harness(
            adapter=adapter,
            system="same system",
            tools=[echo_spec],
            max_turns=5,
            run_dir=str(tmp_path),
        )
        result = harness.run_result("go")
        records = read_jsonl(result.run_file)
        hashes = [r["system_hash"] for r in records]
        assert len(set(hashes)) == 1, "system_hash should be identical across turns"


# ---------------------------------------------------------------------------
# RunResult aggregation equals sum of per-turn JSONL metrics
# ---------------------------------------------------------------------------


class TestUsageAggregation:
    def test_aggregated_usage_equals_sum_of_jsonl(self, tmp_path):
        """result.usage must equal the sum of per-turn metrics in JSONL."""
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
        harness = Harness(
            adapter=adapter,
            system="s",
            tools=[echo_spec],
            max_turns=5,
            run_dir=str(tmp_path),
        )
        result = harness.run_result("go")
        records = read_jsonl(result.run_file)

        total_input = sum(r["metrics"]["input_tokens"] for r in records)
        total_output = sum(r["metrics"]["output_tokens"] for r in records)
        total_cache_read = sum(r["metrics"]["cache_read_tokens"] for r in records)
        total_cache_write = sum(r["metrics"]["cache_write_tokens"] for r in records)

        assert result.usage.input_tokens == total_input
        assert result.usage.output_tokens == total_output
        assert result.usage.cache_read_tokens == total_cache_read
        assert result.usage.cache_write_tokens == total_cache_write


# ---------------------------------------------------------------------------
# Visible tools in JSONL
# ---------------------------------------------------------------------------


class TestVisibleToolsLogged:
    def test_visible_tool_names_in_jsonl(self, tmp_path):
        """JSONL should record the names of visible tools for reconstruction."""
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
            adapter=adapter,
            system="s",
            tools=[echo_spec, hidden_spec],
            max_turns=5,
            run_dir=str(tmp_path),
        )
        result = harness.run_result("go")
        records = read_jsonl(result.run_file)
        record = records[0]
        assert "visible_tools" in record
        assert "echo" in record["visible_tools"]
        assert "hidden_tool" not in record["visible_tools"]


# ---------------------------------------------------------------------------
# Tool error counting in JSONL
# ---------------------------------------------------------------------------


class TestToolErrorCount:
    def test_tool_errors_counted_in_jsonl(self, tmp_path):
        """JSONL should record tool_error_count without changing message flow."""

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
        harness = Harness(
            adapter=adapter,
            system="s",
            tools=[error_spec],
            max_turns=5,
            run_dir=str(tmp_path),
        )
        result = harness.run_result("go")
        assert result.status == "success"  # harness continues after tool errors
        records = read_jsonl(result.run_file)
        tool_turn = records[0]
        assert "tool_error_count" in tool_turn
        assert tool_turn["tool_error_count"] == 1
