"""Tests for Phase 1: typed run results (RunResult, Usage, CacheStorageInfo).

TDD: these tests are written before the implementation exists.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from data_harness.core.exceptions import MaxTurnsExceeded
from data_harness.core.result import CacheStorageInfo, RunResult, Usage
from data_harness.data.cache import SessionCache
from data_harness.data.harness import Harness
from data_harness.llm.providers.base import NormalizedResponse, StopReason
from data_harness.llm.testing import FakeAdapter
from data_harness.llm.types import TextBlock, ToolSpec, ToolUseBlock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def make_harness(
    responses: list[NormalizedResponse],
    *,
    tmp_path: Path,
    cache: SessionCache | None = None,
) -> Harness:
    adapter = FakeAdapter(responses)
    return Harness(
        adapter=adapter,
        system="test system",
        tools=[],
        max_turns=5,
        cache=cache,
    )


# ---------------------------------------------------------------------------
# Usage dataclass
# ---------------------------------------------------------------------------


class TestUsage:
    def test_fields_default_to_zero(self):
        u = Usage()
        assert u.input_tokens == 0
        assert u.output_tokens == 0
        assert u.cache_read_tokens == 0
        assert u.cache_write_tokens == 0

    def test_explicit_fields(self):
        u = Usage(
            input_tokens=10, output_tokens=5, cache_read_tokens=2, cache_write_tokens=3
        )
        assert u.input_tokens == 10
        assert u.output_tokens == 5
        assert u.cache_read_tokens == 2
        assert u.cache_write_tokens == 3

    def test_add(self):
        a = Usage(
            input_tokens=10, output_tokens=5, cache_read_tokens=1, cache_write_tokens=2
        )
        b = Usage(
            input_tokens=3, output_tokens=7, cache_read_tokens=4, cache_write_tokens=0
        )
        total = a + b
        assert total.input_tokens == 13
        assert total.output_tokens == 12
        assert total.cache_read_tokens == 5
        assert total.cache_write_tokens == 2


# ---------------------------------------------------------------------------
# CacheStorageInfo dataclass
# ---------------------------------------------------------------------------


class TestCacheStorageInfo:
    def test_memory_entry(self):
        info = CacheStorageInfo(location="memory", storage_type="memory")
        assert info.location == "memory"
        assert info.storage_type == "memory"

    def test_disk_entry(self):
        info = CacheStorageInfo(location="disk", storage_type="pickle")
        assert info.location == "disk"
        assert info.storage_type == "pickle"

    def test_invalid_location_raises(self):
        with pytest.raises((ValueError, TypeError)):
            CacheStorageInfo(location="cloud", storage_type="memory")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# RunResult dataclass
# ---------------------------------------------------------------------------


class TestRunResult:
    def _make(self, **kwargs) -> RunResult:
        defaults = dict(
            text="done",
            status="success",
            turns=1,
            stop_reason=None,
            usage=Usage(),
            cache_snapshots={},
            cache_storage={},
            error=None,
        )
        defaults.update(kwargs)
        return RunResult(**defaults)

    def test_basic_fields(self):
        r = self._make()
        assert r.text == "done"
        assert r.status == "success"
        assert r.turns == 1
        assert r.stop_reason is None
        assert isinstance(r.usage, Usage)
        assert r.cache_snapshots == {}
        assert r.cache_storage == {}
        assert r.error is None

    def test_max_turns_status(self):
        r = self._make(status="max_turns_exceeded", text="partial")
        assert r.status == "max_turns_exceeded"

    def test_error_status(self):
        r = self._make(status="error", error="boom", text="")
        assert r.status == "error"
        assert r.error == "boom"

    def test_cache_storage_typed(self):
        storage = {"df": CacheStorageInfo(location="disk", storage_type="pickle")}
        r = self._make(cache_storage=storage)
        assert r.cache_storage["df"].location == "disk"


# ---------------------------------------------------------------------------
# Harness.run_result()
# ---------------------------------------------------------------------------


class TestHarnessRunResult:
    def test_returns_run_result(self, tmp_path):
        harness = make_harness([make_text_response("hello")], tmp_path=tmp_path)
        result = harness.run_result("hi")
        assert isinstance(result, RunResult)

    def test_text_matches_run(self, tmp_path):
        harness_a = make_harness([make_text_response("hello")], tmp_path=tmp_path)
        harness_b = make_harness([make_text_response("hello")], tmp_path=tmp_path)
        assert harness_a.run_result("hi").text == harness_b.run("hi")

    def test_status_success(self, tmp_path):
        result = make_harness([make_text_response("ok")], tmp_path=tmp_path).run_result(
            "x"
        )
        assert result.status == "success"

    def test_usage_aggregated(self, tmp_path):
        # Two-turn run with tool use then end
        tool_resp = NormalizedResponse(
            stop_reason=StopReason.TOOL_USE,
            content=[
                ToolUseBlock(
                    tool_use_id="1", tool_name="echo", tool_input={"text": "hi"}
                )
            ],
            input_tokens=10,
            output_tokens=5,
            cache_read_tokens=2,
            cache_write_tokens=1,
        )
        final_resp = make_text_response("done", input_tokens=8, output_tokens=3)
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
        )
        result = harness.run_result("go")
        assert result.usage.input_tokens == 18
        assert result.usage.output_tokens == 8
        assert result.usage.cache_read_tokens == 2
        assert result.usage.cache_write_tokens == 1
        assert result.turns == 2

    def test_cache_snapshots_compact_strings(self, tmp_path):
        cache = SessionCache()
        cache.put("mylist", [1, 2, 3])
        harness = make_harness(
            [make_text_response("done")], tmp_path=tmp_path, cache=cache
        )
        result = harness.run_result("go")
        assert "mylist" in result.cache_snapshots
        snapshot = result.cache_snapshots["mylist"]
        assert isinstance(snapshot, str)
        # Must be a compact snapshot, not the raw list value
        assert snapshot != str([1, 2, 3])

    def test_cache_storage_typed_metadata(self, tmp_path):
        cache = SessionCache()
        cache.put("x", 42)
        harness = make_harness(
            [make_text_response("ok")], tmp_path=tmp_path, cache=cache
        )
        result = harness.run_result("go")
        assert "x" in result.cache_storage
        info = result.cache_storage["x"]
        assert isinstance(info, CacheStorageInfo)
        assert info.location == "memory"

    def test_stop_reason_populated(self, tmp_path):
        result = make_harness([make_text_response("ok")], tmp_path=tmp_path).run_result(
            "x"
        )
        from data_harness.llm.providers.base import StopReason

        assert result.stop_reason == StopReason.END_TURN

    def test_max_turns_returns_result_not_raises(self, tmp_path):
        # run_result should NOT raise MaxTurnsExceeded - it returns status instead
        # All responses are TOOL_USE to force max turns
        from data_harness.llm.providers.base import NormalizedResponse as NR

        tool_responses = [
            NR(
                stop_reason=StopReason.TOOL_USE,
                content=[
                    ToolUseBlock(
                        tool_use_id=f"t{i}", tool_name="echo", tool_input={"text": "x"}
                    )
                ],
                input_tokens=5,
                output_tokens=2,
                cache_read_tokens=0,
                cache_write_tokens=0,
            )
            for i in range(10)
        ]
        echo_spec = ToolSpec(
            name="echo",
            description="echo",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            handler=lambda text: text,
        )
        adapter = FakeAdapter(tool_responses)
        harness = Harness(
            adapter=adapter,
            system="s",
            tools=[echo_spec],
            max_turns=3,
        )
        result = harness.run_result("go")
        assert result.status == "max_turns_exceeded"
        assert result.turns == 3

    def test_run_still_returns_string(self, tmp_path):
        harness = make_harness([make_text_response("hello")], tmp_path=tmp_path)
        text = harness.run("hi")
        assert text == "hello"

    def test_run_still_raises_max_turns_exceeded(self, tmp_path):
        tool_responses = [
            NormalizedResponse(
                stop_reason=StopReason.TOOL_USE,
                content=[
                    ToolUseBlock(
                        tool_use_id=f"t{i}", tool_name="echo", tool_input={"text": "x"}
                    )
                ],
                input_tokens=5,
                output_tokens=2,
                cache_read_tokens=0,
                cache_write_tokens=0,
            )
            for i in range(10)
        ]
        echo_spec = ToolSpec(
            name="echo",
            description="echo",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            handler=lambda text: text,
        )
        adapter = FakeAdapter(tool_responses)
        harness = Harness(
            adapter=adapter,
            system="s",
            tools=[echo_spec],
            max_turns=3,
        )
        with pytest.raises(MaxTurnsExceeded):
            harness.run("go")


# ---------------------------------------------------------------------------
# Harness.ask_result()
# ---------------------------------------------------------------------------


class TestHarnessAskResult:
    def test_returns_run_result(self, tmp_path):
        harness = make_harness(
            [make_text_response("hi"), make_text_response("bye")], tmp_path=tmp_path
        )
        harness.ask("first")
        result = harness.ask_result("second")
        assert isinstance(result, RunResult)
        assert result.text == "bye"

    def test_usage_is_per_ask_not_cumulative(self, tmp_path):
        harness = make_harness(
            [
                make_text_response("first", input_tokens=10, output_tokens=5),
                make_text_response("second", input_tokens=7, output_tokens=3),
            ],
            tmp_path=tmp_path,
        )
        harness.ask("q1")
        result = harness.ask_result("q2")
        # usage should reflect only the second ask
        assert result.usage.input_tokens == 7
        assert result.usage.output_tokens == 3

    def test_ask_still_returns_string(self, tmp_path):
        harness = make_harness([make_text_response("hi")], tmp_path=tmp_path)
        text = harness.ask("q")
        assert text == "hi"


# ---------------------------------------------------------------------------
# Agent.run_result()
# ---------------------------------------------------------------------------


class TestAgentRunResult:
    def test_returns_run_result(self, tmp_path):
        from data_harness.app.agent import Agent

        adapter = FakeAdapter([FakeAdapter.text("done")])
        agent = Agent(adapter=adapter, system="sys", run_dir=str(tmp_path))
        result = agent.run_result("hi")
        assert isinstance(result, RunResult)

    def test_text_matches_run(self, tmp_path):
        from data_harness.app.agent import Agent

        adapter_a = FakeAdapter([FakeAdapter.text("hello")])
        adapter_b = FakeAdapter([FakeAdapter.text("hello")])
        agent_a = Agent(adapter=adapter_a, system="sys", run_dir=str(tmp_path))
        agent_b = Agent(adapter=adapter_b, system="sys", run_dir=str(tmp_path))
        assert agent_a.run_result("hi").text == agent_b.run("hi")

    def test_status_success(self, tmp_path):
        from data_harness.app.agent import Agent

        adapter = FakeAdapter([FakeAdapter.text("ok")])
        agent = Agent(adapter=adapter, system="sys", run_dir=str(tmp_path))
        result = agent.run_result("x")
        assert result.status == "success"

    def test_run_still_returns_string(self, tmp_path):
        from data_harness.app.agent import Agent

        adapter = FakeAdapter([FakeAdapter.text("hello")])
        agent = Agent(adapter=adapter, system="sys", run_dir=str(tmp_path))
        assert agent.run("hi") == "hello"


# ---------------------------------------------------------------------------
# AgentSession.ask_result()
# ---------------------------------------------------------------------------


class TestAgentSessionAskResult:
    def test_returns_run_result(self, tmp_path):
        from data_harness.app.agent import Agent

        adapter = FakeAdapter([FakeAdapter.text("first"), FakeAdapter.text("second")])
        agent = Agent(adapter=adapter, system="sys", run_dir=str(tmp_path))
        session = agent.session()
        session.ask("q1")
        result = session.ask_result("q2")
        assert isinstance(result, RunResult)
        assert result.text == "second"

    def test_usage_per_ask(self, tmp_path):
        from data_harness.app.agent import Agent
        from data_harness.llm.providers.base import NormalizedResponse as NR

        r1 = NR(
            stop_reason=StopReason.END_TURN,
            content=[TextBlock(text="a")],
            input_tokens=10,
            output_tokens=5,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )
        r2 = NR(
            stop_reason=StopReason.END_TURN,
            content=[TextBlock(text="b")],
            input_tokens=7,
            output_tokens=3,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )
        adapter = FakeAdapter([r1, r2])
        agent = Agent(adapter=adapter, system="sys", run_dir=str(tmp_path))
        session = agent.session()
        session.ask("q1")
        result = session.ask_result("q2")
        assert result.usage.input_tokens == 7
        assert result.usage.output_tokens == 3

    def test_ask_still_returns_string(self, tmp_path):
        from data_harness.app.agent import Agent

        adapter = FakeAdapter([FakeAdapter.text("hi")])
        agent = Agent(adapter=adapter, system="sys", run_dir=str(tmp_path))
        session = agent.session()
        assert session.ask("q") == "hi"
