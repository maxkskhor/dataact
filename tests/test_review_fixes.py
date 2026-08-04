"""Tests for bugs and gaps identified in the second code review.

Written before the fixes; all should fail initially.
"""

from __future__ import annotations

import pytest

from data_harness.data.harness import Harness
from data_harness.llm.providers.base import (
    NormalizedResponse,
    ProviderAdapter,
    StopReason,
)
from data_harness.llm.testing import FakeAdapter
from data_harness.llm.types import TextBlock


def make_text_response(text: str) -> NormalizedResponse:
    return NormalizedResponse(
        stop_reason=StopReason.END_TURN,
        content=[TextBlock(text=text)],
        input_tokens=5,
        output_tokens=2,
        cache_read_tokens=0,
        cache_write_tokens=0,
    )


# ---------------------------------------------------------------------------
# Bug: AgentSession.ask() silently swallows status="error"
# ---------------------------------------------------------------------------


class TestAgentSessionAskRaisesOnError:
    def test_ask_raises_runtime_error_on_adapter_exception(self, tmp_path):
        """AgentSession.ask() must raise RuntimeError when adapter raises."""
        from data_harness.app.agent import Agent

        class BoomAdapter(ProviderAdapter):
            def chat(self, system, messages, tools):
                raise RuntimeError("session boom")

            def format_cache_control(self, obj):
                return obj

        agent = Agent(adapter=BoomAdapter(), system="s", run_dir=str(tmp_path))
        session = agent.session()
        with pytest.raises(RuntimeError, match="session boom"):
            session.ask("go")

    def test_ask_result_still_returns_error_status_not_raises(self, tmp_path):
        """ask_result() must return RunResult(status='error'), not raise."""
        from data_harness.app.agent import Agent

        class BoomAdapter(ProviderAdapter):
            def chat(self, system, messages, tools):
                raise RuntimeError("boom2")

            def format_cache_control(self, obj):
                return obj

        agent = Agent(adapter=BoomAdapter(), system="s", run_dir=str(tmp_path))
        session = agent.session()
        result = session.ask_result("go")
        assert result.status == "error"
        assert result.error is not None
        assert "boom2" in result.error


# ---------------------------------------------------------------------------
# Bug: Error path writes no JSONL record
# ---------------------------------------------------------------------------


class TestErrorPathReportsFailure:
    def test_result_carries_the_provider_error(self, tmp_path):
        """A failed provider call must be reported on the `RunResult`, not lost."""

        class BoomAdapter(ProviderAdapter):
            def chat(self, system, messages, tools):
                raise RuntimeError("log me")

            def format_cache_control(self, obj):
                return obj

        harness = Harness(
            adapter=BoomAdapter(),
            system="s",
            tools=[],
            max_turns=5,
        )
        result = harness.run_result("go")
        assert result.status == "error"
        assert "log me" in (result.error or "")


# ---------------------------------------------------------------------------
# Bug: MAX_TOKENS stop reason burns turns instead of terminating
# ---------------------------------------------------------------------------


class TestMaxTokensTerminates:
    def test_max_tokens_terminates_immediately(self, tmp_path):
        """A MAX_TOKENS response must end the run on that turn, not continue looping."""
        max_tokens_resp = NormalizedResponse(
            stop_reason=StopReason.MAX_TOKENS,
            content=[TextBlock(text="truncated output")],
            input_tokens=5,
            output_tokens=2,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )
        harness = Harness(
            adapter=FakeAdapter([max_tokens_resp]),
            system="s",
            tools=[],
            max_turns=10,
        )
        result = harness.run_result("go")
        # Must not spin through remaining turns
        assert result.turns == 1
        assert result.stop_reason == StopReason.MAX_TOKENS

    def test_stop_sequence_terminates_immediately(self, tmp_path):
        """A STOP_SEQUENCE response must also terminate immediately."""
        stop_seq_resp = NormalizedResponse(
            stop_reason=StopReason.STOP_SEQUENCE,
            content=[TextBlock(text="partial")],
            input_tokens=5,
            output_tokens=2,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )
        harness = Harness(
            adapter=FakeAdapter([stop_seq_resp]),
            system="s",
            tools=[],
            max_turns=10,
        )
        result = harness.run_result("go")
        assert result.turns == 1
        assert result.stop_reason == StopReason.STOP_SEQUENCE


# ---------------------------------------------------------------------------
# Bug: AgentSession.__init__ uses shallow cache copy
# ---------------------------------------------------------------------------


class TestAgentSessionDeepCacheIsolation:
    def test_session_cache_is_isolated_from_agent_cache(self, tmp_path):
        """Mutating session cache must not affect the agent-level cache."""
        from data_harness.app.agent import Agent

        agent = Agent(
            adapter=FakeAdapter([make_text_response("ok")]),
            system="s",
            run_dir=str(tmp_path),
        )
        original_data = [1, 2, 3]
        agent.cache.put("data", original_data)

        session = agent.session()
        # Mutate through session cache — should not affect agent cache
        session_data = session.cache.get("data")
        session_data.append(99)
        session.cache.put("data", session_data, overwrite=True)

        # Agent-level cache value must be unaffected
        assert agent.cache.get("data") == [1, 2, 3]

    def test_agent_cache_mutation_does_not_affect_session(self, tmp_path):
        """Mutating agent cache after session creation must not affect session cache."""
        from data_harness.app.agent import Agent

        shared_list = [10, 20]
        agent = Agent(
            adapter=FakeAdapter([make_text_response("ok")]),
            system="s",
            run_dir=str(tmp_path),
        )
        agent.cache.put("vals", shared_list)

        session = agent.session()
        # Now mutate the original object
        shared_list.append(30)

        # Session cache must not see the mutation
        session_vals = session.cache.get("vals")
        assert 30 not in session_vals


# ---------------------------------------------------------------------------
# Bug: ToolAnnotations() all-None appears as empty dict in JSONL
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Bug: Harness(max_turns=0) silently produces broken run
# ---------------------------------------------------------------------------


class TestMaxTurnsValidation:
    def test_max_turns_zero_raises_value_error(self):
        """Harness(max_turns=0) must raise ValueError."""
        with pytest.raises(ValueError, match="max_turns"):
            Harness(
                adapter=FakeAdapter([]),
                system="s",
                tools=[],
                max_turns=0,
            )

    def test_max_turns_negative_raises_value_error(self):
        """Harness(max_turns=-1) must raise ValueError."""
        with pytest.raises(ValueError, match="max_turns"):
            Harness(
                adapter=FakeAdapter([]),
                system="s",
                tools=[],
                max_turns=-1,
            )

    def test_max_turns_one_is_valid(self, tmp_path):
        """Harness(max_turns=1) must be accepted."""
        harness = Harness(
            adapter=FakeAdapter([make_text_response("ok")]),
            system="s",
            tools=[],
            max_turns=1,
        )
        result = harness.run_result("go")
        assert result.status == "success"
