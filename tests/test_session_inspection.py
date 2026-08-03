"""Tests for Phase 4: session and run ids, AgentSession inspection.

TDD: written before implementation.
"""

from __future__ import annotations

from data_harness.agent import Agent
from data_harness.providers.base import NormalizedResponse, StopReason
from data_harness.result import RunResult
from data_harness.testing import FakeAdapter
from data_harness.types import TextBlock


def make_text_response(
    text: str, *, input_tokens: int = 5, output_tokens: int = 2
) -> NormalizedResponse:
    return NormalizedResponse(
        stop_reason=StopReason.END_TURN,
        content=[TextBlock(text=text)],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=0,
        cache_write_tokens=0,
    )


# ---------------------------------------------------------------------------
# AgentSession.id
# ---------------------------------------------------------------------------


class TestAgentSessionId:
    def test_session_has_id(self, tmp_path):
        agent = Agent(
            adapter=FakeAdapter([make_text_response("hi")]),
            system="s",
            run_dir=str(tmp_path),
        )
        session = agent.session()
        assert hasattr(session, "id")
        assert session.id is not None

    def test_session_id_is_string(self, tmp_path):
        agent = Agent(
            adapter=FakeAdapter([make_text_response("hi")]),
            system="s",
            run_dir=str(tmp_path),
        )
        assert isinstance(agent.session().id, str)

    def test_session_id_stable_across_asks(self, tmp_path):
        agent = Agent(
            adapter=FakeAdapter(
                [
                    make_text_response("first"),
                    make_text_response("second"),
                ]
            ),
            system="s",
            run_dir=str(tmp_path),
        )
        session = agent.session()
        session.ask("q1")
        id_after_first = session.id
        session.ask("q2")
        assert session.id == id_after_first

    def test_two_sessions_have_different_ids(self, tmp_path):
        adapter_a = FakeAdapter([make_text_response("a")])
        adapter_b = FakeAdapter([make_text_response("b")])
        agent_a = Agent(adapter=adapter_a, system="s", run_dir=str(tmp_path))
        agent_b = Agent(adapter=adapter_b, system="s", run_dir=str(tmp_path))
        assert agent_a.session().id != agent_b.session().id

    def test_session_id_in_run_result(self, tmp_path):
        agent = Agent(
            adapter=FakeAdapter([make_text_response("hi")]),
            system="s",
            run_dir=str(tmp_path),
        )
        session = agent.session()
        result = session.ask_result("q")
        assert result.session_id == session.id

    def test_agent_run_result_has_run_id(self, tmp_path):
        agent = Agent(
            adapter=FakeAdapter([make_text_response("hi")]),
            system="s",
            run_dir=str(tmp_path),
        )
        result = agent.run_result("q")
        assert result.run_id is not None
        assert isinstance(result.run_id, str)

    def test_two_one_shot_runs_have_different_run_ids(self, tmp_path):
        agent = Agent(
            adapter=FakeAdapter(
                [
                    make_text_response("first"),
                    make_text_response("second"),
                ]
            ),
            system="s",
            run_dir=str(tmp_path),
        )
        r1 = agent.run_result("q1")
        r2 = agent.run_result("q2")
        assert r1.run_id != r2.run_id

    def test_one_shot_run_result_has_no_session_id(self, tmp_path):
        agent = Agent(
            adapter=FakeAdapter([make_text_response("hi")]),
            system="s",
            run_dir=str(tmp_path),
        )
        result = agent.run_result("q")
        assert result.session_id is None

    def test_agent_has_no_last_result_attribute(self, tmp_path):
        """Per plan: Agent should NOT grow last_result in this phase."""
        agent = Agent(
            adapter=FakeAdapter([make_text_response("hi")]),
            system="s",
            run_dir=str(tmp_path),
        )
        agent.run_result("q")
        assert not hasattr(agent, "last_result")


# ---------------------------------------------------------------------------
# AgentSession.last_result
# ---------------------------------------------------------------------------


class TestAgentSessionLastResult:
    def test_last_result_none_before_any_ask(self, tmp_path):
        agent = Agent(
            adapter=FakeAdapter([]),
            system="s",
            run_dir=str(tmp_path),
        )
        session = agent.session()
        assert session.last_result is None

    def test_last_result_updated_after_ask_result(self, tmp_path):
        agent = Agent(
            adapter=FakeAdapter([make_text_response("hello")]),
            system="s",
            run_dir=str(tmp_path),
        )
        session = agent.session()
        result = session.ask_result("q")
        assert session.last_result is result

    def test_last_result_updated_after_ask(self, tmp_path):
        agent = Agent(
            adapter=FakeAdapter([make_text_response("hello")]),
            system="s",
            run_dir=str(tmp_path),
        )
        session = agent.session()
        session.ask("q")
        assert isinstance(session.last_result, RunResult)
        assert session.last_result.text == "hello"

    def test_last_result_updates_on_each_call(self, tmp_path):
        agent = Agent(
            adapter=FakeAdapter(
                [
                    make_text_response("first"),
                    make_text_response("second"),
                ]
            ),
            system="s",
            run_dir=str(tmp_path),
        )
        session = agent.session()
        session.ask("q1")
        assert session.last_result.text == "first"
        session.ask("q2")
        assert session.last_result.text == "second"


# ---------------------------------------------------------------------------
# AgentSession.turns
# ---------------------------------------------------------------------------


class TestAgentSessionTurns:
    def test_turns_zero_before_first_ask(self, tmp_path):
        agent = Agent(
            adapter=FakeAdapter([]),
            system="s",
            run_dir=str(tmp_path),
        )
        assert agent.session().turns == 0

    def test_turns_increments_after_each_ask(self, tmp_path):
        agent = Agent(
            adapter=FakeAdapter(
                [
                    make_text_response("first"),
                    make_text_response("second"),
                ]
            ),
            system="s",
            run_dir=str(tmp_path),
        )
        session = agent.session()
        session.ask("q1")
        assert session.turns == 1
        session.ask("q2")
        assert session.turns == 2

    def test_turns_counts_model_turns_not_asks(self, tmp_path):
        """Turns should count total model calls, not number of ask() calls.

        A two-turn run (tool-use + end-turn) triggered by one ask() should
        contribute 2 to session.turns.
        """
        from data_harness.types import ToolSpec, ToolUseBlock

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
        agent = Agent(adapter=adapter, system="s", run_dir=str(tmp_path))
        # Connector tools aren't easily injectable via Agent here; use workaround
        # by using the harness directly through session
        session = agent.session()
        # Patch: add the echo spec to the harness tools for this test
        session.harness.tools.append(echo_spec)
        session.ask("go")
        assert session.turns == 2
