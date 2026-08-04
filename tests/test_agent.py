"""Tests for the high-level `Agent` convenience class (PLAN_SDK Phase 1)."""

from __future__ import annotations

import pytest

from data_harness.app.agent import Agent
from data_harness.core.hooks import BeforeTurn
from data_harness.data.cache import SessionCache
from data_harness.data.harness import Harness
from data_harness.llm.testing import FakeAdapter
from data_harness.llm.types import ToolResultBlock, ToolSpec


def test_agent_is_exported_from_top_level_package():
    from data_harness import Agent as TopLevelAgent
    from data_harness import AgentSession as TopLevelAgentSession
    from data_harness.app.agent import Agent as ModuleAgent
    from data_harness.app.agent import AgentSession as ModuleAgentSession

    assert TopLevelAgent is ModuleAgent
    assert TopLevelAgentSession is ModuleAgentSession


class TestAgentPhase1:
    def test_run_returns_text(self, tmp_path):
        adapter = FakeAdapter([FakeAdapter.text("done")])
        agent = Agent(adapter=adapter, system="sys", run_dir=str(tmp_path))
        assert agent.run("hi") == "done"

    def test_default_tools_include_python_interpreter_and_list_variables(
        self, tmp_path
    ):
        adapter = FakeAdapter([FakeAdapter.text("done")])
        agent = Agent(adapter=adapter, system="sys", run_dir=str(tmp_path))
        agent.run("hi")
        tools_seen = adapter.calls[0]["tools"]
        names = {t.name for t in tools_seen}
        assert "python_interpreter" in names
        assert "list_variables" in names

    def test_run_is_one_shot_resets_history(self, tmp_path):
        adapter = FakeAdapter([FakeAdapter.text("first"), FakeAdapter.text("second")])
        agent = Agent(adapter=adapter, system="sys", run_dir=str(tmp_path))
        agent.run("hello")
        agent.run("again")
        # Each run should start with exactly one user message in the message list
        assert len(adapter.calls) == 2
        for call in adapter.calls:
            user_msgs = [m for m in call["messages"] if m.role == "user"]
            assert len(user_msgs) == 1

    def test_last_harness_points_to_most_recent(self, tmp_path):
        adapter = FakeAdapter([FakeAdapter.text("a"), FakeAdapter.text("b")])
        agent = Agent(adapter=adapter, system="sys", run_dir=str(tmp_path))
        assert agent.last_harness is None
        agent.run("one")
        first_h = agent.last_harness
        assert isinstance(first_h, Harness)
        agent.run("two")
        # A fresh harness is built per run; reference should change
        assert agent.last_harness is not first_h

    def test_cache_is_exposed(self, tmp_path):
        adapter = FakeAdapter([FakeAdapter.text("done")])
        agent = Agent(adapter=adapter, system="sys", run_dir=str(tmp_path))
        assert isinstance(agent.cache, SessionCache)

    def test_cache_can_be_passed_in(self, tmp_path):
        cache = SessionCache()
        cache.put("preloaded", [1, 2, 3])
        adapter = FakeAdapter([FakeAdapter.text("done")])
        agent = Agent(adapter=adapter, system="sys", cache=cache, run_dir=str(tmp_path))
        assert agent.cache is cache
        assert "preloaded" in agent.cache.list_handles()

    def test_run_dir_optional(self, tmp_path, monkeypatch):
        # When run_dir is omitted, chart artefacts fall back to "./runs/charts".
        # cd into tmp_path so a run with no run_dir doesn't litter the repo.
        monkeypatch.chdir(tmp_path)
        adapter = FakeAdapter([FakeAdapter.text("done")])
        agent = Agent(adapter=adapter, system="sys")
        assert agent.run("hi") == "done"

    def test_max_turns_propagates(self, tmp_path):
        adapter = FakeAdapter([FakeAdapter.text("done")])
        agent = Agent(adapter=adapter, system="sys", max_turns=7, run_dir=str(tmp_path))
        agent.run("hi")
        assert agent.last_harness.max_turns == 7

    def test_explain_returns_readable_sketch(self, tmp_path):
        adapter = FakeAdapter([FakeAdapter.text("done")])
        agent = Agent(adapter=adapter, system="sys", run_dir=str(tmp_path))
        sketch = agent.explain()
        assert isinstance(sketch, str)
        # Should mention the core primitives a reader needs to find
        assert "SessionCache" in sketch
        assert "python_interpreter" in sketch
        assert "list_variables" in sketch
        assert "Harness" in sketch

    def test_explain_works_without_running(self, tmp_path):
        adapter = FakeAdapter([])
        agent = Agent(adapter=adapter, system="sys", run_dir=str(tmp_path))
        # explain() must not require a prior run
        agent.explain()


class TestAgentModelShortcut:
    """Agent(system=..., model=...) resolves an adapter without the caller
    constructing one, the same way Agent.from_dataframe(model=...) always has."""

    def test_model_resolves_an_adapter(self, monkeypatch):
        from data_harness.llm.providers.anthropic import AnthropicAdapter

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        agent = Agent(system="sys", model="claude-sonnet-4-6")
        assert isinstance(agent._adapter, AnthropicAdapter)

    def test_no_adapter_or_model_falls_back_to_the_environment(self, monkeypatch):
        from data_harness.llm.providers.anthropic import AnthropicAdapter

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        agent = Agent(system="sys")
        assert isinstance(agent._adapter, AnthropicAdapter)

    def test_explicit_adapter_wins_over_model(self):
        adapter = FakeAdapter([])
        agent = Agent(system="sys", adapter=adapter, model="claude-sonnet-4-6")
        assert agent._adapter is adapter

    def test_model_shortcut_agent_runs(self, monkeypatch):
        from data_harness.app import quickstart

        monkeypatch.setattr(
            quickstart,
            "resolve_adapter",
            lambda model=None: FakeAdapter([FakeAdapter.text("done")]),
        )
        agent = Agent(system="sys", model="claude-sonnet-4-6")
        assert agent.run("hi") == "done"

    def test_no_provider_configured_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="No provider configured"):
            Agent(system="sys")


class TestAgentPhase1OneShotInvariant:
    def test_second_run_does_not_see_first_run_messages(self, tmp_path):
        """Each Agent.run() builds a fresh Harness; messages must not leak across."""
        adapter = FakeAdapter([FakeAdapter.text("alpha"), FakeAdapter.text("beta")])
        agent = Agent(adapter=adapter, system="sys", run_dir=str(tmp_path))
        agent.run("first user prompt")
        agent.run("second user prompt")

        from data_harness.llm.types import TextBlock

        second_call_msgs = adapter.calls[1]["messages"]
        all_text = " ".join(
            b.text
            for m in second_call_msgs
            for b in m.content
            if isinstance(b, TextBlock)
        )
        assert "first user prompt" not in all_text
        assert "second user prompt" in all_text


class TestAgentSession:
    def test_session_preserves_message_history_across_asks(self, tmp_path):
        adapter = FakeAdapter([FakeAdapter.text("first"), FakeAdapter.text("second")])
        agent = Agent(adapter=adapter, system="sys", run_dir=str(tmp_path))
        session = agent.session()

        assert session.ask("first question") == "first"
        assert session.ask("follow-up question") == "second"

        from data_harness.llm.types import TextBlock

        second_call_msgs = adapter.calls[1]["messages"]
        all_text = " ".join(
            b.text
            for m in second_call_msgs
            for b in m.content
            if isinstance(b, TextBlock)
        )
        assert "first question" in all_text
        assert "follow-up question" in all_text
        assert "first" in all_text

    def test_session_uses_same_cache_for_uploaded_handles(self, tmp_path):
        adapter = FakeAdapter(
            [
                FakeAdapter.tool_use("tu_1", "list_variables", {}),
                FakeAdapter.text("done"),
            ]
        )
        agent = Agent(adapter=adapter, system="sys", run_dir=str(tmp_path))
        session = agent.session()
        session.put("uploaded_data", [1, 2, 3])

        session.ask("what data is available?")

        tool_results = [
            block
            for message in adapter.calls[1]["messages"]
            for block in message.content
            if isinstance(block, ToolResultBlock)
        ]
        assert tool_results
        assert "uploaded_data" in tool_results[-1].content
        assert "uploaded_data" in session.list_handles()

    def test_session_does_not_change_agent_run_one_shot_behaviour(self, tmp_path):
        adapter = FakeAdapter(
            [
                FakeAdapter.text("session"),
                FakeAdapter.text("run"),
            ]
        )
        agent = Agent(adapter=adapter, system="sys", run_dir=str(tmp_path))

        agent.session().ask("session question")
        agent.run("standalone question")

        from data_harness.llm.types import TextBlock

        standalone_call_msgs = adapter.calls[1]["messages"]
        all_text = " ".join(
            b.text
            for m in standalone_call_msgs
            for b in m.content
            if isinstance(b, TextBlock)
        )
        assert "session question" not in all_text
        assert "standalone question" in all_text

    def test_session_updates_agent_debug_affordances(self, tmp_path):
        adapter = FakeAdapter([FakeAdapter.text("done")])
        agent = Agent(adapter=adapter, system="sys", run_dir=str(tmp_path))
        session = agent.session()

        assert agent.last_harness is None
        session.ask("hi")

        assert agent.last_harness is session.harness


class TestAgentConnectors:
    def test_connector_returns_builder_and_tool_returns_function(self, tmp_path):
        adapter = FakeAdapter([FakeAdapter.text("done")])
        agent = Agent(adapter=adapter, system="sys", run_dir=str(tmp_path))

        def fetch_ohlcv(symbol: str) -> list[str]:
            return [symbol]

        builder = agent.connector("market_data", description="Market data tools.")
        returned = builder.tool(
            fetch_ohlcv, description="Fetch OHLCV data for a ticker."
        )

        assert returned is fetch_ohlcv

    def test_connector_tools_start_hidden_and_load_makes_visible(self, tmp_path):
        adapter = FakeAdapter(
            [
                FakeAdapter.tool_use(
                    "tu_1", "load_connectors", {"name": "market_data"}
                ),
                FakeAdapter.text("done"),
            ]
        )
        agent = Agent(adapter=adapter, system="sys", run_dir=str(tmp_path))

        def fetch_ohlcv(symbol: str) -> list[str]:
            return [symbol]

        agent.connector("market_data", description="Market data tools.").tool(
            fetch_ohlcv,
            description="Fetch OHLCV data for a ticker.",
        )

        agent.run("load market data")

        first_names = {t.name for t in adapter.calls[0]["tools"]}
        second_names = {t.name for t in adapter.calls[1]["tools"]}
        assert "load_connectors" in first_names
        assert "market_data__fetch_ohlcv" not in first_names
        assert "market_data__fetch_ohlcv" in second_names

    def test_connector_specs_are_fresh_per_run(self, tmp_path):
        adapter = FakeAdapter(
            [
                FakeAdapter.tool_use(
                    "tu_1", "load_connectors", {"name": "market_data"}
                ),
                FakeAdapter.text("first"),
                FakeAdapter.text("second"),
            ]
        )
        agent = Agent(adapter=adapter, system="sys", run_dir=str(tmp_path))

        def fetch_ohlcv(symbol: str) -> list[str]:
            return [symbol]

        agent.connector("market_data", description="Market data tools.").tool(
            fetch_ohlcv,
            description="Fetch OHLCV data for a ticker.",
        )

        agent.run("first")
        agent.run("second")

        second_run_names = {t.name for t in adapter.calls[2]["tools"]}
        assert "load_connectors" in second_run_names
        assert "market_data__fetch_ohlcv" not in second_run_names

    def test_connector_return_values_flow_through_cache_formatter(self, tmp_path):
        adapter = FakeAdapter(
            [
                FakeAdapter.tool_use(
                    "tu_1", "load_connectors", {"name": "market_data"}
                ),
                FakeAdapter.tool_use("tu_2", "market_data__fetch_many", {}),
                FakeAdapter.text("done"),
            ]
        )
        agent = Agent(adapter=adapter, system="sys", run_dir=str(tmp_path))

        def fetch_many() -> list[int]:
            return list(range(100))

        agent.connector("market_data", description="Market data tools.").tool(
            fetch_many,
            description="Fetch many rows.",
        )

        agent.run("fetch data")

        assert "fetch_many" in agent.cache.list_handles()

    def test_connector_tool_name_uses_connector_prefix(self, tmp_path):
        adapter = FakeAdapter([FakeAdapter.text("done")])
        agent = Agent(adapter=adapter, system="sys", run_dir=str(tmp_path))

        def fetch_ohlcv(symbol: str) -> list[str]:
            return [symbol]

        agent.connector("market_data", description="Market data tools.").tool(
            fetch_ohlcv,
            description="Fetch OHLCV data for a ticker.",
        )

        agent.run("hi")

        names = {t.name for t in agent.last_harness.tools}
        assert "market_data__fetch_ohlcv" in names

    def test_explicit_input_schema_override_bypasses_inference(self, tmp_path):
        adapter = FakeAdapter([FakeAdapter.text("done")])
        agent = Agent(adapter=adapter, system="sys", run_dir=str(tmp_path))

        def fetch(payload: dict) -> str:
            return str(payload)

        schema = {
            "type": "object",
            "properties": {"payload": {"type": "object"}},
            "required": ["payload"],
        }
        agent.connector("market_data", description="Market data tools.").tool(
            fetch,
            description="Fetch with a custom payload.",
            input_schema=schema,
        )

        agent.run("hi")

        specs = {t.name: t for t in agent.last_harness.tools}
        assert specs["market_data__fetch"].input_schema is schema


class TestAgentPlanner:
    def test_enable_planner_adds_planner_tools(self, tmp_path):
        adapter = FakeAdapter([FakeAdapter.text("done")])
        agent = Agent(adapter=adapter, system="sys", run_dir=str(tmp_path))

        agent.enable_planner()
        agent.run("plan")

        names = {t.name for t in adapter.calls[0]["tools"]}
        assert {"planner__add", "planner__update", "planner__list"} <= names

    def test_enable_planner_registers_reminder_hook(self, tmp_path):
        adapter = FakeAdapter([FakeAdapter.text("done")])
        agent = Agent(adapter=adapter, system="sys", run_dir=str(tmp_path))

        agent.enable_planner()
        agent.run("plan")

        before_turn_hooks = agent.last_harness.hooks.hooks.get(BeforeTurn, [])
        assert len(before_turn_hooks) == 1

    def test_planner_absent_when_not_enabled(self, tmp_path):
        adapter = FakeAdapter([FakeAdapter.text("done")])
        agent = Agent(adapter=adapter, system="sys", run_dir=str(tmp_path))

        agent.run("no plan")

        names = {t.name for t in adapter.calls[0]["tools"]}
        assert not any(name.startswith("planner__") for name in names)
        assert agent.last_harness.hooks.hooks.get(BeforeTurn, []) == []

    def test_enable_planner_twice_does_not_duplicate_specs_or_hooks(self, tmp_path):
        adapter = FakeAdapter([FakeAdapter.text("done")])
        agent = Agent(adapter=adapter, system="sys", run_dir=str(tmp_path))

        agent.enable_planner()
        agent.enable_planner()
        agent.run("plan")

        names = [t.name for t in adapter.calls[0]["tools"]]
        assert names.count("planner__add") == 1
        assert names.count("planner__update") == 1
        assert names.count("planner__list") == 1
        assert len(agent.last_harness.hooks.hooks.get(BeforeTurn, [])) == 1

    def test_planner_state_does_not_leak_across_runs(self, tmp_path):
        adapter = FakeAdapter(
            [
                FakeAdapter.tool_use("tu_1", "planner__add", {"items": ["task A"]}),
                FakeAdapter.text("first done"),
                FakeAdapter.tool_use("tu_2", "planner__list", {}),
                FakeAdapter.text("second done"),
            ]
        )
        agent = Agent(adapter=adapter, system="sys", run_dir=str(tmp_path))
        agent.enable_planner()

        agent.run("first")
        agent.run("second")

        second_run_final_call = adapter.calls[3]
        tool_results = [
            block
            for message in second_run_final_call["messages"]
            for block in message.content
            if isinstance(block, ToolResultBlock)
        ]
        assert tool_results
        assert "Todo list is empty." in tool_results[-1].content
        assert "task A" not in tool_results[-1].content


def _subagent_harness(captured):
    """Pick the spawned subagent's harness out of every harness constructed.

    Selects by the worker system prompt rather than by construction order,
    which is an implementation detail of where the patch happens to bite.
    """
    subs = [
        h for h in captured if h.system.startswith("You are a clean-context worker")
    ]
    assert subs, "no subagent harness was constructed"
    return subs[0]


class TestAgentSubagents:
    def test_enable_subagents_adds_subagent_tool(self, tmp_path):
        adapter = FakeAdapter([FakeAdapter.text("done")])
        agent = Agent(adapter=adapter, system="sys", run_dir=str(tmp_path))

        agent.enable_subagents(adapter_factory=lambda: FakeAdapter([]))
        agent.run("delegate")

        names = {t.name for t in adapter.calls[0]["tools"]}
        assert "subagent" in names

    def test_subagent_absent_when_not_enabled(self, tmp_path):
        adapter = FakeAdapter([FakeAdapter.text("done")])
        agent = Agent(adapter=adapter, system="sys", run_dir=str(tmp_path))

        agent.run("no delegation")

        names = {t.name for t in adapter.calls[0]["tools"]}
        assert "subagent" not in names

    def test_subagent_adapter_factory_called_per_spawn(self, tmp_path):
        adapter = FakeAdapter(
            [
                FakeAdapter.tool_use("tu_1", "subagent", {"task": "one"}),
                FakeAdapter.tool_use("tu_2", "subagent", {"task": "two"}),
                FakeAdapter.text("done"),
            ]
        )
        created = []

        def factory():
            sub = FakeAdapter([FakeAdapter.text("sub done")])
            created.append(sub)
            return sub

        agent = Agent(adapter=adapter, system="sys", run_dir=str(tmp_path))
        agent.enable_subagents(adapter_factory=factory)

        agent.run("delegate twice")

        assert len(created) == 2
        assert created[0] is not created[1]

    def test_subagent_parent_tools_exclude_subagent_spec(self, monkeypatch, tmp_path):
        captured_names = []

        def recording_make_subagent_spec(
            adapter_factory,
            parent_tools,
            parent_cache,
            get_sub_cache=None,
            make_sub_tools=None,
        ):
            captured_names.extend(t.name for t in parent_tools)
            return ToolSpec(
                name="subagent",
                description="subagent",
                input_schema={"type": "object", "properties": {}},
                handler=lambda **kwargs: "done",
            )

        monkeypatch.setattr(
            "data_harness.app.agent.make_subagent_spec", recording_make_subagent_spec
        )
        adapter = FakeAdapter([FakeAdapter.text("done")])
        agent = Agent(adapter=adapter, system="sys", run_dir=str(tmp_path))

        agent.enable_subagents(adapter_factory=lambda: FakeAdapter([]))
        agent.run("delegate")

        assert "subagent" not in captured_names

    def test_subagent_does_not_inherit_planner_hooks(self, monkeypatch, tmp_path):
        from data_harness.data.harness import Harness as RealHarness

        captured = []

        def recording_harness(*args, **kwargs):
            harness = RealHarness(*args, **kwargs)
            captured.append(harness)
            return harness

        monkeypatch.setattr("data_harness.data.harness.Harness", recording_harness)
        adapter = FakeAdapter(
            [
                FakeAdapter.tool_use("tu_1", "subagent", {"task": "work"}),
                FakeAdapter.text("done"),
            ]
        )

        agent = Agent(adapter=adapter, system="sys", run_dir=str(tmp_path))
        agent.enable_planner()
        agent.enable_subagents(
            adapter_factory=lambda: FakeAdapter([FakeAdapter.text("sub done")])
        )

        agent.run("delegate")

        assert _subagent_harness(captured).hooks.hooks.get(BeforeTurn, []) == []

    def test_subagent_connector_tools_are_fresh_and_hidden(self, monkeypatch, tmp_path):
        from data_harness.data.harness import Harness as RealHarness

        captured = []

        def recording_harness(*args, **kwargs):
            harness = RealHarness(*args, **kwargs)
            captured.append(harness)
            return harness

        monkeypatch.setattr("data_harness.data.harness.Harness", recording_harness)
        adapter = FakeAdapter(
            [
                FakeAdapter.tool_use(
                    "tu_1", "load_connectors", {"name": "market_data"}
                ),
                FakeAdapter.tool_use("tu_2", "subagent", {"task": "work"}),
                FakeAdapter.text("done"),
                FakeAdapter.text("second done"),
            ]
        )

        def fetch_ohlcv(symbol: str) -> list[str]:
            return [symbol]

        agent = Agent(adapter=adapter, system="sys", run_dir=str(tmp_path))
        agent.connector("market_data", description="Market data tools.").tool(
            fetch_ohlcv,
            description="Fetch OHLCV data for a ticker.",
        )
        agent.enable_subagents(
            adapter_factory=lambda: FakeAdapter([FakeAdapter.text("sub done")])
        )

        agent.run("load then delegate")

        sub_harness = _subagent_harness(captured)
        sub_specs = {t.name: t for t in sub_harness.tools}
        parent_specs = {t.name: t for t in agent.last_harness.tools}
        assert sub_specs["load_connectors"].visible is True
        assert sub_specs["market_data__fetch_ohlcv"].visible is False
        assert parent_specs["market_data__fetch_ohlcv"].visible is True
        assert id(sub_specs["market_data__fetch_ohlcv"]) != id(
            parent_specs["market_data__fetch_ohlcv"]
        )

        first_sub_connector_id = id(sub_specs["market_data__fetch_ohlcv"])
        agent.run("fresh second run")
        second_parent_specs = {t.name: t for t in agent.last_harness.tools}
        assert id(second_parent_specs["market_data__fetch_ohlcv"]) != id(
            parent_specs["market_data__fetch_ohlcv"]
        )
        assert id(second_parent_specs["market_data__fetch_ohlcv"]) != (
            first_sub_connector_id
        )


def test_fake_adapter_drives_quickstart_snippet(tmp_path):
    """README quick-start should be runnable with a fake adapter (Phase 1 docs goal)."""
    adapter = FakeAdapter([FakeAdapter.text("The mean is 3.0")])
    agent = Agent(
        adapter=adapter,
        system="You are a data analyst.",
        run_dir=str(tmp_path),
    )
    result = agent.run("Compute the mean of [1, 2, 3, 4, 5].")
    assert result == "The mean is 3.0"


@pytest.mark.parametrize("n_runs", [1, 3])
def test_agent_run_count_matches_call_count(tmp_path, n_runs):
    adapter = FakeAdapter([FakeAdapter.text(f"r{i}") for i in range(n_runs)])
    agent = Agent(adapter=adapter, system="sys", run_dir=str(tmp_path))
    for i in range(n_runs):
        agent.run(f"msg {i}")
    assert len(adapter.calls) == n_runs
