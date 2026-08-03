"""Tests for documented examples."""

from __future__ import annotations

from data_harness.data.cache import SessionCache
from data_harness.data.harness import Harness
from data_harness.llm.testing import FakeAdapter
from data_harness.llm.types import ToolResultBlock
from examples.advanced_wiring import build_base_tools, load_unemployment_rate
from examples.quickstart import build_agent


def test_quickstart_build_agent_runs_with_fake_adapter(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    adapter = FakeAdapter([FakeAdapter.text("The mean is 3.0")])

    agent = build_agent(adapter)
    result = agent.run("Compute the mean of [1, 2, 3, 4, 5].")

    assert result == "The mean is 3.0"


def test_advanced_wiring_loads_checked_in_fred_sample():
    df = load_unemployment_rate()

    assert list(df.columns) == ["date", "series", "value"]
    assert len(df) == 12
    assert set(df["series"]) == {"UNRATE"}


def test_advanced_wiring_scripted_flow_uses_real_dataset(tmp_path):
    cache = SessionCache()
    adapter = FakeAdapter(
        [
            FakeAdapter.tool_use("tu_1", "load_connectors", {"name": "macro_data"}),
            FakeAdapter.tool_use("tu_2", "macro_data__load_unemployment_rate", {}),
            FakeAdapter.tool_use(
                "tu_3",
                "python_interpreter",
                {"code": "print(round(load_unemployment_rate['value'].mean(), 4))"},
            ),
            FakeAdapter.text("done"),
        ]
    )
    harness = Harness(
        adapter=adapter,
        system="You are a data analyst.",
        tools=build_base_tools(cache),
        run_dir=str(tmp_path),
        cache=cache,
    )

    assert harness.run("analyse UNRATE") == "done"

    assert "load_unemployment_rate" in cache.list_handles()
    tool_result_blocks = [
        block
        for call in adapter.calls
        for message in call["messages"]
        for block in message.content
        if isinstance(block, ToolResultBlock) and block.tool_use_id == "tu_3"
    ]
    assert any("4.0333" in block.content for block in tool_result_blocks)
