"""MCP bridge tests — fully offline, against a fake MCP client (no real server)."""

from __future__ import annotations

from types import SimpleNamespace

from data_harness import Agent
from data_harness.data.mcp import _result_to_text, mcp_tool_specs
from data_harness.llm.testing import FakeAdapter


class _FakeMCPClient:
    """Stands in for a connected MCPClient."""

    def __init__(self):
        self.tools = [
            SimpleNamespace(
                name="get_time",
                description="Get the current time.",
                inputSchema={
                    "type": "object",
                    "properties": {"tz": {"type": "string"}},
                },
            )
        ]
        self.calls = []
        self.closed = False

    def call(self, name, arguments):
        self.calls.append((name, arguments))
        return f"time in {arguments.get('tz')}: 12:00"

    def close(self):
        self.closed = True


def test_mcp_tool_specs_prefix_and_handler():
    client = _FakeMCPClient()
    specs = mcp_tool_specs(client, prefix="clock")
    assert [s.name for s in specs] == ["clock__get_time"]
    spec = specs[0]
    assert spec.visible is False
    assert spec.input_schema["properties"]["tz"]["type"] == "string"
    assert spec.handler(tz="UTC") == "time in UTC: 12:00"
    assert client.calls == [("get_time", {"tz": "UTC"})]


def test_result_to_text_flattens_and_flags_errors():
    ok = SimpleNamespace(
        content=[SimpleNamespace(text="hello"), SimpleNamespace(text="world")],
        isError=False,
    )
    assert _result_to_text(ok) == "hello\nworld"
    err = SimpleNamespace(content=[SimpleNamespace(text="boom")], isError=True)
    assert _result_to_text(err) == "Error: boom"
    empty = SimpleNamespace(content=[], isError=False)
    assert _result_to_text(empty) == "(no content)"


def test_agent_mcp_server_is_a_loadable_connector(tmp_path):
    fake = _FakeMCPClient()
    adapter = FakeAdapter(
        [
            FakeAdapter.tool_use("t1", "load_connectors", {"name": "clock"}),
            FakeAdapter.tool_use("t2", "clock__get_time", {"tz": "UTC"}),
            FakeAdapter.text("It is 12:00 UTC."),
        ]
    )
    agent = Agent(adapter=adapter, system="s", run_dir=str(tmp_path))
    agent.add_mcp_server("clock", client=fake)

    # the MCP tools start hidden (progressive disclosure)
    tools = agent._build_tools()
    assert any(t.name == "load_connectors" for t in tools)
    assert all(t.visible is False for t in tools if t.name == "clock__get_time")

    result = agent.run_result("what time is it?")
    assert result.status == "success"
    assert fake.calls == [("get_time", {"tz": "UTC"})]  # MCP tool was dispatched

    agent.close()
    assert fake.closed is True
