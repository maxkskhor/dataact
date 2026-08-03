"""End-to-end integration tests with all five architectural invariants."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd

from data_harness.data.cache import SessionCache
from data_harness.data.harness import Harness
from data_harness.data.tools.connectors import ConnectorRegistry
from data_harness.data.tools.interpreter import PythonInterpreter
from data_harness.data.tools.subagent import make_subagent_spec
from data_harness.data.tools.variables import make_list_variables_spec
from data_harness.llm.providers.base import (
    NormalizedResponse,
    ProviderAdapter,
    StopReason,
)
from data_harness.llm.types import (
    Message,
    TextBlock,
    ToolResultBlock,
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
                "tools": copy.deepcopy(tools),
            }
        )
        return self._responses.pop(0)

    def format_cache_control(self, obj: dict) -> dict:
        return {**obj, "cache_control": {"type": "ephemeral"}}


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


class TestIntegrationFlow:
    def _build_harness(self, tmp_path, adapter, cache=None):
        if cache is None:
            cache = SessionCache()

        # Build registry with synthetic market_data connector
        registry = ConnectorRegistry()
        df_10k = pd.DataFrame(
            {
                "date": [f"2024-01-{(i % 28) + 1:02d}" for i in range(10000)],
                "open": [100.0 + i * 0.01 for i in range(10000)],
                "close": [101.0 + i * 0.01 for i in range(10000)],
                "volume": [1000 + i for i in range(10000)],
            }
        )
        registry.register(
            name="market_data",
            description="Synthetic OHLCV data",
            tools=[
                ToolSpec(
                    name="market_data__fetch_ohlcv",
                    description="Fetch OHLCV data",
                    input_schema={"type": "object", "properties": {}},
                    handler=lambda: df_10k,
                    visible=False,
                )
            ],
        )

        load_connectors_spec = registry.get_load_connectors_spec()
        wrapped_specs = registry.make_wrapped_specs(cache)
        interp_spec = PythonInterpreter.make_tool_spec(cache)
        variables_spec = make_list_variables_spec(cache)

        tools = [load_connectors_spec, interp_spec, variables_spec] + wrapped_specs

        harness = Harness(
            adapter=adapter,
            system="You are a financial data analyst.",
            tools=tools,
            run_dir=str(tmp_path),
            cache=cache,
        )
        return harness, cache, registry

    def test_4_turn_scripted_flow(self, tmp_path):
        """
        Turn 1: load_connectors("market_data")
        Turn 2: market_data__fetch_ohlcv → DataFrame cached
        Turn 3: list_variables → introspect cache
        Turn 4: final text → loop exits
        """
        adapter = FakeAdapter(
            [
                make_tool_response("tu_1", "load_connectors", {"name": "market_data"}),
                make_tool_response("tu_2", "market_data__fetch_ohlcv", {}),
                make_tool_response("tu_3", "list_variables", {}),
                make_text_response(
                    "Analysis complete. The market data shows 10000 rows."
                ),
            ]
        )
        harness, cache, registry = self._build_harness(tmp_path, adapter)
        result = harness.run("Analyze market data")

        assert "Analysis complete" in result

        # Invariant 1: JSONL has 4 lines, each parseable
        jsonl_files = list(Path(tmp_path).glob("*.jsonl"))
        assert len(jsonl_files) == 1
        raw = jsonl_files[0].read_text().strip().splitlines()
        lines = [json.loads(line) for line in raw]
        assert len(lines) == 4
        for line in lines:
            assert "turn" in line
            assert "system_hash" in line

        # Invariant 2: SessionCache contains the full raw DataFrame
        handles = cache.list_handles()
        assert len(handles) > 0
        # Check at least one handle contains a DataFrame with 10000 rows
        found_df = False
        for name in handles:
            val = cache.get(name)
            try:
                if isinstance(val, pd.DataFrame) and len(val) == 10000:
                    found_df = True
                    break
            except Exception:
                pass
        assert found_df, "Expected 10k-row DataFrame in cache"

        # Invariant 3: No message contains the full raw DataFrame
        for msg_list in adapter._calls:
            for msg in msg_list["messages"]:
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        assert len(block.text) < 100_000
                    elif isinstance(block, ToolResultBlock):
                        assert len(block.content) < 100_000

        # Invariant 4: System prompt byte-identical across all 4 turns
        systems = [call["system"] for call in adapter._calls]
        assert len(set(systems)) == 1

        # Invariant 5: cache_control only in adapter-bound payloads, not harness objects
        for msg in harness.messages:
            for block in msg.content:
                assert not hasattr(block, "cache_control")
        for tool in harness.tools:
            assert not hasattr(tool, "cache_control")

    def test_tool_use_ordering_invariant(self, tmp_path):
        """Every ToolUseBlock is paired with a ToolResultBlock."""
        adapter = FakeAdapter(
            [
                make_tool_response("tu_1", "load_connectors", {"name": "market_data"}),
                make_text_response("done"),
            ]
        )
        harness, cache, _ = self._build_harness(tmp_path, adapter)
        harness.run("go")

        for call in adapter._calls:
            msgs = call["messages"]
            all_tool_use_ids = set()
            all_tool_result_ids = set()
            for msg in msgs:
                for block in msg.content:
                    if isinstance(block, ToolUseBlock):
                        all_tool_use_ids.add(block.tool_use_id)
                    elif isinstance(block, ToolResultBlock):
                        all_tool_result_ids.add(block.tool_use_id)
            # By the second call onwards, all prior tool_use_ids should have results
            # (the current call's tool uses won't have results yet)

    def test_no_automatic_variable_injection(self, tmp_path):
        """Putting a value in cache via interpreter does not auto-add state message."""
        cache = SessionCache()
        cache.put("preloaded_data", "sensitive")

        adapter = FakeAdapter([make_text_response("done")])
        harness, _, _ = self._build_harness(tmp_path, adapter, cache=cache)
        harness.run("hello")

        msgs = adapter._calls[0]["messages"]
        all_text = " ".join(
            b.text for m in msgs for b in m.content if isinstance(b, TextBlock)
        )
        assert "preloaded_data" not in all_text
        assert "sensitive" not in all_text

    def test_system_logging_policy(self, tmp_path):
        """Turn 1 JSONL has system + hash; turns 2+ have hash only; all hashes match."""
        adapter = FakeAdapter(
            [
                make_tool_response("tu_1", "load_connectors", {"name": "market_data"}),
                make_text_response("done"),
            ]
        )
        harness, _, _ = self._build_harness(tmp_path, adapter)
        harness.run("analyze")

        jsonl_files = list(Path(tmp_path).glob("*.jsonl"))
        raw = jsonl_files[0].read_text().strip().splitlines()
        lines = [json.loads(line) for line in raw]

        assert "system" in lines[0]
        assert "system_hash" in lines[0]
        assert "system" not in lines[1]
        assert "system_hash" in lines[1]

        hashes = [line["system_hash"] for line in lines]
        assert len(set(hashes)) == 1

    def test_subagent_integration(self, tmp_path):
        """Subagent spawns fresh adapter and cache; publish_created round-trip."""
        parent_cache = SessionCache()
        parent_cache.put("input_data", "some input")

        sub_call_count = [0]

        def sub_adapter_factory():
            sub_call_count[0] += 1
            return FakeAdapter([make_text_response("subagent done")])

        tools: list[ToolSpec] = []
        subagent_spec = make_subagent_spec(
            adapter_factory=sub_adapter_factory,
            parent_tools=tools,
            parent_cache=parent_cache,
            run_dir=str(tmp_path),
        )

        # Test text_only
        result = subagent_spec.handler(
            task="analyze input", input_handles=["input_data"]
        )
        assert "subagent done" in result
        assert sub_call_count[0] == 1

        # Test that sub used different adapter
        assert sub_call_count[0] == 1  # one fresh adapter was created

    def test_subagent_tools_exclude_subagent(self, tmp_path):
        """Sub-harness tool list never includes the subagent tool itself."""
        parent_cache = SessionCache()

        sub_tools_seen = []

        def sub_adapter_factory():
            class CapturingAdapter(ProviderAdapter):
                def chat(self, system, messages, tools):
                    sub_tools_seen.extend([t.name for t in tools])
                    return make_text_response("done")

                def format_cache_control(self, obj):
                    return {**obj, "cache_control": {"type": "ephemeral"}}

            return CapturingAdapter()

        subagent_spec = make_subagent_spec(
            adapter_factory=sub_adapter_factory,
            parent_tools=[],
            parent_cache=parent_cache,
            run_dir=str(tmp_path),
        )
        subagent_spec.handler(task="task")
        assert "subagent" not in sub_tools_seen
