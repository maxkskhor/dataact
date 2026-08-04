"""Tests for the ConnectorRegistry and progressive disclosure."""

import json

from data_harness.data.cache import SessionCache
from data_harness.data.tools.connectors import ConnectorRegistry
from data_harness.llm.types import ToolSpec


def make_market_data_connector():
    """Returns a registry with a mock 'market_data' connector."""
    import pandas as pd

    def fetch_ohlcv(symbol: str = "AAPL") -> "pd.DataFrame":
        return pd.DataFrame(
            {
                "date": [f"2024-01-{i:02d}" for i in range(1, 10001)],
                "open": [100.0] * 10000,
                "close": [101.0] * 10000,
            }
        )

    connector_spec = ToolSpec(
        name="market_data__fetch_ohlcv",
        description="Fetch OHLCV data for a symbol.",
        input_schema={"type": "object", "properties": {"symbol": {"type": "string"}}},
        handler=fetch_ohlcv,
        visible=False,
    )
    registry = ConnectorRegistry()
    registry.register(
        name="market_data",
        description="Fetches OHLCV market data.",
        tools=[connector_spec],
    )
    return registry


class TestDirectoryRendering:
    def test_directory_in_load_connectors_schema(self):
        registry = make_market_data_connector()
        spec = registry.get_load_connectors_spec()
        assert "market_data" in spec.description or "market_data" in json.dumps(
            spec.input_schema
        )

    def test_hidden_tools_not_in_all_tools_initially(self):
        registry = make_market_data_connector()
        all_specs = registry.all_tool_specs()
        visible_names = [s.name for s in all_specs if s.visible]
        assert "market_data__fetch_ohlcv" not in visible_names

    def test_load_connectors_makes_tools_visible(self):
        registry = make_market_data_connector()
        load_spec = registry.get_load_connectors_spec()
        load_spec.handler(name="market_data")
        all_specs = registry.all_tool_specs()
        visible_names = [s.name for s in all_specs if s.visible]
        assert "market_data__fetch_ohlcv" in visible_names

    def test_unknown_connector_returns_error(self):
        registry = make_market_data_connector()
        load_spec = registry.get_load_connectors_spec()
        result = load_spec.handler(name="nonexistent")
        assert (
            "error" in result.lower()
            or "not found" in result.lower()
            or "unknown" in result.lower()
        )

    def test_connector_tool_handler_dispatchable_after_load(self):
        registry = make_market_data_connector()
        load_spec = registry.get_load_connectors_spec()
        load_spec.handler(name="market_data")
        all_specs = {s.name: s for s in registry.all_tool_specs()}
        fetch_spec = all_specs["market_data__fetch_ohlcv"]
        # Should be callable
        assert fetch_spec.handler is not None


class TestLargeDataframeHandling:
    def test_connector_result_cached_not_inline(self):
        registry = make_market_data_connector()
        cache = SessionCache()
        # Load connector
        registry.get_load_connectors_spec().handler(name="market_data")
        # Call the connector tool with cache
        # The connector tool's handler needs access to cache for format_tool_output
        # We test via the ConnectorRegistry's wrapped handler
        result = registry.call_connector(
            "market_data__fetch_ohlcv", {"symbol": "AAPL"}, cache=cache
        )
        assert "Saved as" in result or "dataframe" in result.lower()
        # The cache should have the full DataFrame, not the message
        handles = cache.list_handles()
        assert len(handles) > 0

    def test_full_dataframe_not_in_result_string(self):
        registry = make_market_data_connector()
        cache = SessionCache()
        registry.get_load_connectors_spec().handler(name="market_data")
        result = registry.call_connector(
            "market_data__fetch_ohlcv", {"symbol": "AAPL"}, cache=cache
        )
        # Result string should not contain thousands of rows
        assert len(result) < 5000
