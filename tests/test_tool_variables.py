"""Tests for the list_variables tool."""

import pytest

from data_harness.data.cache import SessionCache
from data_harness.data.tools.variables import make_list_variables_spec


class TestListVariables:
    def test_returns_all_handles_with_snapshots(self):
        cache = SessionCache()
        cache.put("x", 42)
        cache.put("y", [1, 2, 3])
        spec = make_list_variables_spec(cache)
        result = spec.handler()
        assert "x" in result
        assert "y" in result

    def test_empty_cache_readable_message(self):
        cache = SessionCache()
        spec = make_list_variables_spec(cache)
        result = spec.handler()
        assert isinstance(result, str)
        assert len(result) > 0
        assert "empty" in result.lower() or "no" in result.lower() or "0" in result

    def test_large_dataframe_only_snapshot_not_full_data(self):
        pytest.importorskip("pandas")
        import pandas as pd

        cache = SessionCache(sample_size=5)
        df = pd.DataFrame({"v": range(10000)})
        cache.put("big_df", df)
        spec = make_list_variables_spec(cache)
        result = spec.handler()
        assert "big_df" in result
        # Result should not contain all 10000 rows
        assert len(result) < 50000
        # Should contain snapshot info
        assert "shape" in result or "10000" in result or "dataframe" in result.lower()
