import pytest

from data_harness.data.format import format_tool_output

# ──────────────────────────────────────────────
# Phase 0: inline-path tests (no cache needed)
# ──────────────────────────────────────────────


def test_short_string_inline():
    result = format_tool_output("hello world", cache=None)
    assert result == "hello world"


def test_short_dict_inline():
    d = {"key": "value", "num": 42}
    result = format_tool_output(d, cache=None)
    assert "key" in result
    assert "value" in result


def test_scalar_int_inline():
    result = format_tool_output(42, cache=None)
    assert result == "42"


def test_scalar_float_inline():
    result = format_tool_output(3.14, cache=None)
    assert result == "3.14"


def test_scalar_none_inline():
    result = format_tool_output(None, cache=None)
    assert result == "None"


def test_exception_friendly():
    exc = ValueError("something went wrong")
    result = format_tool_output(exc, cache=None)
    assert "Error" in result
    assert "ValueError" in result
    assert "something went wrong" in result


def test_short_list_inline():
    result = format_tool_output([1, 2, 3], cache=None)
    assert "1" in result
    assert "2" in result


# ──────────────────────────────────────────────
# Phase 2: cache-path tests (added after SessionCache exists)
# ──────────────────────────────────────────────


def test_large_dict_cached():
    harness_cache = pytest.importorskip("harness.cache")
    SessionCache = harness_cache.SessionCache
    cache = SessionCache()
    large = {f"key_{i}": i for i in range(100)}
    result = format_tool_output(large, cache=cache, preferred_name="big_dict")
    assert "Saved as" in result
    assert "big_dict" in result


def test_dataframe_always_cached():
    pd = pytest.importorskip("pandas")
    harness_cache = pytest.importorskip("harness.cache")
    SessionCache = harness_cache.SessionCache
    cache = SessionCache()
    df = pd.DataFrame({"a": range(10), "b": range(10)})
    result = format_tool_output(df, cache=cache, preferred_name="market_data")
    assert "Saved as" in result
    assert "market_data" in result


def test_collision_suffixed_name():
    pd = pytest.importorskip("pandas")
    harness_cache = pytest.importorskip("harness.cache")
    SessionCache = harness_cache.SessionCache
    cache = SessionCache()
    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = pd.DataFrame({"b": [3, 4]})
    r1 = format_tool_output(df1, cache=cache, preferred_name="market_data")
    r2 = format_tool_output(df2, cache=cache, preferred_name="market_data")
    assert "market_data" in r1
    assert "market_data_2" in r2


def test_no_preferred_name_defaults():
    pd = pytest.importorskip("pandas")
    harness_cache = pytest.importorskip("harness.cache")
    SessionCache = harness_cache.SessionCache
    cache = SessionCache()
    df = pd.DataFrame({"x": [1, 2, 3]})
    result = format_tool_output(df, cache=cache)
    assert "Saved as" in result
    # cache should have something in it
    handles = cache.list_handles()
    assert len(handles) == 1
