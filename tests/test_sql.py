"""Tier 2: sql_query tool (DuckDB + SQLAlchemy), Agent.from_* and enable_sql."""

from __future__ import annotations

import json

import pandas as pd

from data_harness import Agent
from data_harness.data.cache import SessionCache
from data_harness.data.tools.sql import make_sql_query_spec
from data_harness.llm.testing import FakeAdapter


def _sales() -> pd.DataFrame:
    return pd.DataFrame({"product": ["a", "b", "a", "c"], "revenue": [10, 20, 30, 5]})


def test_duckdb_query_over_cache_frames():
    cache = SessionCache()
    cache.put("sales", _sales())
    spec = make_sql_query_spec(cache)
    out = spec.handler(
        query="SELECT product, SUM(revenue) AS total FROM sales GROUP BY product "
        "ORDER BY total DESC"
    )
    assert "Saved as `query_result`" in out
    result = cache.get("query_result")
    assert result.iloc[0]["product"] == "a"
    assert int(result.iloc[0]["total"]) == 40


def test_duckdb_result_snapshot_is_compact():
    cache = SessionCache()
    cache.put("sales", _sales())
    spec = make_sql_query_spec(cache)
    spec.handler(query="SELECT * FROM sales")
    snap = json.loads(cache.snapshot("query_result"))
    assert snap["type"] == "dataframe"
    assert snap["shape"] == [4, 2]


def test_sqlalchemy_query_in_memory():
    cache = SessionCache()
    spec = make_sql_query_spec(cache, engine_url="sqlite://")
    out = spec.handler(query="SELECT 1 AS x, 2 AS y")
    assert "query_result" in out
    assert int(cache.get("query_result").iloc[0]["x"]) == 1


def test_sql_annotations_open_world_flag():
    cache = SessionCache()
    duck = make_sql_query_spec(cache)
    sa = make_sql_query_spec(cache, engine_url="sqlite://")
    assert duck.annotations.open_world is False
    assert sa.annotations.open_world is True


def test_enable_sql_adds_tool():
    agent = Agent(adapter=FakeAdapter([]), system="s").enable_sql()
    tools = agent._build_tools()
    assert any(t.name == "sql_query" for t in tools)


def test_agent_not_sql_by_default():
    agent = Agent(adapter=FakeAdapter([]), system="s")
    tools = agent._build_tools()
    assert not any(t.name == "sql_query" for t in tools)


def test_from_dataframe_preloads_handle():
    agent = Agent.from_dataframe(_sales(), adapter=FakeAdapter([]))
    assert "df" in agent.cache.handle_names()


def test_from_csv(tmp_path):
    path = tmp_path / "sales.csv"
    _sales().to_csv(path, index=False)
    agent = Agent.from_csv(str(path), adapter=FakeAdapter([]))
    assert "sales" in agent.cache.handle_names()


def test_sql_used_through_agent_run(tmp_path):
    code_call = FakeAdapter.tool_use(
        "t1", "sql_query", {"query": "SELECT SUM(revenue) AS s FROM df"}
    )
    adapter = FakeAdapter([code_call, FakeAdapter.text("done")])
    agent = Agent.from_dataframe(
        _sales(), adapter=adapter, run_dir=str(tmp_path)
    ).enable_sql()
    res = agent.run_result("total revenue via sql")
    assert res.status == "success"
    assert "query_result" in res.cache_snapshots
