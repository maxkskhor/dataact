"""SQL execution surface.

``make_sql_query_spec`` builds a ``sql_query`` tool. With no engine URL it runs
DuckDB in-process over the DataFrame handles already in the `SessionCache`
(zero setup, reads nothing from disk). With a SQLAlchemy URL it queries that
database instead. Results are stored back into the cache as a DataFrame handle,
so only a compact snapshot reaches the model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from data_harness.data.format import format_tool_output
from data_harness.llm.types import ToolAnnotations, ToolSpec

if TYPE_CHECKING:
    from data_harness.data.cache import SessionCache


def _is_dataframe(value: object) -> bool:
    try:
        import pandas as pd

        return isinstance(value, pd.DataFrame)
    except ImportError:
        return False


def _run_duckdb(cache: SessionCache, query: str):
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - install-matrix dependent
        raise RuntimeError(
            "DuckDB SQL requires the 'duckdb' extra: pip install "
            "'data-harness[duckdb]'."
        ) from exc

    con = duckdb.connect()
    try:
        for name, value in cache.items():
            if _is_dataframe(value):
                con.register(name, value)
        return con.execute(query).df()
    finally:
        con.close()


def _run_sqlalchemy(engine_url: str, query: str):
    try:
        import pandas as pd
        from sqlalchemy import create_engine, text
    except ImportError as exc:  # pragma: no cover - install-matrix dependent
        raise RuntimeError(
            "SQLAlchemy SQL requires the 'sql' extra: pip install 'data-harness[sql]'."
        ) from exc

    engine = create_engine(engine_url)
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)


def make_sql_query_spec(
    cache: SessionCache, *, engine_url: str | None = None
) -> ToolSpec:
    """Build a ``sql_query`` ToolSpec.

    Args:
        cache: The session cache; DataFrame handles are queryable by name when
            running over DuckDB, and results are written back as handles.
        engine_url: Optional SQLAlchemy URL. When ``None``, DuckDB runs
            in-process over the cache's DataFrame handles.
    """

    def sql_query(query: str) -> str:
        if engine_url is None:
            result = _run_duckdb(cache, query)
        else:
            result = _run_sqlalchemy(engine_url, query)
        return format_tool_output(result, cache=cache, preferred_name="query_result")

    if engine_url is None:
        backend = "DuckDB (in-process)"
        scope = (
            "Query the DataFrame handles in the session by name, e.g. "
            "SELECT * FROM sales WHERE revenue > 100."
        )
        open_world = False
    else:
        backend = "SQLAlchemy"
        scope = "Query the configured SQL database."
        open_world = True

    return ToolSpec(
        name="sql_query",
        description=(
            f"Run a SQL query ({backend}) and store the result as a DataFrame "
            f"handle. {scope} The result snapshot is returned; operate on the "
            f"full result via the python_interpreter handle."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The SQL query to run."}
            },
            "required": ["query"],
        },
        handler=sql_query,
        annotations=ToolAnnotations(
            title="SQL Query",
            read_only=False,
            cache_mutating=True,
            destructive=False,
            open_world=open_world,
        ),
    )
