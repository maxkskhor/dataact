# Connectors

Connectors are groups of data tools that start *hidden* and become visible
only when the model calls `load_connectors`. This is the **progressive
disclosure** pattern: a shorter tool list means the model makes better routing
decisions at each turn.

---

## Registering a connector

```python
from data_harness import Agent
from data_harness.llm.providers.anthropic import AnthropicAdapter

agent = Agent(
    adapter=AnthropicAdapter(model="claude-sonnet-4-6"),
    system="You are a data analyst.",
)

# Register the connector
market_data = agent.connector(
    "market_data",
    description="Market data tools for equities and ETFs.",
)

# Attach a tool to it
def fetch_ohlcv(symbol: str) -> list[dict]:
    """Fetch daily OHLCV data for a ticker symbol."""
    return [{"symbol": symbol, "date": "2024-01-01", "close": 101.2}]

market_data.tool(
    fetch_ohlcv,
    description="Fetch daily OHLCV data for a ticker symbol.",
)

result = agent.run("Load the market_data connector and inspect AAPL.")
```

---

## How it works

1. At run time, `Agent` builds a `ConnectorRegistry` and registers all
   connectors with their tools marked `visible=False`.
2. The `load_connectors` tool is added to the visible tool list. It takes a
   connector name and flips its tools to visible.
3. The model must call `load_connectors(connector_name="market_data")` before
   it can see and use `fetch_ohlcv`.

This means the full tool list is never dumped to the model upfront. Only the
tools the model has chosen to load are in scope.

---

## Multiple connectors

```python
market_data = agent.connector("market_data", description="Equity price data.")
macro_data  = agent.connector("macro_data",  description="FRED macroeconomic series.")

market_data.tool(fetch_ohlcv,         description="Fetch OHLCV data.")
macro_data.tool(fetch_fred_series,    description="Fetch a FRED series.")
```

Each connector is independent. The model loads them selectively based on
which data sources it needs for the current task.

---

## Input schema inference

`ConnectorBuilder.tool()` infers the input schema from the function's type
annotations by default. Provide `input_schema` explicitly for full control:

```python
market_data.tool(
    fetch_ohlcv,
    description="Fetch OHLCV data.",
    input_schema={
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "description": "Ticker symbol, e.g. AAPL"},
        },
        "required": ["symbol"],
    },
)
```

---

## MCP servers

data-harness is also an **MCP client**: point it at any [Model Context
Protocol](https://modelcontextprotocol.io) server and its tools become a
connector — same progressive disclosure, same handle/snapshot discipline (large
tool results land in the `SessionCache`, not the prompt). Needs the `[mcp]`
extra.

```python
agent = Agent.from_dataframe(df)            # or Agent(adapter=..., system=...)
agent.add_mcp_server("time", "uvx", args=["mcp-server-time"])
agent.run("What time is it in Tokyo? Load the time connector.")
agent.close()                                # shuts the server subprocess down
```

The server's tools register under the connector name (`time__get_current_time`,
…), hidden until the model calls `load_connectors("time")`. **Any** stdio MCP
server works the same way — swap the command/args for a Postgres, SQLite, or
filesystem server:

```python
agent.add_mcp_server(
    "warehouse", "uvx",
    args=["postgres-mcp", "postgresql://user:pass@host/db"],
)
```

So you reach the whole MCP ecosystem (databases, SaaS APIs, …) without writing a
connector per source — the bridge gives every MCP tool the cache-backed,
progressively-disclosed treatment. See `examples/mcp_demo.py`.

`MCPClient` / `MCPServer` / `mcp_tool_specs` are also exported for wiring an MCP
server into a `Harness` directly.

---

## Low-level: ConnectorRegistry

`Agent.connector()` is a convenience layer. For full control, use
`ConnectorRegistry` and `ToolSpec` directly — as shown in
`examples/advanced_wiring.py`:

```python
from data_harness.data.tools.connectors import ConnectorRegistry
from data_harness.llm.types import ToolSpec

registry = ConnectorRegistry()
registry.register(
    name="macro_data",
    description="FRED macroeconomic data.",
    tools=[
        ToolSpec(
            name="macro_data__load_unrate",
            description="Load the FRED UNRATE series.",
            input_schema={"type": "object", "properties": {}},
            handler=load_unrate,
            visible=False,
        )
    ],
)

tools = [
    registry.get_load_connectors_spec(),
    *registry.make_wrapped_specs(cache),
]
```
