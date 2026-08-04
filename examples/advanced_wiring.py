"""
advanced_wiring.py - Wires the harness around a real checked-in FRED sample.

Requires ANTHROPIC_API_KEY to be set. Run:
    uv run python examples/advanced_wiring.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

from data_harness.data.cache import SessionCache
from data_harness.data.harness import Harness
from data_harness.data.tools.connectors import ConnectorRegistry
from data_harness.data.tools.interpreter import PythonInterpreter
from data_harness.data.tools.planner import Planner
from data_harness.data.tools.subagent import make_subagent_spec
from data_harness.data.tools.variables import make_list_variables_spec
from data_harness.llm.types import ToolSpec

DATA_PATH = Path(__file__).parent / "data" / "fred_unrate_2024.csv"


def load_local_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def load_unemployment_rate(series: str = "UNRATE") -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    filtered = df[df["series"] == series].copy()
    if filtered.empty:
        available = sorted(df.series.unique())
        raise ValueError(f"Unknown series {series!r}; available: {available}")
    return filtered


def build_base_tools(session_cache: SessionCache) -> list[ToolSpec]:
    registry = ConnectorRegistry()
    registry.register(
        name="macro_data",
        description="Checked-in FRED macroeconomic data samples.",
        tools=[
            ToolSpec(
                name="macro_data__load_unemployment_rate",
                description=(
                    "Load the FRED UNRATE sample. Returns monthly 2024 unemployment"
                    " rate observations as a DataFrame."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "series": {
                            "type": "string",
                            "description": "FRED series code; only UNRATE is included",
                        }
                    },
                },
                handler=load_unemployment_rate,
                visible=False,
            )
        ],
    )

    return [
        registry.get_load_connectors_spec(),
        PythonInterpreter.make_tool_spec(session_cache),
        make_list_variables_spec(session_cache),
        *registry.make_wrapped_specs(session_cache),
    ]


def main() -> None:
    load_local_env()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set. Skipping live demo.")
        sys.exit(0)

    from data_harness.llm.providers.anthropic import AnthropicAdapter

    session_cache = SessionCache(sample_size=5)

    # Planner
    planner = Planner()
    planner_specs = planner.make_tool_specs()

    # Subagent
    def adapter_factory():
        return AnthropicAdapter(model="claude-haiku-4-5-20251001", max_tokens=2048)

    base_tools = build_base_tools(session_cache)
    all_tools = base_tools + planner_specs

    subagent_spec = make_subagent_spec(
        adapter_factory=adapter_factory,
        parent_tools=base_tools,
        parent_cache=session_cache,
        run_dir="./runs",
        make_sub_tools=build_base_tools,
    )
    all_tools.append(subagent_spec)

    # Main adapter
    adapter = AnthropicAdapter(model="claude-sonnet-4-6", max_tokens=4096)

    harness = Harness(
        adapter=adapter,
        system=(
            "You are a data analyst with access to macro data tools. "
            "Use load_connectors to access data sources, "
            "python_interpreter to analyze data, "
            "and list_variables to check what's cached. "
            "Always produce a clear final answer."
        ),
        tools=all_tools,
        max_turns=10,
        run_dir="./runs",
        cache=session_cache,
    )

    # Register planner reminder
    harness.register_reminder(planner.reminder_hook)

    print("Starting agent-harness demo...")
    print("=" * 60)

    result = harness.run(
        "Load the macro_data connector and load the UNRATE sample. "
        "Use Python to compute the average unemployment rate. "
        "Then spawn a subagent with the cached handle to identify the highest "
        "unemployment month. Summarize both findings."
    )

    print("\nFinal response:")
    print(result)

    run_files = list(Path("./runs").glob("*.jsonl"))
    if run_files:
        latest = max(run_files, key=lambda f: f.stat().st_mtime)
        print(f"\nJSONL log: {latest}")


if __name__ == "__main__":
    main()
