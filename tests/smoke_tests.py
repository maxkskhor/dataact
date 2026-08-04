"""Reusable live smoke tests for the released SDK surface.

These tests call a real model via OpenRouter (one key, many providers) and are
excluded from normal unit-test runs. They cover the quick SDK path, explicit
harness wiring, real checked-in data, connector/cache behaviour, subagents, and
the per-turn accounting recorded on the session tree.

Required:
    OPENROUTER_API_KEY

Optional:
    DATA_HARNESS_SMOKE_MODEL=openai/gpt-4o-mini

Run:
    uv run pytest tests/smoke_tests.py -v -m live

The default model is a cheap OpenAI model routed through OpenRouter. Override
DATA_HARNESS_SMOKE_MODEL (e.g. deepseek/deepseek-chat) to compare providers.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from data_harness import Agent
from data_harness.core.hooks import BeforeTurn, Reminder
from data_harness.core.result import CacheStorageInfo, RunResult
from data_harness.core.session.entries import TurnEntry
from data_harness.data.cache import SessionCache
from data_harness.data.harness import Harness
from data_harness.data.tools.planner import Planner
from data_harness.data.tools.subagent import make_subagent_spec
from data_harness.llm.providers.base import StopReason
from data_harness.llm.providers.openai import OpenRouterAdapter
from data_harness.llm.types import (
    ToolAnnotations,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)
from examples.advanced_wiring import build_base_tools, load_unemployment_rate

pytestmark = pytest.mark.live

DEFAULT_SMOKE_MODEL = "deepseek/deepseek-v4-flash"


def _load_env() -> None:
    if os.environ.get("OPENROUTER_API_KEY"):
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path(__file__).parent.parent / ".env")


def _require_key() -> None:
    _load_env()
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")


def _smoke_model_id() -> str:
    return os.environ.get("DATA_HARNESS_SMOKE_MODEL", DEFAULT_SMOKE_MODEL)


def _smoke_adapter(max_tokens: int = 512) -> OpenRouterAdapter:
    return OpenRouterAdapter(model=_smoke_model_id(), max_tokens=max_tokens)


def _turn_entries(harness: Harness) -> list[TurnEntry]:
    entries = [e for e in harness.session.store.entries() if isinstance(e, TurnEntry)]
    assert entries, "No TurnEntry records on the session"
    return entries


def _tool_use_blocks(harness: Harness) -> list[ToolUseBlock]:
    return [
        block
        for message in harness.messages
        for block in message.content
        if isinstance(block, ToolUseBlock)
    ]


def _tool_result_blocks(harness: Harness) -> list[ToolResultBlock]:
    return [
        block
        for message in harness.messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
    ]


def _all_text_from_messages(harness: Harness) -> str:
    parts: list[str] = []
    for message in harness.messages:
        for block in message.content:
            text = getattr(block, "text", None) or getattr(block, "content", None)
            if text:
                parts.append(text)
    return "\n".join(parts)


def test_openai_basic_harness_completion(tmp_path):
    _require_key()
    harness = Harness(
        adapter=_smoke_adapter(max_tokens=64),
        system="Answer concisely.",
        tools=[],
        max_turns=2,
    )

    result = harness.run("What is 7 times 8? Reply with only the number.")

    assert "56" in result
    turns = _turn_entries(harness)
    assert len(turns) == 1
    assert turns[0].stop_reason == "end_turn"
    assert turns[0].input_tokens > 0


def test_openai_agent_can_ask_clarifying_question(tmp_path):
    _require_key()
    agent = Agent(
        adapter=_smoke_adapter(max_tokens=128),
        system=(
            "You are a careful data analyst. If the request lacks the dataset or "
            "metric needed for analysis, ask one concise clarifying question and "
            "do not use tools."
        ),
        max_turns=2,
        run_dir=str(tmp_path),
    )

    result = agent.run("Analyse the macro data for me.")

    assert "?" in result
    assert agent.last_harness is not None
    assert not _tool_use_blocks(agent.last_harness)


def test_openai_agent_simple_sdk_uses_python_and_saves_handle(tmp_path):
    _require_key()
    agent = Agent(
        adapter=_smoke_adapter(max_tokens=384),
        system=(
            "You are validating an SDK. Use python_interpreter for arithmetic. "
            "When the user gives exact Python code, pass that code to the tool "
            "unchanged."
        ),
        max_turns=6,
        run_dir=str(tmp_path),
    )

    result = agent.run(
        "Call python_interpreter with exactly this code:\n"
        "value = sum([1, 2, 3, 4, 5])\n"
        "save('total_sum', value)\n"
        "print(value)\n"
        "Then answer with the numeric value."
    )

    assert "15" in result
    assert agent.cache.get("total_sum") == 15
    assert agent.last_harness is not None
    assert _tool_use_blocks(agent.last_harness)


def test_openai_agent_connector_real_fred_data(tmp_path):
    _require_key()
    agent = Agent(
        adapter=_smoke_adapter(max_tokens=768),
        system=(
            "You are a macro data analyst. Load connectors before using their "
            "tools. The connector snapshot is incomplete sample text; never "
            "reconstruct data from it. Use python_interpreter against the cached "
            "DataFrame handle named load_unemployment_rate."
        ),
        max_turns=8,
        run_dir=str(tmp_path),
    )

    macro = agent.connector(
        "macro_data",
        description="Checked-in FRED unemployment-rate sample.",
    )
    macro.tool(
        load_unemployment_rate,
        description="Load monthly 2024 FRED UNRATE observations as a DataFrame.",
    )

    result = agent.run(
        "Load macro_data, call load_unemployment_rate for UNRATE, then call "
        "python_interpreter with code equivalent to:\n"
        "df = load_unemployment_rate\n"
        "summary = {\n"
        "    'row_count': len(df),\n"
        "    'average': round(float(df['value'].mean()), 3),\n"
        "    'highest_date': str(df.loc[df['value'].idxmax(), 'date'].date()),\n"
        "    'highest_value': float(df['value'].max()),\n"
        "}\n"
        "save('unrate_summary', summary)\n"
        "print(summary)\n"
        "Final answer must include row_count, average, highest_date, and "
        "highest_value."
    )

    assert isinstance(result, str) and result

    df = agent.cache.get("load_unemployment_rate")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 12
    assert round(float(df["value"].mean()), 3) == 4.033
    highest = df.loc[df["value"].idxmax()]
    assert str(highest["date"].date()) == "2024-07-01"
    assert float(highest["value"]) == 4.3


def test_openai_explicit_harness_real_data_planner_and_subagent(tmp_path):
    _require_key()
    cache = SessionCache(sample_size=4)
    planner = Planner()
    base_tools = build_base_tools(cache)
    tools = base_tools + planner.make_tool_specs()

    subagent_spec = make_subagent_spec(
        adapter_factory=lambda: _smoke_adapter(max_tokens=512),
        parent_tools=base_tools,
        parent_cache=cache,
        make_sub_tools=build_base_tools,
    )
    tools.append(subagent_spec)

    harness = Harness(
        adapter=_smoke_adapter(max_tokens=1024),
        system=(
            "You are validating explicit data-harness wiring. Use the planner "
            "for a short checklist, load macro_data, use Python for aggregate "
            "calculations, and use a subagent for the highest-month check."
        ),
        tools=tools,
        max_turns=12,
        cache=cache,
    )

    def _planner_reminder(event: BeforeTurn) -> Reminder | None:
        text = planner.reminder_hook(event.turn, event.max_turns)
        return Reminder(text) if text else None

    harness.on(BeforeTurn, _planner_reminder)

    result = harness.run(
        "Add a plan item for loading UNRATE and another for analysing it. "
        "Load macro_data, load the UNRATE sample, compute the average value in "
        "Python, then ask a subagent with the cached data handle to identify the "
        "highest unemployment month. Final answer must include the average and "
        "highest month/value."
    )

    assert isinstance(result, str) and result
    assert len(planner._items) >= 2

    df = cache.get("load_unemployment_rate")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 12
    assert round(float(df["value"].mean()), 3) == 4.033
    highest = df.loc[df["value"].idxmax()]
    assert str(highest["date"].date()) == "2024-07-01"
    assert float(highest["value"]) == 4.3

    tool_names = [block.tool_name for block in _tool_use_blocks(harness)]
    assert "planner__add" in tool_names
    assert "subagent" in tool_names
    assert "load_unemployment_rate" in cache.storage_metadata()


def test_openai_disk_backed_cache_keeps_raw_data_out_of_messages(tmp_path):
    _require_key()
    cache = SessionCache(sample_size=3, storage_dir=tmp_path / "cache", hot_limit=1)
    tools = build_base_tools(cache)
    harness = Harness(
        adapter=_smoke_adapter(max_tokens=768),
        system=(
            "Use macro_data and python_interpreter. Do not paste raw tables; "
            "summarise the cached handles."
        ),
        tools=tools,
        max_turns=8,
        cache=cache,
    )

    result = harness.run(
        "Load macro_data, load UNRATE, save a separate Python summary dict named "
        "unrate_summary with row_count, average, and max_value, then answer."
    )

    assert "unrate_summary" in cache.handle_names()
    assert any(meta["location"] == "disk" for meta in cache.storage_metadata().values())
    assert all("path" not in meta for meta in cache.storage_metadata().values())
    assert "12" in result

    message_text = _all_text_from_messages(harness)
    assert "2024-01-01,UNRATE,3.7" not in message_text
    assert len(message_text) < 30_000


def test_openai_tool_error_is_reported_and_recorded(tmp_path):
    _require_key()

    def failing_tool() -> str:
        raise RuntimeError("intentional smoke failure")

    bad_tool = ToolSpec(
        name="bad_tool",
        description="A tool that always raises RuntimeError.",
        input_schema={"type": "object", "properties": {}},
        handler=failing_tool,
    )
    harness = Harness(
        adapter=_smoke_adapter(max_tokens=256),
        system="Call bad_tool exactly once, then explain that it failed.",
        tools=[bad_tool],
        max_turns=4,
    )

    result = harness.run("Call bad_tool once and report the error.")

    assert "fail" in result.lower() or "error" in result.lower()
    tool_results = _tool_result_blocks(harness)
    assert any(block.is_error for block in tool_results)
    assert any(t.tool_error_count > 0 for t in _turn_entries(harness))


# ---------------------------------------------------------------------------
# PLAN_v5: RunResult typed return surface
# ---------------------------------------------------------------------------


def test_run_result_typed_return(tmp_path):
    _require_key()
    agent = Agent(
        adapter=_smoke_adapter(max_tokens=64),
        system="Answer concisely.",
        max_turns=2,
        run_dir=str(tmp_path),
    )

    result = agent.run_result("What is 6 times 7? Reply with only the number.")

    assert isinstance(result, RunResult)
    assert result.status == "success"
    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens > 0
    assert result.stop_reason == StopReason.END_TURN
    assert result.run_id is not None
    assert "42" in result.text


# ---------------------------------------------------------------------------
# PLAN_v5: AgentSession multi-turn conversation
# ---------------------------------------------------------------------------


def test_agent_session_multiturn(tmp_path):
    _require_key()
    agent = Agent(
        adapter=_smoke_adapter(max_tokens=128),
        system=(
            "You are a helpful assistant with a good memory. "
            "When asked what the user told you, repeat it exactly."
        ),
        max_turns=3,
        run_dir=str(tmp_path),
    )

    session = agent.session()
    r1 = session.ask_result("My favourite number is 7. Please remember that.")
    assert isinstance(r1, RunResult)
    assert r1.status == "success"
    assert r1.run_id is not None

    r2 = session.ask_result("What is my favourite number?")
    assert isinstance(r2, RunResult)
    assert r2.status == "success"
    assert "7" in r2.text

    assert session.turns == r1.turns + r2.turns
    assert session.last_result is r2
    assert session.id is not None


# ---------------------------------------------------------------------------
# PLAN_v5: per-turn accounting fields (visible_tools, tool_error_count)
# ---------------------------------------------------------------------------


def test_turn_entry_records_visible_tools_and_error_count(tmp_path):
    _require_key()

    get_pi = ToolSpec(
        name="get_pi",
        description="Returns the mathematical constant pi as a string.",
        input_schema={"type": "object", "properties": {}},
        handler=lambda: "3.14159265",
        annotations=ToolAnnotations(title="Get Pi", read_only=True),
    )
    harness = Harness(
        adapter=_smoke_adapter(max_tokens=128),
        system="Use get_pi once when asked about pi, then answer.",
        tools=[get_pi],
        max_turns=4,
    )

    harness.run("What is pi? Use the get_pi tool, then tell me the value.")

    turns = _turn_entries(harness)
    assert any("get_pi" in t.visible_tools for t in turns)
    assert all(t.tool_error_count == 0 for t in turns)


# ---------------------------------------------------------------------------
# PLAN_v5: RunResult.cache_storage returns typed CacheStorageInfo
# ---------------------------------------------------------------------------


def test_cache_storage_info_live(tmp_path):
    _require_key()
    agent = Agent(
        adapter=_smoke_adapter(max_tokens=256),
        system=(
            "Use python_interpreter for calculations. "
            "When given exact Python code, execute it unchanged."
        ),
        max_turns=4,
        run_dir=str(tmp_path),
    )

    result = agent.run_result(
        "Call python_interpreter with this exact code:\n"
        "save('answer', 42)\n"
        "print(42)\n"
        "Then reply with the number."
    )

    assert result.status == "success"
    assert "answer" in result.cache_storage, (
        f"'answer' not in cache_storage. Keys: {list(result.cache_storage.keys())}"
    )
    info = result.cache_storage["answer"]
    assert isinstance(info, CacheStorageInfo)
    assert info.location == "memory"
    assert info.storage_type == "memory"


# --- v0.5 entry points (Tiers 1-3) -----------------------------------------
def _smoke_model() -> str:
    return _smoke_model_id()


def _sales_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "month": ["Jan", "Feb", "Mar", "Apr"],
            "revenue": [120, 150, 90, 200],
            "region": ["NA", "NA", "EU", "EU"],
        }
    )


def test_live_ask_returns_structured_value(tmp_path):
    _require_key()
    from data_harness import ask

    result = ask(
        _sales_frame(),
        "What is the total revenue? Use answer() with the number.",
        model=_smoke_model(),
        run_dir=str(tmp_path),
    )
    assert result.status == "success"
    assert int(result.value) == 560


def test_live_ask_renders_chart(tmp_path):
    _require_key()
    from data_harness import ask

    result = ask(
        _sales_frame(),
        "Make a bar chart of revenue by month using matplotlib.",
        model=_smoke_model(),
        run_dir=str(tmp_path),
    )
    assert result.charts, "expected at least one chart artefact"
    assert result.charts[0].read_bytes()[:4] == b"\x89PNG"


def test_live_sql_over_dataframe(tmp_path):
    _require_key()
    from data_harness import ask

    result = ask(
        _sales_frame(),
        "Use sql_query to compute SUM(revenue) grouped by region, then report the "
        "region with the highest total.",
        model=_smoke_model(),
        run_dir=str(tmp_path),
    )
    assert result.status == "success"
    assert "query_result" in result.cache_snapshots
    assert "EU" in result.text  # EU = 90 + 200 = 290 is the top region


def test_live_replay_cache_skips_model(tmp_path):
    _require_key()
    from data_harness import Agent, ExecutionCache

    store = ExecutionCache()
    a1 = Agent.from_dataframe(
        _sales_frame(), model=_smoke_model(), run_dir=str(tmp_path)
    ).enable_cache(store)
    r1 = a1.run_result("Total revenue? Use answer().")
    assert r1.turns > 0

    a2 = Agent.from_dataframe(
        _sales_frame(), model=_smoke_model(), run_dir=str(tmp_path)
    ).enable_cache(store)
    r2 = a2.run_result("Total revenue? Use answer().")
    assert r2.turns == 0  # served from the replay cache, no model call
    assert int(r2.value) == int(r1.value)


# --- OpenRouter cross-provider tests ---------------------------------------
@pytest.mark.parametrize(
    "model",
    [
        "openai/gpt-4o-mini",
        "anthropic/claude-haiku-4.5",
        "deepseek/deepseek-v4-flash",
    ],
)
def test_live_openrouter_cross_provider(tmp_path, model):
    """One key, many providers: the same agent works across OpenAI, Anthropic,
    and DeepSeek via OpenRouter.

    Asserts the correct total (560) appears in either the structured value or the
    prose — models vary in how reliably they call answer().
    """
    _require_key()
    from data_harness import ask

    result = ask(
        _sales_frame(),
        "What is the total revenue? Use answer() with the number.",
        model=model,
        run_dir=str(tmp_path),
    )
    assert result.status == "success", result.error
    assert result.value == 560 or "560" in result.text


def test_live_wtq_benchmark_runs(tmp_path):
    """The WikiTableQuestions public benchmark loads and grades end-to-end."""
    _require_key()
    from data_harness.eval import evaluate, load_wikitablequestions

    cases = load_wikitablequestions(limit=5, seed=1)
    assert len(cases) == 5
    report = evaluate(cases, model=_smoke_model_id(), run_dir=str(tmp_path))
    assert len(report.results) == 5
    assert all(r.status == "success" for r in report.results)
    # JSON report is well-formed for tracking
    assert report.to_dict()["n_runs"] == 5
