"""Tier 1: ask()/Chat/SmartFrame entry points, provider resolution, IO."""

from __future__ import annotations

import pandas as pd
import pytest

from data_harness import Chat, SmartFrame, ask
from data_harness.app.quickstart import resolve_adapter
from data_harness.data.io import load_dataframe, sanitise_handle, to_handles
from data_harness.llm.testing import FakeAdapter


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"month": ["Jan", "Feb", "Mar"], "revenue": [100, 140, 90]})


def _answer_adapter(code: str, final: str) -> FakeAdapter:
    return FakeAdapter(
        [
            FakeAdapter.tool_use("t1", "python_interpreter", {"code": code}),
            FakeAdapter.text(final),
        ]
    )


# --- provider resolution ---------------------------------------------------
def test_resolve_adapter_routes_by_model_name(monkeypatch):
    from data_harness.llm.providers.anthropic import AnthropicAdapter
    from data_harness.llm.providers.openai import OpenAIAdapter

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert isinstance(resolve_adapter("gpt-4o-mini"), OpenAIAdapter)
    assert isinstance(resolve_adapter("o3-mini"), OpenAIAdapter)
    assert isinstance(resolve_adapter("claude-sonnet-4-6"), AnthropicAdapter)


def test_resolve_adapter_prefers_anthropic_env(monkeypatch):
    from data_harness.llm.providers.anthropic import AnthropicAdapter

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert isinstance(resolve_adapter(), AnthropicAdapter)


def test_resolve_adapter_falls_back_to_openai(monkeypatch):
    from data_harness.llm.providers.openai import OpenAIAdapter

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert isinstance(resolve_adapter(), OpenAIAdapter)


def test_resolve_adapter_raises_without_keys(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="No provider configured"):
        resolve_adapter()


def test_resolve_adapter_routes_slash_model_to_openrouter(monkeypatch):
    from data_harness.llm.providers.openai import OPENROUTER_BASE_URL, OpenRouterAdapter

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    adapter = resolve_adapter("anthropic/claude-3.5-sonnet")
    assert isinstance(adapter, OpenRouterAdapter)
    assert str(adapter._client.base_url).rstrip("/") == OPENROUTER_BASE_URL


def test_resolve_adapter_falls_back_to_openrouter(monkeypatch):
    from data_harness.llm.providers.openai import OpenRouterAdapter

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    assert isinstance(resolve_adapter(), OpenRouterAdapter)


def test_openrouter_adapter_uses_env_key(monkeypatch):
    from data_harness.llm.providers.openai import OpenRouterAdapter

    monkeypatch.setenv("OPENROUTER_API_KEY", "router-secret")
    adapter = OpenRouterAdapter(model="openai/gpt-4o-mini")
    assert adapter._client.api_key == "router-secret"


def test_resolve_adapter_routes_deepseek_direct(monkeypatch):
    from data_harness.llm.providers.openai import DEEPSEEK_BASE_URL, DeepSeekAdapter

    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
    adapter = resolve_adapter("deepseek-chat")
    assert isinstance(adapter, DeepSeekAdapter)
    assert str(adapter._client.base_url).rstrip("/") == DEEPSEEK_BASE_URL


def test_resolve_adapter_deepseek_slash_goes_to_openrouter(monkeypatch):
    from data_harness.llm.providers.openai import OpenRouterAdapter

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    # a slash means OpenRouter, even for deepseek
    assert isinstance(resolve_adapter("deepseek/deepseek-chat"), OpenRouterAdapter)


def test_resolve_adapter_falls_back_to_deepseek(monkeypatch):
    from data_harness.llm.providers.openai import DeepSeekAdapter

    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
    assert isinstance(resolve_adapter(), DeepSeekAdapter)


# --- ask() -----------------------------------------------------------------
def test_ask_returns_structured_value(tmp_path):
    code = "total = int(df['revenue'].sum())\nanswer(total)"
    res = ask(
        _frame(),
        "total revenue",
        adapter=_answer_adapter(code, "Total is 330."),
        run_dir=str(tmp_path),
    )
    assert res.status == "success"
    assert res.value == 330
    assert res.text == "Total is 330."


def test_ask_captures_chart(tmp_path):
    code = (
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots()\n"
        "ax.bar(df['month'], df['revenue'])\n"
        "ax.set_title('Revenue')\n"
    )
    res = ask(
        _frame(),
        "plot revenue",
        adapter=_answer_adapter(code, "Here is the chart."),
        run_dir=str(tmp_path),
    )
    assert len(res.charts) == 1
    chart = res.charts[0]
    assert chart.title == "Revenue"
    assert chart.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_ask_preamble_lists_handles(tmp_path):
    adapter = FakeAdapter([FakeAdapter.text("ok")])
    ask(_frame(), "hello", adapter=adapter, run_dir=str(tmp_path))
    sent = adapter.calls[0]["messages"][0].content[0].text
    assert "Available data handles" in sent
    assert "df:" in sent


def test_ask_accepts_csv_path(tmp_path):
    path = tmp_path / "sales.csv"
    _frame().to_csv(path, index=False)
    adapter = FakeAdapter([FakeAdapter.text("ok")])
    ask(str(path), "hi", adapter=adapter, run_dir=str(tmp_path))
    sent = adapter.calls[0]["messages"][0].content[0].text
    assert "sales:" in sent  # handle derived from filename stem


# --- Chat / SmartFrame -----------------------------------------------------
def test_chat_is_multiturn(tmp_path):
    adapter = FakeAdapter([FakeAdapter.text("first"), FakeAdapter.text("second")])
    chat = Chat(_frame(), adapter=adapter, run_dir=str(tmp_path))
    r1 = chat.ask("q1")
    r2 = chat.ask("q2")
    assert r1.text == "first"
    assert r2.text == "second"
    # second turn shares the same session id
    assert r1.session_id == r2.session_id
    # preamble only on the first turn
    assert "Available data handles" in adapter.calls[0]["messages"][0].content[0].text


def test_smartframe_chat(tmp_path):
    adapter = FakeAdapter([FakeAdapter.text("done")])
    res = SmartFrame(_frame(), adapter=adapter, run_dir=str(tmp_path)).chat("q")
    assert res.text == "done"


# --- IO helpers ------------------------------------------------------------
def test_to_handles_dataframe():
    assert set(to_handles(_frame())) == {"df"}


def test_to_handles_dict():
    handles = to_handles({"sales 2024": _frame(), "costs": _frame()})
    assert set(handles) == {"sales_2024", "costs"}


def test_sanitise_handle():
    assert sanitise_handle("2023 sales!") == "d_2023_sales"
    assert sanitise_handle("class") == "class_"


def test_load_dataframe_roundtrip(tmp_path):
    path = tmp_path / "x.csv"
    _frame().to_csv(path, index=False)
    loaded = load_dataframe(path)
    assert list(loaded.columns) == ["month", "revenue"]


def test_load_dataframe_unsupported(tmp_path):
    bad = tmp_path / "x.zzz"
    bad.write_text("nope")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_dataframe(bad)


# --- pandas accessor + notebook magic --------------------------------------
def test_pandas_chat_accessor(tmp_path):
    import data_harness.app.pandas  # noqa: F401  registers the accessor

    df = _frame()
    adapter = FakeAdapter([FakeAdapter.text("via accessor")])
    res = df.chat("q", adapter=adapter, run_dir=str(tmp_path))
    assert res.text == "via accessor"


def test_notebook_magic_class_builds():
    from data_harness.app.notebook import _load_magics_class

    cls = _load_magics_class()
    assert hasattr(cls, "ask")


# --- answer-reliability finalize -------------------------------------------
def test_ask_finalizes_when_answer_missing(tmp_path):
    # run 1: prose only, no answer() -> value None; finalize: tool call records it
    adapter = FakeAdapter(
        [
            FakeAdapter.text("The total is 6."),  # initial run, no answer()
            FakeAdapter.tool_use("f", "python_interpreter", {"code": "answer(6)"}),
            FakeAdapter.text("Recorded."),
        ]
    )
    res = ask(_frame(), "total revenue", adapter=adapter, run_dir=str(tmp_path))
    assert res.value == 6  # recovered by the finalize turn
    assert res.turns == 3  # 1 initial + 2 finalize


def test_ask_no_finalize_when_answer_present(tmp_path):
    # exactly two responses: a third would only be popped if finalize fired
    adapter = FakeAdapter(
        [
            FakeAdapter.tool_use("t", "python_interpreter", {"code": "answer(5)"}),
            FakeAdapter.text("done"),
        ]
    )
    res = ask(_frame(), "q", adapter=adapter, run_dir=str(tmp_path))
    assert res.value == 5
    assert res.turns == 2  # no finalize turn


def test_require_answer_false_skips_finalize(tmp_path):
    adapter = FakeAdapter([FakeAdapter.text("no structured answer here")])
    res = ask(
        _frame(), "q", adapter=adapter, require_answer=False, run_dir=str(tmp_path)
    )
    assert res.value is None
    assert res.turns == 1


def test_chat_finalizes(tmp_path):
    adapter = FakeAdapter(
        [
            FakeAdapter.text("It is 6."),
            FakeAdapter.tool_use("f", "python_interpreter", {"code": "answer(6)"}),
            FakeAdapter.text("Recorded."),
        ]
    )
    chat = Chat(_frame(), adapter=adapter, require_answer=True, run_dir=str(tmp_path))
    res = chat.ask("total revenue")
    assert res.value == 6


def test_finalize_skipped_when_chart_produced(tmp_path):
    # a chart is the deliverable -> no finalize turn (only 2 responses scripted)
    code = (
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots()\n"
        "ax.bar(df['month'], df['revenue'])\n"
    )
    adapter = _answer_adapter(code, "Here is the chart.")
    res = ask(_frame(), "plot revenue", adapter=adapter, run_dir=str(tmp_path))
    assert res.charts and res.value is None
    assert res.turns == 2  # no finalize


def test_finalize_skipped_on_refusal(tmp_path):
    # a refusal must not be turned into a fabricated value (only 1 response)
    adapter = FakeAdapter(
        [FakeAdapter.text("I cannot answer: there is no churn column in the data.")]
    )
    res = ask(
        _frame(), "what is the churn rate?", adapter=adapter, run_dir=str(tmp_path)
    )
    assert res.value is None
    assert res.turns == 1  # no finalize
