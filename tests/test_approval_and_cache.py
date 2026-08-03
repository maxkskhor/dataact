"""Tier 3: approval gate, code-only dry run, and the code-replay cache."""

from __future__ import annotations

import pandas as pd

from data_harness import Agent, ExecutionCache
from data_harness.llm.testing import FakeAdapter


def _df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3]})


def _run_adapter() -> FakeAdapter:
    return FakeAdapter(
        [
            FakeAdapter.tool_use(
                "t1", "python_interpreter", {"code": "answer(int(df['a'].sum()))"}
            ),
            FakeAdapter.text("the sum is 6"),
        ]
    )


# --- approval gate ---------------------------------------------------------
def test_approval_gate_blocks_execution(tmp_path):
    seen = []

    def deny(code: str) -> bool:
        seen.append(code)
        return False

    agent = Agent.from_dataframe(
        _df(), adapter=_run_adapter(), run_dir=str(tmp_path), on_code=deny
    )
    res = agent.run_result("sum")
    # code was offered to the gate but never executed
    assert seen and "answer(" in seen[0]
    assert res.value is None  # answer() never ran


def test_approval_gate_allows_execution(tmp_path):
    agent = Agent.from_dataframe(
        _df(), adapter=_run_adapter(), run_dir=str(tmp_path), on_code=lambda c: True
    )
    res = agent.run_result("sum")
    assert res.value == 6


def test_code_only_dry_run(tmp_path):
    agent = Agent.from_dataframe(
        _df(), adapter=_run_adapter(), run_dir=str(tmp_path), code_only=True
    )
    res = agent.run_result("sum")
    assert res.value is None
    # the interpreter result echoed the code rather than running it
    last_user = agent.last_harness.messages[-2]
    contents = " ".join(b.content for b in last_user.content if hasattr(b, "content"))
    assert "DRY RUN" in contents


# --- code-replay cache -----------------------------------------------------
def test_cache_hit_skips_model(tmp_path):
    store = ExecutionCache()

    a1 = Agent.from_dataframe(
        _df(), adapter=_run_adapter(), run_dir=str(tmp_path)
    ).enable_cache(store)
    r1 = a1.run_result("total of a")
    assert r1.value == 6
    assert r1.turns == 2
    assert len(store) == 1

    # fresh agent, fresh adapter that would RAISE if called
    adapter2 = FakeAdapter([])  # empty -> .pop(0) would IndexError if used
    a2 = Agent.from_dataframe(
        _df(), adapter=adapter2, run_dir=str(tmp_path)
    ).enable_cache(store)
    r2 = a2.run_result("total of a")
    assert r2.value == 6  # replayed
    assert r2.turns == 0  # no model turns
    assert r2.usage.input_tokens == 0
    assert adapter2.calls == []  # model never invoked


def test_cache_replays_on_new_data(tmp_path):
    store = ExecutionCache()
    Agent.from_dataframe(
        _df(), adapter=_run_adapter(), run_dir=str(tmp_path)
    ).enable_cache(store).run_result("total of a")

    # same schema, different values -> replay recomputes (4+5+6 = 15)
    a2 = Agent.from_dataframe(
        pd.DataFrame({"a": [4, 5, 6]}), adapter=FakeAdapter([]), run_dir=str(tmp_path)
    ).enable_cache(store)
    r2 = a2.run_result("total of a")
    assert r2.value == 15


def test_cache_persists_to_disk(tmp_path):
    path = tmp_path / "cache.json"
    Agent.from_dataframe(
        _df(), adapter=_run_adapter(), run_dir=str(tmp_path)
    ).enable_cache(str(path)).run_result("total of a")
    assert path.exists()
    reloaded = ExecutionCache(str(path))
    assert len(reloaded) == 1


def test_extract_steps_skips_errored_calls():
    from data_harness.data.exec_cache import extract_steps
    from data_harness.llm.types import (
        Message,
        TextBlock,
        ToolResultBlock,
        ToolUseBlock,
    )

    messages = [
        Message(role="user", content=[TextBlock(text="q")]),
        Message(
            role="assistant",
            content=[
                ToolUseBlock("bad", "python_interpreter", {"code": "boom"}),
                ToolUseBlock("good", "python_interpreter", {"code": "answer(1)"}),
            ],
        ),
        Message(
            role="user",
            content=[
                ToolResultBlock("bad", "NameError", is_error=True),
                ToolResultBlock("good", "Recorded answer: 1", is_error=False),
            ],
        ),
    ]
    steps = extract_steps(messages)
    assert [s["input"]["code"] for s in steps] == ["answer(1)"]


def test_cache_replay_tolerates_failing_step(tmp_path):
    # First run: the model's first interpreter call errors, then it recovers.
    adapter = FakeAdapter(
        [
            FakeAdapter.tool_use("bad", "python_interpreter", {"code": "undef_var"}),
            FakeAdapter.tool_use(
                "good", "python_interpreter", {"code": "answer(int(df['a'].sum()))"}
            ),
            FakeAdapter.text("6"),
        ]
    )
    store = ExecutionCache()
    Agent.from_dataframe(_df(), adapter=adapter, run_dir=str(tmp_path)).enable_cache(
        store
    ).run_result("total")

    # Replay must succeed and reproduce the answer.
    replayed = (
        Agent.from_dataframe(_df(), adapter=FakeAdapter([]), run_dir=str(tmp_path))
        .enable_cache(store)
        .run_result("total")
    )
    assert replayed.value == 6
    assert replayed.turns == 0
