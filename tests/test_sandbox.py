"""Tier 3: subprocess sandbox isolation."""

from __future__ import annotations

import pandas as pd
import pytest

from data_harness import Agent
from data_harness.data.cache import SessionCache
from data_harness.data.tools.interpreter import PythonInterpreterError
from data_harness.data.tools.sandbox import SubprocessPythonInterpreter
from data_harness.llm.testing import FakeAdapter


def _interp(tmp_path, *, timeout=30, **kw) -> SubprocessPythonInterpreter:
    cache = SessionCache()
    cache.put("df", pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    return SubprocessPythonInterpreter(
        cache=cache, artifacts_dir=str(tmp_path), timeout=timeout, **kw
    )


def test_sandbox_computes_and_saves(tmp_path):
    interp = _interp(tmp_path)
    out = interp.run("total = int(df['a'].sum())\nsave('total', total)\nprint(total)")
    assert "6" in out
    assert interp._cache.get("total") == 6


def test_sandbox_records_answer(tmp_path):
    interp = _interp(tmp_path)
    interp.run("answer(int(df['b'].sum()))")
    assert interp._cache.get_answer() == 15


def test_sandbox_captures_chart(tmp_path):
    interp = _interp(tmp_path)
    interp.run(
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots()\n"
        "ax.plot(df['a'])\n"
        "ax.set_title('sb')\n"
    )
    charts = interp._cache.list_charts()
    assert len(charts) == 1
    assert charts[0].title == "sb"
    assert charts[0].read_bytes()[:4] == b"\x89PNG"


def test_sandbox_blocks_forbidden_import(tmp_path):
    interp = _interp(tmp_path)
    with pytest.raises(PythonInterpreterError, match="Import not allowed"):
        interp.run("import os")


def test_sandbox_propagates_runtime_error(tmp_path):
    interp = _interp(tmp_path)
    with pytest.raises(PythonInterpreterError, match="ZeroDivision"):
        interp.run("x = 1 / 0")


def test_sandbox_timeout(tmp_path):
    interp = _interp(tmp_path, timeout=2, cpu_seconds=1)
    with pytest.raises(PythonInterpreterError):
        # busy loop exceeds both the CPU and wall-clock limit
        interp.run("while True:\n    pass")


def test_sandbox_handles_roundtrip_dataframe(tmp_path):
    interp = _interp(tmp_path)
    interp.run("save('doubled', df * 2)")
    doubled = interp._cache.get("doubled")
    assert int(doubled.iloc[0]["a"]) == 2


def test_agent_subprocess_execution(tmp_path):
    code = "answer(int(df['a'].sum()))"
    adapter = FakeAdapter(
        [
            FakeAdapter.tool_use("t1", "python_interpreter", {"code": code}),
            FakeAdapter.text("sum is 6"),
        ]
    )
    agent = Agent.from_dataframe(
        pd.DataFrame({"a": [1, 2, 3]}),
        adapter=adapter,
        run_dir=str(tmp_path),
        execution="subprocess",
        sandbox_options={"timeout": 30},
    )
    res = agent.run_result("sum of a")
    assert res.status == "success"
    assert res.value == 6
