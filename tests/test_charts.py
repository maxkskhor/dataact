"""Tier 1: chart capture, ChartArtifact, cache integration, RunResult display."""

from __future__ import annotations

from data_harness.core.artifacts import ChartArtifact
from data_harness.core.result import RunResult, Usage
from data_harness.data.cache import SessionCache
from data_harness.data.tools.interpreter import PythonInterpreter


def _interp(tmp_path) -> PythonInterpreter:
    return PythonInterpreter(cache=SessionCache(), artifacts_dir=str(tmp_path))


def test_interpreter_captures_matplotlib_figure(tmp_path):
    interp = _interp(tmp_path)
    out = interp.run(
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots()\n"
        "ax.plot([1, 2, 3])\n"
        "ax.set_title('demo')\n"
    )
    assert "Rendered chart saved to handle" in out
    charts = interp._cache.list_charts()
    assert len(charts) == 1
    assert charts[0].title == "demo"
    assert charts[0].read_bytes()[:4] == b"\x89PNG"


def test_no_chart_when_no_plot(tmp_path):
    interp = _interp(tmp_path)
    interp.run("x = 1 + 1\nprint(x)")
    assert interp._cache.list_charts() == []


def test_answer_records_value(tmp_path):
    interp = _interp(tmp_path)
    out = interp.run("answer(42)")
    assert interp._cache.get_answer() == 42
    assert "Recorded answer" in out


def test_answer_returns_dataframe(tmp_path):
    cache = SessionCache()
    import pandas as pd

    cache.put("df", pd.DataFrame({"a": [1, 2]}))
    interp = PythonInterpreter(cache=cache, artifacts_dir=str(tmp_path))
    interp.run("answer(df.head())")
    ans = cache.get_answer()
    assert isinstance(ans, pd.DataFrame)


def test_chart_artifact_snapshot_has_no_bytes():
    art = ChartArtifact(path="/tmp/x.png", format="png", title="T")
    snap = art.snapshot()
    assert "chart" in snap and "png" in snap
    assert "base64" not in snap


def test_cache_chart_snapshot_is_payload_free(tmp_path):
    cache = SessionCache()
    p = tmp_path / "c.png"
    p.write_bytes(b"\x89PNG fake")
    handle = cache.put("chart", ChartArtifact(path=str(p)))
    snap = cache.snapshot(handle)
    assert "chart" in snap
    assert "PNG" not in snap  # raw bytes never enter the snapshot


def test_runresult_repr_html_includes_chart(tmp_path):
    p = tmp_path / "c.png"
    p.write_bytes(b"\x89PNG fake-bytes")
    res = RunResult(
        text="done",
        status="success",
        turns=1,
        stop_reason=None,
        usage=Usage(),
        value=7,
        charts=[ChartArtifact(path=str(p))],
    )
    html = res._repr_html_()
    assert "done" in html
    assert "<code>7</code>" in html
    assert "data:image/png;base64" in html


def test_runresult_repr_omits_value_and_charts_from_console():
    res = RunResult(
        text="x",
        status="success",
        turns=1,
        stop_reason=None,
        usage=Usage(),
        value="secret-big-object",
    )
    assert "secret-big-object" not in repr(res)
