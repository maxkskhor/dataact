"""Evaluation harness: graders, runner, report, suites (all offline)."""

from __future__ import annotations

import pandas as pd

from data_harness.core.artifacts import ChartArtifact
from data_harness.core.result import RunResult, Usage
from data_harness.eval import (
    EvalCase,
    EvalReport,
    all_of,
    bespoke_suite,
    chart_produced,
    contains,
    dataframe_equals,
    evaluate,
    evaluate_matrix,
    exact,
    extract_numbers,
    numeric,
    refuses,
    wtq_row_to_case,
)
from data_harness.llm.testing import FakeAdapter


def _result(text="", value=None, charts=None) -> RunResult:
    return RunResult(
        text=text,
        status="success",
        turns=1,
        stop_reason=None,
        usage=Usage(),
        value=value,
        charts=charts or [],
    )


_CASE = EvalCase("c", "q", pd.DataFrame({"a": [1]}), numeric(1))


# --- graders ---------------------------------------------------------------
def test_numeric_from_value():
    assert numeric(965)(_result(value=965), _CASE).passed


def test_numeric_from_text_with_commas():
    assert numeric(1234.5)(_result(text="The total is 1,234.5 dollars."), _CASE).passed


def test_numeric_tolerance_and_fail():
    assert numeric(30.0, tol=0.6)(_result(value=30.05), _CASE).passed
    assert not numeric(30.0, tol=0.01)(_result(value=40), _CASE).passed


def test_numeric_ignores_bool():
    # bool is not accepted as a numeric value match
    assert not numeric(1)(_result(value=True), _CASE).passed


def test_extract_numbers():
    assert extract_numbers("a 3, then 4.5 and 1,000") == [3.0, 4.5, 1000.0]


def test_contains_value_and_text():
    assert contains("APAC")(_result(text="Top region is APAC."), _CASE).passed
    assert contains(["Jun", "June"])(_result(value="June"), _CASE).passed
    assert not contains("EU")(_result(text="region NA"), _CASE).passed


def test_exact():
    assert exact("yes")(_result(value="Yes"), _CASE).passed
    assert not exact("yes")(_result(value="no"), _CASE).passed


def test_chart_produced():
    art = ChartArtifact(path="/tmp/x.png")
    assert chart_produced()(_result(charts=[art]), _CASE).passed
    assert not chart_produced()(_result(), _CASE).passed


def test_refuses():
    assert refuses()(_result(text="I cannot answer; no such column."), _CASE).passed
    assert not refuses()(_result(text="The answer is 42."), _CASE).passed


def test_all_of():
    g = all_of(contains("Eve"), contains("Ann"))
    assert g(_result(text="Eve and Ann"), _CASE).passed
    assert not g(_result(text="Eve only"), _CASE).passed


def test_dataframe_equals():
    df = pd.DataFrame({"region": ["EU"], "total": [290]})
    assert dataframe_equals(df)(_result(value=df.copy()), _CASE).passed
    assert not dataframe_equals(df)(_result(value=42), _CASE).passed


# --- runner + report -------------------------------------------------------
def _answer_adapter(*pairs) -> FakeAdapter:
    """pairs: (code, final_text) per case, in order."""
    responses = []
    for i, (code, final) in enumerate(pairs):
        responses.append(
            FakeAdapter.tool_use(f"t{i}", "python_interpreter", {"code": code})
        )
        responses.append(FakeAdapter.text(final))
    return FakeAdapter(responses)


def test_evaluate_scores_cases(tmp_path):
    cases = [
        EvalCase("good", "sum", pd.DataFrame({"a": [1, 2, 3]}), numeric(6)),
        EvalCase("bad", "sum", pd.DataFrame({"a": [1, 2, 3]}), numeric(999)),
    ]
    adapter = _answer_adapter(("answer(6)", "six"), ("answer(6)", "six"))
    report = evaluate(cases, adapter=adapter, model_label="fake", run_dir=str(tmp_path))
    assert report.accuracy() == 0.5
    assert {r.case_id: r.passed for r in report.results} == {"good": True, "bad": False}
    assert report.results[0].model == "fake"
    assert report.results[0].turns == 2


def test_evaluate_records_run_errors(tmp_path):
    # adapter with no scripted responses -> ask() raises -> recorded as error
    cases = [EvalCase("x", "q", pd.DataFrame({"a": [1]}), numeric(1))]
    report = evaluate(cases, adapter=FakeAdapter([]), run_dir=str(tmp_path))
    assert report.accuracy() == 0.0
    assert report.results[0].status == "error"


def test_evaluate_matrix(tmp_path):
    cases = [EvalCase("good", "sum", pd.DataFrame({"a": [1, 2, 3]}), numeric(6))]
    report = evaluate_matrix(
        cases,
        [
            ("m1", _answer_adapter(("answer(6)", "six"))),
            ("m2", _answer_adapter(("answer(7)", "seven"))),
        ],
        run_dir=str(tmp_path),
    )
    assert set(report.models) == {"m1", "m2"}
    by_model = {r.model: r.passed for r in report.results}
    assert by_model == {"m1": True, "m2": False}
    assert "m1" in report.leaderboard()


def test_report_aggregation():
    rows = [
        ("m", "agg", True),
        ("m", "agg", False),
        ("m", "chart", True),
    ]
    from data_harness.eval.report import CaseResult

    report = EvalReport(
        [
            CaseResult(f"c{i}", cat, model, passed, "", 1, 0, 0, 0.0, "success")
            for i, (model, cat, passed) in enumerate(rows)
        ]
    )
    assert abs(report.accuracy() - 2 / 3) < 1e-9
    assert "category" in report.by_category()
    assert len(report.failures()) == 1


def test_report_cost():
    from data_harness.eval.report import CaseResult

    rows = [
        CaseResult("c1", "agg", "m", True, "", 1, 1000, 100, 0.0, "success"),
        CaseResult("c2", "agg", "m", True, "", 1, 2000, 200, 0.0, "success"),
    ]
    report = EvalReport(rows)
    prices = {"m": (0.2, 0.8)}  # ($/Mtok prompt, $/Mtok completion)
    # (3000 * 0.2 + 300 * 0.8) / 1e6 = 840 / 1e6
    assert abs(report.total_cost(prices) - 0.00084) < 1e-12
    lb = report.leaderboard(prices)
    assert "cost ($)" in lb and "0.0008" in lb
    # no prices -> no cost column
    assert "cost ($)" not in report.leaderboard()


def test_report_to_dict_and_json():
    import json

    from data_harness.eval.report import CaseResult

    report = EvalReport(
        [
            CaseResult("c1", "agg", "m", True, "", 2, 1000, 100, 1.0, "success"),
            CaseResult("c2", "agg", "m", False, "x", 1, 500, 50, 1.0, "success"),
        ]
    )
    d = report.to_dict({"m": (0.2, 0.8)})
    assert d["n_runs"] == 2
    assert d["accuracy"] == 0.5
    assert d["models"]["m"]["passed"] == 1
    assert d["models"]["m"]["cost_usd"] > 0
    assert d["by_category"]["agg"]["m"] == 0.5
    assert len(d["results"]) == 2
    # round-trips through JSON
    assert json.loads(report.to_json())["n_runs"] == 2


# --- suites ----------------------------------------------------------------
def test_bespoke_suite_well_formed():
    cases = bespoke_suite()
    assert len(cases) >= 12
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))  # unique
    assert {"aggregation", "chart", "adversarial"} <= {c.category for c in cases}


def test_evalcase_identity_equality_with_dataframes():
    # field-wise dataclass eq would raise on DataFrame truthiness; identity is safe
    a = EvalCase("c", "q", pd.DataFrame({"x": [1]}), numeric(1))
    b = EvalCase("c", "q", pd.DataFrame({"x": [1]}), numeric(1))
    assert a != b  # identity, not field-wise
    assert a in [a]  # membership uses identity, does not raise


def test_hard_suite_well_formed():
    from data_harness.eval import ConversationCase, hard_suite

    cases = hard_suite()
    assert any(isinstance(c, ConversationCase) for c in cases)
    assert any(isinstance(c, EvalCase) for c in cases)
    cats = {c.category for c in cases}
    assert {"join", "multi_step", "stateful"} <= cats


def test_conversation_case_runs_multiturn(tmp_path):
    from data_harness.eval import ConversationCase, Turn

    adapter = FakeAdapter(
        [
            FakeAdapter.tool_use("a", "python_interpreter", {"code": "answer(5)"}),
            FakeAdapter.text("five"),
            FakeAdapter.tool_use("b", "python_interpreter", {"code": "answer(10)"}),
            FakeAdapter.text("ten"),
        ]
    )
    conv = ConversationCase(
        "c",
        pd.DataFrame({"a": [1, 2]}),
        [Turn("first?", numeric(5)), Turn("second?", numeric(10))],
    )
    report = evaluate([conv], adapter=adapter, run_dir=str(tmp_path))
    assert [r.case_id for r in report.results] == ["c#t1", "c#t2"]
    assert all(r.passed for r in report.results)
    assert report.accuracy() == 1.0


def test_large_data_suite_well_formed():
    from data_harness.eval import large_data_suite

    cases = large_data_suite()
    assert len(cases) >= 4
    # genuinely large: can't be answered from a 5-row snapshot
    assert all(len(c.data) >= 10_000 for c in cases)
    assert "snapshot_trap" in {c.category for c in cases}


def test_leaderboard_markdown_and_summary(tmp_path):
    import json

    from data_harness.eval import leaderboard_markdown, write_summary

    report = {
        "n_runs": 2,
        "accuracy": 0.75,
        "models": {
            "m1": {
                "accuracy": 1.0,
                "avg_turns": 2.0,
                "tokens": 1000,
                "cost_usd": 0.001,
            },
            "m2": {"accuracy": 0.5, "avg_turns": 3.0, "tokens": 2000},  # no cost
        },
    }
    md = leaderboard_markdown(report)
    assert "| model | accuracy" in md and "100%" in md
    assert md.index("m1") < md.index("m2")  # higher accuracy sorted first
    assert "0.0010" in md and "n/a" in md  # cost shown / missing

    results = tmp_path / "results"
    results.mkdir()
    (results / "hard_20260101t000000.json").write_text(
        json.dumps(
            {"suite": "hard", "generated_at": "2026-01-01T00:00:00Z", "report": report}
        )
    )
    path = write_summary(results)
    text = path.read_text()
    assert path.name == "SUMMARY.md"
    assert "hard" in text and "m1" in text and "100%" in text


def test_messy_suite_well_formed():
    from data_harness.eval import messy_suite

    cases = messy_suite()
    assert len(cases) == 4
    assert {"messy_parse", "messy_normalise", "messy_dates"} <= {
        c.category for c in cases
    }
    # genuinely messy: amounts are strings with currency/separators (dtype-agnostic)
    amounts = [str(v) for v in cases[0].data["amount"].tolist()]
    assert any("$" in v for v in amounts) and any("," in v for v in amounts)


def test_snapshot_trap_actually_misleads():
    from data_harness.eval.suites.large_data import _snapshot_trap

    df = _snapshot_trap()
    # the snapshot (head rows) suggests EU dominates...
    assert df.head(5)["region"].unique().tolist() == ["EU"]
    # ...but the full-data answer is NA — so reading the snapshot fails the case
    assert df.groupby("region")["amount"].sum().idxmax() == "NA"


def test_wtq_row_to_case():
    row = {
        "question": "Which country has the most golds?",
        "answers": ["France"],
        "table": {
            "header": ["country", "golds"],
            "rows": [["France", "10"], ["Spain", "3"]],
        },
    }
    case = wtq_row_to_case(7, row)
    assert case.id == "wtq-7"
    assert case.category == "wtq"
    assert list(case.data.columns) == ["country", "golds"]
    # grader matches the accepted answer in prose
    assert case.grader(_result(text="The answer is France."), case).passed
    assert not case.grader(_result(text="Spain"), case).passed
