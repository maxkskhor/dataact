"""Tests for the data-harness CLI (with `ask` monkeypatched — no live calls)."""

from __future__ import annotations

import json

import pandas as pd

from data_harness import cli
from data_harness.core.result import RunResult, Usage


def _fake_result(text="the answer is 6", value=6, charts=None) -> RunResult:
    return RunResult(
        text=text,
        status="success",
        turns=2,
        run_file=None,
        stop_reason=None,
        usage=Usage(input_tokens=10, output_tokens=5),
        value=value,
        charts=charts or [],
    )


def _csv(tmp_path) -> str:
    path = tmp_path / "sales.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(path, index=False)
    return str(path)


def test_parser_basics():
    args = cli.build_parser().parse_args(["q", "a.csv", "b.csv", "--no-sql"])
    assert args.question == "q"
    assert args.paths == ["a.csv", "b.csv"]
    assert args.no_sql is True


def test_cli_human_output(tmp_path, monkeypatch, capsys):
    seen = {}

    def fake_ask(data, question, **kwargs):
        seen["data"] = data
        seen["question"] = question
        return _fake_result()

    monkeypatch.setattr(cli, "ask", fake_ask)
    code = cli.main(["total of a", _csv(tmp_path)])

    assert code == 0
    out = capsys.readouterr().out
    assert "the answer is 6" in out and "value: 6" in out
    assert seen["question"] == "total of a"
    assert seen["data"] == [_csv(tmp_path)]  # paths passed through to ask


def test_cli_json_output(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "ask", lambda *a, **k: _fake_result(value=42))
    code = cli.main(["q", _csv(tmp_path), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "success"
    assert payload["value"] == 42
    assert payload["turns"] == 2


def test_cli_no_sql_passthrough(tmp_path, monkeypatch):
    seen = {}

    def fake_ask(data, question, **kwargs):
        seen.update(kwargs)
        return _fake_result()

    monkeypatch.setattr(cli, "ask", fake_ask)
    cli.main(["q", _csv(tmp_path), "--no-sql"])
    assert seen["sql"] is False


def test_cli_requires_data(monkeypatch, capsys):
    class _Tty:
        def isatty(self):
            return True

    monkeypatch.setattr("sys.stdin", _Tty())
    code = cli.main(["q"])
    assert code == 2
    assert "provide a data file" in capsys.readouterr().err
