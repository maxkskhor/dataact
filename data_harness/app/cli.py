"""Command-line interface: ask questions about data files from the shell.

    dh "What was total revenue?" sales.csv
    dh "Join these and find the top region" orders.csv customers.csv
    cat sales.csv | dh "median order amount" --json

Installed as both ``dh`` (short) and ``data-harness``. Resolves a provider from
the environment (or ``--model``), runs the agent, and prints the answer (and any
chart paths). Use ``--json`` for machine-readable output.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from data_harness.app.quickstart import ask


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ask natural-language questions about your data files.",
    )
    parser.add_argument("question", help="The question to ask.")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Data file(s): .csv/.parquet/.json/.xlsx. Omit to read CSV from stdin.",
    )
    parser.add_argument("--model", default=None, help="Model id (routes by name).")
    parser.add_argument("--no-sql", action="store_true", help="Disable the SQL tool.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    parser.add_argument(
        "--run-dir", default=None, help="Directory for chart artefacts."
    )
    return parser


def _jsonable(value: Any) -> Any:
    try:
        import pandas as pd

        if isinstance(value, pd.DataFrame):
            return value.to_dict(orient="records")
    except ImportError:
        pass
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def _result_dict(result) -> dict:
    return {
        "status": result.status,
        "text": result.text,
        "value": None if result.value is None else _jsonable(result.value),
        "charts": [c.path for c in result.charts],
        "turns": result.turns,
        "usage": {
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
        },
        "error": result.error,
    }


def _print_human(result) -> None:
    if result.text:
        print(result.text)
    if result.value is not None and not _is_dataframe(result.value):
        print(f"\nvalue: {result.value!r}")
    for chart in result.charts:
        print(f"chart: {chart.path}")
    if result.status != "success":
        print(f"[{result.status}] {result.error or ''}", file=sys.stderr)


def _is_dataframe(value: Any) -> bool:
    try:
        import pandas as pd

        return isinstance(value, pd.DataFrame)
    except ImportError:
        return False


def _load_data(paths: list[str]) -> Any:
    if paths:
        return paths  # to_handles (inside ask) reads each file into a handle
    if sys.stdin.isatty():
        return None
    import pandas as pd

    return pd.read_csv(sys.stdin)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    data = _load_data(args.paths)
    if data is None:
        print("error: provide a data file, or pipe CSV via stdin", file=sys.stderr)
        return 2

    result = ask(
        data,
        args.question,
        model=args.model,
        sql=False if args.no_sql else None,
        run_dir=args.run_dir,
    )

    if args.json:
        print(json.dumps(_result_dict(result), indent=2, default=str))
    else:
        _print_human(result)
    return 0 if result.status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
