"""Live end-to-end demo of the v0.5 entry points.

Runs real model calls on a cheap model, so it needs an API key (ANTHROPIC or
OPENAI) and spends a few cents. Override the model with --model.

    uv run python examples/live_demo.py
    uv run python examples/live_demo.py --model claude-haiku-4-5-20251001
"""

from __future__ import annotations

import argparse

import pandas as pd
from dotenv import load_dotenv

from data_harness import Chat, ask

SALES = pd.DataFrame(
    {
        "month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
        "revenue": [120, 150, 90, 200, 175, 230],
        "region": ["NA", "NA", "EU", "EU", "APAC", "APAC"],
    }
)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    print(f"== Tier 1: ask() with a structured answer ({args.model}) ==")
    r1 = ask(
        SALES, "What is the total revenue? Reply with the number.", model=args.model
    )
    print("value:", r1.value)
    print("text :", r1.text.strip()[:200])

    print("\n== Tier 1: ask() that renders a chart ==")
    r2 = ask(SALES, "Plot total revenue by region as a bar chart.", model=args.model)
    print("charts:", [(c.title, c.path) for c in r2.charts])

    print("\n== Tier 2: SQL via DuckDB over the DataFrame ==")
    r3 = ask(
        SALES,
        "Use sql_query to get total revenue per region, then report the top region.",
        model=args.model,
    )
    print("text :", r3.text.strip()[:200])
    print("result handles:", list(r3.cache_snapshots))

    print("\n== Multi-turn Chat ==")
    chat = Chat(SALES, model=args.model)
    print("q1:", chat.ask("Which month had the highest revenue?").text.strip()[:150])
    print("q2:", chat.ask("And the lowest?").text.strip()[:150])

    print("\nDone. Charts written under ./runs/charts/.")


if __name__ == "__main__":
    main()
