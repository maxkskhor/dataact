"""Tier 3 evidence: the code-replay cache turns a repeat question into a
zero-token, zero-turn replay.

This uses a scripted FakeAdapter so it runs without an API key and is fully
deterministic. It prints a before/after table proving the second run never
touches the model.

    uv run python examples/cache_benchmark.py
"""

from __future__ import annotations

import dataclasses
import time

import pandas as pd

from data_harness import Agent, ExecutionCache
from data_harness.llm.testing import FakeAdapter

SALES = pd.DataFrame(
    {"month": ["Jan", "Feb", "Mar", "Apr"], "revenue": [120, 150, 90, 200]}
)


def _with_tokens(response, input_tokens, output_tokens):
    return dataclasses.replace(
        response, input_tokens=input_tokens, output_tokens=output_tokens
    )


def _scripted_adapter() -> FakeAdapter:
    # Realistic token counts so the cache hit's savings are visible.
    return FakeAdapter(
        [
            _with_tokens(
                FakeAdapter.tool_use(
                    "t1",
                    "python_interpreter",
                    {"code": "answer(int(sales['revenue'].sum()))"},
                ),
                input_tokens=1200,
                output_tokens=40,
            ),
            _with_tokens(
                FakeAdapter.text("Total revenue across the four months is 560."),
                input_tokens=1300,
                output_tokens=25,
            ),
        ]
    )


def main() -> None:
    store = ExecutionCache()
    question = "What is the total revenue?"

    # First run: full agent loop, model invoked.
    a1 = Agent.from_dataframe(
        {"sales": SALES}, adapter=_scripted_adapter(), run_dir="./runs"
    ).enable_cache(store)
    t0 = time.perf_counter()
    r1 = a1.run_result(question)
    miss_ms = (time.perf_counter() - t0) * 1000

    # Second run: fresh agent + an EMPTY adapter that would error if called.
    a2 = Agent.from_dataframe(
        {"sales": SALES}, adapter=FakeAdapter([]), run_dir="./runs"
    ).enable_cache(store)
    t0 = time.perf_counter()
    r2 = a2.run_result(question)
    hit_ms = (time.perf_counter() - t0) * 1000

    print("Code-replay cache benchmark")
    print("-" * 52)
    print(f"{'':14}{'MISS (run 1)':>18}{'HIT (run 2)':>18}")
    print(f"{'answer':14}{str(r1.value):>18}{str(r2.value):>18}")
    print(f"{'turns':14}{r1.turns:>18}{r2.turns:>18}")
    print(f"{'input tokens':14}{r1.usage.input_tokens:>18}{r2.usage.input_tokens:>18}")
    print(f"{'model called':14}{'yes':>18}{'no':>18}")
    print(f"{'latency (ms)':14}{miss_ms:>18.2f}{hit_ms:>18.2f}")
    print("-" * 52)
    assert r2.value == r1.value, "replay must reproduce the answer"
    assert r2.turns == 0 and r2.usage.input_tokens == 0, "hit must be free"
    print("OK: identical answer, zero turns and zero tokens on the cache hit.")


if __name__ == "__main__":
    main()
