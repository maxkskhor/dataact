# Evaluation

`data_harness.eval` measures how well an agent answers **real** data questions —
across models, with **programmatic grading**. It leans on the structured
`RunResult.value` produced by `answer()`, so most cases grade without an LLM
judge.

## What the suite is designed to measure

Simple table lookups don't exercise what this harness is for. The suite targets
the three axes where the design earns its keep:

- **Depth** — multi-step dependent reasoning (compute → filter → aggregate) that
  runs over **many turns** of the ReAct loop.
- **Breadth** — **joins across several DataFrame handles**, the reason the
  `SessionCache` exists.
- **State** — **multi-turn conversations** where later questions build on handles
  saved in earlier turns. No single-shot benchmark can probe this; it's the
  clearest differentiator of an agent + cache design over one-shot code-gen.

## Suites

| Suite | What it tests | Use |
|---|---|---|
| `bespoke_suite()` | single-shot aggregation/filter/chart/refusal | quick smoke; capable models saturate it |
| `hard_suite()` | multi-table joins, deep multi-step, **stateful multi-turn** | exercises the agent loop + `SessionCache` |
| `large_data_suite()` | large frames that can only be answered via the **handle** (incl. a snapshot trap) | stresses the handle/snapshot design |
| `messy_suite()` | real-world cleaning: string amounts, mixed date formats, inconsistent labels, missing values | **differentiates models** (cleaning is where they diverge) |
| `load_wikitablequestions(...)` | public table-QA over real Wikipedia tables | external credibility; needs the `[eval]` extra |

```python
from data_harness.eval import evaluate, hard_suite

report = evaluate(hard_suite(), model="deepseek/deepseek-v4-flash")
print(report.to_markdown())
```

`evaluate` runs each case, grades it, and records pass/fail, turns, tokens,
latency, and status.

## Case types

### Single-shot — `EvalCase`

A question + dataset + grader. `data` is anything `ask` accepts: a DataFrame, a
`{name: frame}` mapping (→ multiple handles, for joins), a path, or a list.

```python
from data_harness.eval import EvalCase, numeric, contains

db = {"orders": orders_df, "customers": customers_df}
EvalCase(
    "top_region",
    "Join orders to customers; which region has the highest total revenue?",
    db, contains("EU"), category="join",
)
```

### Multi-turn — `ConversationCase`

A sequence of graded `Turn`s run over **one `Chat` session**, so later turns
reuse handles saved earlier — exercising the `SessionCache` across turns. Each
turn is graded and reported as `<id>#t<n>`.

```python
from data_harness.eval import ConversationCase, Turn, numeric, contains

ConversationCase(
    "customer_revenue",
    db,
    [
        Turn("Compute revenue per customer (join orders+customers), save as "
             "`cust_rev`. How many customers are there?", numeric(5)),
        Turn("Using cust_rev, which customer spent the most?", contains("Cara")),
        Turn("What is the average revenue across those customers?", numeric(428)),
    ],
    category="stateful",
)
```

Suites can freely mix `EvalCase` and `ConversationCase`; `evaluate` dispatches on
the type.

## Large-data: stressing the handle/snapshot design

`large_data_suite()` puts ~100k-row frames in the cache. The model only ever
sees the compact snapshot (shape + a few sample rows), so answering **requires**
computing over the full data through the interpreter handle — you can't eyeball
it, and a naive tool that read the rows into the prompt would blow the context
window. It also includes a **snapshot trap**: the few sample rows are
deliberately misleading, so a model that answers from the snapshot instead of
running code on the handle gets it wrong. This is the suite that directly
exercises the design's core bet — large data stays in `SessionCache`, never in
the transcript.

## Graders

Each grader checks `result.value` first (the executed answer), then falls back to
parsing the prose:

| Grader | Passes when |
|---|---|
| `numeric(expected, tol=…)` | the computed number matches within tolerance |
| `contains(expected)` | any expected string appears in value/prose |
| `exact(expected)` | `result.value` equals `expected` (normalised) |
| `dataframe_equals(df)` | `result.value` is an equal DataFrame |
| `chart_produced()` | the run rendered ≥1 chart |
| `refuses()` | the answer signals it can't/shouldn't answer (adversarial) |
| `all_of(...)` / `any_of(...)` | combine graders |

## Multi-model leaderboard + cost

OpenRouter makes the model matrix one key away. Pass a price map (fetched live)
to add a USD cost column:

```python
from data_harness.eval import evaluate_matrix, fetch_openrouter_prices, hard_suite

models = ["deepseek/deepseek-v4-flash", "qwen/qwen3.5-flash-02-23",
          "anthropic/claude-haiku-4.5"]
report = evaluate_matrix(hard_suite(), models)
prices = fetch_openrouter_prices(models)
print(report.to_markdown(prices))   # accuracy / turns / tokens / cost ($)
print(report.by_category())         # accuracy per category × model
```

`models` may also be `(label, adapter)` tuples for custom clients or offline
tests with `FakeAdapter`. Use recent models for a meaningful comparison.

## Tracking results over time

`EvalReport.to_dict()` / `to_json()` produce a machine-readable summary
(accuracy, per-model & per-category, cost, every case result). The example
runners write a timestamped JSON into the tracked `evals/results/` directory, so
runs are diffable in git (chart artefacts stay in gitignored `runs/`):

```bash
uv run python examples/eval_demo.py --suite hard   # → evals/results/hard_<ts>.{json,md}
uv run python examples/eval_wtq.py --limit 50      # → evals/results/wtq_<ts>.{json,md}
```

Each run writes a machine-readable `.json` **and** a human-readable `.md`
leaderboard, and refreshes [`evals/results/SUMMARY.md`](https://github.com/maxkskhor/data-harness/blob/main/evals/results/SUMMARY.md)
— one readable table per suite, so you can skim the numbers without parsing JSON.
Regenerate it any time with `python -m data_harness.eval.summary`.

Commit the JSON to record a run; a live, key-gated smoke test
(`tests/smoke_tests.py -m live`) runs small slices end-to-end, so the benchmark
can be wired into CI-with-secrets or a nightly job.

## Results

A snapshot of recent runs (full, regenerated tables live in
[`evals/results/SUMMARY.md`](https://github.com/maxkskhor/data-harness/blob/main/evals/results/SUMMARY.md)).

**WikiTableQuestions** (25 cases × 5 providers) — the public differentiator:

| model | accuracy | avg turns | cost ($) |
|---|---|---|---|
| deepseek/deepseek-v4-flash | 96% | 4.5 | 0.0222 |
| qwen/qwen3.5-flash | 88% | 6.5 | 0.0373 |
| openai/gpt-5-nano | 88% | 3.0 | 0.0400 |
| z-ai/glm-4.7-flash | 76% | 5.8 | 0.0247 |
| google/gemini-2.5-flash-lite | 56% | 3.2 | 0.0224 |

**messy** (real-world cleaning) — where models also diverge:

| model | accuracy | cost ($) |
|---|---|---|
| deepseek/deepseek-v4-flash · qwen3.5-flash · gpt-5-nano | 100% | ~0.006 |
| google/gemini-2.5-flash-lite · z-ai/glm-4.7-flash | 75% | ~0.004 |

The **hard** and **large-data** suites saturate at ~100% across recent models —
they validate the *design* (joins, multi-step, stateful multi-turn, and 100k-row
handle work for ~$0.002, passing the snapshot trap), rather than separating
models. Model differentiation lives in messy/real-world data.

## Why this fits data-harness

- `answer()` → `.value` gives a **checkable** result, so grading is programmatic.
- **OpenRouter** turns the model matrix into a one-key, cost-aware leaderboard.
- The **session tree** makes every graded case reconstructable.
- The stateful, multi-table cases exercise the **agent loop + SessionCache** —
  turning "it felt better" into a number on the work the harness is actually for.
