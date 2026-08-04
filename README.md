<p align="center">
<img src="docs/assets/data-harness-full.png" alt="data-harness — The controlled data-agent SDK" width="460">
</p>

<p align="center">
Python, not bash. Large data stays in a cache as handles, never in the prompt.<br/>
Every run is logged — and eval-backed.
</p>

<p align="center">
<a href="https://pypi.org/project/data-harness/"><img src="https://img.shields.io/pypi/v/data-harness.svg" alt="PyPI"></a>
<a href="https://pypi.org/project/data-harness/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
<a href="https://maxkskhor.github.io/data-harness/"><img src="https://img.shields.io/badge/docs-mkdocs-blue.svg" alt="Docs"></a>
</p>

---

Most data-agent tooling makes you pick between giving a model a **shell** (unsafe, irreproducible) and **single-shot code-gen** (no state, no multi-step). `data-harness` is the controlled middle path: the model works through a constrained Python interpreter, large objects live in a `SessionCache` and are exposed as compact handle snapshots — so a 100k-row table never hits the context window — every run is recorded as an append-only session tree, and a built-in **evaluation harness** measures quality and cost across providers.

### Principles

- **Python, not bash** — one controlled execution surface: no shell side-effects, no destructive commands, reproducible runs.
- **Handles, not payloads** — large data lives in the cache; only snapshots reach the model, so context (and cost) stay flat as data grows.
- **Measured, not vibes** — a first-class eval harness with programmatic graders, multi-turn cases, cost, and tracked leaderboards.

### Features

- **One-liner** — `ask(df, "...")` in Python, or `dh "..." data.csv` from the shell.
- **Charts & SQL** — automatic matplotlib capture; a DuckDB / SQLAlchemy `sql_query` tool.
- **Many providers, one key** — OpenAI, Anthropic, DeepSeek, Qwen, Google, Z.ai… via OpenRouter.
- **MCP bridge** — connect any [MCP](https://modelcontextprotocol.io) server (Postgres, SQLite, filesystem…) and use its tools, with progressive disclosure + handle/snapshot.
- **Production controls** — subprocess sandbox, an approval gate, and a zero-token replay cache.
- **Evaluation** — bespoke / hard / large-data suites + WikiTableQuestions, with multi-turn cases, cost, and JSON-tracked results.
- **Composable** — `ask`/`Chat` over `Agent` over `Harness`; async + streaming; subagents; progressive connectors.

---

## Install

```bash
pip install data-harness          # core
pip install "data-harness[all]"   # + openai, charts, duckdb, sqlalchemy, notebook, eval
```

Pick individual extras as needed: `[openai]`, `[viz]`, `[duckdb]`, `[sql]`, `[notebook]`, `[eval]`. Requires Python 3.10+.

---

## Quickstart

Ask a question about a DataFrame in one line. `ask()` resolves a provider from your environment, loads the data into the session cache, runs the agent, and returns a `RunResult`:

```python
import pandas as pd
from data_harness import ask

df = pd.read_csv("sales.csv")
result = ask(df, "What was total revenue, and which month was highest?")

print(result.text)      # the written answer
print(result.value)     # the structured result the model computed via answer()
result.charts           # any charts it rendered (notebook-friendly)
```

Reach many providers through one key with [OpenRouter](https://openrouter.ai) — a `provider/model` id auto-routes there. Set `OPENROUTER_API_KEY`:

```python
ask(df, "plot revenue by month", model="deepseek/deepseek-v4-flash")
ask(df, "summarise the data",   model="google/gemini-2.5-flash-lite")
ask(df, "which region grew fastest?", model="qwen/qwen3.5-flash-02-23")
```

Without OpenRouter, `ask()` falls back to `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `DEEPSEEK_API_KEY`. In a notebook, the returned `RunResult` renders prose, the value, and charts inline (there's also a `%%ask` magic via `%load_ext data_harness.app.notebook`).

---

## Command line & demo

Ask from the shell with **`dh`** (also installed as `data-harness`) — point at one or more files, or pipe CSV via stdin:

```bash
dh "What was total revenue?" sales.csv
dh "Join these and find the top region" orders.csv customers.csv
cat sales.csv | dh "median order amount" --json
```

A Streamlit demo app (`pip install "data-harness[demo]"`):

```bash
uv run streamlit run examples/streamlit_app.py
```

<p align="center">
<img src="docs/assets/streamlit-demo.png" alt="data-harness Streamlit demo" width="700">
</p>

## Multi-turn chat

```python
from data_harness import Chat

chat = Chat(df)
chat.ask("What was total revenue?")
chat.ask("Which month was highest?")   # remembers context (shared cache + history)
```

---

## Charts & SQL

matplotlib runs inside the interpreter; open figures are captured automatically as artefacts — the image bytes live on disk and never enter the message history or logs (only a path does):

```python
result = ask(df, "Plot revenue by region as a bar chart.")
result.charts[0]        # a ChartArtifact; renders inline in Jupyter
```

With DuckDB installed, `ask` exposes a `sql_query` tool over your DataFrames; point it at a real database with a SQLAlchemy URL:

```python
ask(df, "Use SQL to get total revenue per region.")          # DuckDB, in-process

from data_harness import Agent
agent = Agent.from_dataframe(df).enable_sql(engine_url="postgresql://...")
agent.run("Top 5 customers by spend last quarter?")
```

---

## Production controls

```python
from data_harness import Agent, ExecutionCache

agent = Agent.from_dataframe(df).enable_cache(ExecutionCache("cache.json"))  # 0-token replays
sandboxed = Agent.from_dataframe(df, execution="subprocess")                 # isolated process
gated = Agent.from_dataframe(df, on_code=lambda code: (print(code), True)[1]) # approve code
preview = Agent.from_dataframe(df, code_only=True)                            # dry-run, never executes
```

- **Code-replay cache** — a repeat question over the same data *schema* replays the recorded code with no model call (zero turns, zero tokens), and stays correct when the data changes.
- **Subprocess sandbox** — interpreter code runs in a separate process with networking disabled and CPU/wall-clock limits; handles cross by value, results merge back.
- **Approval gate** — `on_code` sees every code block before execution and can block it; `code_only=True` returns the code without running it.

---

## Evaluation

A first-class harness to measure how well an agent answers **real** data questions — across models, with **programmatic grading** that leans on the structured `.value` (no LLM judge needed for most cases).

```python
from data_harness.eval import evaluate_matrix, fetch_openrouter_prices, hard_suite

models = ["deepseek/deepseek-v4-flash", "qwen/qwen3.5-flash-02-23",
          "openai/gpt-5-nano", "google/gemini-2.5-flash-lite"]
report = evaluate_matrix(hard_suite(), models)
print(report.to_markdown(fetch_openrouter_prices(models)))  # accuracy / turns / tokens / cost
```

- **Suites** — `bespoke_suite()` (smoke), `hard_suite()` (multi-table joins, deep multi-step, **stateful multi-turn**), `large_data_suite()` (100k-row frames answerable only via the handle, with a **snapshot trap**), and `load_wikitablequestions()` (public table-QA, the model differentiator).
- **Case types** — single-shot `EvalCase` and multi-turn `ConversationCase` (graded turns over one `Chat` session, testing `SessionCache` persistence).
- **Graders** — `numeric`, `contains`, `exact`, `dataframe_equals`, `chart_produced`, `refuses`, `all_of`/`any_of`.
- **Reporting** — leaderboards with per-model **cost**, per-category breakdowns, and `to_dict()`/`to_json()` for results tracked in `evals/results/`.

Results are committed as readable leaderboards — see [`evals/results/SUMMARY.md`](evals/results/SUMMARY.md) (a table per suite: accuracy, turns, tokens, cost).

What the runs show: the structured/large/stateful suites **saturate at ~100% across recent models** — i.e. the *design* is robust (even small, cheap models handle 100k-row data via the handle for ~$0.002 and pass the snapshot trap). Model *differentiation* shows up on messy real-world data — WikiTableQuestions spreads recent models **64%→96%**. See the [Evaluation guide](https://maxkskhor.github.io/data-harness/guide/evaluation/).

---

## Lower-level `Agent` and `Harness`

`ask`/`Chat` are conveniences over `Agent`, itself a thin layer over `Harness`. Drop down for full control:

```python
from data_harness import Agent

agent = Agent(system="You are a data analyst.", model="claude-sonnet-4-6")
print(agent.run("Compute the mean of [1, 2, 3, 4, 5]."))
```

| Component | Role |
|---|---|
| `Harness` | The ReAct loop — messages, tool dispatch, reminders, session recording |
| `SessionCache` | Handle-based store; keeps large objects out of message history |
| `ProviderAdapter` | Translates provider SDK responses into harness types |
| `python_interpreter` | The model's only execution surface |
| `ConnectorRegistry` | Hides connector tools until the model loads them |
| `Subagent` | Isolated worker with explicit state transfer |

Async + streaming (`AsyncAgent.run_stream`), progressive **connectors**, and **subagents** are all supported — see [`examples/advanced_wiring.py`](examples/advanced_wiring.py) and the [docs](https://maxkskhor.github.io/data-harness/).

---

## Examples & tests

```bash
uv run python examples/live_demo.py          # ask()/charts/SQL on a cheap model
uv run python examples/eval_demo.py --suite hard   # multi-model eval leaderboard (cost)
uv run python examples/cache_benchmark.py    # replay-cache benchmark (no API key)
uv run python -m pytest tests/ -m "not live"       # offline test suite
```

[`examples/demo.ipynb`](examples/demo.ipynb) is an executed end-to-end notebook.

---

## Sandbox disclaimer

The in-process interpreter uses AST checks and restricted globals to reduce accidental misuse — it is **not** a container sandbox. For stronger isolation use `execution="subprocess"` (separate process, no network, resource limits). Neither is hardened for untrusted input.

---

## Links

- **Docs:** <https://maxkskhor.github.io/data-harness/>
- **Changelog:** [CHANGELOG.md](CHANGELOG.md)
- **Design series:** [three-part write-up](https://maxkskhor.substack.com/p/designing-a-react-harness-for-data)
- **License:** [MIT](LICENSE)
