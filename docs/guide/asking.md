# Asking questions about data

The fastest way to use `data-harness` is `ask`. It resolves a provider from your
environment, loads your data into a `SessionCache` as handles, runs the agent,
and returns a [`RunResult`](../api/agent.md).

```python
import pandas as pd
from data_harness import ask

df = pd.read_csv("sales.csv")
result = ask(df, "What was total revenue, and which month was highest?")

print(result.text)    # the written answer
print(result.value)   # the structured value the model computed via answer()
result.charts         # any rendered charts
```

`ask` accepts a DataFrame, a `{name: value}` mapping, a file path, or a list of
paths. Provider resolution prefers `ANTHROPIC_API_KEY`, then `OPENAI_API_KEY`,
then `OPENROUTER_API_KEY`; pass `model=` to choose explicitly (the name routes to
the matching provider) or `adapter=` to supply one directly.

### Many providers through one key (OpenRouter)

[OpenRouter](https://openrouter.ai) exposes OpenAI, Anthropic, Google, Meta and
more behind one OpenAI-compatible endpoint — ideal for cross-model testing. Set
`OPENROUTER_API_KEY`; any `provider/model` id (containing `/`) auto-routes there.

```python
ask(df, "summarise the data", model="anthropic/claude-3.5-sonnet")
ask(df, "summarise the data", model="google/gemini-2.0-flash-001")
ask(df, "summarise the data", model="deepseek/deepseek-chat")  # cheap

# or construct it directly:
from data_harness.llm.providers.openai import OpenRouterAdapter
ask(df, "...", adapter=OpenRouterAdapter(model="openai/gpt-4o-mini"))
```

DeepSeek's own (very cheap) API works directly too — set `DEEPSEEK_API_KEY` and
use a bare `model="deepseek-chat"` (or `DeepSeekAdapter`). A `deepseek/...` id
with a slash routes via OpenRouter instead.

### Adding a provider without writing an adapter

Most LLM providers speak the OpenAI-compatible chat completions format — only
Anthropic's Messages API is different enough to need its own adapter. Groq,
Together, Fireworks, Cerebras, and xAI are pre-registered, so they work the
same way as OpenRouter/DeepSeek:

```python
from data_harness.llm.providers.openai import OpenAIAdapter

ask(df, "...", adapter=OpenAIAdapter(model="llama-3.3-70b-versatile", provider="groq"))
```

Register any other OpenAI-compatible provider with a plain config, no adapter
class required:

```python
from data_harness.llm.providers.openai import (
    OpenAICompatibleProvider,
    register_openai_compatible_provider,
)

register_openai_compatible_provider(
    OpenAICompatibleProvider(
        name="my-provider",
        base_url="https://api.my-provider.example/v1",
        api_key_env="MY_PROVIDER_API_KEY",
        default_model="my-model-large",
    )
)
ask(df, "...", adapter=OpenAIAdapter(model="my-model-large", provider="my-provider"))
```

## Structured answers

The interpreter exposes an `answer(value)` helper. When the model calls it, the
value is returned as `RunResult.value` — a number, a DataFrame, anything. This
is the trustworthy, *executed* result, as opposed to free-form prose which a
model can occasionally get wrong. Prefer `.value` for anything programmatic.

Models don't always remember to call `answer()`. By default `ask` **finalises**:
if a run succeeds without a recorded answer, it runs one short follow-up turn
asking the model to record it, so `.value` is populated more reliably. This is
guarded — it never fires when a chart was produced (the chart is the answer) or
the response reads as a refusal (so an unanswerable question isn't coerced into a
fabricated value). Disable with `ask(..., require_answer=False)`. `Chat` is
conversational, so it defaults to `require_answer=False`; pass `require_answer=True`
to opt in.

## Charts

matplotlib is available inside the interpreter. The model builds a figure and it
is captured automatically as a `ChartArtifact`. The image bytes are written to
disk and **never** enter the message history or the session log — only a path
does, exactly like a cache handle. In Jupyter, `RunResult` and `ChartArtifact`
render inline.

```python
result = ask(df, "Plot revenue by region as a bar chart.")
result.charts[0]   # renders inline in a notebook
```

## Multi-turn chat

```python
from data_harness import Chat

chat = Chat(df)
chat.ask("What was total revenue?")
chat.ask("Which month was highest?")   # remembers context
```

`SmartFrame(df).chat("...")` is a pandasai-style wrapper over the same machinery.
There is also an opt-in pandas accessor — `import data_harness.app.pandas` then
`df.chat("...")` — and a `%%ask` notebook magic (`%load_ext data_harness.app.notebook`).

## SQL

With DuckDB installed, `ask` exposes a `sql_query` tool that runs SQL directly
against your DataFrame handles; results become new handles. Point it at a real
database with a SQLAlchemy URL:

```python
ask(df, "Use SQL to compute total revenue per region.")   # DuckDB, in-process

from data_harness import Agent
agent = Agent.from_dataframe(df).enable_sql(engine_url="postgresql://...")
agent.run("Top 5 customers by spend last quarter?")
```

## The semantic layer

Attach column descriptions, units, or business definitions so the model reasons
correctly. They are folded into the snapshot the model sees.

```python
ask(
    df,
    "What is churn?",
    semantics={"df": {"columns": {"churn": "1 if the customer cancelled in-month"}}},
)
```

## Production controls

```python
from data_harness import Agent, ExecutionCache

# Replay repeat questions with no model call (zero turns, zero tokens):
agent = Agent.from_dataframe(df).enable_cache(ExecutionCache("cache.json"))

# Run interpreter code in an isolated process (no network, CPU/time limits):
sandboxed = Agent.from_dataframe(df, execution="subprocess")

# Approve or block code before it runs:
gated = Agent.from_dataframe(df, on_code=lambda code: (print(code), True)[1])

# Dry run — return the code without executing it:
preview = Agent.from_dataframe(df, code_only=True)
```

- **Code-replay cache** keys off the question and the data *schema* (not the
  values), so a repeat question replays the recorded code against fresh data —
  correct and free.
- **Subprocess sandbox** isolates execution in a child process with networking
  disabled and resource limits; handles cross by value and results merge back.
  It shares the exact AST/security core with the in-process interpreter. It is
  not a container/VM sandbox — that remains future work.
- **Approval gate** lets a human (or policy) see every code block before it runs.
