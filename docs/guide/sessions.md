# Sessions

`Agent.run()` is intentionally one-shot — it resets the message history on
every call. For chatbot or workbench applications where follow-up questions
need to refer back to earlier results, use `AgentSession`.

---

## Creating a session

```python
from data_harness import Agent

agent = Agent(system="You are a data analyst.", model="claude-sonnet-4-6")
session = agent.session()
```

`agent.session()` creates an `AgentSession` backed by a single `Harness` and
a single `SessionCache`. Message history accumulates across every `ask()` call
on the same session.

---

## Asking follow-up questions

```python
answer1 = session.ask("Load the sales data and compute total revenue.")
print(answer1)

answer2 = session.ask("Which product category had the highest revenue?")
print(answer2)
```

Both questions share the same message history, so the second question can
refer to results from the first.

---

## Pre-loading data

Use `session.put()` to store a value in the session cache before the first
question. The model can then access it by handle name:

```python
import pandas as pd

df = pd.read_csv("sales.csv")
session.put("sales", df)

answer = session.ask("What is the total revenue in the sales data?")
```

The model sees a compact snapshot (shape, columns, sample rows) rather than
the raw DataFrame. It accesses the data by writing Python against the
`sales` handle.

---

## Inspecting session state

```python
# All handles currently in the cache
print(session.list_handles())

# Total turns used across all ask() calls
print(session.turns)

# The RunResult from the last ask_result() call
print(session.last_result)
```

---

## Full result from a turn

```python
result = session.ask_result("Summarise the analysis so far.")

print(result.text)
print(result.turns)   # turns in THIS question, not cumulative
print(result.usage)
```

---

## When to use `run()` vs `session()`

| `Agent.run()` | `AgentSession.ask()` |
|---|---|
| Fresh message history each call | Shared history across calls |
| No persistent cache | Cache persists across turns |
| Good for one-shot scripts and tests | Good for UIs and workbenches |
