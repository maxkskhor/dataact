# Architecture

This page explains the core design decisions in `data-harness` and why they
exist. Understanding them makes the API surface predictable.

---

## Layers

The package is layered, bottom up. A layer may import the ones below it and
none above; `tests/test_layers.py` enforces that statically.

```
data_harness/
  llm/    provider adapters and the wire types they speak
  core/   the loop, the session tree, RunResult, hooks, compaction
  data/   the session cache, interpreter, SQL, connectors, MCP
  app/    Agent, ask(), the CLI
```

The boundary that matters is `core` not knowing what a DataFrame is. The loop
takes a `RunEnvironment` supplying two things it cannot decide for itself: how
to render a tool's return value, and what the run's final state was. The data
layer's implementation is backed by the `SessionCache`; `NullEnvironment` is
the domain-free default. That is why the harness can be read, tested, and
reused without pandas.

Every pre-layering import path (`data_harness.loop`, `data_harness.types`, …)
still resolves, to the same module object rather than a copy.

---

## One loop, two drivers

`_HarnessBase._plan` is the only ReAct loop. It is a generator that owns every
decision in a turn and performs no network I/O: it *asks* for I/O by yielding
`CallProvider`, `CallTool`, or `ToolFinished`, and the driver answers with
`Ok(value)` or `Failed(exc)`.

- `Harness` performs those inline on the calling thread. No event loop is
  created, so Ctrl-C lands promptly and a tool handler keeps its thread
  affinity: a `sqlite3` connection opened at setup still works.
- `AsyncHarness` awaits the provider and offloads blocking handlers to a
  worker thread, so a long pandas call cannot stall a shared event loop.

There used to be four near-copies of this loop (sync/async x result/stream).
They had drifted: the streaming copy discarded token usage on provider errors,
and `AsyncAgent` was missing eight features `Agent` had.

---

## The session tree

A session is an append-only tree of typed entries, each naming its parent,
with a movable leaf. **The conversation the model sees is derived by walking
root to leaf, and is never stored**, so there is no second copy to disagree
with the log.

Everything else follows from that:

- **Resume** — reopen a session file in another process and carry on.
- **Forking** — move the leaf and append; both branches survive, so a turn can
  be retried with a different model without destroying the first attempt.
- **Compaction is an entry, not an edit.** The compacted turns stay in the
  tree; moving the leaf back restores them.
- Writing is one line per entry, where the older `runs/*.jsonl` turn log
  re-serialises the whole history every turn.

A derived context is always something a provider will accept: a tool call with
no result, or a result with no call, is dropped. Both are reachable without a
bug, because a run killed mid-tool leaves an orphaned call on disk.

---

## Hooks

Four events, each with a decision a hook may return:

```
BeforeTurn     -> Reminder(text) | Stop(reason)
BeforeToolCall -> Block(reason, is_error)
AfterToolCall  -> Replace(content, is_error)
AfterTurn      -> Stop(reason)
```

The interpreter approval gate is built from exactly this mechanism rather than
being special-cased inside the loop, which is the evidence that the mechanism
is sufficient. `AfterTurn` is where a spend cap belongs: the tokens are already
counted, so the decision is made on real numbers.

Hooks must not raise. One that does is reported as `HookError`, and the run
fails with its usage intact rather than vanishing.

---

## Compaction

`max_turns` is a wall, not a strategy: a run needing thirty turns fails at
twenty-five having paid for all of them. Compaction summarises older turns and
replays the summary in their place.

The cut always lands on a turn boundary, so an assistant tool call is never
separated from the result answering it. When there is nowhere safe to cut,
nothing is cut.

Compaction costs this library less than it costs a coding agent: the data
lives in the cache under handles, not in the transcript, so compacting away
the turn that loaded a DataFrame loses the discussion of it, not the frame.

---

## Errors

Every failure carries a stable `code`, because a caller deciding between
retrying, showing the user an error, and billing the attempt needs to tell a
rate-limited provider from a typo in the model's pandas from a sandbox
timeout. Codes are API; messages are not.

`ExecutionError` means the code never ran (timeout, killed process).
`PythonInterpreterError` means it ran and failed, which is the model's problem
and is handed back to it as a tool result.

---

## No bash

Giving an agent shell access is the path of least resistance, but it creates
real problems in data workflows:

- **Unpredictable side effects** — shell commands can touch files, processes,
  and network resources in ways that are hard to audit.
- **Security exposure** — prompt injection or a confused model can run
  destructive commands.
- **Reproducibility** — shell state is implicit and hard to reconstruct from
  logs.

`data-harness` constrains the model to a Python interpreter only. Python is
expressive enough for all data work, and the interpreter can be given explicit
globals and have dangerous operations blocked.

---

## Handle/snapshot pattern

Large objects (DataFrames, arrays, query results) never appear in message
history. Instead:

1. The tool result calls `SessionCache.put(name, value)`.
2. `format_tool_output` returns a compact snapshot — shape, columns, a few
   sample rows — as the tool result string.
3. The model writes Python against the handle name (`sales_df`, `result_2`,
   etc.) to operate on the data.

This keeps context lean without hiding data from the model.

```
Tool result →  {"type": "dataframe", "shape": [1200, 5],
                "columns": ["date", "revenue", ...], "sample": [...]}

Model code  →  result = sales_df[sales_df.revenue > 1000].groupby("category").sum()
               save("top_categories", result)
```

---

## Prefix-stable system prompt

The system prompt is never mutated between turns. Only the conversation suffix
changes. This is a KV-cache discipline: a stable prefix means the provider
can cache it, reducing latency and cost on long runs with many turns.

Reminders, nags, and dynamic state updates are always appended to the suffix.
The `Harness` enforces this invariant — it has no API to modify the system
prompt after construction.

---

## Progressive connector disclosure

Connector tools start hidden (`visible=False`). The model must call
`load_connectors(connector_name="...")` before the tools for that connector
appear in its tool list.

A shorter tool list means the model makes better routing decisions. Loading
all 40 connectors upfront would overwhelm the tool selection at each turn.
The model loads only what it needs for the current task.

---

## Suffix-only reminders

The `Planner` escalates reminders when the model has not made progress for
several turns. These reminders are always appended to the conversation suffix
as `TextBlock` items. They are never inserted into the prefix.

This preserves the stable-prefix invariant and avoids invalidating the
provider's KV cache on every reminder injection.

---

## Subagent isolation

Spawned subagents get:

- A fresh `AsyncProviderAdapter` (or `ProviderAdapter`)
- A fresh message history (empty)
- A fresh `SessionCache`

Parent state crosses the boundary only through explicit `input_handles`.
The parent cache is not inherited. This makes subagent behaviour reproducible
and debuggable independently of the parent run.

Subagents cannot spawn further subagents — `SubagentRecursionError` is raised
if they try.

---

## JSONL turn logging

Every turn is logged to a `.jsonl` file from the start of the run, not bolted
on later. Each line is a complete turn record:

- The system prompt and message history
- The provider response
- Tool results
- Latency and token counts
- Cache storage metadata

The log is designed to reconstruct a run without dumping raw dataset payloads.
Cache handles are logged as snapshots, not raw values.

---

## Key invariants

These are design constraints, not incidental behaviour:

- The system prompt is byte-identical across turns.
- Adapters never mutate harness-owned state.
- Dynamic reminders are suffix-only.
- Tool-use messages are always followed by matching tool-result messages before
  the next assistant call.
- Raw large payloads stay in `SessionCache`; messages and logs receive
  snapshots only.
- Cache handles are valid Python identifiers.
- `python_interpreter` uses fresh locals per call.
- Subagents do not inherit parent cache implicitly.
- JSONL logs support run reconstruction without raw payload leakage.

Tests in `tests/` assert these invariants directly.
