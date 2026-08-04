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

Both drivers run the same generator, so a sync and an async run make
identical decisions about reminders, tool gating, and when to stop.

---

## The session tree

A session is an append-only tree of typed entries, each naming its parent,
with a movable leaf. The conversation the model sees is derived by walking
root to leaf. There is only ever one copy of the conversation: the derived
context, computed from the tree each time it is needed.

Everything else follows from that:

- **Resume** — reopen a session file in another process and carry on.
- **Forking** — move the leaf and append; both branches survive, so a turn can
  be retried with a different model without losing the first attempt.
- **Reversible compaction** — a compaction is its own entry. The compacted
  turns stay in the tree, so moving the leaf back restores them.
- **Linear writes** — one line per entry, so writing a session costs is
  proportional to how many entries it has, independent of how long the
  conversation already is.

A derived context is always something a provider will accept: a tool call
with no result, or a result with no call, is dropped. Both are reachable
without a bug in this library, because a run killed mid-tool leaves an
orphaned call on disk.

---

## Hooks

Four events, each with a decision a hook may return:

```
BeforeTurn     -> Reminder(text) | Stop(reason)
BeforeToolCall -> Block(reason, is_error)
AfterToolCall  -> Replace(content, is_error)
AfterTurn      -> Stop(reason)
```

The interpreter approval gate is built from exactly this mechanism. `AfterTurn`
is where a spend cap belongs: the tokens are already counted by then, so the
decision is made on real numbers.

Hooks must not raise. One that does is reported as `HookError`, and the run
ends with its usage intact.

---

## Compaction

`max_turns` caps how many turns a run gets. Compaction gives a long run a way
to keep going within that cap: it summarises older turns and replays the
summary in their place, freeing up room for more turns.

The cut always lands on a turn boundary, so an assistant tool call is never
separated from the result answering it. When there is nowhere safe to cut,
compaction does nothing that turn.

The data a compacted turn discussed is unaffected: it lives in the cache
under handles, separate from the transcript. Compacting away the turn that
loaded a DataFrame removes the discussion of it from the context; the
DataFrame itself is still there under its handle.

---

## Errors

Every failure carries a stable `code`. A caller deciding between retrying,
showing the user an error, and billing for the attempt can act on the code
directly, distinguishing a rate-limited provider from a bug in the model's
pandas from a sandbox timeout. Codes are part of the API; messages can be
reworded freely.

`ExecutionError` means the sandboxed code never ran to completion — a
timeout, or the process was killed. `PythonInterpreterError` means it ran and
raised, which is the model's problem: it comes back as a tool result so the
model can see the traceback and fix its own code.

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

Large objects (DataFrames, arrays, query results) live in the `SessionCache`
and are represented in message history by a compact snapshot:

1. The tool result calls `SessionCache.put(name, value)`.
2. `format_tool_output` returns a compact snapshot — shape, columns, a few
   sample rows — as the tool result string.
3. The model writes Python against the handle name (`sales_df`, `result_2`,
   etc.) to operate on the data.

This keeps context small while still showing the model enough of the data to
work with it.

```
Tool result →  {"type": "dataframe", "shape": [1200, 5],
                "columns": ["date", "revenue", ...], "sample": [...]}

Model code  →  result = sales_df[sales_df.revenue > 1000].groupby("category").sum()
               save("top_categories", result)
```

---

## Prefix-stable system prompt

The system prompt is byte-identical across every turn of a run. Only the
conversation suffix changes. This is a KV-cache discipline: a stable prefix
lets the provider cache it, reducing latency and cost on long runs with many
turns.

Reminders, nags, and dynamic state updates are always appended to the suffix.
The `Harness` enforces this — it has no API to modify the system prompt after
construction.

---

## Progressive connector disclosure

Connector tools start hidden (`visible=False`). The model must call
`load_connectors(connector_name="...")` before the tools for that connector
appear in its tool list.

A shorter tool list means the model makes better routing decisions at each
turn. The model loads only the connectors it needs for the current task,
instead of choosing from every registered connector's tools on every call.

---

## Suffix-only reminders

The `Planner` escalates reminders when the model has made no progress for
several turns. These reminders are always appended to the conversation suffix
as `TextBlock` items, so they change the working set the model reasons over
without touching the cached prefix.

This preserves the stable-prefix invariant above and keeps the provider's KV
cache valid across a reminder injection.

---

## Subagent isolation

Spawned subagents get:

- A fresh `AsyncProviderAdapter` (or `ProviderAdapter`)
- A fresh message history (empty)
- A fresh `SessionCache`

Parent state crosses the boundary only through explicit `input_handles`.
The parent cache is not shared. This makes subagent behaviour reproducible
and debuggable independently of the parent run.

Subagents cannot spawn further subagents — `SubagentRecursionError` is raised
if they try.

---

## Per-turn accounting

Every turn is recorded on the session tree as a `TurnEntry`, as part of the
same append that records the turn's messages:

- Token counts (input, output, cache read/write) and latency
- The provider's stop reason
- Tool error count and the names of the tools visible on that turn

A `TurnEntry` carries these counts and metadata. The actual message content —
the same `Message` objects the model saw, snapshots included — lives
separately in the tree's `MessageEntry` nodes. `JsonlSessionStore` persists
the whole tree to disk, one line per entry, so a run can be reconstructed
from disk in full: what was asked, what the model did, what it cost, and how
long each turn took, all without reading a dataset's actual values back out
of the log.

---

## Key invariants

These are design constraints that tests assert directly, in `tests/`:

- The system prompt is byte-identical across turns.
- Adapters never mutate harness-owned state.
- Dynamic reminders are suffix-only.
- Tool-use messages are always followed by matching tool-result messages before
  the next assistant call.
- Large values stay in `SessionCache`. The session tree and the messages sent
  to the provider hold snapshots of them, never the values themselves.
- Cache handles are valid Python identifiers.
- `python_interpreter` uses fresh locals per call.
- Subagents do not inherit the parent cache automatically.
- The session tree can be replayed into the same conversation a run actually
  had, from disk, without ever reading a cached value's contents back out of
  the log file.
