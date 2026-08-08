# AGENTS.md / CLAUDE.md

This file provides guidance to coding agents working in this repository.

## What this is

`data-harness` (data + harness) is a Python SDK for controlled data-agent
workflows: the model works through a constrained Python interpreter only, no
bash tool. Large objects live in `SessionCache` and are exposed to the model
as named handles with compact snapshots, never as raw blobs in message
history. It is published on PyPI as `data-harness`; see `CHANGELOG.md` for
release history and `pyproject.toml` for the current version — don't
duplicate either here, they drift out of sync with this file immediately.

[`data-harness-ui`](https://github.com/maxkskhor/data-harness-ui) is a
deployed full-stack reference app built on this SDK (live at
https://data-harness-ui.vercel.app) — the concrete proof this is a real,
usable library, not just a design exercise.

## Motivation

This project is motivated by data workflows where:

- unrestricted bash access is undesirable
- tool results can be large
- many data connectors may exist
- long-running conversations need disciplined context management
- debugging and reproducibility matter

The core design ideas:

- the ReAct loop stays small and readable
- Python is the controlled execution surface for data work
- large tool outputs become handles plus snapshots
- connector schemas are progressively disclosed
- the system prompt stays prefix-stable
- reminders and dynamic state are appended to the conversation suffix
- subagent state transfer is explicit
- the session tree reconstructs runs without dumping raw datasets
- architectural invariants are tested directly

## Where things live

A stable map, not a feature log — check the file itself for current
behaviour rather than trusting a description here to have kept up.

| Concern | Where |
|---|---|
| Provider adapters (Anthropic, OpenAI-compatible registry) | `llm/providers/` |
| Wire types, streaming events | `llm/types.py`, `llm/streaming.py` |
| The ReAct loop (single generator, sync + async drivers) | `core/loop.py` |
| Session tree (append-only typed entries, JSONL persistence) | `core/session/` |
| Hooks (`BeforeTurn`/`BeforeToolCall`/`AfterToolCall`/`AfterTurn`) | `core/hooks.py` |
| Compaction | `core/compaction.py` |
| Typed errors, results (`RunResult`, `Usage`, `CacheStorageInfo`) | `core/exceptions.py`, `core/result.py` |
| Chart capture | `core/artifacts.py` |
| `SessionCache` (handles, hot/cold spilling, semantics) | `data/cache.py` |
| `Harness`/`AsyncHarness` (wires the core loop to a cache) | `data/harness.py` |
| Built-in tools (`python_interpreter`, `list_variables`, `sql_query`, connectors, subagent) | `data/tools/` |
| Subprocess sandbox | `data/tools/sandbox.py`, `data/_sandbox_runner.py` |
| Code-replay cache | `data/exec_cache.py` |
| File ingestion | `data/io.py` |
| High-level API (`Agent`/`AsyncAgent`, `AgentSession`) | `app/agent.py` |
| Zero-config entry points (`ask`, `Chat`, `SmartFrame`, `resolve_adapter`) | `app/quickstart.py` |
| CLI (`dh`) | `app/cli.py` |
| Notebook / pandas integration | `app/notebook.py`, `app/pandas.py` |
| Evaluation harness | `eval/` |
| Docs site (MkDocs) | `docs/` |
| Runnable examples | `examples/` |

Layers import strictly downward — `llm` -> `core` -> `data` -> `app`, each
importing only the layers below it, enforced statically by
`tests/test_layers.py`. There is no compatibility shim for pre-layering flat
import paths; import from the layered path (`data_harness.core.loop`,
`data_harness.data.cache`, ...) or the top-level `data_harness` re-exports in
`__init__.py`. Do not add a flat-path shim for backward compatibility — this
project has deliberately chosen not to carry legacy surface.

## Architecture

The core loop lives in `core/loop.py`: `_HarnessBase._plan` is the single
generator, driven by `Harness` (sync) and `AsyncHarness` (async, adds
streaming). It owns the message list, dispatches tools, applies hooks,
filters visible tools, and records every turn onto the session tree
(`core/session/`). `data/harness.py`'s `Harness`/`AsyncHarness` wire the
domain-free core loop to a `SessionCache`; that pair is what `Agent` builds
and what most callers actually use.

`run_result()` / `ask_result()` return a typed `RunResult` (`core/result.py`)
carrying text, token `Usage`, and `CacheStorageInfo` — never raw payloads.
The model-visible built-in tools are `python_interpreter` and
`list_variables` (`data/tools/variables.py`); the planner, connectors, and
subagent are opt-in.

The harness never mutates the system prompt. Reminders, nags, final-turn
warnings, tool results, and dynamic state belong in the conversation suffix —
reminders arrive as `BeforeTurn` hooks returning `Reminder` (`core/hooks.py`).

`SessionCache` (`data/cache.py`) is shared state between the harness and
tools. Tools store large objects by handle name. The model sees snapshots and
can operate on handles through the Python interpreter.

`python_interpreter` injects cache handles as local variables and exposes
`save(name, value)` for explicit persistence. It should not expose the cache
object itself.

`ToolSpec` (`llm/types.py`) carries the model-visible tool contract plus the
already-bound handler callable. The loop dispatches with
`handler(**tool_input)` and does not know how dependencies were wired.

Connectors (`data/tools/connectors.py`) are registered hidden and flipped
visible on demand through `load_connectors`. This is the progressive
disclosure pattern.

Provider adapters (`llm/providers/base.py`) normalize external provider APIs
into `NormalizedResponse`. Adapters copy and transform inputs; they must not
mutate harness-owned `system`, `messages`, or `tools`.

Subagents (`data/tools/subagent.py`) get a fresh adapter, fresh message
history, and fresh cache per spawn. Parent state crosses the boundary only
through explicit `input_handles`, and created outputs return only through
`publish_created`.

The session tree (`core/session/`) is an append-only tree of typed entries
(`MessageEntry`, `TurnEntry`, `CompactionEntry`, ...) with a movable leaf; the
conversation is derived by walking root to leaf, never stored separately.
`JsonlSessionStore` persists it to disk (one line per entry, via the public
`encode_entry`/`FORMAT_VERSION`); `MemorySessionStore` is the default.

## Key invariants

These are deliberate design constraints. Tests should assert them directly:

- System prompt is byte-identical across turns.
- Adapters never mutate harness-owned state.
- Dynamic reminders are suffix-only.
- Tool-use messages are followed by matching tool-result messages before the
  next assistant call.
- Large values live in `SessionCache`. Messages and the session tree carry
  only compact snapshots of them.
- Cache handles are valid Python identifiers.
- `python_interpreter` uses fresh locals per call.
- Subagents do not inherit parent cache implicitly.
- Subagents cannot recursively register or call `subagent`.
- A session can be replayed from disk into the exact conversation a run had,
  without reading any cached value's contents out of the log file.

## Commands

```bash
uv sync
uv run pytest tests/ -m "not live"                              # full offline suite
uv run pytest tests/smoke_tests.py -v -m live                   # live provider smoke tests
uv run python examples/quickstart.py                            # minimal Agent (ANTHROPIC_API_KEY)
uv run python examples/advanced_wiring.py                       # connectors + planner + subagents
uv run python examples/inspect_run.py                           # RunResult inspection
uv run mkdocs serve                                             # preview the docs site
```

Live smoke tests run through OpenRouter: they require `OPENROUTER_API_KEY`
and default to `openai/gpt-4o-mini`; set `DATA_HARNESS_SMOKE_MODEL` to
override (e.g. `deepseek/deepseek-chat`). The advanced demo requires
`ANTHROPIC_API_KEY`. Both may cost tokens.

## Writing context

This repo supports writing about harness and SDK engineering. Two tracks:

1. **Architecture** — why this harness design exists. Topics: no bash, Python
   execution, progressive disclosure, handle/snapshot, prefix-stable
   suffix-mutable context, planner reminders, subagents, `list_variables`,
   failure modes.
2. **Implementation invariants** — the technical details that keep the
   design coherent. Topics: typed content blocks, adapter mutation boundary,
   tool-use ordering, inline-vs-cache formatting, handle naming, interpreter
   `save`, suffix-only reminders, explicit subagent state transfer, the
   session tree, invariant tests.

Do not frame `data-harness` as a teaching/reference implementation — it is a
published SDK with a real downstream app (`data-harness-ui`). If a
beginner-oriented walkthrough is ever wanted, it belongs in `docs/guide/`
alongside the existing docs, not as a reason to hold this repo back from
normal SDK development.

All writing in this repo uses British English spelling (e.g. behaviour,
normalise, serialise, catalogue, artefact).

## Implementation guidance

Prefer clarity over accidental abstraction, but this is framework code, not a
minimal teaching example — indirection is acceptable when it supports a real
SDK responsibility: stable public API, provider integration, session/result
metadata, reproducibility, observability, sandboxing, async execution, or
safe extensibility. It is not acceptable when it hides state, weakens cache
boundaries, or makes run reconstruction harder.

Keep the public surface coherent and tested. It may grow beyond the original
`Agent` convenience layer, but new APIs should map cleanly onto the
underlying runtime concepts. The high-level class is `Agent`; `Harness`
remains the central implementation boundary unless a future change
explicitly replaces it.

For connector convenience, store immutable connector definitions on `Agent`
and build a fresh `ConnectorRegistry` plus fresh `ToolSpec` instances for
every `run()`. Do not reset visibility on long-lived specs.

`Agent.run()` is one-shot, matching `Harness.run()`: it starts a fresh
message history each call. Subagents should remain fresh workers and should
not inherit planner state or the planner's `BeforeTurn` reminder hook by
default.

The important boundaries:

- Execution boundary: Python interpreter, not bash.
- Context boundary: snapshots, not raw payloads.
- Provider boundary: adapters copy and transform; they do not mutate harness
  state.
- State boundary: handles are explicit and valid Python identifiers.
- Subagent boundary: no implicit parent state; input/output handles are
  explicit.
- Logging boundary: reconstruct behaviour without dumping full datasets.

Still genuinely missing / deferred: container/VM/WASM-level sandboxing (the
subprocess sandbox — no network, CPU/wall-clock limits — is the current
isolation level).

## Release process

Before tagging a release:

1. `uv run ruff check data_harness tests` — must exit 0.
2. `uv run ruff format --check data_harness tests` — must exit 0.
3. `uv run pytest tests/ -m "not live"` — must exit 0.
4. Bump `version` in `pyproject.toml` and add a `CHANGELOG.md` entry
   describing what changed and why, not just what.
5. `uv build`, then install the built wheel in a clean venv and smoke-test
   the specific thing that changed before publishing — a passing test suite
   against source is not the same guarantee as the actual artifact working.
6. Commit, push to `main`, tag `vX.Y.Z`, push the tag.

Pushing the tag triggers `release.yml`, which re-runs steps 1-3, then `uv
build` and `uv publish` — a failure in any step blocks the publish. Fix lint
and format issues locally rather than pushing to trigger the workflow
speculatively. Confirm the new version is live at
`https://pypi.org/pypi/data-harness/json` before relying on it from a
downstream project (PyPI propagation can lag ~15-30s after the workflow
reports success).
