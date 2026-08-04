# AGENTS.md / CLAUDE.md

This file provides guidance to coding agents working in this repository.

## What this is

`data-harness` (data + harness) is developing into a full SDK/framework for controlled data-agent workflows.

The repository began as a readable reference implementation of harness design. That teaching role is now being split out: after the full SDK stabilises, create a separate `learn-data-harness` repository that extracts the basic principles without async, production sandboxing, broader SDK ergonomics, or other framework-heavy concerns.

This repository is now the implementation and framework track. It should still make the core boundaries inspectable, but it no longer needs to stay artificially small for tutorial clarity.

The model operates through a constrained Python interpreter only — no bash tool. Large objects live in `SessionCache` and are exposed to the model as named handles with compact snapshots, never as raw blobs in message history.

## Current Direction

Build `data-harness` as the full SDK proper. Keep the foundational design visible, but allow the API surface to grow when it supports real data-agent workflows, debuggability, integration, or reproducibility.

When there is tension between SDK completeness and teaching simplicity, keep the full implementation in `data-harness` and record the distilled teaching version for the future `learn-data-harness` repo. Do not downgrade or avoid useful SDK features merely to keep this repository looking like a small tutorial.

The future `learn-data-harness` repo should be the clean teaching resource: basic ReAct loop, Python tool boundary, handle/snapshot cache, progressive tools, logging, and explicit subagent transfer. It should be created after `data-harness` has enough stable SDK shape that the guide will not immediately go stale.

Do not erase the core harness invariants. `data-harness` can become a framework, but it should remain explicit about execution, context, provider, state, subagent, and logging boundaries.

## Shipped surface (as of v1.0.0)

A snapshot so agents stop re-proposing work that already exists. Confirm against the code before relying on it.

- Layers: `llm` -> `core` -> `data` -> `app`, each importing only the layers below it, enforced statically by `tests/test_layers.py`. There is no compatibility shim for the pre-layering flat import paths (`data_harness.loop`, `data_harness.cache`, ...) — they were removed in v1.0.0. Import from the layered path (`data_harness.core.loop`, `data_harness.data.cache`, ...) or the top-level `data_harness` re-exports in `__init__.py`.
- Entry points: `ask(df, "...")` one-liner, `Chat` / `SmartFrame`, `Agent.from_dataframe` / `from_csv`, `resolve_adapter` (env-based provider resolution), and a `%%ask` notebook magic (`data_harness/app/notebook.py`, `data_harness/app/pandas.py`).
- Providers: `AnthropicAdapter`, `OpenAIAdapter` (accepts `base_url`/`api_key`), `OpenRouterAdapter` (OpenAI-compatible, `provider/model` ids, `OPENROUTER_API_KEY`), and `DeepSeekAdapter` (direct, `deepseek-*` ids, `DEEPSEEK_API_KEY`) — all behind `ProviderAdapter` / `AsyncProviderAdapter`. `resolve_adapter` routes `provider/model` ids to OpenRouter and `deepseek-*` to DeepSeek direct.
- One loop, two drivers: `_HarnessBase._plan` in `core/loop.py` is the single generator; `Harness` runs it inline (sync), `AsyncHarness` awaits it and adds streaming (`run_stream()` / `ask_stream()`, SSE event types in `llm/streaming.py`).
- Multi-turn: `AgentSession` / `AsyncAgentSession` over a shared cache and history.
- Session tree: `core/session/` — an append-only tree of typed entries (`MessageEntry`, `TurnEntry`, `CompactionEntry`, ...) with a movable leaf; the conversation is derived by walking root to leaf, never stored separately. `JsonlSessionStore` persists it to disk (one line per entry); `MemorySessionStore` is the default.
- Hooks: `BeforeTurn` / `BeforeToolCall` / `AfterToolCall` / `AfterTurn` returning `Reminder` / `Block` / `Replace` / `Stop` (`core/hooks.py`). The interpreter approval gate (`on_code`, `code_only`) is built on this.
- Compaction: `core/compaction.py` — cuts land on turn boundaries, summarised turns stay in the tree so moving the leaf back restores them (opt-in via `compactor=`).
- Typed errors: `DataHarnessError` taxonomy with stable `code` strings (`core/exceptions.py`); existing `RuntimeError`/`ValueError`/`KeyError` bases kept for backward compatibility.
- Typed results: `RunResult` (`value` + `charts` + `stopped_by`), `Usage`, `CacheStorageInfo` (`core/result.py`); `run_result()` / `ask_result()`.
- Structured answers: `answer(value)` interpreter helper → `RunResult.value`.
- Charts: matplotlib captured as `ChartArtifact` handles (`core/artifacts.py`); image bytes never enter messages or the session tree.
- State: `SessionCache` with hot/cold disk spilling (Parquet / `.npy` / pickle), an answer slot, chart tracking, and a semantic layer (`put(..., semantics=...)`, `describe`).
- Tools: `python_interpreter`, `list_variables`, `sql_query` (DuckDB / SQLAlchemy, `tools/sql.py`), opt-in `Planner`, `ConnectorRegistry` progressive disclosure, and isolated `subagent`.
- Execution: in-process or `execution="subprocess"` (`tools/sandbox.py` + `_sandbox_runner.py`) — isolated process, no network, CPU/time limits; shares the security core via `interpreter.execute_namespace`.
- Controls: `on_code` approval gate + `code_only` dry-run (in `Harness`); code-replay cache (`exec_cache.py`, `Agent.enable_cache`).
- File ingestion: `io.py` (`load_dataframe`, `to_handles`).
- Tool metadata: `ToolAnnotations` on `ToolSpec` (not leaked to providers).
- Observability: the session tree is the durable run record; `Agent(run_dir=...)` still controls where chart artefacts and subagent working state land. See `examples/inspect_run.py` for `RunResult` inspection and `data_harness/core/observe.py` for turn-latency timing.
- Packaging: published to PyPI as `data-harness`; optional extras `[openai]`, `[viz]`, `[duckdb]`, `[sql]`, `[notebook]`, `[mcp]`, `[eval]`, `[demo]`, `[all]`; MkDocs site under `docs/`.
- **Removed in v1.0.0 (breaking, no compat shims kept):** the per-turn JSONL log (`RunResult.run_file`, `Harness(run_dir=...)`, `Agent.last_run_file`); the flat pre-layering import paths (`data_harness/_legacy_paths.py` and everything it aliased); `Harness.register_reminder()` / `.reminders` (use `harness.on(BeforeTurn, hook)` returning a `Reminder`); `AsyncProviderAdapter.stream()` (use `stream_events()`). Do not re-add any of these for backward compatibility — the project has deliberately chosen not to carry legacy surface.

Still genuinely missing / deferred: container/VM/WASM-level sandboxing (the subprocess sandbox is the current isolation; PLAN_v4).

## Motivation

This project is motivated by data workflows where:

- unrestricted bash access is undesirable
- tool results can be large
- many data connectors may exist
- long-running conversations need disciplined context management
- debugging and reproducibility matter

The core design ideas remain:

- the ReAct loop should stay small and readable
- Python is the controlled execution surface for data work
- large tool outputs become handles plus snapshots
- connector schemas are progressively disclosed
- the system prompt stays prefix-stable
- reminders and dynamic state are appended to the conversation suffix
- subagent state transfer is explicit
- logs reconstruct runs without dumping raw datasets
- architectural invariants are tested directly

## Commands

```bash
uv sync
uv run pytest tests/ -m "not live"                              # full offline suite (~800 tests)
uv run pytest tests/smoke_tests.py -v -m live                   # live provider smoke tests
uv run pytest tests/test_loop.py::TestLoopBasic::test_exits_on_end_turn -v
uv run python examples/quickstart.py                            # minimal Agent (ANTHROPIC_API_KEY)
uv run python examples/advanced_wiring.py                       # connectors + planner + subagents
uv run python examples/inspect_run.py                           # RunResult inspection
uv run mkdocs serve                                             # preview the docs site
```

Live smoke tests run through OpenRouter: they require `OPENROUTER_API_KEY` and
default to `openai/gpt-4o-mini`; set `DATA_HARNESS_SMOKE_MODEL` to override (e.g.
`deepseek/deepseek-chat`). The advanced demo requires `ANTHROPIC_API_KEY`. Both
may cost tokens.

## Architecture

The core loop lives in `core/loop.py`: `_HarnessBase._plan` is the single generator, driven by `Harness` (sync) and `AsyncHarness` (async, adds streaming). It owns the message list, dispatches tools, applies hooks, filters visible tools, and records every turn onto the session tree (`core/session/`). `data/harness.py`'s `Harness`/`AsyncHarness` wire the domain-free core loop to a `SessionCache`; that pair is what `Agent` builds and what most callers actually use.

`run_result()` / `ask_result()` return a typed `RunResult` (`core/result.py`) carrying text, token `Usage`, and `CacheStorageInfo` — never raw payloads. The model-visible built-in tools are `python_interpreter` and `list_variables` (`data/tools/variables.py`); the planner, connectors, and subagent are opt-in.

The harness never mutates the system prompt. Reminders, nags, final-turn warnings, tool results, and dynamic state belong in the conversation suffix — reminders arrive as `BeforeTurn` hooks returning `Reminder` (`core/hooks.py`).

`SessionCache` (`data/cache.py`) is shared state between the harness and tools. Tools store large objects by handle name. The model sees snapshots and can operate on handles through the Python interpreter.

`python_interpreter` injects cache handles as local variables and exposes `save(name, value)` for explicit persistence. It should not expose the cache object itself.

`ToolSpec` (`llm/types.py`) carries the model-visible tool contract plus the already-bound handler callable. The loop dispatches with `handler(**tool_input)` and does not know how dependencies were wired.

Connectors (`data/tools/connectors.py`) are registered hidden and flipped visible on demand through `load_connectors`. This is the progressive disclosure pattern.

Provider adapters (`llm/providers/base.py`) normalize external provider APIs into `NormalizedResponse`. Adapters copy and transform inputs; they must not mutate harness-owned `system`, `messages`, or `tools`.

Subagents (`data/tools/subagent.py`) get a fresh adapter, fresh message history, and fresh cache per spawn. Parent state crosses the boundary only through explicit `input_handles`, and created outputs return only through `publish_created`.

## Key invariants

These are design constraints, not incidental behavior:

- System prompt is byte-identical across turns.
- Adapters never mutate harness-owned state.
- Dynamic reminders are suffix-only.
- Tool-use messages are followed by matching tool-result messages before the next assistant call.
- Raw large payloads stay in `SessionCache`; messages and the session tree receive snapshots only.
- Cache handles are valid Python identifiers.
- `python_interpreter` uses fresh locals per call.
- Subagents do not inherit parent cache implicitly.
- Subagents cannot recursively register or call `subagent`.
- The session tree must support run reconstruction without raw payload leakage.

Tests should assert these invariants directly.

## Writing Context

This repo supports writing about harness and SDK engineering. The posts should be clear about the split: `data-harness` is the fuller implementation/framework track, while `learn-data-harness` is the planned teaching guide for the distilled principles.

There are two intended writing tracks:

1. Architecture: why this harness design exists.
   Topics: no bash, Python execution, progressive disclosure, handle/snapshot, prefix-stable suffix-mutable context, planner reminders, subagents, list_variables, failure modes.

2. Implementation invariants: the technical details that keep the design coherent.
   Topics: typed content blocks, adapter mutation boundary, tool-use ordering, inline-vs-cache formatting, handle naming, interpreter `save`, suffix-only reminders, explicit subagent state transfer, the session tree, invariant tests.

When updating README or docs, avoid claiming `data-harness` is still only a teaching/reference implementation. It is now developing into the full SDK/framework. If a simpler teaching resource is needed, point to the planned `learn-data-harness` split rather than forcing `data-harness` docs to stay beginner-oriented.

All writing in this repo uses British English spelling (e.g. behaviour, normalise, serialise, catalogue, artefact).

## Implementation Guidance

Prefer clarity over accidental abstraction, but do not impose the old "small enough for an afternoon" constraint on framework work.

Framework-style indirection is acceptable when it supports a real SDK responsibility: stable public API, provider integration, session/result metadata, reproducibility, observability, sandboxing, async execution, or safe extensibility. It is not acceptable when it hides state, weakens cache boundaries, or makes run reconstruction harder.

For SDK work, keep the public surface coherent and tested. It may grow beyond the original `Agent` convenience layer, but new APIs should map cleanly onto the underlying runtime concepts.

The high-level class is still `Agent`, and `Harness` remains the central implementation boundary unless a later plan explicitly replaces it.

The old strict SDK complexity budget is superseded. Use the current plan files to decide scope:

- `plan/PLAN_v4.md`: runtime roadmap. Async (v0.2.0) and streaming (v0.3.0) have shipped; container-level sandboxing remains deferred.
- `plan/PLAN_v5.md`: shipped — typed `RunResult`, shared turn records, `ToolAnnotations`, and session inspection.
- `plan/PLAN_TEACHING.md`: future `learn-data-harness` teaching split after the SDK stabilises.

Note: the `plan/` and `blogs/` files still use the old `dataact` package name in places. The package is `data-harness` (`data_harness`); treat `dataact` references there as historical.

For connector convenience, store immutable connector definitions on `Agent` and build a fresh `ConnectorRegistry` plus fresh `ToolSpec` instances for every `run()`. Do not reset visibility on long-lived specs.

`Agent.run()` is one-shot, matching `Harness.run()`: it starts a fresh message history each call. Subagents should remain fresh workers and should not inherit planner state or the planner's `BeforeTurn` reminder hook by default.

The important boundaries are:

- Execution boundary: Python interpreter, not bash.
- Context boundary: snapshots, not raw payloads.
- Provider boundary: adapters copy and transform; they do not mutate harness state.
- State boundary: handles are explicit and valid Python identifiers.
- Subagent boundary: no implicit parent state; input/output handles are explicit.
- Logging boundary: reconstruct behavior without dumping full datasets.

## Release checklist

Before merging any branch that will trigger a PyPI release:

1. Run `uv run ruff check data_harness tests` — must exit 0.
2. Run `uv run ruff format --check data_harness tests` — must exit 0.
3. Run `uv run pytest tests/ -m "not live"` — must exit 0.

The `release.yml` workflow runs all three steps before `uv build` and `uv publish`. A failure in any step blocks the publish. Fix lint and format issues locally before pushing release tags or triggering the workflow manually.
