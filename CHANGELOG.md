# Changelog

### 1.3.3

- **`encode_entry`/`FORMAT_VERSION` are now public**, exported from
  `data_harness.core.session`. A caller with a live, in-memory-only session
  (no `JsonlSessionStore`, no file path) can now reproduce byte-for-byte what
  would have been written to disk — e.g. a web backend rendering "view this
  session as JSONL" for a session it never persisted to a file:
  ```python
  from data_harness.core.session import FORMAT_VERSION, encode_entry
  lines = [json.dumps({"type": "session", "version": FORMAT_VERSION, "id": sid})]
  lines += [json.dumps(encode_entry(e)) for e in session.branch()]
  ```
  Previously this required reaching into `data_harness.core.session.jsonl`
  directly, an internal module with no stability guarantee.

### 1.3.2

- **`ChartArtifact.handle` is now populated.** Both chart-capture sites
  (`PythonInterpreter._capture_charts`, `SubprocessPythonInterpreter._merge_result`)
  construct the artifact, `cache.put()` it, then set `.handle` to the actual
  returned name (accounting for `SessionCache`'s auto-suffix on a name
  collision, e.g. `chart_2`). The field existed in the dataclass since it was
  first added — documented as "the cache handle it's stored under" — but was
  never actually set, so it silently stayed `None`. A caller with just a
  `ChartArtifact` (e.g. from `RunResult.charts`) had no way back to the name
  it's addressable by without re-deriving it from `SessionCache.list_charts()`
  order.

### 1.3.1

- **`resolve_adapter`/`resolve_async_adapter` forward extra keyword arguments**
  to the adapter class they resolve to, e.g. `resolve_adapter("claude-haiku-4-5-20251001", max_tokens=2048)`.
  Previously they only accepted `model`, so tuning anything else (`max_tokens`,
  ...) meant importing the specific provider class by hand — exactly the
  "why do I need to write different adaptors" friction from 1.1.0's provider
  registry, just one layer up. `examples/advanced_wiring.py` and
  `examples/quickstart.py` updated to use `resolve_adapter`/`Agent(model=...)`
  instead of manually importing `AnthropicAdapter`.

### 1.3.0

- **No more `**kwargs` in the public constructors.** `Agent.__init__`,
  `AsyncAgent.__init__`, `Agent.from_dataframe`, and `Agent.from_csv` now
  name every parameter explicitly instead of forwarding an opaque
  `**kwargs: Any` to `_AgentBase`. This was a real, not cosmetic, problem:
  the API reference (mkdocstrings/griffe) could not see through the
  `**kwargs` forwarding, so it rendered an incomplete signature and warned
  about documented parameters ("`max_turns`", "`run_dir`", ...) that
  appeared to not exist. The docs now match the actual code exactly, with
  zero griffe warnings on a full site build. Every call site — positional
  or keyword, in this codebase or a caller's — is unaffected; only the
  internal `**kwargs` plumbing changed.

### 1.2.0

- **`Agent(system=..., model=...)`.** The constructor now resolves an adapter
  from a model id directly, the same way `Agent.from_dataframe(model=...)`
  and `ask()` always have. Previously the plain constructor required a
  pre-built `adapter=`, even though the resolution logic already existed
  elsewhere in the codebase — `Agent(system="...", model="claude-sonnet-4-6")`
  now works without constructing `AnthropicAdapter` by hand. `adapter=` still
  works and takes priority when both are given. `AsyncAgent` gets the same
  `model=`. Fully backward compatible: every existing `Agent(adapter=...)`
  call is unaffected.
- `plan/` moved to `archive/plan/` (gitignored, not pushed) — it was already
  untracked as of 1.1.0; this just gives it a clearer home on disk instead of
  a directory that looks like it vanished.

### 1.1.0

- **Provider registry.** Adding an OpenAI-compatible provider (most of them
  are) no longer needs a new adapter class — it's a
  `register_openai_compatible_provider(OpenAICompatibleProvider(...))` call.
  `OpenAIAdapter`/`AsyncOpenAIAdapter` gain `provider=`, which resolves
  `base_url`/`api_key` from the `PROVIDERS` registry (`llm/providers/openai.py`).
  Pre-registered: `openrouter`, `deepseek`, `groq`, `together`, `fireworks`,
  `cerebras`, `xai`. `OpenRouterAdapter`/`DeepSeekAdapter` are unchanged for
  existing callers — they're now implemented on top of the same mechanism
  instead of duplicating `os.environ.get(...)` lookups. Anthropic keeps its own
  adapter; its Messages API is the one genuinely different wire protocol.

### 1.0.0

Structural refactor, taking its shape from the pi agent harness. One breaking
change beyond the layer move itself: the per-turn JSONL log is gone (see
below), and the flat pre-layering import paths no longer resolve — there is
no compatibility shim.

- **Layers.** The package was one flat namespace where the loop constructed a
  `SessionCache` and called `format_tool_output`. It is now `llm` -> `core` ->
  `data` -> `app`, each importing only the layers below it, enforced
  statically by `tests/test_layers.py`. The loop takes a `RunEnvironment`
  instead of a cache, so `data_harness.core` runs an agent with no pandas
  anywhere.
- **One loop.** Four near-copies (sync/async x result/stream) collapsed into
  one generator driven by two drivers. They had drifted: the streaming copy
  discarded token usage on provider errors, and `AsyncAgent` was missing
  `from_dataframe`, subagents, MCP, the replay cache, and the approval gate.
  The sync driver runs inline, so Ctrl-C lands promptly and a handler holding
  a `sqlite3` connection still works.
- **Session tree.** A session is an append-only tree of typed entries and the
  conversation is *derived* from it, never stored. Brings resume across
  processes, forking a conversation without losing the original, reversible
  compaction, and linear rather than quadratic writes.
- **Hooks.** `BeforeTurn`, `BeforeToolCall`, `AfterToolCall`, `AfterTurn`,
  returning `Reminder`, `Block`, `Replace`, or `Stop`. The interpreter
  approval gate is now built from this mechanism rather than hardcoded inside
  the dispatcher. `AfterTurn` is where a spend cap belongs.
- **Compaction.** `max_turns` was a wall, not a strategy. Cuts land on turn
  boundaries so a tool call is never separated from its result, and nothing is
  deleted: the summary is an entry, so stepping the leaf back undoes it.
- **Typed errors.** Every failure carries a stable `code` under a common
  `DataHarnessError`, so a caller can tell a rate-limited provider from a bug
  in the model's code from a sandbox timeout. Existing base classes are kept,
  so code catching `RuntimeError`/`ValueError` still works.
- `RunResult` gains `stopped_by`; `Agent` gains `last_result` and `hooks`;
  `resolve_async_adapter` mirrors `resolve_adapter`.
- **Breaking: the `runs/*.jsonl` turn log is gone.** The session tree is now
  the sole durable record of a run, and it doesn't pay the old log's
  quadratic per-turn write cost. Removed: `RunResult.run_file`,
  `Harness`/`AsyncHarness`'s `run_file` property and `run_dir` constructor
  argument, `Agent.last_run_file`. `Agent(run_dir=...)` is unaffected — it
  still controls where chart artefacts and subagent working state land.
  Anything that read `result.run_file` for post-hoc inspection should read
  `harness.session` instead (`session.store.entries()`, or `JsonlSessionStore`
  to persist it).
- **Breaking: no legacy-path compatibility shim.** The pre-layering flat
  import paths (`data_harness.loop`, `data_harness.cache`, ...) are gone —
  import from the layered path (`data_harness.core.loop`,
  `data_harness.data.cache`, ...) or the top-level `data_harness` re-exports.
  Also removed, not deprecated: `Harness.register_reminder()` / `.reminders`
  (use `harness.on(BeforeTurn, hook)` returning `Reminder`) and
  `AsyncProviderAdapter.stream()` (use `stream_events()`).

823 tests, up from 547.


### 0.13.0
- **MCP bridge** (`[mcp]` extra): data-harness is now an MCP client. `Agent.add_mcp_server(name, command, args=...)` connects any stdio [MCP](https://modelcontextprotocol.io) server and exposes its tools as a connector — hidden until `load_connectors`, with large results routed through the `SessionCache`. Swap the command for a Postgres/SQLite/filesystem server; no per-source code.
- `MCPClient` / `MCPServer` / `mcp_tool_specs` exported for wiring MCP into a `Harness` directly; `Agent.close()` shuts servers down. `examples/mcp_demo.py` shows it live against `mcp-server-time`.

### 0.12.0
- **CLI:** a console command installed as **`dh`** (short) and `data-harness` — `dh "question" data.csv` (multiple files, or pipe CSV via stdin), with `--model`, `--no-sql`, `--json`, `--run-dir`
- **Streamlit demo app** (`examples/streamlit_app.py`, `[demo]` extra): upload a CSV, ask, see the answer + structured value + charts inline (charts passed as bytes so they always render)
- **Readable eval results:** each run now writes a human-readable `.md` leaderboard alongside the `.json`, and a committed `evals/results/SUMMARY.md` aggregates all results into one table per suite (`data_harness.eval.write_summary` / `python -m data_harness.eval.summary`)

### 0.11.0
- **Messy-data eval suite** (`messy_suite()`): real-world cleaning — amounts as strings with `$`/separators, dates in several formats, inconsistent country labels, missing values — the kind of friction that actually differentiates models. Ground truth computed in-suite by a reference cleaner. `eval_demo --suite messy`.
- **README**: centered hero (deepagents-style) and a static `python 3.10+` badge (the dynamic `pyversions` badge showed "missing" with no trove classifiers); added Python/license trove classifiers to package metadata.

### 0.10.0
- **Large-data eval suite** (`large_data_suite()`): ~100k-row frames answerable only via the cache **handle**, plus a **snapshot trap** that fails any model reading the sample rows instead of computing over the full data — directly stresses the handle/snapshot design
- **Cheaper, more diverse default lineup:** dropped `claude-haiku-4.5` (far pricier than comparable open models) and standardised on recent models across five providers — DeepSeek, Qwen, OpenAI (`gpt-5-nano`), Google (`gemini-2.5-flash-lite`), Z.ai (`glm-4.7-flash`)
- `eval_demo` gains `--suite large`

### 0.9.0
- **Revamped eval suite to stretch the design:** a new `hard_suite()` with multi-table joins, deep multi-step reasoning, and **stateful multi-turn conversations**
- **Multi-turn eval primitive:** `ConversationCase` + `Turn` run graded turns over one `Chat` session, testing `SessionCache` persistence across turns (what single-shot benchmarks can't)
- **Tracked results in-repo:** example runners write timestamped JSON to a committed `evals/results/` directory (diffable over time); `runs/` stays gitignored
- Dropped `gpt-4o-mini` from eval lineups (too old for a meaningful comparison); defaults are recent models only
- Expanded the [Evaluation guide](https://maxkskhor.github.io/data-harness/guide/evaluation/) with a full explanation of the suite

### 0.8.0
- **WikiTableQuestions as a tracked metric:** `load_wikitablequestions()` now uses the parquet-native `lighteval/wikitablequestions` mirror (the old script-based dataset no longer loads); a harder public benchmark that differentiates models the bespoke suite saturates
- **Machine-readable reports:** `EvalReport.to_dict()` / `to_json()` (accuracy, per-model/per-category, cost, every case result) for tracking results over time
- `examples/eval_wtq.py` runs the benchmark across models and writes a timestamped JSON report; a live, key-gated WTQ smoke test enables CI/nightly tracking

### 0.7.0
- **`answer()` reliability:** `ask()` now finalises by default — if a successful run produced no structured answer, it runs one focused follow-up turn asking the model to record it via `answer()`, so `.value` is populated more reliably (`require_answer=True`, default)
- The finalize step is **guarded**: it never fires when a chart was produced (the chart is the deliverable) or the answer reads as a refusal (so unanswerable questions aren't turned into fabricated values)
- `Chat`/`SmartFrame` keep `require_answer=False` by default (conversational); opt in per instance
- **Eval cost reporting:** `EvalReport` leaderboards can show per-model USD cost; `fetch_openrouter_prices()` pulls live prices, and `eval_demo` includes a cost column
- Refreshed default eval lineup to current models (DeepSeek V4, recent Qwen); dropped the older `deepseek-chat` (V3) alias from examples/tests
- `EvalCase` now uses identity equality (avoids DataFrame-truthiness errors when comparing cases)

### 0.6.0
- **Evaluation harness (`data_harness.eval`):** define `EvalCase`s with programmatic graders (`numeric`, `contains`, `exact`, `dataframe_equals`, `chart_produced`, `refuses`, `all_of`/`any_of`), run with `evaluate` / `evaluate_matrix`, and read an `EvalReport` (accuracy, leaderboard, per-category, failures)
- Grading leans on the structured `RunResult.value`; the model matrix runs across providers via OpenRouter
- Built-in `bespoke_suite()` plus a public-benchmark loader `load_wikitablequestions()` (`[eval]` extra)

### 0.5.0
- **Entry points:** `ask(df, "...")` one-liner, `Chat`/`SmartFrame`, zero-config provider resolution, `Agent.from_dataframe` / `from_csv`, and a `%%ask` notebook magic
- **OpenRouter & DeepSeek:** `OpenRouterAdapter` + `OpenAIAdapter(base_url=...)`; `provider/model` ids (e.g. `anthropic/claude-3.5-sonnet`) auto-route to OpenRouter, `deepseek-*` ids to DeepSeek's direct API, with `OPENROUTER_API_KEY` / `DEEPSEEK_API_KEY` picked up automatically — one key for many providers
- **Charts:** matplotlib in the interpreter; open figures captured as `ChartArtifact` handles (bytes stay out of messages/logs); `RunResult.charts` + rich Jupyter display
- **Structured results:** `answer(value)` interpreter helper → `RunResult.value`
- **SQL:** `sql_query` tool (DuckDB in-process over cached frames, or a SQLAlchemy URL); `Agent.enable_sql`
- **Semantic layer:** per-handle column/units descriptions folded into snapshots (`cache.put(..., semantics=...)`, `cache.describe`)
- **Subprocess sandbox:** `execution="subprocess"` runs interpreter code in an isolated process (no network, CPU/time limits)
- **Approval gate:** `on_code` callback and `code_only` dry-run
- **Code-replay cache:** `Agent.enable_cache(...)` replays repeat questions with zero model calls
- New optional extras: `[viz]`, `[duckdb]`, `[sql]`, `[notebook]`, `[all]`

### 0.4.0
- `python_interpreter`: runtime errors now raise `PythonInterpreterError` so the harness marks `ToolResultBlock.is_error=True`
- `python_interpreter`: final-expression capture — bare expressions return their repr automatically (notebook-like behaviour)
- `python_interpreter`: `locals()` usage detected at AST level and returns a targeted error with `list_variables` guidance
- `python_interpreter`: improved empty-output message directs the model to `print(...)` or `save(name, value)`
- `python_interpreter`: strengthened tool description with explicit guidance on handle usage, stdout capture, fresh locals, and `save()`

### 0.3.0
- Streaming protocol: SSE event types, `stream_events()`, `AsyncAgent.run_stream()`

### 0.2.0
- Async support: `AsyncAgent`, `AsyncAgentSession`, `AsyncHarness`
- `AgentSession` for multi-turn conversations
- `RunResult` with token usage and cache state

### 0.1.0
- Initial release: `Agent`, `Harness`, `SessionCache`, `ProviderAdapter`
