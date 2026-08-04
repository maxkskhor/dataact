"""High-level `Agent` and `AsyncAgent` convenience layers.

`Agent` wraps `Harness` for sync workflows. `AsyncAgent` wraps `AsyncHarness`
for async and streaming workflows.

Both are one-shot per `run()` call. Use `session()` / `async_session()` for
multi-turn conversations over a shared message history and cache.

Configuration, tool wiring, connectors, MCP, subagents, and the replay cache
all live once in `_AgentBase`. The two public classes differ only in which
adapter kind they take and whether their run methods are coroutines. They used
to be independent copies, which is how `AsyncAgent` silently ended up missing
`from_dataframe`, subagents, MCP, the replay cache, and the approval gate.
"""

from __future__ import annotations

import asyncio
import functools
import uuid
from collections.abc import AsyncGenerator, Callable
from contextlib import aclosing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from data_harness.core.hooks import Decision, Event, HookRegistry
from data_harness.core.result import CacheStorageInfo, RunResult, Usage, unwrap_text
from data_harness.core.schema import infer_input_schema
from data_harness.data.cache import SessionCache
from data_harness.data.harness import AsyncHarness, Harness, run_coroutine_blocking
from data_harness.data.tools.connectors import ConnectorRegistry
from data_harness.data.tools.interpreter import PythonInterpreter
from data_harness.data.tools.planner import Planner
from data_harness.data.tools.subagent import _copy_cache_value, make_subagent_spec
from data_harness.data.tools.variables import make_list_variables_spec
from data_harness.llm.providers.base import AsyncProviderAdapter, ProviderAdapter
from data_harness.llm.streaming import StreamEvent
from data_harness.llm.types import ToolAnnotations, ToolSpec


@dataclass(frozen=True)
class _ConnectorToolDefinition:
    connector_name: str
    fn: Callable[..., Any]
    description: str
    input_schema: dict
    annotations: ToolAnnotations | None = None


@dataclass(frozen=True)
class _ConnectorDefinition:
    name: str
    description: str


class ConnectorBuilder:
    """Fluent builder for attaching tools to a named connector on an `Agent`.

    Obtain an instance via `Agent.connector` rather than constructing directly.
    """

    def __init__(self, agent: _AgentBase, name: str) -> None:
        self._agent = agent
        self._name = name

    def tool(
        self,
        fn: Callable[..., Any],
        *,
        description: str,
        input_schema: dict | None = None,
        annotations: ToolAnnotations | None = None,
    ) -> Callable[..., Any]:
        """Register ``fn`` as a tool under this connector.

        Args:
            fn: The callable to expose as a tool. Its name becomes the tool
                name (prefixed with the connector name).
            description: Natural-language description shown to the model.
            input_schema: JSON Schema for the tool's parameters. Inferred from
                ``fn``'s type annotations when ``None``.
            annotations: Optional `ToolAnnotations` side-effect hints.

        Returns:
            ``fn`` unchanged, so the method can be used as a decorator.
        """
        schema = input_schema if input_schema is not None else infer_input_schema(fn)
        self._agent._connector_tools.append(
            _ConnectorToolDefinition(
                connector_name=self._name,
                fn=fn,
                description=description,
                input_schema=schema,
                annotations=annotations,
            )
        )
        return fn


_SelfT = TypeVar("_SelfT", bound="_AgentBase")


class _AgentBase:
    """Configuration, tool wiring, and feature toggles shared by both agents.

    Holds no provider I/O: subclasses own the adapter and the run methods.
    Every ``enable_*`` toggle, connector, MCP server, and the replay cache is
    defined here exactly once, so the sync and async agents cannot drift.
    """

    def __init__(
        self,
        system: str,
        *,
        max_turns: int = 25,
        cache: SessionCache | None = None,
        run_dir: str | Path | None = None,
        execution: str = "inprocess",
        sandbox_options: dict[str, Any] | None = None,
        on_code: Callable[[str], Any] | None = None,
        code_only: bool = False,
        hooks: HookRegistry | None = None,
    ) -> None:
        self._system = system
        self._max_turns = max_turns
        self._cache = cache if cache is not None else SessionCache()
        self._run_dir = run_dir
        self._execution = execution
        self._sandbox_options = sandbox_options
        self._on_code = on_code
        self._code_only = code_only
        self._last_run_file: str | None = None
        self._connectors: dict[str, _ConnectorDefinition] = {}
        self._connector_tools: list[_ConnectorToolDefinition] = []
        self._planner_enabled = False
        self._subagent_factory: Callable[[], Any] | None = None
        self._sql_enabled = False
        self._sql_engine_url: str | None = None
        self._exec_cache: Any = None
        self._mcp_clients: dict[str, Any] = {}
        self._last_result: RunResult | None = None
        self._hooks = hooks if hooks is not None else HookRegistry()

    # ── construction helpers ────────────────────────────────────────────────

    @classmethod
    def _default_adapter(cls, model: str | None) -> Any:
        raise NotImplementedError

    @classmethod
    def from_dataframe(
        cls: type[_SelfT],
        data: Any,
        *,
        adapter: Any = None,
        model: str | None = None,
        system: str | None = None,
        semantics: dict[str, dict] | None = None,
        **kwargs: Any,
    ) -> _SelfT:
        """Build an agent with ``data`` preloaded as cache handles.

        Accepts a DataFrame, a ``{name: value}`` mapping, a file path, or a list
        of paths. Resolves an adapter from ``model``/the environment and applies
        the default analyst system prompt unless overridden.
        """
        from data_harness.app.quickstart import _DEFAULT_SYSTEM
        from data_harness.data.io import to_handles

        agent = cls(
            adapter=adapter if adapter is not None else cls._default_adapter(model),
            system=system if system is not None else _DEFAULT_SYSTEM,
            **kwargs,
        )
        sem = semantics or {}
        for name, value in to_handles(data).items():
            agent.cache.put(name, value, semantics=sem.get(name))
        return agent

    @classmethod
    def from_csv(cls: type[_SelfT], path: str | Path, **kwargs: Any) -> _SelfT:
        """Build an agent from a CSV (or other supported file) path."""
        return cls.from_dataframe(str(path), **kwargs)

    # ── inspection ──────────────────────────────────────────────────────────

    @property
    def cache(self) -> SessionCache:
        return self._cache

    @property
    def last_run_file(self) -> str | None:
        return self._last_run_file

    @property
    def last_result(self) -> RunResult | None:
        """`RunResult` for the most recent run, however it was executed.

        Set by streamed runs and replay-cache hits too, neither of which
        returns a result to the caller directly.
        """
        return self._last_result

    @property
    def exec_cache(self) -> Any:
        """The `ExecutionCache`, or ``None`` if caching is disabled."""
        return self._exec_cache

    @property
    def hooks(self) -> HookRegistry:
        """Hooks applied to every harness this agent builds.

        The agent constructs a fresh `Harness` per run, so hooks have to live
        here rather than on the harness: one registered on a harness would be
        gone by the next call.
        """
        return self._hooks

    def on(
        self, event_type: type[Event], hook: Callable[[Any], Decision | None]
    ) -> None:
        """Register ``hook`` for ``event_type``. See `data_harness.core.hooks`."""
        self._hooks.add(event_type, hook)

    # ── feature toggles ─────────────────────────────────────────────────────

    def connector(self, name: str, *, description: str) -> ConnectorBuilder:
        """Register a named connector and return a builder for attaching tools.

        Connector tools start hidden; the model must call ``load_connectors``
        before it can use them (progressive disclosure).

        Args:
            name: Unique connector name. Used as the tool-name prefix.
            description: Shown to the model when it lists available connectors.

        Returns:
            A `ConnectorBuilder` for registering tools under this connector.
        """
        self._connectors[name] = _ConnectorDefinition(
            name=name, description=description
        )
        return ConnectorBuilder(self, name)

    def enable_planner(self: _SelfT) -> _SelfT:
        """Enable the planning tool and suffix-based nag reminders.

        The planner escalates reminders at turns 4, 8, and 12 when no progress
        has been recorded. Call this before `run` or `session`.

        Returns:
            ``self``, for method chaining.
        """
        self._planner_enabled = True
        return self

    def enable_subagents(self: _SelfT, *, adapter_factory: Callable[[], Any]) -> _SelfT:
        """Enable the subagent tool, using ``adapter_factory`` for spawned agents.

        Each spawned subagent gets a fresh adapter, fresh message history, and
        fresh cache. State crosses the boundary only through explicit
        ``input_handles``.

        Args:
            adapter_factory: Zero-argument callable returning a fresh adapter
                for each subagent. Either a `ProviderAdapter` or an
                `AsyncProviderAdapter`; whichever it returns picks the matching
                driver, so a sync adapter keeps the parent's threading
                semantics.

        Returns:
            ``self``, for method chaining.
        """
        self._subagent_factory = adapter_factory
        return self

    def enable_sql(self: _SelfT, *, engine_url: str | None = None) -> _SelfT:
        """Enable the ``sql_query`` tool.

        With no ``engine_url``, queries run via DuckDB in-process over the
        DataFrame handles in the cache. With a SQLAlchemy URL, queries run
        against that database instead.

        Args:
            engine_url: Optional SQLAlchemy connection URL.

        Returns:
            ``self``, for method chaining.
        """
        self._sql_enabled = True
        self._sql_engine_url = engine_url
        return self

    def enable_cache(self: _SelfT, path: Any = None) -> _SelfT:
        """Enable the code-replay cache.

        On a repeat ``run``/``run_result`` with the same question and data
        schema, the previously recorded interpreter/SQL code is replayed against
        the current cache without calling the model (zero tokens, no turns).

        Args:
            path: A JSON file path to persist the cache across processes, an
                existing `ExecutionCache` to share in-process, or ``None`` for an
                in-memory cache.

        Returns:
            ``self``, for method chaining.
        """
        from data_harness.data.exec_cache import ExecutionCache

        if isinstance(path, ExecutionCache):
            self._exec_cache = path
        else:
            self._exec_cache = ExecutionCache(path)
        return self

    def add_mcp_server(
        self: _SelfT,
        name: str,
        command: str | None = None,
        *,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        client: Any = None,
    ) -> _SelfT:
        """Connect an MCP server and expose its tools (via progressive disclosure).

        The server's tools become a connector named ``name`` — hidden until the
        model calls ``load_connectors``. Large tool results flow through the
        `SessionCache` like any other tool output.

        Args:
            name: Connector name for this server (also the tool-name prefix).
            command: Executable to launch the stdio MCP server (e.g. ``"uvx"``).
            args: Arguments for the server command.
            env: Extra environment for the server process.
            client: A pre-connected ``MCPClient`` (mainly for testing); if given,
                ``command``/``args``/``env`` are ignored.

        Returns:
            ``self``, for chaining.
        """
        if client is None:
            from data_harness.data.mcp import MCPClient, MCPServer

            client = MCPClient(
                MCPServer(command=command or "", args=args or [], env=env)
            ).connect()
        self._mcp_clients[name] = client
        return self

    def close(self) -> None:
        """Shut down any connected MCP servers."""
        for client in self._mcp_clients.values():
            try:
                client.close()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
        self._mcp_clients.clear()

    def explain(self) -> str:
        """Return a string showing the equivalent explicit `Harness` wiring."""
        return _EXPLAIN_TEMPLATE.format(
            system=_truncate(self._system),
            max_turns=self._max_turns,
            run_dir=self._run_dir if self._run_dir is not None else "./runs",
        )

    # ── tool wiring ─────────────────────────────────────────────────────────

    @property
    def _effective_run_dir(self) -> str:
        return str(self._run_dir) if self._run_dir is not None else "./runs"

    def _build_tools(
        self,
        *,
        planner: Planner | None = None,
        cache: SessionCache | None = None,
    ) -> list[ToolSpec]:
        """Assemble the full tool list for one harness."""
        target_cache = cache if cache is not None else self._cache
        artifacts_dir = str(Path(self._effective_run_dir) / "charts")

        if self._execution == "subprocess":
            from data_harness.data.tools.sandbox import SubprocessPythonInterpreter

            interpreter_spec = SubprocessPythonInterpreter.make_tool_spec(
                target_cache,
                artifacts_dir=artifacts_dir,
                **(self._sandbox_options or {}),
            )
        else:
            interpreter_spec = PythonInterpreter.make_tool_spec(
                target_cache, artifacts_dir=artifacts_dir
            )

        tools = [interpreter_spec, make_list_variables_spec(target_cache)]

        if planner is not None:
            tools.extend(planner.make_tool_specs())

        if self._sql_enabled:
            from data_harness.data.tools.sql import make_sql_query_spec

            tools.append(
                make_sql_query_spec(target_cache, engine_url=self._sql_engine_url)
            )

        if self._connectors or self._mcp_clients:
            tools.extend(self._build_connector_tools(target_cache))

        return tools

    def _build_connector_tools(self, cache: SessionCache) -> list[ToolSpec]:
        from data_harness.data.mcp import mcp_tool_specs

        registry = ConnectorRegistry()
        for connector_name, connector in self._connectors.items():
            registry.register(
                name=connector_name,
                description=connector.description,
                tools=[
                    ToolSpec(
                        name=f"{connector_name}__{definition.fn.__name__}",
                        description=definition.description,
                        input_schema=definition.input_schema,
                        handler=definition.fn,
                        visible=False,
                        annotations=definition.annotations,
                    )
                    for definition in self._connector_tools
                    if definition.connector_name == connector_name
                ],
            )
        for server_name, mcp_client in self._mcp_clients.items():
            tool_count = len(mcp_client.tools)
            registry.register(
                name=server_name,
                description=f"MCP server '{server_name}' ({tool_count} tools)",
                tools=mcp_tool_specs(mcp_client, prefix=server_name),
            )
        return [
            registry.get_load_connectors_spec(),
            *registry.make_wrapped_specs(cache),
        ]

    def _resolve_planner(self, planner: Planner | None) -> Planner | None:
        if planner is not None:
            return planner
        return Planner() if self._planner_enabled else None

    def _tools_for_harness(
        self, *, cache: SessionCache, planner: Planner | None
    ) -> list[ToolSpec]:
        """Tool list plus the subagent tool, when subagents are enabled."""
        tools = self._build_tools(planner=planner, cache=cache)
        if self._subagent_factory is None:
            return tools
        tools.append(
            make_subagent_spec(
                adapter_factory=self._subagent_factory,
                parent_tools=self._build_tools(planner=None, cache=cache),
                parent_cache=cache,
                run_dir=self._effective_run_dir,
                make_sub_tools=lambda sub_cache: self._build_tools(
                    planner=None, cache=sub_cache
                ),
            )
        )
        return tools

    def _harness_kwargs(
        self, *, cache: SessionCache | None, planner: Planner | None
    ) -> tuple[dict[str, Any], Planner | None]:
        effective_cache = cache if cache is not None else self._cache
        effective_planner = self._resolve_planner(planner)
        kwargs: dict[str, Any] = {
            "system": self._system,
            "tools": self._tools_for_harness(
                cache=effective_cache, planner=effective_planner
            ),
            "max_turns": self._max_turns,
            "cache": effective_cache,
            "on_code": self._on_code,
            "code_only": self._code_only,
            "hooks": self._hooks,
        }
        if self._run_dir is not None:
            kwargs["run_dir"] = str(self._run_dir)
        return kwargs, effective_planner

    # ── replay cache ────────────────────────────────────────────────────────

    def _replay_key(self, user_message: str) -> str | None:
        if self._exec_cache is None:
            return None
        from data_harness.data.exec_cache import make_key

        return make_key(user_message, self._cache, self._system)

    def _replay_steps(self, cached: Any) -> list[tuple[Callable[..., Any], dict]]:
        """The dispatchable ``(handler, input)`` pairs recorded by a cached run."""
        tool_map = {t.name: t for t in self._build_tools(cache=self._cache)}
        steps = []
        for step in cached.steps:
            spec = tool_map.get(step["tool"])
            if spec is not None and spec.handler is not None:
                steps.append((spec.handler, step["input"]))
        return steps

    def _replay_result(self, text: str) -> RunResult:
        result = self._build_replay_result(text)
        self._last_result = result
        return result

    def _build_replay_result(self, text: str) -> RunResult:
        return RunResult(
            text=text,
            status="success",
            turns=0,
            run_file=None,
            stop_reason=None,
            usage=Usage(),
            cache_snapshots=self._cache.list_handles(),
            cache_storage={
                name: CacheStorageInfo(
                    location=meta["location"], storage_type=meta["storage_type"]
                )
                for name, meta in self._cache.storage_metadata().items()
            },
            value=self._cache.get_answer(),
            charts=self._cache.list_charts(),
            run_id=str(uuid.uuid4()),
        )

    def _record_replay(self, key: str | None, harness: Any, result: RunResult) -> None:
        if key is None or result.status != "success":
            return
        from data_harness.data.exec_cache import CachedRun, extract_steps

        self._exec_cache.put(
            key, CachedRun(steps=extract_steps(harness.messages), text=result.text)
        )


class Agent(_AgentBase):
    """High-level synchronous agent.

    `Agent` composes a `Harness`, a `SessionCache`, and optional tools from a
    single configuration. Each call to `run` builds a fresh `Harness` with a
    fresh message history. Use `session` when you need multi-turn conversation
    state to persist across questions.

    Example::

        from data_harness import Agent
        from data_harness.llm.providers.anthropic import AnthropicAdapter

        agent = Agent(
            adapter=AnthropicAdapter(model="claude-sonnet-4-6"),
            system="You are a data analyst.",
        )
        print(agent.run("Compute the mean of [1, 2, 3]."))

    Args:
        adapter: Synchronous provider adapter.
        system: System prompt passed unchanged to every `Harness` run.
        max_turns: Hard cap on provider turns per `run` call.
        cache: Shared `SessionCache`. A fresh cache is created when ``None``.
        run_dir: Directory for JSONL logs. Defaults to ``./runs``.
        execution: ``"inprocess"`` or ``"subprocess"`` interpreter isolation.
        sandbox_options: Extra options for the subprocess interpreter.
        on_code: Approval gate called with interpreter code before it runs.
        code_only: When ``True``, interpreter code is echoed, never executed.
    """

    def __init__(
        self,
        adapter: ProviderAdapter,
        system: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(system, **kwargs)
        self._adapter = adapter
        self._last_harness: Harness | None = None

    @classmethod
    def _default_adapter(cls, model: str | None) -> ProviderAdapter:
        from data_harness.app.quickstart import resolve_adapter

        return resolve_adapter(model)

    @property
    def last_harness(self) -> Harness | None:
        return self._last_harness

    def _replay(self, cached: Any) -> RunResult:
        """Re-execute a cached run's tool steps inline, without calling the model."""
        for handler, tool_input in self._replay_steps(cached):
            try:
                if asyncio.iscoroutinefunction(handler):
                    run_coroutine_blocking(handler(**tool_input))
                else:
                    handler(**tool_input)
            except Exception:  # noqa: BLE001
                # A recorded step may fail against fresh data; skip it rather
                # than aborting the whole replay.
                continue
        return self._replay_result(cached.text)

    def session(self) -> AgentSession:
        """Create a stateful `AgentSession` for multi-turn conversations.

        Returns:
            A new `AgentSession` backed by a copy of this agent's cache.
        """
        return AgentSession(self)

    def run_result(self, user_message: str) -> RunResult:
        """Run the agent and return the full `RunResult`.

        Builds a fresh `Harness` with a fresh message history for each call.

        Args:
            user_message: The user prompt to send.

        Returns:
            A `RunResult` with the text response, token usage, and cache state.
        """
        key = self._replay_key(user_message)
        if key is not None:
            cached = self._exec_cache.get(key)
            if cached is not None:
                return self._replay(cached)

        harness = self._make_harness()
        self._last_harness = harness
        result = harness.run_result(
            user_message, run_id=str(uuid.uuid4()), session_id=None
        )
        self._last_run_file = harness.run_file
        self._last_result = result
        self._record_replay(key, harness, result)
        return result

    def run(self, user_message: str) -> str:
        """Run the agent and return the final text response.

        Args:
            user_message: The user prompt to send.

        Returns:
            The model's final text response.

        Raises:
            MaxTurnsExceeded: If the loop reaches ``max_turns``.
            RuntimeError: If the provider raises an exception.
        """
        return unwrap_text(self.run_result(user_message))

    def _make_harness(
        self,
        *,
        cache: SessionCache | None = None,
        planner: Planner | None = None,
    ) -> Harness:
        kwargs, effective_planner = self._harness_kwargs(cache=cache, planner=planner)
        harness = Harness(adapter=self._adapter, **kwargs)
        if effective_planner is not None:
            harness.register_reminder(effective_planner.reminder_hook)
        return harness


class AsyncAgent(_AgentBase):
    """Async agent for use with `AsyncProviderAdapter`.

    `run()` and `run_result()` are coroutines. `run_stream()` is an async
    generator that yields `StreamEvent`s as they arrive from the provider.
    Use `async_session()` for multi-turn streaming conversations.

    Takes the same configuration and supports the same features as `Agent`:
    connectors, MCP servers, SQL, the planner, subagents, the replay cache,
    and the interpreter approval gate.
    """

    def __init__(
        self,
        adapter: AsyncProviderAdapter,
        system: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(system, **kwargs)
        self._adapter = adapter
        self._last_harness: AsyncHarness | None = None

    @classmethod
    def _default_adapter(cls, model: str | None) -> AsyncProviderAdapter:
        from data_harness.app.quickstart import resolve_async_adapter

        return resolve_async_adapter(model)

    @property
    def last_harness(self) -> AsyncHarness | None:
        return self._last_harness

    async def _replay(self, cached: Any) -> RunResult:
        """Re-execute a cached run's tool steps without stalling the event loop."""
        for handler, tool_input in self._replay_steps(cached):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(**tool_input)
                else:
                    await asyncio.to_thread(functools.partial(handler, **tool_input))
            except Exception:  # noqa: BLE001
                # A recorded step may fail against fresh data; skip it rather
                # than aborting the whole replay.
                continue
        return self._replay_result(cached.text)

    def async_session(self) -> AsyncAgentSession:
        """Create a stateful `AsyncAgentSession` for multi-turn conversations."""
        return AsyncAgentSession(self)

    async def run_result(self, user_message: str) -> RunResult:
        """Run the agent and return the full `RunResult`."""
        key = self._replay_key(user_message)
        if key is not None:
            cached = self._exec_cache.get(key)
            if cached is not None:
                return await self._replay(cached)

        harness = self._make_harness()
        self._last_harness = harness
        result = await harness.run_result(
            user_message, run_id=str(uuid.uuid4()), session_id=None
        )
        self._last_run_file = harness.run_file
        self._last_result = result
        self._record_replay(key, harness, result)
        return result

    async def run(self, user_message: str) -> str:
        """Run the agent and return the final text response.

        Raises:
            MaxTurnsExceeded: If the loop reaches ``max_turns``.
            RuntimeError: If the provider raises an exception.
        """
        return unwrap_text(await self.run_result(user_message))

    async def run_stream(self, user_message: str) -> AsyncGenerator[StreamEvent, None]:
        """Stream events for a one-shot run.

        Yields StreamEvent objects (message_start, content_block_*, message_delta,
        message_stop, tool_result) following the Claude Agent SDK protocol.

        After the generator is exhausted, ``agent.last_result`` holds the
        `RunResult` for the run, including token usage. Read it there rather
        than through ``last_harness``: a replay-cache hit builds no harness, so
        ``last_harness`` is either ``None`` or left over from an earlier run.

        A cache hit yields the cached answer as a single synthetic text block
        and reports zero usage, so a streaming caller sees the same answer a
        non-streaming one would.

        Usage::

            async for event in agent.run_stream("hello"):
                if event.type == "content_block_delta":
                    from data_harness.llm.streaming import TextDelta
                    if isinstance(event.delta, TextDelta):
                        print(event.delta.text, end="", flush=True)
        """
        key = self._replay_key(user_message)
        if key is not None:
            cached = self._exec_cache.get(key)
            if cached is not None:
                # `_replay` mints its own run id; nothing here to stamp.
                result = await self._replay(cached)
                for event in _synthetic_text_events(result.text):
                    yield event
                return

        run_id = str(uuid.uuid4())
        harness = self._make_harness()
        self._last_harness = harness
        # `aclosing`, not a bare `async for`: closing this generator does not
        # close the one it iterates, so abandoning the stream here would leave
        # the harness (and the provider's HTTP stream) suspended.
        async with aclosing(harness.run_stream(user_message)) as events:
            async for event in events:
                yield event
        self._last_run_file = harness.run_file
        if harness.last_result is not None:
            self._last_result = harness._stamp(harness.last_result, run_id, None)
            self._record_replay(key, harness, harness.last_result)

    def _make_harness(
        self,
        *,
        cache: SessionCache | None = None,
        planner: Planner | None = None,
    ) -> AsyncHarness:
        kwargs, effective_planner = self._harness_kwargs(cache=cache, planner=planner)
        harness = AsyncHarness(adapter=self._adapter, **kwargs)
        if effective_planner is not None:
            harness.register_reminder(effective_planner.reminder_hook)
        return harness


class _SessionBase:
    """Shared state for `AgentSession` and `AsyncAgentSession`.

    A session owns its own `SessionCache`, seeded with a deep copy of the
    agent's handles so a session cannot mutate the agent definition it was
    built from.
    """

    def __init__(self, agent: _AgentBase) -> None:
        self._agent = agent
        self._cache = SessionCache(
            sample_size=agent.cache.sample_size,
            storage_dir=None,
            hot_limit=agent.cache.hot_limit,
        )
        for name, value in agent.cache.items():
            self._cache.put(
                name,
                _copy_cache_value(value),
                semantics=agent.cache.get_semantics(name),
            )
        self._harness = agent._make_harness(cache=self._cache)  # type: ignore[attr-defined]
        self._id: str = str(uuid.uuid4())
        self._last_result: RunResult | None = None
        self._turns: int = 0

    @property
    def id(self) -> str:
        return self._id

    @property
    def last_result(self) -> RunResult | None:
        return self._last_result

    @property
    def turns(self) -> int:
        return self._turns

    @property
    def cache(self) -> SessionCache:
        return self._cache

    @property
    def run_file(self) -> str | None:
        return self._harness.run_file

    def put(self, name: str, value: Any, *, overwrite: bool = False) -> str:
        """Store a value in the session cache and return the handle used.

        Args:
            name: Desired handle name. Must be a valid Python identifier.
            value: Any Python object to store.
            overwrite: Replace the existing handle if ``True``.

        Returns:
            The handle name under which the value was stored.
        """
        return self._cache.put(name, value, overwrite=overwrite)

    def list_handles(self) -> dict[str, str]:
        """Return a mapping of all cache handle names to their snapshot strings."""
        return self._cache.list_handles()

    def _record(self, result: RunResult) -> RunResult:
        self._last_result = result
        self._turns += result.turns
        self._agent._last_harness = self._harness  # type: ignore[attr-defined]
        self._agent._last_run_file = self._harness.run_file
        self._agent._last_result = result
        return result


class AgentSession(_SessionBase):
    """Stateful chat session built from an `Agent` definition.

    `Agent.run()` intentionally stays one-shot for examples and tests. Use
    `Agent.session()` when an application needs follow-up questions over the
    same message history and cache handles.
    """

    @property
    def harness(self) -> Harness:
        return self._harness

    def ask_result(self, user_message: str) -> RunResult:
        """Send a follow-up message and return the full `RunResult`.

        Args:
            user_message: The follow-up user prompt.

        Returns:
            A `RunResult` for this turn sequence.
        """
        return self._record(
            self._harness.ask_result(
                user_message, run_id=str(uuid.uuid4()), session_id=self._id
            )
        )

    def ask(self, user_message: str) -> str:
        """Send a follow-up message and return the final text response.

        Args:
            user_message: The follow-up user prompt.

        Returns:
            The model's final text response.

        Raises:
            MaxTurnsExceeded: If the loop reaches ``max_turns``.
            RuntimeError: If the provider raises an exception.
        """
        return unwrap_text(self.ask_result(user_message))


class AsyncAgentSession(_SessionBase):
    """Stateful async chat session built from an `AsyncAgent` definition."""

    @property
    def harness(self) -> AsyncHarness:
        return self._harness

    async def ask_result(self, user_message: str) -> RunResult:
        """Send a follow-up message and return the full `RunResult`."""
        return self._record(
            await self._harness.ask_result(
                user_message, run_id=str(uuid.uuid4()), session_id=self._id
            )
        )

    async def ask(self, user_message: str) -> str:
        """Send a follow-up message and return the final text response.

        Raises:
            MaxTurnsExceeded: If the loop reaches ``max_turns``.
            RuntimeError: If the provider raises an exception.
        """
        return unwrap_text(await self.ask_result(user_message))

    async def ask_stream(self, user_message: str) -> AsyncGenerator[StreamEvent, None]:
        """Stream events for a follow-up turn.

        After the generator is exhausted, ``session.last_result`` holds the
        `RunResult` for the turn, stamped with the session and run ids exactly
        as a non-streamed turn would be.
        """
        run_id = str(uuid.uuid4())
        async with aclosing(self._harness.ask_stream(user_message)) as events:
            async for event in events:
                yield event
        result = self._harness.last_result
        if result is not None:
            self._record(self._harness._stamp(result, run_id, self._id))
        else:  # pragma: no cover - the loop always sets a result
            self._agent._last_harness = self._harness
            self._agent._last_run_file = self._harness.run_file


def _synthetic_text_events(text: str) -> list[StreamEvent]:
    """A minimal, well-formed event sequence carrying ``text`` and no usage.

    Used when a streamed run is served from the replay cache: there is no
    provider stream to relay, but a streaming caller still renders from
    deltas, so it must receive the answer the same way.
    """
    from data_harness.llm.providers.base import StopReason
    from data_harness.llm.streaming import (
        ContentBlockDeltaEvent,
        ContentBlockStartEvent,
        ContentBlockStopEvent,
        MessageDeltaEvent,
        MessageStartEvent,
        MessageStopEvent,
        TextDelta,
    )
    from data_harness.llm.types import TextBlock

    return [
        MessageStartEvent(),
        ContentBlockStartEvent(index=0, content_block=TextBlock(text="")),
        ContentBlockDeltaEvent(index=0, delta=TextDelta(text=text)),
        ContentBlockStopEvent(index=0),
        MessageDeltaEvent(
            stop_reason=StopReason.END_TURN,
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
        ),
        MessageStopEvent(),
    ]


_EXPLAIN_TEMPLATE = """\
Agent is a thin composition layer. The equivalent explicit wiring is:

    from data_harness.data.cache import SessionCache
    from data_harness.data.harness import Harness
    from data_harness.data.tools.interpreter import PythonInterpreter
    from data_harness.data.tools.variables import make_list_variables_spec

    cache = SessionCache()
    tools = [
        PythonInterpreter.make_tool_spec(cache),
        make_list_variables_spec(cache),
    ]
    harness = Harness(
        adapter=adapter,
        system={system!r},
        tools=tools,
        max_turns={max_turns},
        run_dir={run_dir!r},
        cache=cache,
    )
    harness.run(user_message)

Each call to Agent.run() builds a fresh Harness with fresh tool specs.
Model-visible tools include python_interpreter and list_variables.
The message history resets per run; use agent.session().ask(...) for follow-up.
"""


def _truncate(text: str, limit: int = 80) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"
