"""The data-flavoured harness: `core`'s loop wired to a `SessionCache`.

`data_harness.core.loop` is domain-free — it takes a `RunEnvironment` and has
no idea what a DataFrame is. Almost nobody wants that directly. These two
classes are the same loop with the data domain plugged in, and they are what
`Agent` builds and what `data_harness.loop` resolves to.

The only difference from the core classes is the constructor: ``cache=`` in
place of ``environment=``, defaulting to a fresh `SessionCache`.
"""

from __future__ import annotations

from collections.abc import Callable

from data_harness.core.hooks import HookRegistry
from data_harness.core.loop import (
    Answer,
    CallProvider,
    CallTool,
    Effect,
    Failed,
    Ok,
    ToolFinished,
    run_coroutine_blocking,
)
from data_harness.core.loop import (
    AsyncHarness as _CoreAsyncHarness,
)
from data_harness.core.loop import (
    Harness as _CoreHarness,
)
from data_harness.core.session import Session
from data_harness.data.cache import SessionCache
from data_harness.data.environment import CacheEnvironment
from data_harness.llm.providers.base import AsyncProviderAdapter, ProviderAdapter
from data_harness.llm.types import ToolSpec


def _environment_for(cache: SessionCache | None) -> CacheEnvironment:
    return CacheEnvironment(cache)


class Harness(_CoreHarness):
    """Synchronous harness over a `SessionCache`. See `core.loop.Harness`.

    Args:
        adapter: Synchronous provider adapter.
        system: System prompt. Kept byte-identical across all turns.
        tools: Full tool list.
        max_turns: Hard cap on provider turns.
        run_dir: Directory where JSONL logs are written.
        cache: Shared `SessionCache`. A fresh one is created if ``None``.
        session: Durable record of the run. Pass one opened from storage
            to resume a conversation across processes.
        hooks: Pre-built `HookRegistry`, so an application can define its
            policy once and reuse it across harnesses.
        on_code: Approval gate called with interpreter code before it runs.
        code_only: When ``True``, interpreter code is echoed, never executed.
    """

    def __init__(
        self,
        adapter: ProviderAdapter,
        system: str,
        tools: list[ToolSpec],
        max_turns: int = 25,
        run_dir: str = "./runs",
        cache: SessionCache | None = None,
        on_code: Callable[[str], object] | None = None,
        code_only: bool = False,
        session: Session | None = None,
        hooks: HookRegistry | None = None,
    ) -> None:
        super().__init__(
            adapter=adapter,
            system=system,
            tools=tools,
            max_turns=max_turns,
            run_dir=run_dir,
            environment=_environment_for(cache),
            on_code=on_code,
            code_only=code_only,
            session=session,
            hooks=hooks,
        )

    @property
    def cache(self) -> SessionCache:
        """The `SessionCache` backing tool results and handles."""
        return self._environment.cache


class AsyncHarness(_CoreAsyncHarness):
    """Async harness over a `SessionCache`. See `core.loop.AsyncHarness`.

    Same arguments as `Harness`, but takes an `AsyncProviderAdapter`.
    """

    def __init__(
        self,
        adapter: AsyncProviderAdapter,
        system: str,
        tools: list[ToolSpec],
        max_turns: int = 25,
        run_dir: str = "./runs",
        cache: SessionCache | None = None,
        on_code: Callable[[str], object] | None = None,
        code_only: bool = False,
        session: Session | None = None,
        hooks: HookRegistry | None = None,
    ) -> None:
        super().__init__(
            adapter=adapter,
            system=system,
            tools=tools,
            max_turns=max_turns,
            run_dir=run_dir,
            environment=_environment_for(cache),
            on_code=on_code,
            code_only=code_only,
            session=session,
            hooks=hooks,
        )

    @property
    def cache(self) -> SessionCache:
        """The `SessionCache` backing tool results and handles."""
        return self._environment.cache


__all__ = [
    "Answer",
    "AsyncHarness",
    "CallProvider",
    "CallTool",
    "Effect",
    "Failed",
    "Harness",
    "Ok",
    "ToolFinished",
    "run_coroutine_blocking",
]
