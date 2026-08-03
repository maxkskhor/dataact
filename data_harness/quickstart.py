"""Zero-config entry points: ``ask()``, ``Chat``, and ``SmartFrame``.

These are the headline conveniences. ``ask(df, "...")`` is the lowest-friction
way to query data: it resolves a provider from the environment, loads the data
into a session cache as handles, runs the agent, and returns a `RunResult`
(which renders richly in notebooks).

``Chat`` keeps a session alive for follow-up questions; ``SmartFrame`` is a
pandasai-style wrapper over a single frame. Neither mutates global state.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from data_harness.agent import Agent
from data_harness.io import to_handles
from data_harness.result import RunResult

if TYPE_CHECKING:
    from data_harness.providers.base import AsyncProviderAdapter, ProviderAdapter

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"

_DEFAULT_SYSTEM = (
    "You are a senior data analyst. You answer questions about the user's data "
    "by writing Python in the python_interpreter tool.\n\n"
    "Guidelines:\n"
    "- The user's data is preloaded as cache handles (named variables). Use them "
    "directly; call list_variables if you need to see what is available.\n"
    "- Inspect data with print(...) before computing.\n"
    "- To make a chart, use matplotlib (import matplotlib.pyplot as plt) and build "
    "the figure; it is captured automatically. Do not call plt.show().\n"
    "- You MUST call answer(value) inside the interpreter with your final result "
    "(a number, a DataFrame, etc.) before you finish — this is how the caller "
    "receives the structured result. Do this even if you also explain in prose.\n"
    "- If a sql_query tool is available, you may use SQL over the data handles.\n"
    "- Finish with a short, clear written summary."
)

_OPENAI_PREFIXES = ("gpt", "o1", "o3", "o4", "chatgpt")


def _is_openai_model(model: str) -> bool:
    return model.lower().startswith(_OPENAI_PREFIXES)


#: provider key -> (sync class, async class, name of the extra that supplies it)
_ADAPTER_CLASSES: dict[str, tuple[str, str, str | None]] = {
    "anthropic": ("AnthropicAdapter", "AsyncAnthropicAdapter", None),
    "openai": ("OpenAIAdapter", "AsyncOpenAIAdapter", "openai"),
    "openrouter": ("OpenRouterAdapter", "AsyncOpenRouterAdapter", "openai"),
    "deepseek": ("DeepSeekAdapter", "AsyncDeepSeekAdapter", "openai"),
}

#: provider key -> environment variable consulted when no model is given,
#: in preference order.
_ENV_PREFERENCE: tuple[tuple[str, str, str], ...] = (
    ("anthropic", "ANTHROPIC_API_KEY", DEFAULT_ANTHROPIC_MODEL),
    ("openai", "OPENAI_API_KEY", DEFAULT_OPENAI_MODEL),
    ("openrouter", "OPENROUTER_API_KEY", DEFAULT_OPENROUTER_MODEL),
    ("deepseek", "DEEPSEEK_API_KEY", DEFAULT_DEEPSEEK_MODEL),
)


def _route(model: str | None) -> tuple[str, str]:
    """Map a model name (or the environment) onto ``(provider_key, model_id)``.

    Sync and async resolution share this routing so the two can never disagree
    about which provider a model name belongs to.

    Raises:
        RuntimeError: If no provider can be resolved (no key, no model).
    """
    if model is not None:
        if "/" in model:
            return "openrouter", model
        if model.startswith("deepseek"):
            return "deepseek", model
        if _is_openai_model(model):
            return "openai", model
        return "anthropic", model

    for provider, env_var, default_model in _ENV_PREFERENCE:
        if os.environ.get(env_var):
            return provider, default_model

    raise RuntimeError(
        "No provider configured. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, "
        "OPENROUTER_API_KEY, or DEEPSEEK_API_KEY, or pass an explicit "
        "adapter=... / model=... to ask()/Chat()."
    )


def _build_adapter(model: str | None, *, is_async: bool) -> Any:
    provider, model_id = _route(model)
    sync_name, async_name, extra = _ADAPTER_CLASSES[provider]
    module_name = (
        "data_harness.providers.anthropic"
        if provider == "anthropic"
        else "data_harness.providers.openai"
    )
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - exercised via install matrix
        raise RuntimeError(
            f"{provider} support requires the {extra!r} extra: pip install "
            f"'data-harness[{extra}]'."
        ) from exc
    cls = getattr(module, async_name if is_async else sync_name)
    return cls(model=model_id)


def resolve_adapter(model: str | None = None) -> ProviderAdapter:
    """Resolve a `ProviderAdapter` from an explicit model or the environment.

    With no ``model``, prefers ``ANTHROPIC_API_KEY``, then ``OPENAI_API_KEY``,
    then ``OPENROUTER_API_KEY``, then ``DEEPSEEK_API_KEY``. With a ``model``,
    routes by name: a ``provider/model`` id (containing ``/``) goes to OpenRouter,
    ``deepseek*`` to DeepSeek's direct API, ``gpt*``/``o*`` to OpenAI, otherwise
    Anthropic.

    Raises:
        RuntimeError: If no provider can be resolved (no key, no model).
    """
    return _build_adapter(model, is_async=False)


def resolve_async_adapter(model: str | None = None) -> AsyncProviderAdapter:
    """Resolve an `AsyncProviderAdapter` using the same routing as `resolve_adapter`.

    Args:
        model: Explicit model id, or ``None`` to pick from the environment.

    Raises:
        RuntimeError: If no provider can be resolved (no key, no model).
    """
    return _build_adapter(model, is_async=True)


def _build_agent(
    handles: dict[str, Any],
    *,
    adapter: ProviderAdapter | None,
    model: str | None,
    system: str | None,
    max_turns: int,
    run_dir: str | None,
    semantics: dict[str, dict] | None,
    sql: bool | None,
) -> Agent:
    agent = Agent(
        adapter=adapter if adapter is not None else resolve_adapter(model),
        system=system if system is not None else _DEFAULT_SYSTEM,
        max_turns=max_turns,
        run_dir=run_dir,
    )
    sem = semantics or {}
    for name, value in handles.items():
        agent.cache.put(name, value, semantics=sem.get(name))
    if sql is None:
        sql = importlib.util.find_spec("duckdb") is not None
    if sql:
        agent.enable_sql()
    return agent


def _handles_preamble(agent: Agent) -> str:
    listing = agent.cache.list_handles()
    if not listing:
        return ""
    lines = "\n".join(f"- {name}: {snap}" for name, snap in listing.items())
    return f"\n\nAvailable data handles:\n{lines}"


_FINALIZE_PROMPT = (
    "You did not record a structured final answer. Call answer(value) in the "
    "python_interpreter now with your final result (a number, a DataFrame, etc.), "
    "computed from the data. Do not add other commentary."
)

# Lightweight refusal markers so finalize never turns a correct "I can't answer
# this" into a fabricated value. (Kept local to avoid importing data_harness.eval,
# which imports this module.)
_REFUSAL_MARKERS = (
    "cannot",
    "can't",
    "not enough",
    "no column",
    "not available",
    "unable",
    "don't have",
    "do not have",
    "no information",
    "not possible",
    "not present",
)


def _looks_like_refusal(text: str | None) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


def _finalize(
    result: RunResult, continue_turn: Callable[[str], RunResult]
) -> RunResult:
    """Coax a structured answer when a successful run produced none.

    If the run succeeded but the model never called ``answer()`` (so
    ``result.value`` is ``None``), run one focused follow-up turn asking it to
    record the value, then merge that value into the original result. The
    original prose is kept; turns/usage are accumulated.

    Guarded so it never fires when a chart was produced (the chart is the
    deliverable) or the answer reads as a refusal (forcing a value there would
    invite a hallucination).
    """
    if (
        result.status != "success"
        or result.value is not None
        or result.charts
        or _looks_like_refusal(result.text)
    ):
        return result
    follow = continue_turn(_FINALIZE_PROMPT)
    if follow.value is None:
        return result
    return dataclasses.replace(
        result,
        value=follow.value,
        turns=result.turns + follow.turns,
        usage=result.usage + follow.usage,
        charts=result.charts or follow.charts,
        cache_snapshots=follow.cache_snapshots,
        cache_storage=follow.cache_storage,
    )


def ask(
    data: Any,
    question: str,
    *,
    model: str | None = None,
    adapter: ProviderAdapter | None = None,
    system: str | None = None,
    semantics: dict[str, dict] | None = None,
    sql: bool | None = None,
    require_answer: bool = True,
    max_turns: int = 12,
    run_dir: str | None = None,
) -> RunResult:
    """Ask a one-shot natural-language question about ``data``.

    Args:
        data: A DataFrame, a ``{name: value}`` mapping, a file path, or a list
            of file paths. Loaded into the session cache as handles.
        question: The natural-language question.
        model: Optional model id; routes to the matching provider. Ignored if
            ``adapter`` is given.
        adapter: Explicit provider adapter, overriding ``model``.
        system: Override the default analyst system prompt.
        semantics: Optional ``{handle: {...}}`` domain context folded into
            snapshots.
        sql: Enable the ``sql_query`` tool. ``None`` auto-enables it when DuckDB
            is installed.
        require_answer: If ``True`` (default) and a successful run produced no
            structured answer, run one follow-up turn asking the model to record
            it via ``answer()`` so ``.value`` is populated.
        max_turns: Hard cap on provider turns.
        run_dir: Directory for JSONL logs and chart artefacts.

    Returns:
        A `RunResult`. Use ``.text`` for prose, ``.value`` for the structured
        answer, and ``.charts`` for rendered charts.
    """
    handles = to_handles(data)
    agent = _build_agent(
        handles,
        adapter=adapter,
        model=model,
        system=system,
        max_turns=max_turns,
        run_dir=run_dir,
        semantics=semantics,
        sql=sql,
    )
    result = agent.run_result(question + _handles_preamble(agent))
    if require_answer and agent.last_harness is not None:
        result = _finalize(result, agent.last_harness.ask_result)
    return result


class Chat:
    """A stateful conversation over a dataset.

    Unlike `ask`, a `Chat` keeps one message history and cache alive so
    follow-up questions can build on earlier turns.

    Example::

        chat = Chat(sales_df)
        chat.ask("What was total revenue?")
        chat.ask("Which month was highest?")  # remembers context
    """

    def __init__(
        self,
        data: Any,
        *,
        model: str | None = None,
        adapter: ProviderAdapter | None = None,
        system: str | None = None,
        semantics: dict[str, dict] | None = None,
        sql: bool | None = None,
        require_answer: bool = False,
        max_turns: int = 12,
        run_dir: str | None = None,
    ) -> None:
        handles = to_handles(data)
        self._agent = _build_agent(
            handles,
            adapter=adapter,
            model=model,
            system=system,
            max_turns=max_turns,
            run_dir=run_dir,
            semantics=semantics,
            sql=sql,
        )
        self._session = self._agent.session()
        self._first = True
        self._require_answer = require_answer

    def ask(self, question: str) -> RunResult:
        """Ask a follow-up question and return its `RunResult`."""
        if self._first:
            question = question + _handles_preamble(self._agent)
            self._first = False
        result = self._session.ask_result(question)
        if self._require_answer:
            result = _finalize(result, self._session.ask_result)
        return result

    @property
    def session(self):
        """The underlying `AgentSession` (cache, history, run file)."""
        return self._session


class SmartFrame:
    """pandasai-style wrapper over a single DataFrame.

    Example::

        SmartFrame(df).chat("plot revenue by month")
    """

    def __init__(self, df: Any, **kwargs: Any) -> None:
        self._chat = Chat(df, **kwargs)

    def chat(self, question: str) -> RunResult:
        """Ask a question about the wrapped frame; returns a `RunResult`."""
        return self._chat.ask(question)
