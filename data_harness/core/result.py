"""Typed result dataclasses for Harness and Agent run outputs."""

from __future__ import annotations

import html as _html
from dataclasses import dataclass, field
from typing import Any, Literal

from data_harness.core.artifacts import ChartArtifact
from data_harness.llm.providers.base import StopReason


@dataclass
class Usage:
    """Accumulated token counts for a run or session turn.

    Attributes:
        input_tokens: Tokens in the prompt sent to the provider.
        output_tokens: Tokens in the provider's response.
        cache_read_tokens: Prompt tokens served from the provider's KV cache.
        cache_write_tokens: Prompt tokens written to the provider's KV cache.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass
class CacheStorageInfo:
    """Where a named handle is physically stored in the `SessionCache`.

    Attributes:
        location: Either ``"memory"`` (hot) or ``"disk"`` (spilled cold).
        storage_type: Format used on disk, e.g. ``"dataframe_parquet"`` or
            ``"numpy_npy"``. Always ``"memory"`` for hot entries.
    """

    location: Literal["memory", "disk"]
    storage_type: str

    def __post_init__(self) -> None:
        if self.location not in ("memory", "disk"):
            from data_harness.core.exceptions import ConfigurationError

            raise ConfigurationError(
                f"Invalid location: {self.location!r}. Must be 'memory' or 'disk'."
            )


@dataclass
class RunResult:
    """The complete outcome of a single `Harness.run()` or `Agent.run()` call.

    Attributes:
        text: The final text response from the model.
        status: ``"success"``, ``"max_turns_exceeded"``, or ``"error"``.
        turns: Number of provider turns executed.
        run_file: Path to the JSONL log for this run, or ``None`` if logging
            was disabled.
        stop_reason: Provider stop reason from the final turn, or ``None`` on
            error/max-turns.
        usage: Cumulative token counts across all turns.
        cache_snapshots: Mapping of handle name → compact snapshot string for
            every value in the session cache at the end of the run.
        cache_storage: Mapping of handle name → `CacheStorageInfo` describing
            where each handle is stored.
        error: Exception repr when ``status == "error"``, otherwise ``None``.
        stopped_by: Why a hook ended the run early, if one did. A capped run
            is otherwise indistinguishable from a model that answered with
            nothing.
        run_id: Optional UUID assigned by `Agent`; ``None`` when using `Harness`
            directly.
        session_id: Optional session UUID when the run is part of an
            `AgentSession`; ``None`` for one-shot runs.
        value: The structured final answer the model recorded via ``answer(...)``
            inside the interpreter (a scalar, DataFrame, etc.), or ``None`` if it
            only produced prose.
        charts: Chart artefacts rendered during the run. Image bytes live on
            disk; only paths are referenced here.
    """

    text: str
    status: Literal["success", "max_turns_exceeded", "error"]
    turns: int
    run_file: str | None
    stop_reason: StopReason | None
    usage: Usage
    cache_snapshots: dict[str, str] = field(default_factory=dict)
    cache_storage: dict[str, CacheStorageInfo] = field(default_factory=dict)
    error: str | None = None
    stopped_by: str | None = None
    run_id: str | None = None
    session_id: str | None = None
    value: Any = field(default=None, repr=False)
    charts: list[ChartArtifact] = field(default_factory=list, repr=False)

    # --- Rich display for Jupyter / IPython --------------------------------
    def _repr_markdown_(self) -> str:
        parts = [self.text] if self.text else []
        if self.value is not None and not _renders_itself(self.value):
            parts.append(f"\n**answer:** `{self.value!r}`")
        return "\n".join(parts) if parts else f"_(status: {self.status})_"

    def _repr_html_(self) -> str:
        parts: list[str] = []
        if self.text:
            parts.append(f"<p>{_html.escape(self.text)}</p>")
        if self.value is not None and not _renders_itself(self.value):
            parts.append(
                f"<p><strong>answer:</strong> <code>"
                f"{_html.escape(repr(self.value))}</code></p>"
            )
        if _renders_itself(self.value):
            parts.append(self.value._repr_html_())
        for chart in self.charts:
            parts.append(chart._repr_html_())
        if not parts:
            parts.append(f"<em>(status: {self.status})</em>")
        return "\n".join(parts)


def _renders_itself(value: Any) -> bool:
    """Whether ``value`` can render its own HTML, as a DataFrame can.

    Duck-typed on purpose: core must not know what a DataFrame is, and this
    way it also works for a polars frame, a styled frame, or anything else
    that follows the IPython display protocol.
    """
    return hasattr(value, "_repr_html_")


def unwrap_text(result: RunResult) -> str:
    """Return a successful run's text, or raise the exception its status means.

    The single place that decides which failure becomes which exception, so
    `run`, `ask`, and their async twins cannot disagree about it.

    Raises:
        MaxTurnsExceeded: If the run hit its turn cap.
        ProviderError: If the run failed. `ProviderError` is a `RuntimeError`,
            so callers written before the taxonomy still catch it.
    """
    from data_harness.core.exceptions import MaxTurnsExceeded, ProviderError

    if result.status == "max_turns_exceeded":
        raise MaxTurnsExceeded(result.turns)
    if result.status == "error":
        raise ProviderError(result.error or "unknown error")
    return result.text
