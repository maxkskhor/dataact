"""The unit of evaluation: an `EvalCase`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from data_harness.core.result import RunResult
    from data_harness.eval.graders import Grade

# A grader inspects the run's outcome and decides pass/fail.
Grader = Callable[["RunResult", "EvalCase"], "Grade"]


@dataclass(eq=False)
class EvalCase:
    """One single-shot evaluation task.

    Uses identity equality (``eq=False``): cases hold DataFrames, and dataclass
    field-wise comparison would raise on DataFrame truthiness.

    Attributes:
        id: Unique, stable identifier.
        question: The natural-language question posed to the agent.
        data: The dataset, passed straight to `ask` (a DataFrame, mapping, path,
            or list of paths).
        grader: Callable ``(RunResult, EvalCase) -> Grade`` deciding pass/fail.
        category: Free-form grouping label used in reports.
        semantics: Optional per-handle semantic context forwarded to `ask`.
        tags: Optional extra labels.
    """

    id: str
    question: str
    data: Any
    grader: Grader
    category: str = "general"
    semantics: dict[str, dict] | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(eq=False)
class Turn:
    """One graded turn within a `ConversationCase`."""

    question: str
    grader: Grader


@dataclass(eq=False)
class ConversationCase:
    """A stateful, multi-turn evaluation task.

    Runs over a single `Chat` session so later turns build on handles saved in
    earlier turns — this is what exercises the `SessionCache` and the harness's
    long-horizon, stateful design (and what single-shot benchmarks cannot test).
    Each turn is graded independently and reported as ``<id>#t<n>``.

    Attributes:
        id: Unique, stable identifier.
        data: The dataset, loaded once into the shared session cache.
        turns: Ordered graded turns.
        category: Grouping label used in reports.
        semantics: Optional per-handle semantic context.
    """

    id: str
    data: Any
    turns: list[Turn]
    category: str = "conversation"
    semantics: dict[str, dict] | None = None
