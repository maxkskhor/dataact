"""Graders: turn a `RunResult` into a pass/fail `Grade`.

Graders deliberately check the structured ``result.value`` first (the value the
model computed via ``answer()``) and fall back to parsing the prose, because
models vary in how reliably they call ``answer()``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from data_harness.core.result import RunResult
    from data_harness.eval.case import EvalCase, Grader

_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")

DEFAULT_REFUSAL_MARKERS = (
    "cannot",
    "can't",
    "not enough",
    "no column",
    "not available",
    "unable",
    "don't have",
    "do not have",
    "clarif",
    "no information",
    "not possible",
    "not present",
)


@dataclass
class Grade:
    """The outcome of grading one case."""

    passed: bool
    detail: str = ""


def _candidates(result: RunResult) -> list[str]:
    out: list[str] = []
    if result.value is not None:
        out.append(str(result.value))
    if result.text:
        out.append(result.text)
    return out


def extract_numbers(text: str) -> list[float]:
    """Pull numeric literals out of free text (handles thousands separators)."""
    nums: list[float] = []
    for match in _NUMBER_RE.findall(text or ""):
        cleaned = match.replace(",", "")
        try:
            nums.append(float(cleaned))
        except ValueError:
            continue
    return nums


def numeric(expected: float, tol: float = 1e-6) -> Grader:
    """Pass if the answer equals ``expected`` within ``tol`` (absolute).

    Checks ``result.value`` when it is a number, otherwise scans the prose.
    """

    def grade(result: RunResult, case: EvalCase) -> Grade:
        value = result.value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if abs(float(value) - expected) <= tol:
                return Grade(True, f"value={value}")
        for num in extract_numbers(result.text or ""):
            if abs(num - expected) <= tol:
                return Grade(True, f"text≈{num}")
        return Grade(
            False,
            f"expected {expected} (±{tol}); value={result.value!r} "
            f"text={(result.text or '')[:80]!r}",
        )

    return grade


def contains(expected: str | list[str], case_insensitive: bool = True) -> Grader:
    """Pass if any of ``expected`` appears in the value or prose."""
    expected_list = [expected] if isinstance(expected, str) else list(expected)

    def grade(result: RunResult, case: EvalCase) -> Grade:
        hay = " ".join(_candidates(result))
        if case_insensitive:
            hay = hay.lower()
        for item in expected_list:
            needle = str(item).lower() if case_insensitive else str(item)
            if needle in hay:
                return Grade(True, f"found {item!r}")
        return Grade(False, f"none of {expected_list} in output")

    return grade


def exact(expected: Any, case_insensitive: bool = True) -> Grader:
    """Pass if ``result.value`` equals ``expected`` (string-normalised)."""

    def grade(result: RunResult, case: EvalCase) -> Grade:
        got = str(result.value).strip()
        want = str(expected).strip()
        if case_insensitive:
            got, want = got.lower(), want.lower()
        return Grade(got == want, f"value={result.value!r} expected={expected!r}")

    return grade


def dataframe_equals(expected: Any, check_like: bool = True) -> Grader:
    """Pass if ``result.value`` is a DataFrame equal to ``expected``."""

    def grade(result: RunResult, case: EvalCase) -> Grade:
        import pandas as pd

        value = result.value
        if not isinstance(value, pd.DataFrame):
            return Grade(False, f"value is not a DataFrame ({type(value).__name__})")
        a, b = value, expected
        if check_like:
            a = a.reset_index(drop=True).sort_index(axis=1)
            b = b.reset_index(drop=True).sort_index(axis=1)
        try:
            pd.testing.assert_frame_equal(
                a, b, check_dtype=False, check_like=check_like
            )
            return Grade(True, "frame matches")
        except AssertionError as exc:
            return Grade(False, f"frame mismatch: {str(exc).splitlines()[0][:80]}")

    return grade


def chart_produced() -> Grader:
    """Pass if the run rendered at least one chart."""

    def grade(result: RunResult, case: EvalCase) -> Grade:
        n = len(result.charts)
        return Grade(n > 0, f"{n} chart(s)")

    return grade


def refuses(markers: tuple[str, ...] = DEFAULT_REFUSAL_MARKERS) -> Grader:
    """Pass if the answer signals it cannot/should not answer (adversarial cases)."""

    def grade(result: RunResult, case: EvalCase) -> Grade:
        text = (result.text or "").lower()
        hit = [m for m in markers if m in text]
        return Grade(bool(hit), f"markers={hit}" if hit else "no refusal markers")

    return grade


def all_of(*graders: Grader) -> Grader:
    """Pass only if every sub-grader passes."""

    def grade(result: RunResult, case: EvalCase) -> Grade:
        details = []
        for g in graders:
            r = g(result, case)
            details.append(r.detail)
            if not r.passed:
                return Grade(False, "; ".join(details))
        return Grade(True, "; ".join(details))

    return grade


def any_of(*graders: Grader) -> Grader:
    """Pass if any sub-grader passes."""

    def grade(result: RunResult, case: EvalCase) -> Grade:
        details = []
        for g in graders:
            r = g(result, case)
            details.append(r.detail)
            if r.passed:
                return Grade(True, r.detail)
        return Grade(False, "; ".join(details))

    return grade
