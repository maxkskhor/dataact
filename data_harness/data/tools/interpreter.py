from __future__ import annotations

import ast
import builtins
import io
import os
import sys
import tempfile
import traceback
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from data_harness.core.artifacts import ChartArtifact
from data_harness.data.cache import SessionCache
from data_harness.llm.types import ToolAnnotations, ToolSpec

# Force a headless backend so plotting works without a display. Set before any
# user code imports matplotlib.
os.environ.setdefault("MPLBACKEND", "Agg")

_INTERPRETER_ANNOTATIONS = ToolAnnotations(
    title="Python Interpreter",
    read_only=False,
    cache_mutating=True,
    destructive=False,
    open_world=False,
)

_DEFAULT_ALLOWLIST = frozenset(
    {
        "pandas",
        "numpy",
        "json",
        "math",
        "datetime",
        "collections",
        "itertools",
        "statistics",
        "matplotlib",
        "seaborn",
        "plotly",
        "pd",
        "np",  # common aliases
    }
)

_FORBIDDEN_NAMES = frozenset({"eval", "exec", "__import__", "open", "compile"})

_EMPTY_OUTPUT_GUIDANCE = (
    "Code ran successfully but produced no stdout. "
    "Use print(...) to inspect values, or save(name, value) to persist results "
    "for later calls."
)

_LOCALS_ERROR = (
    "Error: locals() is not available in python_interpreter. "
    "Use cache handles directly as local variables. "
    "Call list_variables to see available handles."
)

_SENTINEL = object()


class PythonInterpreterError(Exception):
    """Raised by PythonInterpreter.run() on any execution failure."""

    def __repr__(self) -> str:
        return str(self)


class _SecurityVisitor(ast.NodeVisitor):
    """AST visitor that raises ValueError on forbidden patterns."""

    def __init__(self, allowlist: frozenset[str]) -> None:
        self._allowlist = allowlist
        self.errors: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            top = alias.name.split(".")[0]
            if top not in self._allowlist:
                self.errors.append(f"Import not allowed: {alias.name!r}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            top = node.module.split(".")[0]
            if top not in self._allowlist:
                self.errors.append(f"Import not allowed: {node.module!r}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_NAMES:
            self.errors.append(f"Call not allowed: {node.func.id!r}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__") and node.attr.endswith("__"):
            self.errors.append(f"Dunder attribute access not allowed: {node.attr!r}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _FORBIDDEN_NAMES:
            self.errors.append(f"Name not allowed: {node.id!r}")
        self.generic_visit(node)


def _has_locals_call(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "locals"
        ):
            return True
    return False


class PythonInterpreter:
    def __init__(
        self,
        cache: SessionCache,
        allowlist: frozenset[str] | None = None,
        artifacts_dir: str | Path | None = None,
    ) -> None:
        self._cache = cache
        self._allowlist = allowlist if allowlist is not None else _DEFAULT_ALLOWLIST
        self._artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else None
        self._temp_artifacts: tempfile.TemporaryDirectory[str] | None = None

    def _artifacts_path(self) -> Path:
        if self._artifacts_dir is not None:
            self._artifacts_dir.mkdir(parents=True, exist_ok=True)
            return self._artifacts_dir
        if self._temp_artifacts is None:
            self._temp_artifacts = tempfile.TemporaryDirectory(
                prefix="data-harness-charts-"
            )
        return Path(self._temp_artifacts.name)

    def _capture_charts(self) -> list[str]:
        """Save any open matplotlib figures as ChartArtifact handles."""
        if "matplotlib" not in sys.modules:
            return []
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return []
        handles: list[str] = []
        for num in plt.get_fignums():
            fig = plt.figure(num)
            title = None
            if fig.axes:
                title = fig.axes[0].get_title() or None
            path = self._artifacts_path() / f"chart_{uuid.uuid4().hex[:8]}.png"
            fig.savefig(path, format="png", bbox_inches="tight")
            artifact = ChartArtifact(path=str(path), format="png", title=title)
            handles.append(self._cache.put("chart", artifact))
            plt.close(fig)
        return handles

    def run(self, code: str) -> str:
        local_vars: dict[str, Any] = {}

        # Inject cache handles
        for name, value in self._cache.items():
            local_vars[name] = value

        # Inject save() helper
        def save(name: str, value: Any) -> str:
            return self._cache.put(name, value)

        local_vars["save"] = save

        # Inject answer() helper: records the designated final answer.
        def answer(value: Any) -> Any:
            self._cache.set_answer(value)
            print(f"Recorded answer: {_short_answer(value)}")
            return value

        local_vars["answer"] = answer

        output, last_val = execute_namespace(code, local_vars, self._allowlist)

        chart_handles = self._capture_charts()
        chart_note = "\n".join(
            f"Rendered chart saved to handle `{h}`." for h in chart_handles
        )
        return assemble_output(output, chart_note, last_val)

    @staticmethod
    def make_tool_spec(
        cache: SessionCache, artifacts_dir: str | Path | None = None
    ) -> ToolSpec:
        interp = PythonInterpreter(cache=cache, artifacts_dir=artifacts_dir)
        return ToolSpec(
            name="python_interpreter",
            description=_interpreter_description(),
            input_schema={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                },
                "required": ["code"],
            },
            handler=interp.run,
            annotations=_INTERPRETER_ANNOTATIONS,
        )


def _interpreter_description() -> str:
    """Shared model-facing description for the in-process and sandboxed tools."""
    return (
        "Run Python code in a sandboxed interpreter with direct access to "
        "cached data handles.\n\n"
        "• Cache handles are available as local variables — use them directly, "
        "e.g. print(fred_unrate.head()). Do NOT use locals() to look up "
        "handles by name.\n"
        "• The interpreter captures printed stdout only. Always use print(...) "
        "to inspect values, e.g. print(df.describe()).\n"
        "• If the last statement is a bare expression (e.g. df.describe()), "
        "its repr is returned automatically when nothing was printed.\n"
        "• Each call starts with fresh local variables. Any variable not saved "
        "with save(name, value) is discarded at the end of the call.\n"
        "• Call save(name, value) to persist computed artefacts for later calls.\n"
        "• Call answer(value) to record your final result (a number, DataFrame, "
        "etc.) so the caller receives it as structured output.\n"
        "• To make a chart, use matplotlib (import matplotlib.pyplot as plt) and "
        "build a figure; open figures are captured and returned automatically — "
        "do not call plt.show()."
    )


def execute_namespace(
    code: str,
    local_vars: dict[str, Any],
    allowlist: frozenset[str],
) -> tuple[str, object]:
    """Security-check and execute ``code`` against ``local_vars``.

    This is the single shared execution core used by both the in-process
    interpreter and the subprocess sandbox, so the two cannot drift apart on
    the security boundary.

    Returns:
        ``(stdout, last_value)`` where ``last_value`` is `_SENTINEL` unless the
        final statement was a bare expression.

    Raises:
        PythonInterpreterError: On syntax errors, forbidden constructs, or any
            exception raised by the executed code.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise PythonInterpreterError(f"SyntaxError: {exc}") from exc

    visitor = _SecurityVisitor(allowlist)
    visitor.visit(tree)
    if visitor.errors:
        raise PythonInterpreterError(
            "SecurityError: " + "; ".join(visitor.errors) + " — not allowed"
        )

    if _has_locals_call(tree):
        raise PythonInterpreterError(_LOCALS_ERROR)

    globals_dict: dict[str, Any] = {"__builtins__": _safe_builtins(allowlist)}
    buf = io.StringIO()
    last_val: object = _SENTINEL

    if tree.body and isinstance(tree.body[-1], ast.Expr):
        body_tree = ast.Module(body=tree.body[:-1], type_ignores=[])
        expr_tree = ast.Expression(body=tree.body[-1].value)
        ast.fix_missing_locations(body_tree)
        ast.fix_missing_locations(expr_tree)
        try:
            with redirect_stdout(buf):
                exec(  # noqa: S102
                    compile(body_tree, "<code>", "exec"), globals_dict, local_vars
                )
                last_val = eval(  # noqa: S307
                    compile(expr_tree, "<code>", "eval"), globals_dict, local_vars
                )
        except Exception:
            raise PythonInterpreterError(f"Error:\n{traceback.format_exc()}") from None
    else:
        try:
            with redirect_stdout(buf):
                exec(  # noqa: S102
                    compile(tree, "<code>", "exec"), globals_dict, local_vars
                )
        except Exception:
            raise PythonInterpreterError(f"Error:\n{traceback.format_exc()}") from None

    return buf.getvalue(), last_val


def assemble_output(output: str, chart_note: str, last_val: object) -> str:
    """Combine captured stdout, chart notes, and a final-expression repr."""
    if output and chart_note:
        return f"{output}\n{chart_note}"
    if output:
        return output
    if chart_note:
        return chart_note
    if last_val is not _SENTINEL and last_val is not None:
        return repr(last_val)
    return _EMPTY_OUTPUT_GUIDANCE


def _short_answer(value: Any) -> str:
    """Brief, payload-free description of a recorded answer."""
    cls = type(value).__name__
    shape = getattr(value, "shape", None)
    if shape is not None:
        return f"{cls} shape={tuple(shape)}"
    r = repr(value)
    return r if len(r) <= 120 else r[:117] + "..."


def _safe_builtins(allowlist: frozenset[str]) -> dict:
    safe = {
        "print": print,
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "sorted": sorted,
        "reversed": reversed,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "type": type,
        "isinstance": isinstance,
        "hasattr": hasattr,
        "getattr": getattr,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "any": any,
        "all": all,
        "repr": repr,
        "format": format,
        "vars": vars,
        "dir": dir,
        "None": None,
        "True": True,
        "False": False,
        "__import__": _make_safe_import(allowlist),
    }
    return safe


def _make_safe_import(allowlist: frozenset[str]):
    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level != 0:
            raise ImportError("relative imports are not allowed")
        top = name.split(".")[0]
        if top not in allowlist:
            raise ImportError(f"Import not allowed: {name!r}")
        return builtins.__import__(name, globals, locals, fromlist, level)

    return safe_import
