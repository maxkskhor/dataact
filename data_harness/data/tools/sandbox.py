"""Subprocess-isolated Python interpreter.

`SubprocessPythonInterpreter` runs model code in a separate Python process with
networking disabled and CPU/wall-clock (and optional memory) limits. Handle
values cross the boundary as pickles; ``save()`` / ``answer()`` results and
rendered charts are collected from the child and merged back into the parent
`SessionCache`.

This is a real isolation improvement over the in-process AST guard, but it is
not a container/VM sandbox: it shares the host filesystem and Python install.
Container/WASM isolation remains future work (see ``plan/PLAN_v4.md``).
"""

from __future__ import annotations

import pickle
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from data_harness.core.artifacts import ChartArtifact
from data_harness.core.exceptions import ExecutionError
from data_harness.data.cache import SessionCache
from data_harness.data.tools.interpreter import (
    _DEFAULT_ALLOWLIST,
    _EMPTY_OUTPUT_GUIDANCE,
    PythonInterpreterError,
    _interpreter_description,
)
from data_harness.llm.types import ToolAnnotations, ToolSpec

_SANDBOX_ANNOTATIONS = ToolAnnotations(
    title="Python Interpreter (sandboxed)",
    read_only=False,
    cache_mutating=True,
    destructive=False,
    open_world=False,
)


class SubprocessPythonInterpreter:
    """Run interpreter code in an isolated child process."""

    def __init__(
        self,
        cache: SessionCache,
        allowlist: frozenset[str] | None = None,
        artifacts_dir: str | Path | None = None,
        *,
        timeout: float = 30.0,
        cpu_seconds: int | None = 25,
        memory_mb: int | None = None,
        block_network: bool = True,
    ) -> None:
        self._cache = cache
        self._allowlist = allowlist if allowlist is not None else _DEFAULT_ALLOWLIST
        self._artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else None
        self._temp_artifacts: tempfile.TemporaryDirectory[str] | None = None
        self._timeout = timeout
        self._cpu_seconds = cpu_seconds
        self._memory_mb = memory_mb
        self._block_network = block_network

    def _artifacts_path(self) -> Path:
        if self._artifacts_dir is not None:
            self._artifacts_dir.mkdir(parents=True, exist_ok=True)
            return self._artifacts_dir
        if self._temp_artifacts is None:
            self._temp_artifacts = tempfile.TemporaryDirectory(
                prefix="data-harness-charts-"
            )
        return Path(self._temp_artifacts.name)

    def run(self, code: str) -> str:
        handles = {name: self._cache.get(name) for name in self._cache.handle_names()}
        payload = {
            "code": code,
            "handles": handles,
            "allowlist": list(self._allowlist),
            "artifacts_dir": str(self._artifacts_path()),
            "cpu_seconds": self._cpu_seconds,
            "memory_mb": self._memory_mb,
            "block_network": self._block_network,
        }

        with tempfile.TemporaryDirectory(prefix="data-harness-sandbox-") as tmp:
            payload_path = Path(tmp) / "payload.pkl"
            result_path = Path(tmp) / "result.pkl"
            try:
                with payload_path.open("wb") as fh:
                    pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
            except (pickle.PicklingError, TypeError) as exc:
                raise PythonInterpreterError(
                    f"Could not serialise cache handles for the sandbox: {exc}"
                ) from None

            try:
                proc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "data_harness.data._sandbox_runner",
                        str(payload_path),
                        str(result_path),
                    ],
                    capture_output=True,
                    timeout=self._timeout,
                )
            except subprocess.TimeoutExpired:
                # The code never finished, so this is the environment failing
                # rather than the model's code being wrong.
                raise ExecutionError(
                    f"Execution timed out after {self._timeout}s in the sandbox."
                ) from None

            if proc.returncode != 0 or not result_path.exists():
                stderr = proc.stderr.decode("utf-8", "replace")[-800:]
                raise ExecutionError(
                    f"Sandbox process failed (exit {proc.returncode}). {stderr}"
                )

            with result_path.open("rb") as fh:
                result = pickle.load(fh)

        return self._merge_result(result)

    def _merge_result(self, result: dict) -> str:
        if not result.get("ok"):
            raise PythonInterpreterError(result.get("error", "unknown sandbox error"))

        for name, value in result["saved"].items():
            self._cache.put(name, value, overwrite=True)
        if result["answer_set"]:
            self._cache.set_answer(result["answer"])

        chart_handles: list[str] = []
        for path, title in result["charts"]:
            artifact = ChartArtifact(path=path, format="png", title=title)
            handle = self._cache.put("chart", artifact)
            artifact.handle = handle
            chart_handles.append(handle)
        chart_note = "\n".join(
            f"Rendered chart saved to handle `{h}`." for h in chart_handles
        )

        stdout = result["stdout"]
        if stdout and chart_note:
            return f"{stdout}\n{chart_note}"
        if stdout:
            return stdout
        if chart_note:
            return chart_note
        if result["last_repr"] is not None:
            return result["last_repr"]
        return _EMPTY_OUTPUT_GUIDANCE

    @staticmethod
    def make_tool_spec(
        cache: SessionCache,
        artifacts_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> ToolSpec:
        interp = SubprocessPythonInterpreter(
            cache=cache, artifacts_dir=artifacts_dir, **kwargs
        )
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
            annotations=_SANDBOX_ANNOTATIONS,
        )


# Stable handle for tests that need a fresh uuid-based path.
def _fresh_chart_name() -> str:  # pragma: no cover - trivial helper
    return f"chart_{uuid.uuid4().hex[:8]}"
