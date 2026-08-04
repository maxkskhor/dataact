"""Child-process entry point for the subprocess sandbox.

Invoked as ``python -m data_harness.data._sandbox_runner <payload> <result>``. Reads a
pickled payload (code, handle values, allowlist, limits), executes the code
under resource limits with networking disabled, and writes a pickled result.

It deliberately reuses :func:`data_harness.data.tools.interpreter.execute_namespace`
so the sandbox shares the exact same security boundary as the in-process path.
"""

from __future__ import annotations

import os
import pickle
import sys
import uuid
from typing import Any


def _block_network() -> None:
    """Disable Python-level networking inside the child process."""
    import socket

    def _blocked(*args: Any, **kwargs: Any):
        raise RuntimeError("Network access is disabled in the sandbox.")

    socket.socket = _blocked  # type: ignore[assignment,misc]
    socket.create_connection = _blocked  # type: ignore[assignment]


def _apply_limits(cpu_seconds: int | None, memory_mb: int | None) -> None:
    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX
        return
    if cpu_seconds:
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        except (ValueError, OSError):  # pragma: no cover
            pass
    if memory_mb:
        nbytes = memory_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (nbytes, nbytes))
        except (ValueError, OSError):  # pragma: no cover
            pass


def _capture_charts(artifacts_dir: str) -> list[tuple[str, str | None]]:
    if "matplotlib" not in sys.modules:
        return []
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return []
    os.makedirs(artifacts_dir, exist_ok=True)
    charts: list[tuple[str, str | None]] = []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        title = None
        if fig.axes:
            title = fig.axes[0].get_title() or None
        path = os.path.join(artifacts_dir, f"chart_{uuid.uuid4().hex[:8]}.png")
        fig.savefig(path, format="png", bbox_inches="tight")
        charts.append((path, title))
        plt.close(fig)
    return charts


def _run(payload: dict) -> dict:
    from data_harness.data.tools.interpreter import (
        _DEFAULT_ALLOWLIST,
        _SENTINEL,
        PythonInterpreterError,
        _short_answer,
        execute_namespace,
    )

    allowlist = frozenset(payload.get("allowlist") or _DEFAULT_ALLOWLIST)
    saved: dict[str, Any] = {}
    answer_holder: dict[str, Any] = {"set": False, "value": None}

    local_vars: dict[str, Any] = dict(payload["handles"])

    def save(name: str, value: Any) -> str:
        saved[name] = value
        return name

    def answer(value: Any) -> Any:
        answer_holder["set"] = True
        answer_holder["value"] = value
        print(f"Recorded answer: {_short_answer(value)}")
        return value

    local_vars["save"] = save
    local_vars["answer"] = answer

    try:
        output, last_val = execute_namespace(payload["code"], local_vars, allowlist)
    except PythonInterpreterError as exc:
        return {"ok": False, "error": str(exc)}

    last_repr = (
        repr(last_val) if last_val is not _SENTINEL and last_val is not None else None
    )
    return {
        "ok": True,
        "stdout": output,
        "last_repr": last_repr,
        "saved": saved,
        "answer_set": answer_holder["set"],
        "answer": answer_holder["value"],
        "charts": _capture_charts(payload["artifacts_dir"]),
    }


def main() -> None:
    payload_path, result_path = sys.argv[1], sys.argv[2]
    with open(payload_path, "rb") as fh:
        payload = pickle.load(fh)

    if payload.get("block_network", True):
        _block_network()
    _apply_limits(payload.get("cpu_seconds"), payload.get("memory_mb"))

    result = _run(payload)

    with open(result_path, "wb") as fh:
        pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
