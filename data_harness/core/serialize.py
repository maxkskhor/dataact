from __future__ import annotations

import dataclasses
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any


def to_jsonable(obj: Any) -> Any:
    """Recursively convert obj to a JSON-serializable structure. Never raises."""
    try:
        return _convert(obj)
    except Exception as exc:
        return f"<serialization error: {exc!r}>"


def _convert(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, Enum):
        return obj.value

    if isinstance(obj, datetime):
        return obj.isoformat()

    if isinstance(obj, Exception):
        return {"error_type": type(obj).__name__, "error_message": str(obj)}

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        result: dict[str, Any] = {}
        if isinstance(obj, _get_text_block_type()):
            result["type"] = "text"
            result["text"] = _convert(obj.text)  # type: ignore[attr-defined]
        elif isinstance(obj, _get_tool_use_block_type()):
            result["type"] = "tool_use"
            result["id"] = _convert(obj.tool_use_id)  # type: ignore[attr-defined]
            result["name"] = _convert(obj.tool_name)  # type: ignore[attr-defined]
            result["input"] = _convert(obj.tool_input)  # type: ignore[attr-defined]
        elif isinstance(obj, _get_tool_result_block_type()):
            result["type"] = "tool_result"
            result["tool_use_id"] = _convert(obj.tool_use_id)  # type: ignore[attr-defined]
            result["content"] = _convert(obj.content)  # type: ignore[attr-defined]
            result["is_error"] = _convert(obj.is_error)  # type: ignore[attr-defined]
        else:
            for f in dataclasses.fields(obj):
                result[f.name] = _convert(getattr(obj, f.name))
        return result

    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_convert(item) for item in obj]

    for snapshotter in _SNAPSHOTTERS:
        snapshot = snapshotter(obj)
        if snapshot is not None:
            return snapshot

    return repr(obj)


#: Type-specific snapshotters, tried in registration order. Each returns a
#: payload-free dict for a value it recognises, or ``None`` to pass.
#:
#: The log has to describe a DataFrame without embedding one, but knowing what
#: a DataFrame *is* belongs to the data layer. `data_harness.data` registers
#: its own on import; core on its own falls back to `repr`.
_SNAPSHOTTERS: list[Callable[[Any], dict | None]] = []


def register_snapshotter(snapshotter: Callable[[Any], dict | None]) -> None:
    """Teach `to_jsonable` how to describe one more kind of value."""
    _SNAPSHOTTERS.append(snapshotter)


def _get_text_block_type():
    try:
        from data_harness.llm.types import TextBlock

        return TextBlock
    except ImportError:
        return type(None)


def _get_tool_use_block_type():
    try:
        from data_harness.llm.types import ToolUseBlock

        return ToolUseBlock
    except ImportError:
        return type(None)


def _get_tool_result_block_type():
    try:
        from data_harness.llm.types import ToolResultBlock

        return ToolResultBlock
    except ImportError:
        return type(None)
