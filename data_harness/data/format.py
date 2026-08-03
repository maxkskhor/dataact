from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from data_harness.data.cache import SessionCache

_INLINE_STR_MAX = 500
_INLINE_JSON_MAX = 2000
_INLINE_COLLECTION_MAX = 20


def format_tool_output(
    value: Any,
    cache: "SessionCache | None" = None,
    preferred_name: str | None = None,
) -> str:
    """Decide whether to inline output or cache it; return a string for the message."""
    if isinstance(value, Exception):
        return f"Error: {type(value).__name__}: {value}"

    # DataFrame / ndarray → always cache
    if _is_dataframe(value) or _is_ndarray(value):
        return _cache_value(value, cache, preferred_name or _default_name(value))

    # Short string → inline
    if isinstance(value, str):
        if len(value) <= _INLINE_STR_MAX:
            return value
        return _cache_value(value, cache, preferred_name or "text_result")

    # Scalar (int, float, bool, None)
    if isinstance(value, (int, float, bool)) or value is None:
        return str(value)

    # Short dict/list → inline JSON repr
    if isinstance(value, (dict, list)):
        try:
            serialized = json.dumps(value, default=repr)
        except Exception:
            serialized = repr(value)
        if (
            len(serialized) <= _INLINE_JSON_MAX
            and _collection_size(value) <= _INLINE_COLLECTION_MAX
        ):
            return serialized
        return _cache_value(value, cache, preferred_name or _default_name(value))

    # Unknown object with preferred_name → cache
    if preferred_name is not None and cache is not None:
        return _cache_value(value, cache, preferred_name)

    # Unknown object → repr truncated
    r = repr(value)
    if len(r) > _INLINE_STR_MAX:
        return r[:_INLINE_STR_MAX] + "..."
    return r


def _cache_value(value: Any, cache: "SessionCache | None", name: str) -> str:
    if cache is None:
        # No cache available, fall back to repr
        r = repr(value)
        if len(r) > _INLINE_STR_MAX:
            return r[:_INLINE_STR_MAX] + "..."
        return r
    resolved = cache.put(name, value)
    snapshot = cache.snapshot(resolved)
    return f"Saved as `{resolved}`\nSnapshot: {snapshot}"


def _default_name(value: Any) -> str:
    if _is_dataframe(value):
        return "dataframe"
    if _is_ndarray(value):
        return "array"
    if isinstance(value, dict):
        return "result_dict"
    if isinstance(value, list):
        return "result_list"
    return "result"


def _collection_size(value: Any) -> int:
    if isinstance(value, dict):
        return len(value)
    if isinstance(value, list):
        return len(value)
    return 0


def _is_dataframe(value: Any) -> bool:
    try:
        import pandas as pd

        return isinstance(value, pd.DataFrame)
    except ImportError:
        return False


def _is_ndarray(value: Any) -> bool:
    try:
        import numpy as np

        return isinstance(value, np.ndarray)
    except ImportError:
        return False
