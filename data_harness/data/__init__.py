"""The data domain: what makes this a *data* harness rather than a generic one.

Owns the `SessionCache` and its handle/snapshot discipline, the sandboxed
Python interpreter, SQL and connector tools, MCP, and the DataFrame-aware
formatting that keeps large payloads out of the transcript.

May import `llm` and `core`. May not import `app`.
"""

from data_harness.core.serialize import register_snapshotter


def _dataframe_snapshot(obj: object) -> dict | None:
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - exercised via the install matrix
        return None
    if not isinstance(obj, pd.DataFrame):
        return None
    return {
        "type": "dataframe_snapshot",
        "shape": list(obj.shape),
        "columns": list(obj.columns),
        "sample": obj.head(5).to_dict(orient="records"),
    }


def _ndarray_snapshot(obj: object) -> dict | None:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - exercised via the install matrix
        return None
    if not isinstance(obj, np.ndarray):
        return None
    return {
        "type": "ndarray_snapshot",
        "shape": list(obj.shape),
        "dtype": str(obj.dtype),
        "sample": obj.flat[:5].tolist(),
    }


# Registered on import: the run log has to describe a frame without embedding
# one, but only this layer knows what a frame is.
register_snapshotter(_dataframe_snapshot)
register_snapshotter(_ndarray_snapshot)
