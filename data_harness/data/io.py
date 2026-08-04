"""File ingestion helpers.

Lower the friction of getting data into a run: read common tabular formats into
DataFrames and turn arbitrary inputs (a DataFrame, a dict of frames, a path, or
a list of paths) into named cache handles.
"""

from __future__ import annotations

import keyword
import re
from pathlib import Path
from typing import Any

_READERS = {
    ".csv": "read_csv",
    ".tsv": "read_csv",
    ".parquet": "read_parquet",
    ".pq": "read_parquet",
    ".json": "read_json",
    ".xlsx": "read_excel",
    ".xls": "read_excel",
    ".feather": "read_feather",
}


def load_dataframe(path: str | Path) -> Any:
    """Read a tabular file into a pandas DataFrame, dispatching by extension.

    Args:
        path: Path to a ``.csv``, ``.tsv``, ``.parquet``, ``.json``, ``.xlsx``,
            ``.xls`` or ``.feather`` file.

    Returns:
        A pandas DataFrame.

    Raises:
        ValueError: If the file extension is not recognised.
    """
    import pandas as pd

    p = Path(path)
    reader = _READERS.get(p.suffix.lower())
    if reader is None:
        raise ValueError(
            f"Unsupported file type {p.suffix!r}. "
            f"Supported: {sorted(_READERS)}. Pass a DataFrame instead."
        )
    if p.suffix.lower() == ".tsv":
        return pd.read_csv(p, sep="\t")
    return getattr(pd, reader)(p)


def sanitise_handle(name: str) -> str:
    """Coerce an arbitrary string into a valid Python identifier handle."""
    cleaned = re.sub(r"\W", "_", name).strip("_") or "data"
    if cleaned[0].isdigit():
        cleaned = f"d_{cleaned}"
    if keyword.iskeyword(cleaned):
        cleaned = f"{cleaned}_"
    return cleaned


def to_handles(data: Any) -> dict[str, Any]:
    """Normalise a user-supplied ``data`` argument into ``{handle: value}``.

    Accepts a DataFrame, a mapping of name → value, a file path (str/Path), a
    list/tuple of paths, or any other object (stored under ``"data"``).
    """
    if _is_dataframe(data):
        return {"df": data}
    if isinstance(data, dict):
        return {sanitise_handle(str(k)): v for k, v in data.items()}
    if isinstance(data, (str, Path)):
        path = Path(data)
        return {sanitise_handle(path.stem): load_dataframe(path)}
    if isinstance(data, (list, tuple)) and data and _all_paths(data):
        out: dict[str, Any] = {}
        for item in data:
            path = Path(item)
            out[sanitise_handle(path.stem)] = load_dataframe(path)
        return out
    return {"data": data}


def _all_paths(items: Any) -> bool:
    return all(isinstance(i, (str, Path)) for i in items)


def _is_dataframe(value: Any) -> bool:
    try:
        import pandas as pd

        return isinstance(value, pd.DataFrame)
    except ImportError:
        return False
