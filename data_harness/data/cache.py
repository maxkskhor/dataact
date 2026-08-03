from __future__ import annotations

import json
import keyword
import pickle
import re
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_harness.core.artifacts import ChartArtifact

_VALID_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

_ANSWER_UNSET = object()


def _is_valid_identifier(name: str) -> bool:
    return bool(_VALID_IDENTIFIER.match(name)) and not keyword.iskeyword(name)


@dataclass(frozen=True)
class _ColdEntry:
    path: Path
    storage_type: str


class SessionCache:
    """In-process store that exposes large values as named handles with snapshots.

    Large objects (DataFrames, arrays, query results) are stored by name.
    The model only ever sees a compact snapshot — shape, columns, a few sample
    rows — and operates on the data by writing Python against the handle name.
    This keeps message context lean without hiding data from the model.

    When ``hot_limit`` is set, least-recently-used handles are spilled to disk
    automatically. DataFrames are written as Parquet, NumPy arrays as ``.npy``,
    and everything else as pickle.

    Args:
        sample_size: Number of rows/elements to include in each snapshot.
        storage_dir: Directory for disk-spilled handles. If ``None`` and
            ``hot_limit`` is set, a temporary directory is created
            automatically.
        hot_limit: Maximum number of handles kept in memory at once. ``None``
            means unbounded (all handles stay in memory).
    """

    def __init__(
        self,
        sample_size: int = 5,
        storage_dir: str | Path | None = None,
        hot_limit: int | None = None,
    ) -> None:
        if hot_limit is not None and hot_limit < 1:
            raise ValueError("hot_limit must be at least 1")
        self.sample_size = sample_size
        self.hot_limit = hot_limit
        self._store: dict[str, Any] = {}
        self._cold: dict[str, _ColdEntry] = {}
        self._snapshots: dict[str, str] = {}
        self._semantics: dict[str, dict] = {}
        self._chart_handles: list[str] = []
        self._answer: Any = _ANSWER_UNSET
        self._recency: OrderedDict[str, None] = OrderedDict()
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        if storage_dir is None and hot_limit is not None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="data-harness-cache-")
            self._storage_dir = Path(self._temp_dir.name)
        elif storage_dir is not None:
            self._storage_dir = Path(storage_dir)
            self._storage_dir.mkdir(parents=True, exist_ok=True)
            if self.hot_limit is None:
                # Supplying storage_dir opts into disk-backed cache behaviour.
                # Keep the default bounded so a caller does not create a spill
                # directory that is never used.
                self.hot_limit = 10
        else:
            self._storage_dir = None

    def put(
        self,
        name: str,
        value: Any,
        overwrite: bool = False,
        *,
        semantics: dict | None = None,
    ) -> str:
        """Store a value under ``name`` and return the handle actually used.

        If ``name`` is already taken and ``overwrite`` is ``False``, a numeric
        suffix is appended (``name_2``, ``name_3``, …) and the new handle is
        returned.

        Args:
            name: Desired handle name. Must be a valid Python identifier.
            value: Any Python object. DataFrames and NumPy arrays get
                specialised snapshot and spill formats.
            overwrite: Replace the existing handle if ``True``.
            semantics: Optional business/domain context (e.g. column
                descriptions or units) folded into the snapshot the model sees.

        Returns:
            The handle name under which the value was stored.

        Raises:
            ValueError: If ``name`` is not a valid Python identifier.
        """
        if not _is_valid_identifier(name):
            raise ValueError(
                f"Invalid handle name: {name!r}. Must be a valid Python identifier."
            )
        if overwrite or not self.has_handle(name):
            if overwrite:
                self._delete_cold(name)
            self._put_resolved(name, value, semantics)
            return name
        # Auto-suffix on collision
        suffix = 2
        while True:
            candidate = f"{name}_{suffix}"
            if not self.has_handle(candidate):
                self._put_resolved(candidate, value, semantics)
                return candidate
            suffix += 1

    def get(self, name: str) -> Any:
        """Retrieve a value by handle name, promoting cold entries to hot.

        Args:
            name: A handle previously returned by `put`.

        Returns:
            The stored Python object.

        Raises:
            KeyError: If no handle with ``name`` exists.
        """
        if name in self._store:
            self._mark_recent(name)
            return self._store[name]
        if name in self._cold:
            value = self._read_cold(name)
            self._delete_cold(name)
            self._store[name] = value
            self._mark_recent(name)
            self._enforce_hot_limit()
            return value
        raise KeyError(name)

    def snapshot(self, handle: str) -> str:
        """Return the compact snapshot string for a stored handle.

        The snapshot is a JSON string describing the value's type, shape, and a
        few sample elements. It is what the model sees in message history
        instead of the raw object.

        Args:
            handle: A handle previously returned by `put`.

        Returns:
            A JSON string summary of the stored value.
        """
        if handle in self._snapshots:
            return self._snapshots[handle]
        value = self.get(handle)
        snapshot = self._make_snapshot(value)
        self._snapshots[handle] = snapshot
        return snapshot

    def list_handles(self) -> dict[str, str]:
        """Return a mapping of all handle names to their snapshot strings."""
        return {name: self.snapshot(name) for name in self.handle_names()}

    def handle_names(self) -> list[str]:
        """Return all handle names in most-recently-used order."""
        return list(self._recency.keys())

    def has_handle(self, name: str) -> bool:
        """Return ``True`` if ``name`` is a registered handle (hot or cold)."""
        return name in self._store or name in self._cold

    def items(self):
        for name in self.handle_names():
            yield name, self.get(name)

    def storage_metadata(
        self, *, include_paths: bool = False
    ) -> dict[str, dict[str, str]]:
        metadata = {}
        for name in self.handle_names():
            if name in self._cold:
                entry = self._cold[name]
                metadata[name] = {
                    "location": "disk",
                    "storage_type": entry.storage_type,
                }
                if include_paths:
                    metadata[name]["path"] = str(entry.path)
            else:
                metadata[name] = {"location": "memory", "storage_type": "memory"}
        return metadata

    def delete(self, name: str) -> None:
        """Remove a handle and its associated disk artefact (if any).

        Args:
            name: Handle to remove.

        Raises:
            KeyError: If no handle with ``name`` exists.
        """
        if not self.has_handle(name):
            raise KeyError(name)
        self._store.pop(name, None)
        self._delete_cold(name)
        self._snapshots.pop(name, None)
        self._semantics.pop(name, None)
        self._recency.pop(name, None)
        if name in self._chart_handles:
            self._chart_handles.remove(name)

    def close(self) -> None:
        """Release the temporary storage directory, if one was created."""
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None

    def __del__(self) -> None:
        self.close()

    def _put_resolved(
        self, name: str, value: Any, semantics: dict | None = None
    ) -> None:
        self._store[name] = value
        if semantics is not None:
            self._semantics[name] = semantics
        if isinstance(value, ChartArtifact):
            value.handle = name
            if name not in self._chart_handles:
                self._chart_handles.append(name)
        self._snapshots[name] = self._make_snapshot(value, self._semantics.get(name))
        self._mark_recent(name)
        self._enforce_hot_limit()

    # --- Answer slot --------------------------------------------------------
    def set_answer(self, value: Any) -> None:
        """Record the designated final answer for the current run."""
        self._answer = value

    def get_answer(self) -> Any:
        """Return the recorded answer, or ``None`` if none was set."""
        return None if self._answer is _ANSWER_UNSET else self._answer

    @property
    def has_answer(self) -> bool:
        return self._answer is not _ANSWER_UNSET

    def clear_answer(self) -> None:
        self._answer = _ANSWER_UNSET

    # --- Charts -------------------------------------------------------------
    def list_charts(self) -> list[ChartArtifact]:
        """Return all `ChartArtifact` handles still present in the cache."""
        charts = []
        for name in self._chart_handles:
            if self.has_handle(name):
                value = self.get(name)
                if isinstance(value, ChartArtifact):
                    charts.append(value)
        return charts

    # --- Semantics ----------------------------------------------------------
    def describe(self, name: str, semantics: dict) -> None:
        """Attach or update semantic context for an existing handle.

        Args:
            name: An existing handle.
            semantics: Domain context folded into the handle's snapshot.

        Raises:
            KeyError: If no handle with ``name`` exists.
        """
        if not self.has_handle(name):
            raise KeyError(name)
        self._semantics[name] = semantics
        self._snapshots[name] = self._make_snapshot(self.get(name), semantics)

    def get_semantics(self, name: str) -> dict | None:
        """Return the semantic context attached to ``name``, if any."""
        return self._semantics.get(name)

    def _mark_recent(self, name: str) -> None:
        self._recency[name] = None
        self._recency.move_to_end(name)

    def _enforce_hot_limit(self) -> None:
        if self._storage_dir is None or self.hot_limit is None:
            return
        while len(self._store) > self.hot_limit:
            for candidate in self._recency:
                if candidate in self._store:
                    self._spill(candidate)
                    break
            else:
                break

    def _spill(self, name: str) -> None:
        value = self._store.pop(name)
        self._cold[name] = self._write_cold(name, value)

    def _write_cold(self, name: str, value: Any) -> _ColdEntry:
        if self._storage_dir is None:
            raise RuntimeError("storage_dir is required for disk-backed cache")

        try:
            import numpy as np

            if isinstance(value, np.ndarray) and value.dtype != object:
                path = self._storage_dir / f"{name}.npy"
                with path.open("wb") as fh:
                    np.save(fh, value, allow_pickle=False)
                return _ColdEntry(path=path, storage_type="numpy_npy")
        except ImportError:
            pass

        try:
            import pandas as pd

            if isinstance(value, pd.DataFrame):
                parquet_path = self._storage_dir / f"{name}.parquet"
                try:
                    value.to_parquet(parquet_path, index=False)
                    return _ColdEntry(
                        path=parquet_path,
                        storage_type="dataframe_parquet",
                    )
                except (ImportError, TypeError, ValueError):
                    # Parquet is the preferred teaching path, but pyarrow /
                    # fastparquet are not core dependencies for this reference
                    # implementation. Fall back explicitly rather than adding a
                    # heavy storage dependency to the default install.
                    pass

                path = self._storage_dir / f"{name}.pkl"
                value.to_pickle(path)
                return _ColdEntry(path=path, storage_type="dataframe_pickle")
        except ImportError:
            pass

        path = self._storage_dir / f"{name}.pickle"
        with path.open("wb") as fh:
            pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return _ColdEntry(path=path, storage_type="pickle")

    def _read_cold(self, name: str) -> Any:
        entry = self._cold[name]
        if entry.storage_type == "numpy_npy":
            import numpy as np

            with entry.path.open("rb") as fh:
                return np.load(fh, allow_pickle=False)
        if entry.storage_type == "dataframe_parquet":
            import pandas as pd

            return pd.read_parquet(entry.path)
        if entry.storage_type in {"dataframe_pickle", "pandas_pickle"}:
            import pandas as pd

            return pd.read_pickle(entry.path)
        with entry.path.open("rb") as fh:
            return pickle.load(fh)

    def _delete_cold(self, name: str) -> None:
        entry = self._cold.pop(name, None)
        if entry is not None:
            try:
                entry.path.unlink()
            except FileNotFoundError:
                pass

    def _make_snapshot(self, value: Any, semantics: dict | None = None) -> str:
        if isinstance(value, ChartArtifact):
            return value.snapshot()

        try:
            import pandas as pd

            if isinstance(value, pd.DataFrame):
                return self._with_semantics(self._snapshot_dataframe(value), semantics)
        except ImportError:
            pass

        try:
            import numpy as np

            if isinstance(value, np.ndarray):
                return self._with_semantics(self._snapshot_ndarray(value), semantics)
        except ImportError:
            pass

        if isinstance(value, list):
            return self._with_semantics(self._snapshot_list(value), semantics)
        if isinstance(value, dict):
            return self._with_semantics(self._snapshot_dict(value), semantics)
        # Scalar
        scalar = f"value: {value!r}"
        if semantics:
            return f"{scalar} | semantics: {json.dumps(semantics, default=str)}"
        return scalar

    @staticmethod
    def _with_semantics(snapshot: str, semantics: dict | None) -> str:
        if not semantics:
            return snapshot
        try:
            obj = json.loads(snapshot)
            obj["semantics"] = semantics
            return json.dumps(obj, default=str)
        except (ValueError, TypeError):
            return f"{snapshot} | semantics: {json.dumps(semantics, default=str)}"

    def _snapshot_dataframe(self, df) -> str:
        cols = list(df.columns)
        shape = list(df.shape)
        sample = df.head(self.sample_size).to_dict(orient="records")
        return json.dumps(
            {
                "type": "dataframe",
                "shape": shape,
                "columns": cols,
                "sample": sample,
            },
            default=str,
        )

    def _snapshot_ndarray(self, arr) -> str:
        flat = arr.flat
        sample = [
            x.item() if hasattr(x, "item") else x
            for _, x in zip(range(self.sample_size), flat)
        ]
        return json.dumps(
            {
                "type": "ndarray",
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "sample": sample,
            }
        )

    def _snapshot_list(self, lst: list) -> str:
        sample = lst[: self.sample_size]
        try:
            sample_json = json.dumps(sample)
            sample_value = json.loads(sample_json)
        except Exception:
            sample_value = repr(sample)
        return json.dumps(
            {
                "type": "list",
                "length": len(lst),
                "sample": sample_value,
            }
        )

    def _snapshot_dict(self, d: dict) -> str:
        keys = list(d.keys())[: self.sample_size]
        sample = {k: d[k] for k in keys}
        try:
            sample_str = json.dumps(sample, default=repr)
        except Exception:
            sample_str = repr(sample)
        return json.dumps(
            {
                "type": "dict",
                "total_keys": len(d),
                "sample_keys": keys,
                "sample": json.loads(sample_str),
            }
        )
