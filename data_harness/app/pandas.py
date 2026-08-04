"""Opt-in pandas accessor: ``df.chat("...")``.

This is the most pandasai-like entry point, but it mutates the global pandas
namespace, so it is **not** the documented headline (use ``ask`` / ``Chat``).
Enable it explicitly::

    import data_harness.app.pandas  # registers .chat on every DataFrame
    df.chat("plot revenue by month")
"""

from __future__ import annotations

from typing import Any

import pandas as pd


@pd.api.extensions.register_dataframe_accessor("chat")
class _ChatAccessor:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def __call__(self, question: str, **kwargs: Any) -> Any:
        from data_harness.app.quickstart import ask

        return ask(self._df, question, **kwargs)
