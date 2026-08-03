"""IPython integration: the ``%%ask`` cell magic.

Load it in a notebook with::

    %load_ext data_harness.app.notebook

Then ask questions about a DataFrame in the user namespace::

    %%ask sales_df
    What was total revenue, and plot it by month?

The cell body is the question; the line argument names the data variable. The
returned `RunResult` renders richly (prose, structured value, charts).
"""

from __future__ import annotations

from typing import Any


def _load_magics_class() -> Any:
    from IPython.core.magic import Magics, cell_magic, magics_class

    @magics_class
    class DataHarnessMagics(Magics):
        @cell_magic
        def ask(self, line: str, cell: str) -> Any:
            from data_harness.app.quickstart import ask as _ask

            var = line.strip()
            if not var:
                raise UsageError("Usage: %%ask <dataframe-variable>")
            if var not in self.shell.user_ns:
                raise UsageError(f"Variable {var!r} not found in the namespace.")
            return _ask(self.shell.user_ns[var], cell.strip())

    return DataHarnessMagics


try:  # pragma: no cover - exercised only when IPython is installed
    from IPython.core.error import UsageError
except ImportError:  # pragma: no cover

    class UsageError(Exception):  # type: ignore[no-redef]
        pass


def load_ipython_extension(ipython: Any) -> None:
    """Entry point for ``%load_ext data_harness.app.notebook``."""
    ipython.register_magics(_load_magics_class())
