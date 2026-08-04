"""Artefact types produced by tool execution.

A `ChartArtifact` is a reference to a chart rendered to disk. The raw image
bytes never enter message history or the JSONL log — only the on-disk path and a
compact snapshot do, exactly like a `SessionCache` handle. The bytes are read
lazily for notebook display via ``_repr_png_`` / ``_repr_html_``.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ChartArtifact:
    """A chart rendered to disk and referenced by path.

    Attributes:
        path: Absolute path to the rendered image file.
        format: Image format, e.g. ``"png"``.
        title: Optional human-readable title captured from the figure.
        handle: Cache handle the artefact is stored under, if any.
    """

    path: str
    format: str = "png"
    title: str | None = None
    handle: str | None = None

    def read_bytes(self) -> bytes:
        """Read the raw image bytes from disk on demand."""
        return Path(self.path).read_bytes()

    def snapshot(self) -> str:
        """Compact, payload-free description used in messages and logs."""
        title = f", title={self.title!r}" if self.title else ""
        return f'{{"type": "chart", "format": {self.format!r}{title}}}'

    # --- Rich display for Jupyter / IPython ---------------------------------
    def _repr_png_(self) -> bytes | None:
        if self.format == "png":
            return self.read_bytes()
        return None

    def _repr_html_(self) -> str:
        encoded = base64.b64encode(self.read_bytes()).decode("ascii")
        alt = self.title or "chart"
        return (
            f'<img src="data:image/{self.format};base64,{encoded}" '
            f'alt="{alt}" style="max-width:100%;" />'
        )

    def __repr__(self) -> str:
        title = f" title={self.title!r}" if self.title else ""
        return f"<ChartArtifact {self.format}{title} path={self.path!r}>"
