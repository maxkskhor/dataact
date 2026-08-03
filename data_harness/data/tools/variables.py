from __future__ import annotations

from data_harness.data.cache import SessionCache
from data_harness.llm.types import ToolAnnotations, ToolSpec

_LIST_VARIABLES_ANNOTATIONS = ToolAnnotations(
    title="List Variables",
    read_only=True,
    cache_mutating=False,
    open_world=False,
)


def make_list_variables_spec(cache: SessionCache) -> ToolSpec:
    def list_variables() -> str:
        handles = cache.list_handles()
        if not handles:
            return "No variables in session cache."
        lines = [f"Session cache ({len(handles)} handle(s)):"]
        for name, snapshot in handles.items():
            lines.append(f"\n  {name}:\n    {snapshot}")
        return "\n".join(lines)

    return ToolSpec(
        name="list_variables",
        description=(
            "List all variables currently stored in the session cache"
            " with their snapshots."
        ),
        input_schema={"type": "object", "properties": {}},
        handler=list_variables,
        annotations=_LIST_VARIABLES_ANNOTATIONS,
    )
