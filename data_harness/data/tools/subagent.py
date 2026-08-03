from __future__ import annotations

import copy
import dataclasses
from typing import Callable

from data_harness.data.cache import SessionCache
from data_harness.data.tools.interpreter import PythonInterpreter
from data_harness.data.tools.variables import make_list_variables_spec
from data_harness.llm.providers.base import AsyncProviderAdapter, ProviderAdapter
from data_harness.llm.types import ToolSpec

_SUBAGENT_TOOL_NAME = "subagent"

_WORKER_SYSTEM_TEMPLATE = """\
You are a clean-context worker invoked by another agent.

Your task: {task}

Available input handles (already loaded into your cache): {input_handles}

Use `python_interpreter` to inspect cached handles. Call `save(name, value)` for any
computed artifact worth returning. You must produce final text summarizing your
findings. If you save artifacts, mention what they contain and why they matter."""


def make_subagent_spec(
    adapter_factory: Callable[[], ProviderAdapter | AsyncProviderAdapter],
    parent_tools: list[ToolSpec],
    parent_cache: SessionCache,
    run_dir: str = "./runs",
    get_sub_cache: Callable[[], SessionCache] | None = None,
    make_sub_tools: Callable[[SessionCache], list[ToolSpec]] | None = None,
) -> ToolSpec:
    """Create a subagent tool with an explicit cache boundary.

    ``adapter_factory`` may return either a `ProviderAdapter` or an
    `AsyncProviderAdapter`. Whichever it returns picks the matching driver, so
    a sync adapter runs on the calling thread rather than being pushed through
    an event loop. The tool handler itself stays synchronous, so it works
    identically whether the parent is an `Agent` or an `AsyncAgent`.

    If parent_tools include cache-bound wrappers such as ConnectorRegistry
    wrapped specs, pass make_sub_tools so those handlers can be rebuilt against
    the subagent cache. The fallback path only copies cache-independent tools
    and the built-in cache tools it knows how to rebind.
    """

    def subagent(
        task: str,
        input_handles: list[str] | None = None,
        output_policy: str = "text_only",
    ) -> str:
        from data_harness.data.harness import (
            AsyncHarness,
            Harness,
            run_coroutine_blocking,
        )

        # Validate input_handles against parent cache
        if input_handles:
            missing = [h for h in input_handles if not parent_cache.has_handle(h)]
            if missing:
                return f"Error: input handles not found in parent cache: {missing}"

        # Build sub-cache
        if get_sub_cache is not None:
            sub_cache = get_sub_cache()
        else:
            sub_cache = SessionCache(sample_size=parent_cache.sample_size)

        # Copy requested handles into sub-cache
        if input_handles:
            for handle in input_handles:
                try:
                    sub_cache.put(handle, _copy_cache_value(parent_cache.get(handle)))
                except Exception as exc:
                    return (
                        "Error: failed to copy input handle "
                        f"{handle!r}: {type(exc).__name__}: {exc}"
                    )

        # Track pre-run handles to detect newly created ones
        pre_run_handles = set(sub_cache.handle_names())

        # Build sub-tools — exclude subagent to prevent recursion
        try:
            if make_sub_tools is not None:
                sub_tools = [
                    dataclasses.replace(t)
                    for t in make_sub_tools(sub_cache)
                    if t.name != _SUBAGENT_TOOL_NAME
                ]
            else:
                sub_tools = [
                    _copy_tool_for_subcache(t, sub_cache)
                    for t in parent_tools
                    if t.name != _SUBAGENT_TOOL_NAME
                ]
        except ValueError as exc:
            return f"Error: subagent tools are not isolated: {exc}"

        # Build system prompt
        handles_str = str(input_handles) if input_handles else "none"
        system = _WORKER_SYSTEM_TEMPLATE.format(task=task, input_handles=handles_str)

        # Spawn fresh adapter. A sync adapter gets the sync driver so the
        # subagent keeps the parent's threading semantics; only an async
        # adapter needs a loop spun up to drive it from this sync handler.
        sub_adapter = adapter_factory()
        harness_kwargs = {
            "system": system,
            "tools": sub_tools,
            "run_dir": run_dir,
            "cache": sub_cache,
        }

        try:
            if isinstance(sub_adapter, AsyncProviderAdapter):
                final_text = run_coroutine_blocking(
                    AsyncHarness(adapter=sub_adapter, **harness_kwargs).run(task)
                )
            else:
                final_text = Harness(adapter=sub_adapter, **harness_kwargs).run(task)
        except Exception as exc:
            return f"Error: subagent failed: {type(exc).__name__}: {exc}"

        if output_policy == "text_only":
            return f"Subagent final output:\n{final_text}"

        # publish_created: find newly created handles
        new_handles = {}
        for name in sub_cache.handle_names():
            if name in pre_run_handles:
                continue
            new_handles[name] = sub_cache.get(name)

        if not new_handles:
            return f"Subagent final output:\n{final_text}\n\nPublished outputs: none"

        published_lines = []
        for sub_name, value in new_handles.items():
            parent_name = parent_cache.put(sub_name, value)
            snap = parent_cache.snapshot(parent_name)
            published_lines.append(f"- {sub_name} -> {parent_name}\n  Snapshot: {snap}")

        published_str = "\n".join(published_lines)
        return (
            f"Subagent final output:\n{final_text}\n\n"
            f"Published outputs:\n{published_str}"
        )

    return ToolSpec(
        name=_SUBAGENT_TOOL_NAME,
        description=(
            "Spawn a clean-context subagent to handle a subtask. "
            "The subagent has fresh message history and session cache. "
            "Use input_handles to pass data from this cache to the subagent."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Natural-language instruction for the subagent.",
                },
                "input_handles": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Parent cache handle names to copy into the subagent's cache."
                    ),
                },
                "output_policy": {
                    "type": "string",
                    "enum": ["text_only", "publish_created"],
                    "description": (
                        "'text_only': return only the subagent's final text. "
                        "'publish_created': also copy newly-created handles back to"
                        " parent cache."
                    ),
                },
            },
            "required": ["task"],
        },
        handler=subagent,
    )


def _copy_tool_for_subcache(tool: ToolSpec, sub_cache: SessionCache) -> ToolSpec:
    if tool.name == "python_interpreter":
        return PythonInterpreter.make_tool_spec(sub_cache)
    if tool.name == "list_variables":
        return make_list_variables_spec(sub_cache)
    if _handler_closes_over_cache(tool.handler):
        raise ValueError(
            f"{tool.name!r} has a handler closed over a SessionCache. "
            "Pass make_sub_tools to rebuild cache-bound tool specs for the "
            "subagent cache."
        )
    return dataclasses.replace(tool)


def _handler_closes_over_cache(handler) -> bool:
    closure = getattr(handler, "__closure__", None)
    if not closure:
        return False
    for cell in closure:
        try:
            if isinstance(cell.cell_contents, SessionCache):
                return True
        except ValueError:
            continue
    return False


def _copy_cache_value(value):
    try:
        import pandas as pd

        if isinstance(value, pd.DataFrame):
            # Deep copy is deliberate: shallow DataFrame copies can still share
            # underlying blocks, which would break the parent/subagent boundary
            # for representative in-place mutations.
            return value.copy(deep=True)
    except ImportError:
        pass

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.copy()
    except ImportError:
        pass

    return copy.deepcopy(value)
