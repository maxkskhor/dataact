from __future__ import annotations

from typing import Any, Callable

from data_harness.data.cache import SessionCache
from data_harness.data.format import format_tool_output
from data_harness.llm.types import ToolSpec


class ConnectorRegistry:
    def __init__(self) -> None:
        self._directory: dict[str, str] = {}  # name -> one-line description
        self._connector_tools: dict[
            str, list[ToolSpec]
        ] = {}  # name -> list of ToolSpec

    def register(
        self,
        name: str,
        description: str,
        tools: list[ToolSpec],
    ) -> None:
        self._directory[name] = description
        # Ensure all tools are hidden by default
        for spec in tools:
            spec.visible = False
        self._connector_tools[name] = list(tools)

    def get_load_connectors_spec(self) -> ToolSpec:
        directory = dict(self._directory)
        connector_tools = self._connector_tools

        def load_connector(name: str) -> str:
            if name not in connector_tools:
                available = list(directory.keys())
                return f"Error: connector {name!r} not found. Available: {available}"
            for spec in connector_tools[name]:
                spec.visible = True
            desc = directory.get(name, "")
            tool_names = [s.name for s in connector_tools[name]]
            return (
                f"Loaded connector {name!r}.\n"
                f"Description: {desc}\n"
                f"Available tools: {tool_names}"
            )

        dir_lines = "\n".join(f"- {k}: {v}" for k, v in directory.items())
        return ToolSpec(
            name="load_connectors",
            description=(
                f"Load a data connector to make its tools available.\n"
                f"Available connectors:\n{dir_lines}"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            f"Connector name. One of: {list(directory.keys())}"
                        ),
                    }
                },
                "required": ["name"],
            },
            handler=load_connector,
            visible=True,
        )

    def all_tool_specs(self) -> list[ToolSpec]:
        specs = []
        for tool_list in self._connector_tools.values():
            specs.extend(tool_list)
        return specs

    def call_connector(
        self,
        tool_name: str,
        tool_input: dict,
        cache: SessionCache,
    ) -> str:
        for tool_list in self._connector_tools.values():
            for spec in tool_list:
                if spec.name == tool_name and spec.handler is not None:
                    raw = spec.handler(**tool_input)
                    return format_tool_output(
                        raw, cache=cache, preferred_name=tool_name.split("__")[-1]
                    )
        return f"Error: tool {tool_name!r} not found"

    def make_wrapped_specs(self, cache: SessionCache) -> list[ToolSpec]:
        """
        Return ToolSpecs whose handlers auto-cache large results.

        Replaces the specs in the registry in-place so that load_connectors'
        visibility flip applies to the returned (wrapped) specs, not stale originals.
        """
        result = []
        for connector_name, tool_list in self._connector_tools.items():
            new_list = []
            for orig_spec in tool_list:
                handler = orig_spec.handler
                if handler is None:
                    new_list.append(orig_spec)
                    result.append(orig_spec)
                    continue
                preferred = orig_spec.name.split("__")[-1]

                def make_handler(h: Callable, pname: str):
                    def wrapped(**kwargs: Any) -> str:
                        raw = h(**kwargs)
                        return format_tool_output(
                            raw, cache=cache, preferred_name=pname
                        )

                    return wrapped

                new_spec = ToolSpec(
                    name=orig_spec.name,
                    description=orig_spec.description,
                    input_schema=orig_spec.input_schema,
                    handler=make_handler(handler, preferred),
                    visible=orig_spec.visible,
                    annotations=orig_spec.annotations,
                )
                new_list.append(new_spec)
                result.append(new_spec)
            # Replace in registry so load_connectors flips the wrapped specs' visible
            self._connector_tools[connector_name] = new_list
        return result
