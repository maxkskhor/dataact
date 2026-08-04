"""MCP bridge — use any MCP server's tools inside the harness.

data-harness acts as an MCP **client**: it connects to an MCP server (e.g. a
Postgres, SQLite, or filesystem server), lists its tools, and exposes them as
ordinary `ToolSpec`s. Large tool results flow through the `SessionCache` like any
other tool output (snapshot to the model, raw payload kept as a handle), and the
tools are registered behind progressive disclosure via the connector registry.

The MCP Python SDK is async; `MCPClient` runs that session in a background event
loop so the synchronous `Harness` can call MCP tools. Requires the ``[mcp]``
extra.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any

from data_harness.llm.types import ToolAnnotations, ToolSpec


@dataclass
class MCPServer:
    """How to launch a stdio MCP server."""

    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None


def _result_to_text(result: Any) -> str:
    """Flatten an MCP CallToolResult into a string for the harness."""
    parts: list[str] = []
    for item in getattr(result, "content", None) or []:
        text = getattr(item, "text", None)
        if text is not None:
            parts.append(text)
    text = "\n".join(parts)
    if getattr(result, "isError", False):
        return f"Error: {text}"
    return text or "(no content)"


class MCPClient:
    """Synchronous handle to an MCP stdio server (async session on a bg thread).

    Connect once and reuse across tool calls; close to shut the server down.
    Usable as a context manager.
    """

    def __init__(self, server: MCPServer, *, timeout: float = 60.0) -> None:
        self._server = server
        self._timeout = timeout
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: Any = None
        self._stop: asyncio.Event | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self.tools: list[Any] = []

    def connect(self) -> MCPClient:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=self._timeout):
            raise RuntimeError("MCP server did not become ready in time")
        if self._error is not None:
            raise RuntimeError(f"MCP connect failed: {self._error!r}")
        return self

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        finally:
            self._loop.close()

    async def _serve(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=self._server.command,
            args=self._server.args,
            env=self._server.env,
        )
        self._stop = asyncio.Event()
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self.tools = list((await session.list_tools()).tools)
                    self._ready.set()
                    await self._stop.wait()
        except BaseException as exc:  # noqa: BLE001 - surface to connect()
            self._error = exc
            self._ready.set()

    def call(self, name: str, arguments: dict) -> str:
        """Call an MCP tool by its server-side name; return flattened text."""
        if self._loop is None or self._session is None:
            raise RuntimeError("MCP client is not connected")
        future = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(name, arguments), self._loop
        )
        return _result_to_text(future.result(timeout=self._timeout))

    def close(self) -> None:
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread is not None:
            self._thread.join(timeout=self._timeout)
        self._loop = self._thread = self._session = None

    def __enter__(self) -> MCPClient:
        return self.connect()

    def __exit__(self, *exc: object) -> None:
        self.close()


def mcp_tool_specs(client: MCPClient, *, prefix: str | None = None) -> list[ToolSpec]:
    """Adapt an MCP server's tools into `ToolSpec`s.

    Handlers return the raw tool text; the connector registry / harness route
    large results into the `SessionCache`. Tool names are prefixed with the
    server name to avoid collisions.
    """
    specs: list[ToolSpec] = []
    for tool in client.tools:
        spec_name = f"{prefix}__{tool.name}" if prefix else tool.name

        def make_handler(tool_name: str):
            def handler(**kwargs: Any) -> str:
                return client.call(tool_name, kwargs)

            return handler

        specs.append(
            ToolSpec(
                name=spec_name,
                description=tool.description or f"MCP tool {tool.name}",
                input_schema=getattr(tool, "inputSchema", None)
                or {"type": "object", "properties": {}},
                handler=make_handler(tool.name),
                visible=False,
                annotations=ToolAnnotations(
                    title=tool.name,
                    read_only=False,
                    open_world=True,
                ),
            )
        )
    return specs
