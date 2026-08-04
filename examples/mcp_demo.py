"""Live MCP-bridge demo: data-harness using an external MCP server's tools.

    uv run python examples/mcp_demo.py

Connects to a lightweight MCP server (``mcp-server-time`` via uvx) and lets the
agent use its tools through the harness — hidden until the model loads the
connector. **Any** stdio MCP server works the same way, including a Postgres MCP
server (swap the command/args). Needs an API key and the ``[mcp]`` extra.
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from data_harness import Agent
from data_harness.app.quickstart import resolve_adapter


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek/deepseek-v4-flash")
    parser.add_argument("--server-command", default="uvx")
    parser.add_argument("--server-args", nargs="+", default=["mcp-server-time"])
    parser.add_argument("--name", default="time")
    parser.add_argument(
        "--question",
        default="What is the current time in Asia/Tokyo? Use the time connector.",
    )
    args = parser.parse_args()

    agent = Agent(
        adapter=resolve_adapter(args.model),
        system=(
            "You are a helpful assistant. Tools live behind connectors — call "
            "load_connectors to make a connector's tools available, then use them."
        ),
        run_dir="./runs/mcp",
    )
    agent.add_mcp_server(args.name, args.server_command, args=args.server_args)
    try:
        tools = [t.name for t in agent._mcp_clients[args.name].tools]
        print(f"MCP server '{args.name}' exposes: {tools}\n")
        result = agent.run_result(args.question)
        print("answer:", result.text)
        print(f"\n({result.turns} turns)")
    finally:
        agent.close()


if __name__ == "__main__":
    main()
