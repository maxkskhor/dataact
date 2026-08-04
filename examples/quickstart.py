"""Minimal `Agent` quick start.

Requires ANTHROPIC_API_KEY when run as a script. Tests import `build_agent`
and drive it with `FakeAdapter` instead.
"""

from __future__ import annotations

import os
import sys

from data_harness import Agent


def load_local_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def build_agent(adapter, system="You are a data analyst."):
    return Agent(adapter=adapter, system=system)


if __name__ == "__main__":
    load_local_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set. Skipping live quick start.")
        sys.exit(0)

    from data_harness.llm.providers.anthropic import AnthropicAdapter

    agent = build_agent(AnthropicAdapter(model="claude-sonnet-4-6"))
    result = agent.run("Compute the mean of [1, 2, 3, 4, 5] and print it.")
    print(result)
