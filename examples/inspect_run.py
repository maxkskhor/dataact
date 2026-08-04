"""Demonstrate run inspection using run_result() with FakeAdapter.

Runs without any provider API keys.
"""

from data_harness import Agent, RunResult
from data_harness.llm.testing import FakeAdapter

adapter = FakeAdapter([FakeAdapter.text("The mean of [1, 2, 3] is 2.0")])

agent = Agent(
    adapter=adapter,
    system="You are a helpful data assistant.",
    run_dir="/tmp/data-harness-inspect-run-example",
)

result: RunResult = agent.run_result("What is the mean of [1, 2, 3]?")

print(f"text:      {result.text}")
print(f"status:    {result.status}")
print(f"turns:     {result.turns}")
print(f"run_id:    {result.run_id}")
print(
    f"usage:     input={result.usage.input_tokens} output={result.usage.output_tokens}"
)
