"""The boundary is worth something, not just declared.

Splitting the package into layers only means something if `data_harness.core`
can run an agent with no data domain at all — no cache, no pandas, no
interpreter. A layering that no one can use without the layer above it is
decoration.
"""

from __future__ import annotations

import pytest

from data_harness.core.environment import NullEnvironment, RunEnvironment, RunState
from data_harness.core.loop import AsyncHarness, Harness
from data_harness.llm.testing import FakeAdapter, FakeAsyncAdapter
from data_harness.llm.types import ToolSpec


def upper_spec() -> ToolSpec:
    return ToolSpec(
        name="shout",
        description="Upper-case the input.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        handler=lambda text: text.upper(),
    )


# ── core runs on its own ────────────────────────────────────────────────────


def test_core_harness_runs_a_tool_loop_with_no_domain(tmp_path):
    """No cache, no pandas, no interpreter: still a working agent."""
    harness = Harness(
        adapter=FakeAdapter(
            [
                FakeAdapter.tool_use("tu_1", "shout", {"text": "hello"}),
                FakeAdapter.text("I shouted it."),
            ]
        ),
        system="sys",
        tools=[upper_spec()],
    )

    result = harness.run_result("go")

    assert result.status == "success"
    assert result.text == "I shouted it."
    assert harness.messages[2].content[0].content == "HELLO"


@pytest.mark.asyncio
async def test_core_async_harness_runs_with_no_domain(tmp_path):
    harness = AsyncHarness(
        adapter=FakeAsyncAdapter([FakeAsyncAdapter.text("fine")]),
        system="sys",
        tools=[],
    )

    assert await harness.run("go") == "fine"


def test_the_null_environment_contributes_no_run_state(tmp_path):
    """A domain-free run reports no handles, no answer, and no charts."""
    result = Harness(
        adapter=FakeAdapter([FakeAdapter.text("done")]),
        system="sys",
        tools=[],
    ).run_result("go")

    assert result.cache_snapshots == {}
    assert result.cache_storage == {}
    assert result.value is None
    assert result.charts == []


def test_a_custom_environment_shapes_the_transcript_and_the_result(tmp_path):
    """The seam is real: a third domain can plug in without touching core.

    This is the actual test of the boundary. If the loop still reached for a
    `SessionCache`, there would be no way to write this.
    """

    class ShoutingEnvironment:
        """Renders every tool result in caps and records one fixed handle."""

        def render_tool_output(self, value: object) -> str:
            return f"<<{value}>>"

        def capture(self) -> RunState:
            return RunState(snapshots={"answer": "42"}, value=42)

        def storage_metadata(self) -> dict[str, dict[str, str]]:
            return {"answer": {"location": "memory", "storage_type": "memory"}}

    harness = Harness(
        adapter=FakeAdapter(
            [
                FakeAdapter.tool_use("tu_1", "shout", {"text": "hi"}),
                FakeAdapter.text("done"),
            ]
        ),
        system="sys",
        tools=[upper_spec()],
        environment=ShoutingEnvironment(),
    )

    result = harness.run_result("go")

    assert harness.messages[2].content[0].content == "<<HI>>"
    assert result.value == 42
    assert result.cache_snapshots == {"answer": "42"}


def test_null_environment_satisfies_the_protocol():
    assert isinstance(NullEnvironment(), RunEnvironment)


def test_the_cache_environment_also_satisfies_it():
    from data_harness.data.environment import CacheEnvironment

    assert isinstance(CacheEnvironment(), RunEnvironment)
