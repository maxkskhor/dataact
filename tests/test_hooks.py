"""Phase 4: hooks.

The loop used to have exactly one extension point, `register_reminder`, which
could append text to the prompt and nothing else. Everything more interesting
was hardcoded: the interpreter approval gate lived inside the dispatcher and
was keyed on the literal string "python_interpreter".

The test that matters most here is not that hooks fire. It is that the gate is
now built out of the same public mechanism anyone else has, which is what
makes it evidence the mechanism is sufficient rather than decorative.
"""

from __future__ import annotations

import pytest

from data_harness.core.hooks import (
    AfterToolCall,
    AfterTurn,
    BeforeToolCall,
    BeforeTurn,
    Block,
    HookError,
    HookRegistry,
    Reminder,
    Replace,
    Stop,
)
from data_harness.data.harness import AsyncHarness, Harness
from data_harness.llm.testing import FakeAdapter, FakeAsyncAdapter
from data_harness.llm.types import ToolSpec


def echo_spec(calls: list | None = None) -> ToolSpec:
    def handler(value: str) -> str:
        if calls is not None:
            calls.append(value)
        return value

    return ToolSpec(
        name="echo",
        description="Echo the input.",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        handler=handler,
    )


def tool_then_text(cls=FakeAdapter):
    return [
        cls.tool_use("tu_1", "echo", {"value": "hi"}),
        cls.text("done"),
    ]


# ── the registry ────────────────────────────────────────────────────────────


def test_hooks_run_in_registration_order():
    seen: list[str] = []
    registry = HookRegistry()
    registry.add(BeforeTurn, lambda e: seen.append("first") or None)
    registry.add(BeforeTurn, lambda e: seen.append("second") or None)

    registry.emit(BeforeTurn(turn=1, max_turns=5, messages=[]))

    assert seen == ["first", "second"]


def test_every_hook_sees_the_event_even_after_one_decides():
    """A hook that records spend must not be skipped because another stopped.

    Precedence between conflicting decisions belongs to the caller, not to
    whichever hook happened to be registered first.
    """
    seen: list[str] = []
    registry = HookRegistry()
    registry.add(BeforeTurn, lambda e: Stop("out of budget"))
    registry.add(BeforeTurn, lambda e: seen.append("still ran") or None)

    decisions = registry.emit(BeforeTurn(turn=1, max_turns=5, messages=[]))

    assert seen == ["still ran"]
    assert isinstance(decisions[0], Stop)


def test_a_hook_for_another_event_is_not_called():
    registry = HookRegistry()
    registry.add(AfterTurn, lambda e: pytest.fail("wrong event"))

    registry.emit(BeforeTurn(turn=1, max_turns=1, messages=[]))


def test_a_raising_hook_is_named_not_swallowed():
    """Hooks must not raise, so when one does the error says which."""

    def badly_behaved(event):
        raise ValueError("I should have returned a decision")

    registry = HookRegistry()
    registry.add(BeforeTurn, badly_behaved)

    with pytest.raises(HookError) as excinfo:
        registry.emit(BeforeTurn(turn=1, max_turns=1, messages=[]))

    assert "badly_behaved" in str(excinfo.value)
    assert "BeforeTurn" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ValueError)


# ── the gate is built from the public mechanism ─────────────────────────────


def test_the_approval_gate_is_an_ordinary_hook(tmp_path):
    """`code_only` registers a `BeforeToolCall` hook like any other.

    If the gate still lived inside the dispatcher there would be nothing here
    to find, and no reason to believe the hook mechanism was enough for it.
    """
    harness = Harness(
        adapter=FakeAdapter([]),
        system="sys",
        tools=[],
        run_dir=str(tmp_path),
        code_only=True,
    )

    assert len(harness.hooks.hooks[BeforeToolCall]) == 1


def test_no_gate_hook_is_registered_when_the_gate_is_off(tmp_path):
    harness = Harness(
        adapter=FakeAdapter([]), system="sys", tools=[], run_dir=str(tmp_path)
    )
    assert BeforeToolCall not in harness.hooks.hooks


# ── blocking a tool call ────────────────────────────────────────────────────


def test_a_hook_can_refuse_a_tool_call(tmp_path):
    calls: list[str] = []
    harness = Harness(
        adapter=FakeAdapter(tool_then_text()),
        system="sys",
        tools=[echo_spec(calls)],
        run_dir=str(tmp_path),
    )
    harness.on(BeforeToolCall, lambda e: Block("not allowed right now"))

    harness.run("go")

    assert calls == []
    block = harness.messages[2].content[0]
    assert block.content == "not allowed right now"
    assert block.is_error is False


def test_a_refusal_can_be_marked_an_error_when_it_really_is_one(tmp_path):
    harness = Harness(
        adapter=FakeAdapter(tool_then_text()),
        system="sys",
        tools=[echo_spec()],
        run_dir=str(tmp_path),
    )
    harness.on(BeforeToolCall, lambda e: Block("malformed input", is_error=True))

    harness.run("go")

    assert harness.messages[2].content[0].is_error is True


def test_a_hook_sees_what_the_model_asked_for(tmp_path):
    seen: list[tuple[str, dict]] = []
    harness = Harness(
        adapter=FakeAdapter(tool_then_text()),
        system="sys",
        tools=[echo_spec()],
        run_dir=str(tmp_path),
    )
    harness.on(BeforeToolCall, lambda e: seen.append((e.tool_name, e.tool_input)))

    harness.run("go")

    assert seen == [("echo", {"value": "hi"})]


# ── rewriting a tool result ─────────────────────────────────────────────────


def test_a_hook_can_rewrite_a_tool_result(tmp_path):
    harness = Harness(
        adapter=FakeAdapter(tool_then_text()),
        system="sys",
        tools=[echo_spec()],
        run_dir=str(tmp_path),
    )
    harness.on(AfterToolCall, lambda e: Replace(f"[redacted: {len(e.result.content)}]"))

    harness.run("go")

    assert harness.messages[2].content[0].content == "[redacted: 2]"


def test_after_tool_call_sees_the_real_result(tmp_path):
    seen: list[str] = []
    harness = Harness(
        adapter=FakeAdapter(tool_then_text()),
        system="sys",
        tools=[echo_spec()],
        run_dir=str(tmp_path),
    )
    harness.on(AfterToolCall, lambda e: seen.append(e.result.content))

    harness.run("go")

    assert seen == ["hi"]


# ── reminders ───────────────────────────────────────────────────────────────


def test_a_hook_can_append_a_reminder(tmp_path):
    harness = Harness(
        adapter=FakeAdapter([FakeAdapter.text("done")]),
        system="sys",
        tools=[],
        run_dir=str(tmp_path),
    )
    harness.on(BeforeTurn, lambda e: Reminder(f"turn {e.turn} of {e.max_turns}"))

    harness.run("go")

    assert "turn 1 of 25" in harness.messages[0].content[-1].text


def test_the_legacy_reminder_api_still_works(tmp_path):
    """`register_reminder` predates hooks and must keep working."""
    harness = Harness(
        adapter=FakeAdapter([FakeAdapter.text("done")]),
        system="sys",
        tools=[],
        run_dir=str(tmp_path),
    )
    harness.register_reminder(lambda turn, max_turns: "legacy reminder")
    harness.on(BeforeTurn, lambda e: Reminder("hook reminder"))

    harness.run("go")

    text = harness.messages[0].content[-1].text
    assert "legacy reminder" in text
    assert "hook reminder" in text


# ── stopping a run ──────────────────────────────────────────────────────────


def test_an_after_turn_hook_can_stop_the_run(tmp_path):
    """Where a spend cap belongs: the tokens are counted, so the decision is
    made on real numbers rather than a guess before the call."""
    harness = Harness(
        adapter=FakeAdapter(
            [
                FakeAdapter.tool_use("tu_1", "echo", {"value": "hi"}),
                FakeAdapter.text("never reached"),
            ]
        ),
        system="sys",
        tools=[echo_spec()],
        run_dir=str(tmp_path),
    )
    harness.on(AfterTurn, lambda e: Stop("budget exhausted") if e.turn >= 1 else None)

    result = harness.run_result("go")

    assert result.status == "success"
    assert result.turns == 1
    assert len(harness.messages) == 3  # user, assistant, tool results


def test_after_turn_sees_the_tokens_already_spent(tmp_path):
    seen: list[tuple[int, int]] = []
    harness = Harness(
        adapter=FakeAdapter([FakeAdapter.text("done")]),
        system="sys",
        tools=[],
        run_dir=str(tmp_path),
    )
    harness.on(AfterTurn, lambda e: seen.append((e.input_tokens, e.output_tokens)))

    harness.run("go")

    assert seen == [(0, 0)]  # FakeAdapter.text reports no usage


def test_a_before_turn_stop_spends_nothing(tmp_path):
    adapter = FakeAdapter([FakeAdapter.text("should not be called")])
    harness = Harness(adapter=adapter, system="sys", tools=[], run_dir=str(tmp_path))
    harness.on(BeforeTurn, lambda e: Stop("refused before starting"))

    result = harness.run_result("go")

    assert adapter.calls == []
    assert result.turns == 0


def test_a_stop_does_not_leak_into_the_next_run(tmp_path):
    stops = iter([True, False])
    harness = Harness(
        adapter=FakeAdapter([FakeAdapter.text("one"), FakeAdapter.text("two")]),
        system="sys",
        tools=[],
        run_dir=str(tmp_path),
    )
    harness.on(BeforeTurn, lambda e: Stop("first only") if next(stops, False) else None)

    harness.run_result("first")
    second = harness.run_result("second")

    assert second.text == "one"


# ── both drivers ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hooks_work_the_same_under_the_async_driver(tmp_path):
    calls: list[str] = []
    harness = AsyncHarness(
        adapter=FakeAsyncAdapter(tool_then_text(FakeAsyncAdapter)),
        system="sys",
        tools=[echo_spec(calls)],
        run_dir=str(tmp_path),
    )
    harness.on(BeforeToolCall, lambda e: Block("no"))

    await harness.run("go")

    assert calls == []
    assert harness.messages[2].content[0].content == "no"


@pytest.mark.asyncio
async def test_hooks_apply_to_streamed_runs(tmp_path):
    harness = AsyncHarness(
        adapter=FakeAsyncAdapter(tool_then_text(FakeAsyncAdapter)),
        system="sys",
        tools=[echo_spec()],
        run_dir=str(tmp_path),
    )
    harness.on(AfterToolCall, lambda e: Replace("rewritten"))

    from data_harness.llm.streaming import ToolResultEvent

    events = [e async for e in harness.run_stream("go")]
    tool_events = [e for e in events if isinstance(e, ToolResultEvent)]

    assert [e.content for e in tool_events] == ["rewritten"]


# ── a hook registry can be supplied up front ────────────────────────────────


def test_a_registry_can_be_passed_to_the_constructor(tmp_path):
    """So an application can build its policy once and reuse it."""
    registry = HookRegistry()
    registry.add(BeforeToolCall, lambda e: Block("policy says no"))

    harness = Harness(
        adapter=FakeAdapter(tool_then_text()),
        system="sys",
        tools=[echo_spec()],
        run_dir=str(tmp_path),
        hooks=registry,
    )
    harness.run("go")

    assert harness.messages[2].content[0].content == "policy says no"
