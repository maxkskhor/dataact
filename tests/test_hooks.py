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


@pytest.mark.asyncio
async def test_after_turn_sees_the_tokens_already_spent(tmp_path):
    """Cumulative, and real.

    The old version asserted (0, 0) against an adapter that reports (0, 0),
    so it could not tell cumulative from per-turn from hardcoded. A mutation
    making the event always report zero survived the whole suite.
    """
    seen: list[tuple[int, int]] = []
    harness = AsyncHarness(
        adapter=FakeAsyncAdapter(
            [
                FakeAsyncAdapter.tool_use("tu_1", "echo", {"value": "hi"}),
                FakeAsyncAdapter.text("done"),
            ]
        ),
        system="sys",
        tools=[echo_spec()],
        run_dir=str(tmp_path),
    )
    harness.on(AfterTurn, lambda e: seen.append((e.input_tokens, e.output_tokens)))

    await harness.run("go")

    # FakeAsyncAdapter reports 10/5 a turn, and the totals accumulate.
    assert seen == [(10, 5), (20, 10)]


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


# ── review round 1: what the mutation pass showed was unguarded ─────────────


def test_an_after_turn_stop_keeps_the_model_answer(tmp_path):
    """Stopping after a turn must not discard what that turn produced.

    A mutation returning empty text here survived the whole suite, and the
    claim that the answer still stands was the headline of the feature.
    """
    harness = Harness(
        adapter=FakeAdapter([FakeAdapter.text("the answer"), FakeAdapter.text("more")]),
        system="sys",
        tools=[],
        run_dir=str(tmp_path),
    )
    harness.on(AfterTurn, lambda e: Stop("capped"))

    result = harness.run_result("go")

    assert result.text == "the answer"
    assert result.status == "success"


def test_a_before_turn_stop_is_a_success_not_an_error(tmp_path):
    harness = Harness(
        adapter=FakeAdapter([FakeAdapter.text("never")]),
        system="sys",
        tools=[],
        run_dir=str(tmp_path),
    )
    harness.on(BeforeTurn, lambda e: Stop("refused"))

    result = harness.run_result("go")

    assert result.status == "success"
    assert result.error is None


def test_a_mid_run_stop_keeps_the_usage_already_spent(tmp_path):
    """Stopping at turn 3 must still report what turns 1 and 2 cost."""
    harness = AsyncHarness(
        adapter=FakeAsyncAdapter(
            [
                FakeAsyncAdapter.tool_use("tu_1", "echo", {"value": "a"}),
                FakeAsyncAdapter.tool_use("tu_2", "echo", {"value": "b"}),
                FakeAsyncAdapter.text("never reached"),
            ]
        ),
        system="sys",
        tools=[echo_spec()],
        run_dir=str(tmp_path),
    )
    harness.on(BeforeTurn, lambda e: Stop("capped") if e.turn == 3 else None)

    result = run_sync(harness.run_result("go"))

    assert result.turns == 2
    assert result.usage.input_tokens == 20
    assert result.usage.output_tokens == 10


def run_sync(coro):
    from data_harness.core.loop import run_coroutine_blocking

    return run_coroutine_blocking(coro)


# ── why a run stopped is recorded ───────────────────────────────────────────


def test_a_capped_run_says_so(tmp_path):
    """A capped run was otherwise indistinguishable from an empty answer."""
    harness = Harness(
        adapter=FakeAdapter([FakeAdapter.text("never")]),
        system="sys",
        tools=[],
        run_dir=str(tmp_path),
    )
    harness.on(BeforeTurn, lambda e: Stop("out of budget"))

    result = harness.run_result("go")

    assert result.stopped_by == "out of budget"


def test_an_after_turn_stop_also_says_why(tmp_path):
    """Both stop paths record it; a mutation on this one survived without it."""
    harness = Harness(
        adapter=FakeAdapter([FakeAdapter.text("the answer"), FakeAdapter.text("more")]),
        system="sys",
        tools=[],
        run_dir=str(tmp_path),
    )
    harness.on(AfterTurn, lambda e: Stop("spend cap reached"))

    result = harness.run_result("go")

    assert result.stopped_by == "spend cap reached"
    assert result.text == "the answer"


def test_an_ordinary_run_is_not_marked_as_stopped(tmp_path):
    harness = Harness(
        adapter=FakeAdapter([FakeAdapter.text("done")]),
        system="sys",
        tools=[],
        run_dir=str(tmp_path),
    )
    assert harness.run_result("go").stopped_by is None


# ── a raising hook fails the run instead of vanishing with it ───────────────


def test_a_raising_hook_still_produces_a_result(tmp_path):
    """The tokens were billed; losing the RunResult loses the accounting."""

    def bad(event):
        if event.turn == 2:
            raise ValueError("boom")
        return None

    harness = AsyncHarness(
        adapter=FakeAsyncAdapter(
            [
                FakeAsyncAdapter.tool_use("tu_1", "echo", {"value": "hi"}),
                FakeAsyncAdapter.text("done"),
            ]
        ),
        system="sys",
        tools=[echo_spec()],
        run_dir=str(tmp_path),
    )
    harness.on(BeforeTurn, bad)

    with pytest.raises(HookError):
        run_sync(harness.run_result("go"))

    result = harness.last_result
    assert result is not None
    assert result.status == "error"
    assert "boom" in (result.error or "")
    # Turn 1 completed and was billed before the hook broke.
    assert result.usage.input_tokens == 10


# ── a supplied registry is not written to ───────────────────────────────────


def test_a_shared_registry_does_not_leak_one_harness_policy_into_another(tmp_path):
    """The gate is added by the harness, so it must not land in the caller's
    registry and govern every other harness sharing it."""
    shared = HookRegistry()

    gated = Harness(
        adapter=FakeAdapter([]),
        system="sys",
        tools=[],
        run_dir=str(tmp_path),
        code_only=True,
        hooks=shared,
    )
    ungated = Harness(
        adapter=FakeAdapter(tool_then_text()),
        system="sys",
        tools=[echo_spec()],
        run_dir=str(tmp_path),
        hooks=shared,
    )

    assert BeforeToolCall not in shared.hooks
    assert BeforeToolCall in gated.hooks.hooks
    assert BeforeToolCall not in ungated.hooks.hooks

    calls: list[str] = []
    ungated.tools[:] = [echo_spec(calls)]
    ungated.run("go")
    assert calls == ["hi"]


def test_copy_is_independent():
    original = HookRegistry()
    original.add(BeforeTurn, lambda e: None)

    duplicate = original.copy()
    duplicate.add(BeforeTurn, lambda e: None)
    duplicate.add(AfterTurn, lambda e: None)

    assert len(original.hooks[BeforeTurn]) == 1
    assert AfterTurn not in original.hooks


# ── hooks see every tool outcome, not only the happy one ────────────────────


def test_a_failing_tool_result_reaches_after_tool_call(tmp_path):
    """A redaction hook has to see failures: an exception repr can carry a
    connection string, which is exactly what needs redacting."""

    def explode(value: str) -> str:
        raise RuntimeError("postgres://user:hunter2@host/db")

    spec = ToolSpec(
        name="echo",
        description="Fail.",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        handler=explode,
    )
    harness = Harness(
        adapter=FakeAdapter(tool_then_text()),
        system="sys",
        tools=[spec],
        run_dir=str(tmp_path),
    )
    harness.on(
        AfterToolCall,
        lambda e: Replace("[redacted]", is_error=True) if e.result.is_error else None,
    )

    harness.run("go")

    block = harness.messages[2].content[0]
    assert block.content == "[redacted]"
    assert block.is_error is True


def test_a_call_to_an_unknown_tool_reaches_the_hooks(tmp_path):
    """A policy hook needs to see attempts on tools that do not exist."""
    seen: list[str] = []
    harness = Harness(
        adapter=FakeAdapter(
            [
                FakeAdapter.tool_use("tu_1", "no_such_tool", {}),
                FakeAdapter.text("done"),
            ]
        ),
        system="sys",
        tools=[],
        run_dir=str(tmp_path),
    )
    harness.on(BeforeToolCall, lambda e: seen.append(e.tool_name))

    harness.run("go")

    assert seen == ["no_such_tool"]
    assert harness.messages[2].content[0].is_error is True


# ── the agent exposes hooks, since it builds a fresh harness per run ────────


def test_an_agent_applies_its_hooks_to_every_run(tmp_path):
    from data_harness.app.agent import Agent

    agent = Agent(
        adapter=FakeAdapter([FakeAdapter.text("one"), FakeAdapter.text("two")]),
        system="sys",
        run_dir=str(tmp_path),
    )
    seen: list[int] = []
    agent.on(BeforeTurn, lambda e: seen.append(e.turn))

    agent.run("first")
    agent.run("second")

    # A hook put on a harness would be gone by the second run.
    assert seen == [1, 1]
