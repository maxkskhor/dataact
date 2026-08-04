"""Phase 5: compaction and the error taxonomy.

Before this the only context management was `max_turns`, which is a wall
rather than a strategy: a run needing thirty turns failed at twenty-five
having paid for all of them.

The tests that matter most are the two safety properties. A cut that separates
an assistant tool call from its result produces a transcript the provider
rejects outright, and it is the first bug everyone writes here. And nothing
may actually be deleted, because a compaction that cannot be undone is an edit
to history rather than a view of it.
"""

from __future__ import annotations

import pytest

from data_harness.core.compaction import (
    CompactionSettings,
    estimate_tokens,
    find_cut_point,
    make_compactor,
    maybe_compact,
    messages_before,
    should_compact,
    starts_a_turn,
)
from data_harness.core.exceptions import (
    ConfigurationError,
    DataHarnessError,
    ExecutionError,
    MaxTurnsExceeded,
    ProviderError,
    SubagentRecursionError,
    ToolNotFoundError,
)
from data_harness.core.hooks import HookError
from data_harness.core.session import MessageEntry, Session, SessionStoreError
from data_harness.data.harness import Harness
from data_harness.llm.testing import FakeAdapter
from data_harness.llm.types import (
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)


def say(text: str, role: str = "user") -> Message:
    return Message(role=role, content=[TextBlock(text=text)])


def texts(messages: list[Message]) -> list[str]:
    return [m.content[0].text for m in messages]


def small_settings(**overrides) -> CompactionSettings:
    base = dict(context_window=1000, reserve_tokens=200, keep_recent_tokens=100)
    base.update(overrides)
    return CompactionSettings(**base)


def bulk(label: str, tokens: int) -> Message:
    """A message of roughly ``tokens`` tokens."""
    return say(f"{label}:" + "x" * (tokens * 4))


# ── estimation and the trigger ──────────────────────────────────────────────


def test_token_estimate_counts_every_block_kind():
    messages = [
        say("a" * 40),
        Message(
            role="assistant",
            content=[ToolUseBlock(tool_use_id="t1", tool_name="echo", tool_input={})],
        ),
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="t1", content="b" * 40)],
        ),
    ]
    # Not exact by design; what matters is that nothing is counted as zero.
    assert estimate_tokens(messages) >= 20


def test_the_trigger_leaves_room_for_a_reply():
    settings = small_settings()  # window 1000, reserve 200
    assert should_compact(801, settings)
    assert not should_compact(800, settings)


def test_settings_that_could_never_trigger_are_rejected():
    """Keeping more than the trigger point would compact forever."""
    with pytest.raises(ValueError, match="keep_recent_tokens"):
        CompactionSettings(
            context_window=1000, reserve_tokens=200, keep_recent_tokens=900
        )


# ── the cut lands on a turn boundary ────────────────────────────────────────


def test_a_user_message_starts_a_turn():
    session = Session()
    session.append_message(say("q1"))
    entries = session.context_entries()
    assert starts_a_turn(entries, 0)


def test_a_tool_result_message_does_not_start_a_turn():
    """It is a user message, but cutting before it orphans the call above."""
    session = Session()
    session.append_message(
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="t1", content="r")],
        )
    )
    assert not starts_a_turn(session.context_entries(), 0)


def test_an_assistant_message_does_not_start_a_turn():
    """Cutting here would open the conversation with an assistant message."""
    session = Session()
    session.append_message(say("a", "assistant"))
    assert not starts_a_turn(session.context_entries(), 0)


def test_the_cut_never_splits_a_tool_call_from_its_result():
    """The bug everyone writes. A split pair is rejected by the provider."""
    session = Session()
    session.append_message(bulk("q1", 200))
    session.append_message(
        Message(
            role="assistant",
            content=[ToolUseBlock(tool_use_id="t1", tool_name="echo", tool_input={})],
        )
    )
    session.append_message(
        Message(role="user", content=[ToolResultBlock(tool_use_id="t1", content="r")])
    )
    session.append_message(bulk("q2", 200))

    entries = session.context_entries()
    cut_id = find_cut_point(entries, small_settings())

    kept = [e for e in entries if e.id == cut_id]
    assert kept, "expected a cut point"
    entry = kept[0]
    assert isinstance(entry, MessageEntry)
    assert entry.message.role == "user"
    assert not any(isinstance(b, ToolResultBlock) for b in entry.message.content)


def test_no_cut_is_made_when_there_is_nowhere_safe():
    """One indivisible turn is better left alone than cut badly."""
    session = Session()
    session.append_message(bulk("only question", 500))
    session.append_message(say("thinking", "assistant"))

    assert find_cut_point(session.context_entries(), small_settings()) is None


def test_nothing_is_cut_when_the_boundary_is_the_first_entry():
    session = Session()
    session.append_message(bulk("q1", 500))

    assert find_cut_point(session.context_entries(), small_settings()) is None


def test_messages_before_the_cut_are_what_gets_summarised():
    session = Session()
    session.append_message(say("old one"))
    session.append_message(say("old two", "assistant"))
    cut = session.append_message(say("recent"))

    replaced = messages_before(session.context_entries(), cut)

    assert texts(replaced) == ["old one", "old two"]


# ── compacting a session ────────────────────────────────────────────────────


def test_compaction_replaces_old_turns_with_a_summary():
    session = Session()
    for i in range(6):
        session.append_message(bulk(f"q{i}", 150))
        session.append_message(bulk(f"a{i}", 150))

    entry_id = maybe_compact(session, lambda msgs: "they talked", small_settings())

    assert entry_id is not None
    context = texts(session.build_context())
    assert context[0].startswith("Summary of the earlier conversation:")
    assert "they talked" in context[0]
    assert len(context) < 12


def test_the_summariser_sees_exactly_what_it_replaces():
    seen: list[list[str]] = []
    session = Session()
    for i in range(6):
        session.append_message(bulk(f"q{i}", 150))
        session.append_message(bulk(f"a{i}", 150))

    def summarize(messages):
        seen.append([m.content[0].text.split(":")[0] for m in messages])
        return "summary"

    maybe_compact(session, summarize, small_settings())

    kept = {t.split(":")[0] for t in texts(session.build_context())[1:]}
    assert seen, "summariser was not called"
    assert not (set(seen[0]) & kept), "summarised and kept overlap"


def test_a_small_conversation_is_left_alone():
    session = Session()
    session.append_message(say("q1"))

    assert maybe_compact(session, lambda m: "never", small_settings()) is None
    assert texts(session.build_context()) == ["q1"]


def test_nothing_is_deleted_by_compacting():
    """A compaction that cannot be undone is an edit, not a view."""
    session = Session()
    for i in range(6):
        session.append_message(bulk(f"q{i}", 150))
        session.append_message(bulk(f"a{i}", 150))
    before = len(session.store.entries())

    maybe_compact(session, lambda m: "summary", small_settings())

    stored = [e for e in session.store.entries() if isinstance(e, MessageEntry)]
    assert len(stored) == before
    assert any(m.message.content[0].text.startswith("q0") for m in stored)


def test_a_compaction_can_be_undone_by_moving_the_leaf():
    session = Session()
    for i in range(6):
        session.append_message(bulk(f"q{i}", 150))
    last_before = session.leaf_id

    maybe_compact(session, lambda m: "summary", small_settings())
    assert "Summary" in texts(session.build_context())[0]

    session.move_to(last_before)
    restored = texts(session.build_context())
    assert not restored[0].startswith("Summary")
    assert len(restored) == 6


# ── the harness consults a compactor ────────────────────────────────────────


def test_the_harness_compacts_between_turns(tmp_path):
    """A long transcript, not a long tool output.

    A big tool result never reaches the transcript: the cache stores it under
    a handle and the model sees a snapshot. That is the design working, and it
    is why compaction costs this library less than it costs a coding agent.
    So the pressure here comes from the conversation itself.
    """
    calls: list[int] = []

    def summarize(messages):
        calls.append(len(messages))
        return "earlier work"

    tiny = CompactionSettings(
        context_window=400, reserve_tokens=100, keep_recent_tokens=40
    )
    harness = Harness(
        adapter=FakeAdapter(
            [
                FakeAdapter.text("a" * 900),
                FakeAdapter.text("b" * 900),
                FakeAdapter.text("done"),
            ]
        ),
        system="sys",
        tools=[],
        run_dir=str(tmp_path),
        compactor=make_compactor(summarize, tiny),
    )

    harness.run("q" * 900)
    harness.ask("second question")
    harness.ask("third question")

    assert calls, "compactor never fired"
    assert any("earlier work" in str(m.content) for m in harness.messages)


def test_the_working_copy_is_rebuilt_from_the_session_after_compacting(tmp_path):
    """Not edited in place: the conversation stays derived from the log."""
    tiny = CompactionSettings(
        context_window=400, reserve_tokens=100, keep_recent_tokens=40
    )
    harness = Harness(
        adapter=FakeAdapter(
            [
                FakeAdapter.text("a" * 900),
                FakeAdapter.text("b" * 900),
                FakeAdapter.text("done"),
            ]
        ),
        system="sys",
        tools=[],
        run_dir=str(tmp_path),
        compactor=make_compactor(lambda m: "summary", tiny),
    )

    harness.run("q" * 900)
    harness.ask("second question")
    harness.ask("third question")

    assert harness.messages == harness.session.build_context()


def test_a_harness_without_a_compactor_never_compacts(tmp_path):
    harness = Harness(
        adapter=FakeAdapter([FakeAdapter.text("done")]),
        system="sys",
        tools=[],
        run_dir=str(tmp_path),
    )
    harness.run("go")

    from data_harness.core.session import CompactionEntry

    assert not [
        e for e in harness.session.store.entries() if isinstance(e, CompactionEntry)
    ]


# ── the error taxonomy ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (MaxTurnsExceeded(3), "max_turns_exceeded"),
        (ToolNotFoundError("x"), "tool_not_found"),
        (SubagentRecursionError("x"), "subagent_recursion"),
        (ProviderError("rate limited"), "provider_error"),
        (ExecutionError("timed out"), "execution_error"),
        (ConfigurationError("bad wiring"), "configuration_error"),
        (SessionStoreError("not_found", "gone"), "not_found"),
    ],
)
def test_every_error_carries_a_stable_code(error, code):
    """Codes are the API. Messages are not, and may be reworded."""
    assert error.code == code


def test_every_library_error_shares_one_base():
    """`except DataHarnessError` catches the library's failures and no others."""
    for error in [
        MaxTurnsExceeded(1),
        ToolNotFoundError("x"),
        SubagentRecursionError("x"),
        ProviderError("x"),
        ExecutionError("x"),
        ConfigurationError("x"),
        SessionStoreError("not_found", "x"),
    ]:
        assert isinstance(error, DataHarnessError)


def test_a_hook_error_is_part_of_the_taxonomy():
    def bad(event):
        raise ValueError("boom")

    error = HookError("event", bad, ValueError("boom"))
    assert isinstance(error, DataHarnessError)
    assert error.code == "hook_error"


def test_the_legacy_base_classes_still_hold():
    """Callers caught these as RuntimeError/KeyError before the taxonomy."""
    assert isinstance(MaxTurnsExceeded(1), RuntimeError)
    assert isinstance(SubagentRecursionError("x"), RuntimeError)
    assert isinstance(ToolNotFoundError("x"), KeyError)


def test_a_tool_handler_bug_is_not_caught_by_the_library_base():
    """The base must not be so broad it swallows the caller's own mistakes."""
    assert not isinstance(ValueError("caller bug"), DataHarnessError)
