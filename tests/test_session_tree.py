"""Phase 3: the session tree.

The claim is that a session is an append-only tree and the conversation is
*derived* from it. Everything worth having follows from that: resume, forking,
reversible compaction, and a log that is linear rather than quadratic to
write. Each of those is tested here, because each is a claim a README could
make falsely.
"""

from __future__ import annotations

import json

import pytest

from data_harness.core.session import (
    JsonlSessionStore,
    LeafEntry,
    MemorySessionStore,
    MessageEntry,
    Session,
    SessionStore,
    SessionStoreError,
    TurnEntry,
)
from data_harness.data.harness import Harness
from data_harness.llm.testing import FakeAdapter
from data_harness.llm.types import Message, TextBlock, ToolSpec


def say(text: str, role: str = "user") -> Message:
    return Message(role=role, content=[TextBlock(text=text)])


def texts(messages: list[Message]) -> list[str]:
    return [m.content[0].text for m in messages]


def echo_spec() -> ToolSpec:
    return ToolSpec(
        name="echo",
        description="Echo the input.",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        handler=lambda value: value,
    )


@pytest.fixture(params=["memory", "jsonl"])
def store(request, tmp_path) -> SessionStore:
    """Both stores, everywhere, so neither drifts from the protocol."""
    if request.param == "memory":
        return MemorySessionStore("sess")
    return JsonlSessionStore.create(tmp_path / "session.jsonl", "sess")


# ── the tree ────────────────────────────────────────────────────────────────


def test_context_is_derived_from_the_path(store):
    session = Session(store)
    session.append_message(say("q1"))
    session.append_message(say("a1", "assistant"))
    session.append_message(say("q2"))

    assert texts(session.build_context()) == ["q1", "a1", "q2"]


def test_branching_keeps_both_branches(store):
    """The point of a tree. Retrying a turn must not destroy the first try."""
    session = Session(store)
    session.append_message(say("q1"))
    fork_point = session.append_message(say("a1", "assistant"))
    original = session.append_message(say("original follow-up"))

    session.move_to(fork_point)
    session.append_message(say("different follow-up"))

    assert texts(session.build_context()) == ["q1", "a1", "different follow-up"]
    assert texts(session.build_context(original)) == ["q1", "a1", "original follow-up"]


def test_moving_the_leaf_is_itself_recorded(store):
    session = Session(store)
    first = session.append_message(say("q1"))
    session.append_message(say("q2"))
    session.move_to(first)

    moves = [e for e in store.entries() if isinstance(e, LeafEntry)]
    assert [m.target_id for m in moves] == [first]


def test_moving_to_none_starts_a_fresh_root(store):
    session = Session(store)
    session.append_message(say("q1"))
    session.move_to(None)
    session.append_message(say("a new beginning"))

    assert texts(session.build_context()) == ["a new beginning"]


def test_an_unknown_entry_cannot_be_made_the_leaf(store):
    session = Session(store)
    session.append_message(say("q1"))
    with pytest.raises(SessionStoreError) as excinfo:
        session.move_to("nope")
    assert excinfo.value.code == "not_found"


def test_entries_are_never_removed(store):
    """History only grows. That is what makes a run auditable after the fact."""
    session = Session(store)
    session.append_message(say("q1"))
    fork = session.append_message(say("a1", "assistant"))
    session.append_message(say("abandoned"))
    session.move_to(fork)
    session.append_message(say("kept"))

    recorded = [
        e.message.content[0].text
        for e in store.entries()
        if isinstance(e, MessageEntry)
    ]
    assert recorded == ["q1", "a1", "abandoned", "kept"]


# ── compaction ──────────────────────────────────────────────────────────────


def test_compaction_shrinks_the_context_but_not_the_tree(store):
    session = Session(store)
    session.append_message(say("ancient q"))
    session.append_message(say("ancient a", "assistant"))
    keep_from = session.append_message(say("recent q"))
    session.append_compaction(
        summary="They discussed something ancient.",
        first_kept_entry_id=keep_from,
        tokens_before=1234,
    )
    session.append_message(say("newest q"))

    context = texts(session.build_context())
    assert context[0].startswith("Summary of the earlier conversation:")
    assert "recent q" in context
    assert "newest q" in context
    assert "ancient q" not in context

    # Nothing was deleted.
    stored = [
        e.message.content[0].text
        for e in store.entries()
        if isinstance(e, MessageEntry)
    ]
    assert "ancient q" in stored


def test_compaction_is_reversible_by_moving_the_leaf(store):
    """A compaction is an entry, not an edit, so stepping back undoes it."""
    session = Session(store)
    session.append_message(say("ancient q"))
    before_compaction = session.append_message(say("recent q"))
    session.append_compaction("a summary", first_kept_entry_id=None, tokens_before=1)
    session.append_message(say("after"))

    assert "ancient q" not in texts(session.build_context())

    session.move_to(before_compaction)
    assert texts(session.build_context()) == ["ancient q", "recent q"]


def test_compaction_without_a_kept_tail_drops_everything_before_it(store):
    session = Session(store)
    session.append_message(say("q1"))
    session.append_message(say("q2"))
    session.append_compaction("summary", first_kept_entry_id=None, tokens_before=9)

    assert len(session.build_context()) == 1


def test_the_newest_compaction_wins(store):
    session = Session(store)
    session.append_message(say("q1"))
    session.append_compaction("first summary", None, 1)
    session.append_message(say("q2"))
    session.append_compaction("second summary", None, 2)
    session.append_message(say("q3"))

    context = texts(session.build_context())
    assert any("second summary" in c for c in context)
    assert not any("first summary" in c for c in context)


# ── custom entries: the extension point that keeps core domain-free ────────


def test_custom_entries_are_invisible_to_the_model_by_default(store):
    session = Session(store)
    session.append_message(say("q1"))
    session.append_custom("cache_put", {"handle": "sales_df", "snapshot": "3x2"})

    assert texts(session.build_context()) == ["q1"]
    assert [e.data["handle"] for e in session.custom_entries("cache_put")] == [
        "sales_df"
    ]


def test_a_projector_opts_a_custom_entry_into_the_context(store):
    session = Session(
        store,
        projectors={
            "note": lambda entry: [say(f"note: {entry.data['text']}")],
        },
    )
    session.append_message(say("q1"))
    session.append_custom("note", {"text": "remember this"})

    assert texts(session.build_context()) == ["q1", "note: remember this"]


# ── labels ──────────────────────────────────────────────────────────────────


def test_labels_name_a_branch_point(store):
    session = Session(store)
    first = session.append_message(say("q1"))
    session.label(first, "before the detour")

    assert session.labels() == {first: "before the detour"}

    session.label(first, None)
    assert session.labels() == {}


def test_labelling_an_unknown_entry_is_an_error(store):
    session = Session(store)
    with pytest.raises(SessionStoreError):
        session.label("nope", "x")


# ── store integrity ─────────────────────────────────────────────────────────


def test_a_duplicate_entry_id_is_rejected(store):
    entry = MessageEntry(id="fixed", parent_id=None, message=say("q"))
    store.append(entry)
    with pytest.raises(SessionStoreError) as excinfo:
        store.append(MessageEntry(id="fixed", parent_id=None, message=say("q")))
    assert excinfo.value.code == "duplicate_entry"


def test_an_entry_naming_an_unknown_parent_is_rejected(store):
    with pytest.raises(SessionStoreError) as excinfo:
        store.append(MessageEntry(id="a", parent_id="ghost", message=say("q")))
    assert excinfo.value.code == "missing_parent"


def test_both_stores_satisfy_the_protocol(store):
    assert isinstance(store, SessionStore)


# ── persistence ─────────────────────────────────────────────────────────────


def test_a_jsonl_session_round_trips(tmp_path):
    path = tmp_path / "s.jsonl"
    session = Session(JsonlSessionStore.create(path, "sess-1"))
    session.append_message(say("q1"))
    session.append_message(say("a1", "assistant"))
    session.append_turn(
        turn=1, input_tokens=10, output_tokens=5, stop_reason="end_turn"
    )
    session.append_custom("chart", {"path": "/tmp/x.png"})

    reopened = Session(JsonlSessionStore.open(path))

    assert texts(reopened.build_context()) == ["q1", "a1"]
    assert reopened.stats().input_tokens == 10
    assert reopened.custom_entries("chart")[0].data["path"] == "/tmp/x.png"
    assert reopened.leaf_id == session.leaf_id


def test_tool_messages_survive_a_round_trip(tmp_path):
    """Tool blocks are the part a naive text-only encoder would silently lose."""
    from data_harness.llm.types import ToolResultBlock, ToolUseBlock

    path = tmp_path / "s.jsonl"
    session = Session(JsonlSessionStore.create(path, "sess"))
    session.append_message(
        Message(
            role="assistant",
            content=[
                ToolUseBlock(tool_use_id="t1", tool_name="echo", tool_input={"v": 1})
            ],
        )
    )
    session.append_message(
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="t1", content="ok", is_error=False)],
        )
    )

    reopened = Session(JsonlSessionStore.open(path)).build_context()

    use = reopened[0].content[0]
    result = reopened[1].content[0]
    assert isinstance(use, ToolUseBlock)
    assert use.tool_name == "echo" and use.tool_input == {"v": 1}
    assert isinstance(result, ToolResultBlock)
    assert result.tool_use_id == "t1" and result.is_error is False


def test_the_file_grows_linearly_not_quadratically(tmp_path):
    """The log this replaces re-serialised the whole history every turn.

    At 40 messages that is 800 message-copies on disk; here it is 40 lines.
    """
    path = tmp_path / "s.jsonl"
    session = Session(JsonlSessionStore.create(path, "sess"))
    for i in range(40):
        session.append_message(say(f"message {i}"))

    lines = [line for line in path.read_text().splitlines() if line.strip()]
    assert len(lines) == 41  # header + one line per entry

    # Each message appears exactly once in the file.
    assert path.read_text().count('"message 0"') == 1


def test_a_missing_session_file_is_a_typed_error(tmp_path):
    with pytest.raises(SessionStoreError) as excinfo:
        JsonlSessionStore.open(tmp_path / "nope.jsonl")
    assert excinfo.value.code == "not_found"


def test_a_file_without_a_header_is_rejected(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"type": "message", "id": "a"}\n')
    with pytest.raises(SessionStoreError) as excinfo:
        JsonlSessionStore.open(path)
    assert excinfo.value.code == "invalid_session"


def test_a_future_format_version_is_rejected_not_guessed(tmp_path):
    path = tmp_path / "future.jsonl"
    path.write_text(json.dumps({"type": "session", "version": 99, "id": "x"}) + "\n")
    with pytest.raises(SessionStoreError) as excinfo:
        JsonlSessionStore.open(path)
    assert excinfo.value.code == "invalid_session"


def test_a_corrupt_entry_line_names_the_line(tmp_path):
    path = tmp_path / "corrupt.jsonl"
    path.write_text(
        json.dumps({"type": "session", "version": 1, "id": "x"}) + "\nnot json\n"
    )
    with pytest.raises(SessionStoreError) as excinfo:
        JsonlSessionStore.open(path)
    assert excinfo.value.code == "invalid_entry"
    assert ":2" in str(excinfo.value)


def test_an_unknown_entry_type_is_rejected(tmp_path):
    path = tmp_path / "unknown.jsonl"
    path.write_text(
        json.dumps({"type": "session", "version": 1, "id": "x"})
        + "\n"
        + json.dumps({"type": "from_the_future", "id": "a", "parent_id": None})
        + "\n"
    )
    with pytest.raises(SessionStoreError) as excinfo:
        JsonlSessionStore.open(path)
    assert excinfo.value.code == "invalid_entry"


# ── the harness records into the session ────────────────────────────────────


def test_a_run_records_every_message_it_sent(tmp_path):
    harness = Harness(
        adapter=FakeAdapter(
            [
                FakeAdapter.tool_use("tu_1", "echo", {"value": "hi"}),
                FakeAdapter.text("done"),
            ]
        ),
        system="sys",
        tools=[echo_spec()],
        run_dir=str(tmp_path),
    )
    harness.run("go")

    # The working copy and the durable log agree. They are two representations
    # of one thing, and this is what stops them becoming two sources of truth.
    assert harness.session.build_context() == harness.messages
    assert [m.role for m in harness.session.build_context()] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_a_run_records_what_each_turn_cost(tmp_path):
    harness = Harness(
        adapter=FakeAdapter(
            [
                FakeAdapter.tool_use("tu_1", "echo", {"value": "hi"}),
                FakeAdapter.text("done"),
            ]
        ),
        system="sys",
        tools=[echo_spec()],
        run_dir=str(tmp_path),
    )
    harness.run("go")

    turns = [e for e in harness.session.store.entries() if isinstance(e, TurnEntry)]
    assert [t.turn for t in turns] == [1, 2]
    assert turns[0].stop_reason == "tool_use"
    assert turns[1].stop_reason == "end_turn"
    assert turns[0].visible_tools == ["echo"]


def test_a_session_spans_several_runs(tmp_path):
    harness = Harness(
        adapter=FakeAdapter([FakeAdapter.text("one"), FakeAdapter.text("two")]),
        system="sys",
        tools=[],
        run_dir=str(tmp_path),
    )
    harness.run("first")
    harness.run("second")

    recorded = [
        e.message.content[0].text
        for e in harness.session.store.entries()
        if isinstance(e, MessageEntry)
    ]
    assert recorded == ["first", "one", "second", "two"]


def test_a_conversation_resumes_across_a_restart(tmp_path):
    """The headline capability, and what the old write-only log could not do."""
    path = tmp_path / "session.jsonl"

    first = Harness(
        adapter=FakeAdapter([FakeAdapter.text("4")]),
        system="sys",
        tools=[],
        run_dir=str(tmp_path),
        session=Session(JsonlSessionStore.create(path, "sess-1")),
    )
    first.run("what is 2+2")

    # A different process, holding nothing but the file.
    second = Harness(
        adapter=FakeAdapter([FakeAdapter.text("6")]),
        system="sys",
        tools=[],
        run_dir=str(tmp_path),
        session=Session(JsonlSessionStore.open(path)),
    )
    assert texts(second.messages) == ["what is 2+2", "4"]

    second.ask("and 3+3")

    assert texts(second.messages) == ["what is 2+2", "4", "and 3+3", "6"]
    assert second.session.stats().turns == 2


def test_a_resumed_session_can_fork_instead_of_continuing(tmp_path):
    """Re-ask an earlier question differently, keeping the first attempt."""
    path = tmp_path / "session.jsonl"
    harness = Harness(
        adapter=FakeAdapter([FakeAdapter.text("first answer")]),
        system="sys",
        tools=[],
        run_dir=str(tmp_path),
        session=Session(JsonlSessionStore.create(path, "sess-1")),
    )
    harness.run("the question")
    original_leaf = harness.session.leaf_id

    session = Session(JsonlSessionStore.open(path))
    first_message = next(
        e for e in session.store.entries() if isinstance(e, MessageEntry)
    )
    session.move_to(first_message.id)

    retried = Harness(
        adapter=FakeAdapter([FakeAdapter.text("second answer")]),
        system="sys",
        tools=[],
        run_dir=str(tmp_path),
        session=session,
    )
    retried.ask("")

    assert texts(retried.session.build_context())[-1] == "second answer"
    assert texts(retried.session.build_context(original_leaf))[-1] == "first answer"


@pytest.mark.asyncio
async def test_a_streamed_run_records_the_same_way(tmp_path):
    from data_harness.data.harness import AsyncHarness
    from data_harness.llm.testing import FakeAsyncAdapter

    harness = AsyncHarness(
        adapter=FakeAsyncAdapter([FakeAsyncAdapter.text("streamed")]),
        system="sys",
        tools=[],
        run_dir=str(tmp_path),
    )
    async for _ in harness.run_stream("go"):
        pass

    assert texts(harness.session.build_context()) == ["go", "streamed"]
    assert harness.session.stats().turns == 1
