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


def test_both_stores_behave_the_same_way(store):
    """`isinstance` against a Protocol checks names only, so exercise it.

    A class with every method present and every signature wrong passes
    `isinstance(..., SessionStore)`, which makes that assertion close to
    worthless on its own.
    """
    assert isinstance(store, SessionStore)

    first = MessageEntry(id="e1", parent_id=None, message=say("q1"))
    second = MessageEntry(id="e2", parent_id="e1", message=say("a1", "assistant"))
    store.append(first)
    store.append(second)

    assert store.leaf_id == "e2"
    assert store.get("e1") == first
    assert store.get("missing") is None
    assert [e.id for e in store.entries()] == ["e1", "e2"]
    assert [e.id for e in store.path_to_root("e2")] == ["e1", "e2"]
    assert store.path_to_root(None) == []

    store.set_leaf("e1")
    assert store.leaf_id == "e1"


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


def test_a_corrupt_entry_in_the_middle_names_the_line(tmp_path):
    """Corruption anywhere but the tail means the file cannot be trusted."""
    path = tmp_path / "corrupt.jsonl"
    good = json.dumps(
        {"type": "message", "id": "a", "parent_id": None, "timestamp": "t"}
    )
    path.write_text(
        json.dumps({"type": "session", "version": 1, "id": "x"})
        + "\nnot json\n"
        + good
        + "\n"
    )
    with pytest.raises(SessionStoreError) as excinfo:
        JsonlSessionStore.open(path)
    assert excinfo.value.code == "invalid_entry"
    assert ":2" in str(excinfo.value)


def test_a_torn_final_line_costs_one_entry_not_the_file(tmp_path):
    """A crash mid-append must not make every earlier entry unreadable.

    This is an append-only recovery log. Refusing the whole file because the
    process died halfway through the last write is the opposite of the point.
    """
    path = tmp_path / "torn.jsonl"
    session = Session(JsonlSessionStore.create(path, "sess"))
    session.append_message(say("q1"))
    session.append_message(say("a1", "assistant"))

    with path.open("a") as handle:
        handle.write('{"type": "message", "id": "hal')  # power cut

    store = JsonlSessionStore.open(path)
    recovered = Session(store)

    assert store.truncated is True
    assert texts(recovered.build_context()) == ["q1", "a1"]


def test_open_or_create_does_not_destroy_an_existing_session(tmp_path):
    """`create` truncates, which on a restart path is the worst possible move."""
    path = tmp_path / "s.jsonl"
    Session(JsonlSessionStore.create(path, "sess")).append_message(say("q1"))

    reopened = Session(JsonlSessionStore.open_or_create(path, "sess"))
    assert texts(reopened.build_context()) == ["q1"]

    fresh = Session(JsonlSessionStore.open_or_create(tmp_path / "new.jsonl", "sess2"))
    assert fresh.build_context() == []


def test_an_unserializable_custom_payload_is_a_typed_error(tmp_path):
    """Every store failure is a SessionStoreError, not a bare TypeError."""
    session = Session(JsonlSessionStore.create(tmp_path / "s.jsonl", "sess"))
    with pytest.raises(SessionStoreError) as excinfo:
        session.append_custom("bad", {"handle": object()})
    assert excinfo.value.code == "unserializable_entry"


def test_a_cycle_in_a_hand_edited_file_is_detected(store):
    """Unreachable through the API, reachable by editing a file."""
    from data_harness.core.session.entries import MessageEntry as ME

    store.append(ME(id="a", parent_id=None, message=say("q1")))
    store._by_id["a"] = ME(id="a", parent_id="a", message=say("q1"))
    with pytest.raises(SessionStoreError) as excinfo:
        store.path_to_root("a")
    assert excinfo.value.code == "cycle"


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


# ── the review's surviving mutants ──────────────────────────────────────────


def test_message_roles_survive_a_round_trip(tmp_path):
    """Nothing pinned this, so an encoder writing every role as `user` passed."""
    path = tmp_path / "s.jsonl"
    session = Session(JsonlSessionStore.create(path, "sess"))
    session.append_message(say("q", "user"))
    session.append_message(say("a", "assistant"))
    session.append_message(say("q2", "user"))

    roles = [m.role for m in Session(JsonlSessionStore.open(path)).build_context()]
    assert roles == ["user", "assistant", "user"]


def test_a_failed_tool_result_round_trips_as_failed(tmp_path):
    """`is_error` defaults to False, so asserting False proved nothing."""
    from data_harness.llm.types import ToolResultBlock, ToolUseBlock

    path = tmp_path / "s.jsonl"
    session = Session(JsonlSessionStore.create(path, "sess"))
    session.append_message(
        Message(
            role="assistant",
            content=[ToolUseBlock(tool_use_id="t1", tool_name="boom", tool_input={})],
        )
    )
    session.append_message(
        Message(
            role="user",
            content=[
                ToolResultBlock(tool_use_id="t1", content="it broke", is_error=True)
            ],
        )
    )

    result = Session(JsonlSessionStore.open(path)).build_context()[1].content[0]
    assert result.is_error is True
    assert result.content == "it broke"


# ── a derived context is always something a provider will accept ────────────


def test_an_orphaned_tool_call_is_dropped_from_the_context(tmp_path):
    """A run killed mid-tool leaves a call with no result on disk.

    Resuming must not hand the provider a transcript it rejects outright,
    which is precisely the session a user most wants back.
    """
    from data_harness.llm.types import ToolUseBlock

    def explode(value: str) -> str:
        raise KeyboardInterrupt

    spec = ToolSpec(
        name="boom",
        description="Die.",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        handler=explode,
    )
    path = tmp_path / "s.jsonl"
    harness = Harness(
        adapter=FakeAdapter([FakeAdapter.tool_use("t1", "boom", {"value": "x"})]),
        system="sys",
        tools=[spec],
        run_dir=str(tmp_path),
        session=Session(JsonlSessionStore.create(path, "sess")),
    )
    with pytest.raises(KeyboardInterrupt):
        harness.run("go")

    # The orphan really is on disk.
    stored = Session(JsonlSessionStore.open(path))
    raw = [
        b
        for e in stored.store.entries()
        if isinstance(e, MessageEntry)
        for b in e.message.content
    ]
    assert any(isinstance(b, ToolUseBlock) for b in raw)

    # It is not in the context a resumed run would send.
    context = stored.build_context()
    assert not any(isinstance(b, ToolUseBlock) for m in context for b in m.content)
    assert texts(context) == ["go"]


def test_a_compaction_that_orphans_a_tool_result_drops_it(store):
    from data_harness.llm.types import ToolResultBlock, ToolUseBlock

    session = Session(store)
    session.append_message(say("q1"))
    session.append_message(
        Message(
            role="assistant",
            content=[ToolUseBlock(tool_use_id="t1", tool_name="echo", tool_input={})],
        )
    )
    orphan = session.append_message(
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="t1", content="r", is_error=False)],
        )
    )
    session.append_compaction("summary", first_kept_entry_id=orphan, tokens_before=1)
    session.append_message(say("next"))

    context = session.build_context()
    assert not any(isinstance(b, ToolResultBlock) for m in context for b in m.content)


# ── compaction is validated and composes ────────────────────────────────────


def test_compacting_from_an_entry_off_the_path_is_rejected(store):
    """A stale id would silently amnesia the agent rather than fail."""
    session = Session(store)
    root = session.append_message(say("q1"))
    session.append_message(say("a1", "assistant"))
    session.move_to(root)
    abandoned_branch_entry = None
    for entry in store.entries():
        if isinstance(entry, MessageEntry) and entry.message.role == "assistant":
            abandoned_branch_entry = entry.id

    with pytest.raises(SessionStoreError) as excinfo:
        session.append_compaction("s", abandoned_branch_entry, 1)
    assert excinfo.value.code == "invalid_cut"


def test_stacked_compactions_leave_one_summary_in_order(store):
    """An older compaction inside the kept tail must not be replayed.

    The kept tail deliberately starts *before* the first compaction, so the
    first compaction sits inside it. Replaying it emitted the older summary
    after the newer one and resurrected the entries it had dropped.
    """
    session = Session(store)
    session.append_message(say("q1"))
    keep = session.append_message(say("a1", "assistant"))
    session.append_compaction(
        "first summary", first_kept_entry_id=None, tokens_before=1
    )
    session.append_message(say("q2"))
    session.append_compaction(
        "second summary", first_kept_entry_id=keep, tokens_before=2
    )
    session.append_message(say("q3"))

    context = texts(session.build_context())
    assert sum("summary" in c for c in context) == 1, context
    assert "second summary" in context[0]
    assert "first summary" not in " ".join(context)
    assert context[1:] == ["a1", "q2", "q3"]


# ── the working copy and the log agree ──────────────────────────────────────


def test_a_second_run_starts_a_new_branch_not_a_false_continuation(tmp_path):
    """`run` resets the conversation, so the log must not imply otherwise.

    Otherwise the tree claims a continuity the model was never shown, and
    resuming replays a context that was never sent.
    """
    harness = Harness(
        adapter=FakeAdapter([FakeAdapter.text("one"), FakeAdapter.text("two")]),
        system="sys",
        tools=[],
        run_dir=str(tmp_path),
    )
    harness.run("first")
    assert harness.session.build_context() == harness.messages

    harness.run("second")
    assert harness.session.build_context() == harness.messages
    assert texts(harness.session.build_context()) == ["second", "two"]


def test_the_log_and_the_working_copy_agree_on_every_entry_point(tmp_path):
    harness = Harness(
        adapter=FakeAdapter(
            [FakeAdapter.text("a"), FakeAdapter.text("b"), FakeAdapter.text("c")]
        ),
        system="sys",
        tools=[],
        run_dir=str(tmp_path),
    )
    harness.run("q1")
    assert harness.session.build_context() == harness.messages
    harness.ask("q2")
    assert harness.session.build_context() == harness.messages
    harness.ask("q3")
    assert harness.session.build_context() == harness.messages


def test_a_reminder_appended_to_a_recorded_message_is_still_logged(tmp_path):
    """Entries are immutable, so the reminder becomes its own entry.

    Without it the JSONL store, which serialises on write, would show a
    prompt the model never actually saw.
    """
    harness = Harness(
        adapter=FakeAdapter([FakeAdapter.text("done")]),
        system="sys",
        tools=[],
        run_dir=str(tmp_path),
        max_turns=2,
    )
    harness.register_reminder(lambda turn, max_turns: "stay on task")
    harness.run("go")

    reminders = harness.session.custom_entries("reminder")
    assert len(reminders) == 1
    # The built-in max-turn nag rides along in the same suffix block.
    assert reminders[0].data["text"].startswith("stay on task")
    assert reminders[0].data["turn"] == 1


def test_both_stores_agree_after_a_message_is_mutated(tmp_path):
    """Stores snapshot on write, so neither can be rewritten after the fact."""
    message = say("original")
    memory = Session(MemorySessionStore("m"))
    on_disk = Session(JsonlSessionStore.create(tmp_path / "s.jsonl", "d"))
    memory.append_message(message)
    on_disk.append_message(message)

    message.content.append(TextBlock(text="added later"))

    # Compare every block, not just the first: the earlier version of this
    # test used a helper that read `content[0]` and so could not see an
    # appended block at all.
    def blocks(session):
        return [[b.text for b in m.content] for m in session.build_context()]

    assert blocks(memory) == [["original"]]
    assert blocks(on_disk) == [["original"]]
