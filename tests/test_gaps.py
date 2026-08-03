"""Tests exposing gaps identified in PLAN_v5 review.

Written before the fixes; all should fail initially.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from data_harness.data.cache import SessionCache
from data_harness.data.harness import Harness
from data_harness.llm.providers.base import (
    NormalizedResponse,
    ProviderAdapter,
    StopReason,
)
from data_harness.llm.testing import FakeAdapter
from data_harness.llm.types import TextBlock, ToolAnnotations, ToolSpec, ToolUseBlock


def make_text_response(
    text: str, *, input_tokens: int = 5, output_tokens: int = 2
) -> NormalizedResponse:
    return NormalizedResponse(
        stop_reason=StopReason.END_TURN,
        content=[TextBlock(text=text)],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=0,
        cache_write_tokens=0,
    )


def make_harness(
    responses: list[NormalizedResponse],
    *,
    tmp_path: Path,
    tools: list[ToolSpec] | None = None,
    cache: SessionCache | None = None,
) -> Harness:
    return Harness(
        adapter=FakeAdapter(responses),
        system="s",
        tools=tools or [],
        max_turns=5,
        run_dir=str(tmp_path),
        cache=cache,
    )


# ---------------------------------------------------------------------------
# Gap #2: stop_reason semantics
# ---------------------------------------------------------------------------


class TestStopReasonSemantics:
    def test_stop_reason_end_turn_after_multiturn_success(self, tmp_path):
        """After tool-use then end-turn, stop_reason must be END_TURN, not TOOL_USE."""
        tool_resp = NormalizedResponse(
            stop_reason=StopReason.TOOL_USE,
            content=[
                ToolUseBlock(
                    tool_use_id="t1", tool_name="echo", tool_input={"text": "x"}
                )
            ],
            input_tokens=5,
            output_tokens=2,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )
        final_resp = make_text_response("done")
        echo_spec = ToolSpec(
            name="echo",
            description="echo",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            handler=lambda text: text,
        )
        harness = make_harness(
            [tool_resp, final_resp], tmp_path=tmp_path, tools=[echo_spec]
        )
        result = harness.run_result("go")
        assert result.status == "success"
        assert result.stop_reason == StopReason.END_TURN

    def test_stop_reason_none_for_max_turns_exceeded(self, tmp_path):
        """stop_reason must be None when status=max_turns_exceeded.

        The last response had TOOL_USE (that's why turns ran out), but that is
        the adapter's stop reason, not a meaningful terminal stop reason for the
        run as a whole.
        """
        tool_responses = [
            NormalizedResponse(
                stop_reason=StopReason.TOOL_USE,
                content=[
                    ToolUseBlock(
                        tool_use_id=f"t{i}", tool_name="echo", tool_input={"text": "x"}
                    )
                ],
                input_tokens=5,
                output_tokens=2,
                cache_read_tokens=0,
                cache_write_tokens=0,
            )
            for i in range(10)
        ]
        echo_spec = ToolSpec(
            name="echo",
            description="echo",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            handler=lambda text: text,
        )
        harness = Harness(
            adapter=FakeAdapter(tool_responses),
            system="s",
            tools=[echo_spec],
            max_turns=3,
            run_dir=str(tmp_path),
        )
        result = harness.run_result("go")
        assert result.status == "max_turns_exceeded"
        assert result.stop_reason is None


# ---------------------------------------------------------------------------
# Gap #3: status="error" for adapter exceptions
# ---------------------------------------------------------------------------


class TestStatusError:
    def test_adapter_exception_sets_error_status(self, tmp_path):
        """An unhandled exception from adapter.chat() should produce status='error'."""

        class BoomAdapter(ProviderAdapter):
            def chat(self, system, messages, tools):
                raise RuntimeError("network failure")

            def format_cache_control(self, obj):
                return obj

        harness = Harness(
            adapter=BoomAdapter(),
            system="s",
            tools=[],
            max_turns=5,
            run_dir=str(tmp_path),
        )
        result = harness.run_result("go")
        assert result.status == "error"
        assert result.error is not None
        assert "network failure" in result.error

    def test_run_reraises_on_adapter_exception(self, tmp_path):
        """Harness.run() should still raise on adapter error."""

        class BoomAdapter(ProviderAdapter):
            def chat(self, system, messages, tools):
                raise RuntimeError("boom")

            def format_cache_control(self, obj):
                return obj

        harness = Harness(
            adapter=BoomAdapter(),
            system="s",
            tools=[],
            max_turns=5,
            run_dir=str(tmp_path),
        )
        with pytest.raises(RuntimeError, match="boom"):
            harness.run("go")


# ---------------------------------------------------------------------------
# Gap #5: CacheStorageInfo.storage_type "hot" overloading
# ---------------------------------------------------------------------------


class TestCacheStorageTypeSemantics:
    def test_in_memory_storage_type_is_not_hot(self, tmp_path):
        """storage_type for in-memory handles must not be 'hot'.

        'hot' is a cache-eviction tier concept; CacheStorageInfo.storage_type
        should describe the storage format, not the tier.
        """
        cache = SessionCache()
        cache.put("x", 42)
        harness = make_harness(
            [make_text_response("ok")], tmp_path=tmp_path, cache=cache
        )
        result = harness.run_result("go")
        info = result.cache_storage["x"]
        assert info.location == "memory"
        assert info.storage_type != "hot"

    def test_in_memory_storage_type_is_memory(self, tmp_path):
        """storage_type for in-memory handles should be 'memory'."""
        cache = SessionCache()
        cache.put("x", 42)
        harness = make_harness(
            [make_text_response("ok")], tmp_path=tmp_path, cache=cache
        )
        result = harness.run_result("go")
        assert result.cache_storage["x"].storage_type == "memory"


# ---------------------------------------------------------------------------
# Gap #8: cache_snapshots reflect post-mutation state
# ---------------------------------------------------------------------------


class TestCacheSnapshotsPostMutation:
    def test_cache_snapshots_reflect_post_tool_mutation(self, tmp_path):
        """cache_snapshots in RunResult should capture the state AFTER any
        tool calls, not the pre-run state.
        """
        from data_harness.data.cache import SessionCache

        cache = SessionCache()
        initial_data = [1, 2, 3]
        cache.put("data", initial_data)

        def mutating_tool() -> str:
            cache.put("computed", [10, 20, 30], overwrite=True)
            return "saved computed"

        mutate_spec = ToolSpec(
            name="mutate",
            description="mutate",
            input_schema={"type": "object"},
            handler=mutating_tool,
        )
        tool_resp = NormalizedResponse(
            stop_reason=StopReason.TOOL_USE,
            content=[ToolUseBlock(tool_use_id="t1", tool_name="mutate", tool_input={})],
            input_tokens=5,
            output_tokens=2,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )
        final_resp = make_text_response("done")
        harness = make_harness(
            [tool_resp, final_resp], tmp_path=tmp_path, tools=[mutate_spec], cache=cache
        )
        result = harness.run_result("go")
        # computed was created during the run — it must appear in snapshots
        assert "computed" in result.cache_snapshots
        # The snapshot must be a compact string, not the raw list
        assert isinstance(result.cache_snapshots["computed"], str)
        assert result.cache_snapshots["computed"] != str([10, 20, 30])


# ---------------------------------------------------------------------------
# Gap #9: cache_storage doesn't leak raw payloads
# ---------------------------------------------------------------------------


class TestCacheStorageNoLeaks:
    def test_cache_storage_does_not_contain_raw_values(self, tmp_path):
        """cache_storage in RunResult contains only location/type metadata,
        not the raw cached values.
        """
        cache = SessionCache()
        sensitive = {"secret_key": "abc123", "data": list(range(100))}
        cache.put("secret", sensitive)
        harness = make_harness(
            [make_text_response("ok")], tmp_path=tmp_path, cache=cache
        )
        result = harness.run_result("go")
        info = result.cache_storage["secret"]
        # CacheStorageInfo must not have a "value" field
        assert not hasattr(info, "value")
        # The info object is purely metadata
        assert hasattr(info, "location")
        assert hasattr(info, "storage_type")
        # The raw dict values must not appear
        import dataclasses

        fields = {f.name for f in dataclasses.fields(info)}
        assert fields == {"location", "storage_type"}

    def test_cache_snapshots_are_compact_not_full_value(self, tmp_path):
        """cache_snapshots must contain compact strings, not full serialised values."""
        cache = SessionCache()
        large_list = list(range(1000))
        cache.put("large", large_list)
        harness = make_harness(
            [make_text_response("ok")], tmp_path=tmp_path, cache=cache
        )
        result = harness.run_result("go")
        snapshot = result.cache_snapshots["large"]
        # Snapshot is a string
        assert isinstance(snapshot, str)
        # It must not contain the full list (1000 items)
        assert "999" not in snapshot  # last element not in compact snapshot


# ---------------------------------------------------------------------------
# Gap #10: annotations=None not in to_provider_dict
# ---------------------------------------------------------------------------


class TestAnnotationsNoneNotInProviderDict:
    def test_annotations_none_not_in_provider_dict(self):
        """ToolSpec.to_provider_dict() must not include 'annotations' key
        even when annotations=None (the default).
        """
        spec = ToolSpec(
            name="my_tool",
            description="desc",
            input_schema={"type": "object"},
        )
        assert spec.annotations is None
        d = spec.to_provider_dict()
        assert "annotations" not in d

    def test_annotations_set_not_in_provider_dict(self):
        """ToolSpec.to_provider_dict() must not include 'annotations' key
        when annotations is set.
        """
        spec = ToolSpec(
            name="my_tool",
            description="desc",
            input_schema={"type": "object"},
            annotations=ToolAnnotations(read_only=True),
        )
        d = spec.to_provider_dict()
        assert "annotations" not in d
        assert "read_only" not in d


# ---------------------------------------------------------------------------
# Gap #15: ConnectorBuilder.tool() accepts annotations
# ---------------------------------------------------------------------------


class TestConnectorBuilderAnnotations:
    def test_connector_tool_accepts_annotations(self, tmp_path):
        from data_harness.app.agent import Agent
        from data_harness.llm.testing import FakeAdapter

        adapter = FakeAdapter([FakeAdapter.text("done")])
        agent = Agent(adapter=adapter, system="s", run_dir=str(tmp_path))
        conn = agent.connector("db", description="database")

        ann = ToolAnnotations(title="Query DB", read_only=True, open_world=True)

        def query_db(sql: str) -> str:
            return f"result of {sql}"

        # Should not raise — annotations kwarg must be accepted
        conn.tool(query_db, description="query", annotations=ann)
        result = agent.run_result("hi")
        assert result.status == "success"

    def test_connector_tool_annotations_propagate_to_spec(self, tmp_path):
        from data_harness.app.agent import Agent
        from data_harness.llm.testing import FakeAdapter

        adapter = FakeAdapter([FakeAdapter.text("done")])
        agent = Agent(adapter=adapter, system="s", run_dir=str(tmp_path))
        conn = agent.connector("db", description="database")

        ann = ToolAnnotations(read_only=True)

        def query_db(sql: str) -> str:
            return "result"

        conn.tool(query_db, description="query", annotations=ann)

        # Build tools and inspect the connector spec
        tools = agent._build_tools()
        db_spec = next((t for t in tools if t.name == "db__query_db"), None)
        assert db_spec is not None
        assert db_spec.annotations is ann
        assert db_spec.annotations.read_only is True
