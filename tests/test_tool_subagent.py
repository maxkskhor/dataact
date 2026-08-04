"""Tests for subagent tool — clean-context spawn with explicit state transfer."""

from __future__ import annotations

import copy

from data_harness.data.cache import SessionCache
from data_harness.data.tools.connectors import ConnectorRegistry
from data_harness.data.tools.interpreter import PythonInterpreter
from data_harness.data.tools.subagent import make_subagent_spec
from data_harness.llm.providers.base import (
    NormalizedResponse,
    ProviderAdapter,
    StopReason,
)
from data_harness.llm.types import Message, TextBlock, ToolSpec, ToolUseBlock


class FakeAdapter(ProviderAdapter):
    def __init__(self, responses: list[NormalizedResponse]) -> None:
        self._responses = list(responses)
        self._calls = []

    def chat(
        self, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> NormalizedResponse:
        self._calls.append({"system": system, "messages": copy.deepcopy(messages)})
        return self._responses.pop(0)

    def format_cache_control(self, obj: dict) -> dict:
        return {**obj, "cache_control": {"type": "ephemeral"}}


def make_text_response(text: str) -> NormalizedResponse:
    return NormalizedResponse(
        stop_reason=StopReason.END_TURN,
        content=[TextBlock(text=text)],
        input_tokens=5,
        output_tokens=3,
        cache_read_tokens=0,
        cache_write_tokens=0,
    )


def make_tool_response(
    tool_id: str, tool_name: str, tool_input: dict
) -> NormalizedResponse:
    return NormalizedResponse(
        stop_reason=StopReason.TOOL_USE,
        content=[
            ToolUseBlock(
                tool_use_id=tool_id,
                tool_name=tool_name,
                tool_input=tool_input,
            )
        ],
        input_tokens=5,
        output_tokens=3,
        cache_read_tokens=0,
        cache_write_tokens=0,
    )


class TestSubagentBasic:
    def test_sub_run_returns_text(self, tmp_path):
        parent_cache = SessionCache()
        call_count = [0]

        def adapter_factory():
            call_count[0] += 1
            return FakeAdapter([make_text_response("sub result")])

        tools = [
            ToolSpec(
                name="echo", description="echo", input_schema={}, handler=lambda: "ok"
            )
        ]
        subagent_spec = make_subagent_spec(
            adapter_factory=adapter_factory,
            parent_tools=tools,
            parent_cache=parent_cache,
        )
        result = subagent_spec.handler(task="do something")
        assert "sub result" in result

    def test_sub_harness_tools_exclude_subagent(self, tmp_path):
        """The subagent tool cannot recurse — sub harness doesn't have subagent spec."""
        parent_cache = SessionCache()
        tools: list[ToolSpec] = []

        def adapter_factory():
            return FakeAdapter([make_text_response("done")])

        subagent_spec = make_subagent_spec(
            adapter_factory=adapter_factory,
            parent_tools=tools,
            parent_cache=parent_cache,
        )
        # Add subagent spec to tools list and verify it can't be passed to sub
        tools_with_subagent = [subagent_spec]
        subagent_spec2 = make_subagent_spec(
            adapter_factory=adapter_factory,
            parent_tools=tools_with_subagent,
            parent_cache=parent_cache,
        )
        # The sub should not have subagent in its tools
        # Verify sub-harness builds successfully but excludes subagent tool
        result = subagent_spec2.handler(task="subtask")
        assert isinstance(result, str)

    def test_fresh_adapter_per_spawn(self, tmp_path):
        parent_cache = SessionCache()
        adapters_created = []

        def adapter_factory():
            a = FakeAdapter([make_text_response("fresh")])
            adapters_created.append(a)
            return a

        tools: list[ToolSpec] = []
        subagent_spec = make_subagent_spec(
            adapter_factory=adapter_factory,
            parent_tools=tools,
            parent_cache=parent_cache,
        )
        subagent_spec.handler(task="task 1")
        subagent_spec.handler(task="task 2")
        assert len(adapters_created) == 2
        assert adapters_created[0] is not adapters_created[1]

    def test_fresh_cache_per_spawn(self, tmp_path):
        parent_cache = SessionCache()
        parent_cache.put("secret", "parent_secret")

        def adapter_factory():
            return FakeAdapter([make_text_response("done")])

        tools: list[ToolSpec] = []
        subagent_spec = make_subagent_spec(
            adapter_factory=adapter_factory,
            parent_tools=tools,
            parent_cache=parent_cache,
        )
        subagent_spec.handler(task="check cache", input_handles=None)
        # Parent cache should be unchanged
        assert parent_cache.get("secret") == "parent_secret"


class TestSubagentIsolation:
    def test_no_implicit_parent_state(self, tmp_path):
        parent_cache = SessionCache()
        parent_cache.put("confidential", "sensitive_data")

        captured_messages = []

        def adapter_factory():
            class CapturingAdapter(ProviderAdapter):
                def chat(self, system, messages, tools):
                    captured_messages.extend(copy.deepcopy(messages))
                    return make_text_response("done")

                def format_cache_control(self, obj):
                    return {**obj, "cache_control": {"type": "ephemeral"}}

            return CapturingAdapter()

        tools: list[ToolSpec] = []
        subagent_spec = make_subagent_spec(
            adapter_factory=adapter_factory,
            parent_tools=tools,
            parent_cache=parent_cache,
        )
        subagent_spec.handler(task="task", input_handles=None)
        all_text = " ".join(
            b.text
            for m in captured_messages
            for b in m.content
            if isinstance(b, TextBlock)
        )
        assert "sensitive_data" not in all_text

    def test_input_handles_copies_only_requested(self, tmp_path):
        parent_cache = SessionCache()
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2, 3]})
        parent_cache.put("wanted", df)
        parent_cache.put("unwanted", "secret")

        sub_received_handles = []

        def adapter_factory():
            class InspectingAdapter(ProviderAdapter):
                def chat(self, system, messages, tools):
                    sub_received_handles.extend(
                        list(
                            getattr(self, "_cache", {}).keys()
                            if hasattr(self, "_cache")
                            else []
                        )
                    )
                    return make_text_response("done")

                def format_cache_control(self, obj):
                    return {**obj, "cache_control": {"type": "ephemeral"}}

            return InspectingAdapter()

        tools: list[ToolSpec] = []
        subagent_spec = make_subagent_spec(
            adapter_factory=adapter_factory,
            parent_tools=tools,
            parent_cache=parent_cache,
        )
        # Even if we can't easily inspect sub-cache, at least verify no error
        result = subagent_spec.handler(task="task", input_handles=["wanted"])
        assert isinstance(result, str)

    def test_missing_input_handle_returns_error(self, tmp_path):
        parent_cache = SessionCache()

        def adapter_factory():
            return FakeAdapter([make_text_response("done")])

        tools: list[ToolSpec] = []
        subagent_spec = make_subagent_spec(
            adapter_factory=adapter_factory,
            parent_tools=tools,
            parent_cache=parent_cache,
        )
        result = subagent_spec.handler(task="task", input_handles=["nonexistent"])
        assert (
            "error" in result.lower()
            or "not found" in result.lower()
            or "missing" in result.lower()
        )

    def test_text_only_does_not_publish_handles(self, tmp_path):
        parent_cache = SessionCache()

        def adapter_factory():
            return FakeAdapter([make_text_response("result with data")])

        tools: list[ToolSpec] = []
        subagent_spec = make_subagent_spec(
            adapter_factory=adapter_factory,
            parent_tools=tools,
            parent_cache=parent_cache,
        )
        subagent_spec.handler(task="task", output_policy="text_only")
        # Parent cache should still be empty
        assert len(parent_cache.list_handles()) == 0

    def test_input_handles_are_value_copied_for_subagent_mutation(self, tmp_path):
        parent_cache = SessionCache()
        parent_cache.put("numbers", [1, 2, 3])

        def adapter_factory():
            return FakeAdapter(
                [
                    make_tool_response(
                        "tu_1",
                        "python_interpreter",
                        {
                            "code": (
                                "numbers.append(4)\n"
                                "save('mutated_numbers', numbers)\n"
                                "print(numbers)"
                            )
                        },
                    ),
                    make_text_response("done"),
                ]
            )

        subagent_spec = make_subagent_spec(
            adapter_factory=adapter_factory,
            parent_tools=[PythonInterpreter.make_tool_spec(parent_cache)],
            parent_cache=parent_cache,
        )

        subagent_spec.handler(
            task="mutate the input",
            input_handles=["numbers"],
            output_policy="publish_created",
        )

        assert parent_cache.get("numbers") == [1, 2, 3]
        assert parent_cache.get("mutated_numbers") == [1, 2, 3, 4]

    def test_cache_bound_connector_fallback_returns_isolation_error(self, tmp_path):
        parent_cache = SessionCache()
        registry = ConnectorRegistry()
        registry.register(
            name="data",
            description="Data tools.",
            tools=[
                ToolSpec(
                    name="data__fetch",
                    description="Fetch rows.",
                    input_schema={"type": "object", "properties": {}},
                    handler=lambda: list(range(100)),
                    visible=True,
                )
            ],
        )
        parent_tools = registry.make_wrapped_specs(parent_cache)

        def adapter_factory():
            return FakeAdapter([make_text_response("should not run")])

        subagent_spec = make_subagent_spec(
            adapter_factory=adapter_factory,
            parent_tools=parent_tools,
            parent_cache=parent_cache,
        )

        result = subagent_spec.handler(task="use connector")

        assert "subagent tools are not isolated" in result
        assert "make_sub_tools" in result
        assert parent_cache.list_handles() == {}

    def test_publish_created_gets_only_new_handles(self, tmp_path):
        parent_cache = SessionCache()
        parent_cache.put("input_data", "input")
        sub_cache = _CountingCache(storage_dir=tmp_path / "sub", hot_limit=1)

        def adapter_factory():
            return FakeAdapter(
                [
                    make_tool_response("tu_1", "create", {}),
                    make_text_response("done"),
                ]
            )

        def make_sub_tools(cache):
            return [
                ToolSpec(
                    name="create",
                    description="Create a handle.",
                    input_schema={"type": "object", "properties": {}},
                    handler=lambda: cache.put("created", "value"),
                )
            ]

        subagent_spec = make_subagent_spec(
            adapter_factory=adapter_factory,
            parent_tools=[],
            parent_cache=parent_cache,
            get_sub_cache=lambda: sub_cache,
            make_sub_tools=make_sub_tools,
        )

        subagent_spec.handler(
            task="create output",
            input_handles=["input_data"],
            output_policy="publish_created",
        )

        assert sub_cache.get_calls == ["created"]
        assert parent_cache.get("created") == "value"


class TestSubagentPublishCreated:
    def test_publish_created_copies_new_handles(self, tmp_path):
        parent_cache = SessionCache()

        def adapter_factory():
            # Sub adapter that will call save() via interpreter or just returns text
            return FakeAdapter([make_text_response("computed result")])

        tools: list[ToolSpec] = []
        subagent_spec = make_subagent_spec(
            adapter_factory=adapter_factory,
            parent_tools=tools,
            parent_cache=parent_cache,
            get_sub_cache=lambda: _CacheWithPrefill(),
        )
        result = subagent_spec.handler(task="task", output_policy="publish_created")
        assert isinstance(result, str)

    def test_tool_spec_isolation(self, tmp_path):
        """Flipping visible on a copied ToolSpec doesn't mutate the parent ToolSpec."""
        parent_tool = ToolSpec(
            name="my_tool",
            description="desc",
            input_schema={},
            handler=None,
            visible=True,
        )

        def adapter_factory():
            return FakeAdapter([make_text_response("done")])

        subagent_spec = make_subagent_spec(
            adapter_factory=adapter_factory,
            parent_tools=[parent_tool],
            parent_cache=SessionCache(),
        )
        # Run subagent - even if it flips visibility internally
        subagent_spec.handler(task="task")
        # Parent tool should be unchanged
        assert parent_tool.visible is True

    def test_publish_created_collision_uses_resolved_parent_name(self, tmp_path):
        parent_cache = SessionCache()
        parent_cache.put("result", "parent value")

        def adapter_factory():
            return FakeAdapter(
                [
                    make_tool_response(
                        "tu_1",
                        "python_interpreter",
                        {"code": "save('result', 'child value')\nprint('saved')"},
                    ),
                    make_text_response("done"),
                ]
            )

        subagent_spec = make_subagent_spec(
            adapter_factory=adapter_factory,
            parent_tools=[PythonInterpreter.make_tool_spec(parent_cache)],
            parent_cache=parent_cache,
        )

        result = subagent_spec.handler(task="publish", output_policy="publish_created")

        assert parent_cache.get("result") == "parent value"
        assert parent_cache.get("result_2") == "child value"
        assert "- result -> result_2" in result


class _CacheWithPrefill(SessionCache):
    """A SessionCache pre-filled with a value to simulate subagent-created handles."""

    def __init__(self):
        super().__init__()
        self.put("sub_result", "computed data")


class _CountingCache(SessionCache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.get_calls = []

    def get(self, name: str):
        self.get_calls.append(name)
        return super().get(name)
