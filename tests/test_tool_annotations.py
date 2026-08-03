"""Tests for Phase 3: ToolAnnotations on ToolSpec.

TDD: written before implementation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_harness.data.harness import Harness
from data_harness.llm.providers.base import StopReason
from data_harness.llm.testing import FakeAdapter
from data_harness.llm.types import TextBlock, ToolSpec


def make_text_response(text: str):
    from data_harness.llm.providers.base import NormalizedResponse

    return NormalizedResponse(
        stop_reason=StopReason.END_TURN,
        content=[TextBlock(text=text)],
        input_tokens=5,
        output_tokens=2,
        cache_read_tokens=0,
        cache_write_tokens=0,
    )


def read_jsonl(path: str) -> list[dict]:
    lines = Path(path).read_text().strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# ---------------------------------------------------------------------------
# ToolAnnotations dataclass
# ---------------------------------------------------------------------------


class TestToolAnnotations:
    def test_import(self):
        from data_harness.llm.types import ToolAnnotations

        assert ToolAnnotations is not None

    def test_all_fields_optional(self):
        from data_harness.llm.types import ToolAnnotations

        ann = ToolAnnotations()
        assert ann.title is None
        assert ann.read_only is None
        assert ann.cache_mutating is None
        assert ann.destructive is None
        assert ann.open_world is None

    def test_explicit_fields(self):
        from data_harness.llm.types import ToolAnnotations

        ann = ToolAnnotations(
            title="Echo",
            read_only=True,
            cache_mutating=False,
            destructive=False,
            open_world=False,
        )
        assert ann.title == "Echo"
        assert ann.read_only is True
        assert ann.cache_mutating is False

    def test_frozen(self):
        from data_harness.llm.types import ToolAnnotations

        ann = ToolAnnotations(title="Echo")
        with pytest.raises((AttributeError, TypeError)):
            ann.title = "Other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ToolSpec carries annotations
# ---------------------------------------------------------------------------


class TestToolSpecAnnotations:
    def test_annotations_field_defaults_to_none(self):
        spec = ToolSpec(
            name="my_tool",
            description="desc",
            input_schema={"type": "object"},
        )
        assert spec.annotations is None

    def test_annotations_field_set(self):
        from data_harness.llm.types import ToolAnnotations

        ann = ToolAnnotations(read_only=True)
        spec = ToolSpec(
            name="my_tool",
            description="desc",
            input_schema={"type": "object"},
            annotations=ann,
        )
        assert spec.annotations is ann
        assert spec.annotations.read_only is True

    def test_to_provider_dict_excludes_annotations(self):
        from data_harness.llm.types import ToolAnnotations

        ann = ToolAnnotations(read_only=True, destructive=False)
        spec = ToolSpec(
            name="my_tool",
            description="desc",
            input_schema={"type": "object"},
            annotations=ann,
        )
        d = spec.to_provider_dict()
        assert "annotations" not in d
        assert "read_only" not in d
        assert "destructive" not in d
        assert set(d.keys()) == {"name", "description", "input_schema"}


# ---------------------------------------------------------------------------
# Built-in tools carry expected annotations
# ---------------------------------------------------------------------------


class TestBuiltinToolAnnotations:
    def test_list_variables_read_only(self):
        from data_harness.data.cache import SessionCache
        from data_harness.data.tools.variables import make_list_variables_spec

        spec = make_list_variables_spec(SessionCache())
        assert spec.annotations is not None
        assert spec.annotations.read_only is True

    def test_python_interpreter_cache_mutating(self):
        from data_harness.data.cache import SessionCache
        from data_harness.data.tools.interpreter import PythonInterpreter

        spec = PythonInterpreter.make_tool_spec(SessionCache())
        assert spec.annotations is not None
        assert spec.annotations.cache_mutating is True

    def test_python_interpreter_not_open_world(self):
        from data_harness.data.cache import SessionCache
        from data_harness.data.tools.interpreter import PythonInterpreter

        spec = PythonInterpreter.make_tool_spec(SessionCache())
        assert spec.annotations.open_world is False


# ---------------------------------------------------------------------------
# Annotations appear in JSONL logs
# ---------------------------------------------------------------------------


class TestAnnotationsInLog:
    def test_annotations_serialised_in_jsonl(self, tmp_path):
        from data_harness.llm.types import ToolAnnotations

        ann = ToolAnnotations(title="Echo tool", read_only=True)
        echo_spec = ToolSpec(
            name="echo",
            description="echo",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            handler=lambda text: text,
            annotations=ann,
        )
        adapter = FakeAdapter([make_text_response("ok")])
        harness = Harness(
            adapter=adapter,
            system="s",
            tools=[echo_spec],
            max_turns=5,
            run_dir=str(tmp_path),
        )
        result = harness.run_result("go")
        records = read_jsonl(result.run_file)
        record = records[0]
        assert "tool_annotations" in record
        assert "echo" in record["tool_annotations"]
        ann_data = record["tool_annotations"]["echo"]
        assert ann_data["title"] == "Echo tool"
        assert ann_data["read_only"] is True

    def test_tool_without_annotations_omitted_from_map(self, tmp_path):
        """Tools without annotations should not appear in the annotation map."""
        no_ann_spec = ToolSpec(
            name="plain",
            description="plain",
            input_schema={"type": "object"},
            handler=lambda: "ok",
        )
        adapter = FakeAdapter([make_text_response("ok")])
        harness = Harness(
            adapter=adapter,
            system="s",
            tools=[no_ann_spec],
            max_turns=5,
            run_dir=str(tmp_path),
        )
        result = harness.run_result("go")
        records = read_jsonl(result.run_file)
        record = records[0]
        # tool_annotations absent or empty — not populated for unannotated tools
        annotations = record.get("tool_annotations", {})
        assert "plain" not in annotations
