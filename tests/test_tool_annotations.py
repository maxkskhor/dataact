"""Tests for Phase 3: ToolAnnotations on ToolSpec.

TDD: written before implementation.
"""

from __future__ import annotations

import pytest

from data_harness.llm.types import ToolSpec

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
