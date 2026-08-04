"""Tests for the sandboxed Python interpreter tool."""

import pandas as pd
import pytest

from data_harness.data.cache import SessionCache
from data_harness.data.harness import Harness
from data_harness.data.tools.interpreter import (
    _EMPTY_OUTPUT_GUIDANCE,
    _LOCALS_ERROR,
    PythonInterpreter,
    PythonInterpreterError,
)
from data_harness.llm.providers.base import (
    NormalizedResponse,
    ProviderAdapter,
    StopReason,
)
from data_harness.llm.types import TextBlock, ToolResultBlock, ToolUseBlock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeAdapter(ProviderAdapter):
    """Returns scripted responses then a final text turn."""

    def __init__(self, responses: list[NormalizedResponse]) -> None:
        self._responses = list(responses)

    def chat(self, system, messages, tools):
        return self._responses.pop(0)

    def format_cache_control(self, obj):
        return {**obj, "cache_control": {"type": "ephemeral"}}


def _tool_response(
    tool_use_id: str, tool_name: str, tool_input: dict
) -> NormalizedResponse:
    return NormalizedResponse(
        stop_reason=StopReason.TOOL_USE,
        content=[
            ToolUseBlock(
                tool_use_id=tool_use_id, tool_name=tool_name, tool_input=tool_input
            )
        ],
        input_tokens=5,
        output_tokens=5,
        cache_read_tokens=0,
        cache_write_tokens=0,
    )


def _text_response(text: str) -> NormalizedResponse:
    return NormalizedResponse(
        stop_reason=StopReason.END_TURN,
        content=[TextBlock(text=text)],
        input_tokens=5,
        output_tokens=5,
        cache_read_tokens=0,
        cache_write_tokens=0,
    )


def _run_interpreter_via_harness(
    code: str, cache: SessionCache | None = None
) -> ToolResultBlock:
    """Drive one python_interpreter tool call through a real Harness."""
    cache = cache or SessionCache()
    spec = PythonInterpreter.make_tool_spec(cache)
    adapter = FakeAdapter(
        [
            _tool_response("t1", "python_interpreter", {"code": code}),
            _text_response("done"),
        ]
    )
    harness = Harness(adapter=adapter, system="sys", tools=[spec], cache=cache)
    harness.run("go")
    # The tool result is the second-to-last message (user message with tool result)
    for msg in reversed(harness.messages):
        if msg.role == "user":
            for block in msg.content:
                if isinstance(block, ToolResultBlock):
                    return block
    raise AssertionError("no ToolResultBlock found")


# ---------------------------------------------------------------------------
# Allowed / forbidden imports
# ---------------------------------------------------------------------------


class TestAllowedImports:
    def test_allowed_import_passes(self):
        cache = SessionCache()
        interp = PythonInterpreter(cache=cache)
        result = interp.run(
            code="import pandas as pd\nprint(pd.DataFrame({'x': [1]}).shape[0])"
        )
        assert result.strip() == "1"

    def test_disallowed_import_rejected(self):
        cache = SessionCache()
        interp = PythonInterpreter(cache=cache)
        with pytest.raises(PythonInterpreterError, match="not allowed"):
            interp.run(code="import os\nprint(os.getcwd())")

    def test_subprocess_import_rejected(self):
        cache = SessionCache()
        interp = PythonInterpreter(cache=cache)
        with pytest.raises(PythonInterpreterError, match="not allowed"):
            interp.run(code="import subprocess")


# ---------------------------------------------------------------------------
# Forbidden builtins
# ---------------------------------------------------------------------------


class TestForbiddenBuiltins:
    def test_eval_rejected(self):
        cache = SessionCache()
        interp = PythonInterpreter(cache=cache)
        with pytest.raises(PythonInterpreterError):
            interp.run(code="eval('1+1')")

    def test_exec_rejected(self):
        cache = SessionCache()
        interp = PythonInterpreter(cache=cache)
        with pytest.raises(PythonInterpreterError):
            interp.run(code="exec('x=1')")

    def test_import_builtin_rejected(self):
        cache = SessionCache()
        interp = PythonInterpreter(cache=cache)
        with pytest.raises(PythonInterpreterError):
            interp.run(code="__import__('os')")

    def test_open_rejected(self):
        cache = SessionCache()
        interp = PythonInterpreter(cache=cache)
        with pytest.raises(PythonInterpreterError):
            interp.run(code="open('/etc/passwd')")

    def test_dunder_access_rejected(self):
        cache = SessionCache()
        interp = PythonInterpreter(cache=cache)
        with pytest.raises(PythonInterpreterError):
            interp.run(code="x = [].__class__.__bases__")


# ---------------------------------------------------------------------------
# Stdout capture
# ---------------------------------------------------------------------------


class TestStdoutCapture:
    def test_stdout_captured(self):
        cache = SessionCache()
        interp = PythonInterpreter(cache=cache)
        result = interp.run(code="print('hello world')")
        assert "hello world" in result

    def test_multiple_prints(self):
        cache = SessionCache()
        interp = PythonInterpreter(cache=cache)
        result = interp.run(code="print('a')\nprint('b')\nprint('c')")
        assert "a" in result and "b" in result and "c" in result


# ---------------------------------------------------------------------------
# Cache handles
# ---------------------------------------------------------------------------


class TestCacheHandles:
    def test_cache_handle_available_as_local(self):
        cache = SessionCache()
        df = pd.DataFrame({"x": [1, 2, 3]})
        cache.put("market_data", df)
        interp = PythonInterpreter(cache=cache)
        result = interp.run(code="print(len(market_data))")
        assert "3" in result

    def test_save_stores_in_cache(self):
        cache = SessionCache()
        interp = PythonInterpreter(cache=cache)
        result = interp.run(code="save('avg', 42.0)\nprint('saved')")
        assert "saved" in result or "avg" in result
        assert cache.get("avg") == 42.0

    def test_save_collision_auto_suffixed(self):
        cache = SessionCache()
        cache.put("result", "original")
        interp = PythonInterpreter(cache=cache)
        result = interp.run(code="name = save('result', 'new')\nprint(name)")
        assert "result_2" in result

    def test_locals_do_not_persist_between_calls(self):
        cache = SessionCache()
        interp = PythonInterpreter(cache=cache)
        interp.run(code="my_local = 'hello'\nprint('ok')")
        with pytest.raises(PythonInterpreterError, match="Error"):
            interp.run(code="print(my_local)")

    def test_cache_object_not_in_locals(self):
        cache = SessionCache()
        interp = PythonInterpreter(cache=cache)
        result = interp.run(code="print('cache' in dir())")
        assert "False" in result or "cache" not in result


# ---------------------------------------------------------------------------
# Error handling: raise PythonInterpreterError; harness marks is_error=True
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_exception_in_user_code_raises(self):
        cache = SessionCache()
        interp = PythonInterpreter(cache=cache)
        with pytest.raises(PythonInterpreterError) as exc_info:
            interp.run(code="raise ValueError('oops')")
        assert "oops" in str(exc_info.value) or "ValueError" in str(exc_info.value)

    def test_zero_division_raises(self):
        cache = SessionCache()
        interp = PythonInterpreter(cache=cache)
        with pytest.raises(PythonInterpreterError) as exc_info:
            interp.run(code="1/0")
        assert "ZeroDivision" in str(exc_info.value) or "Error" in str(exc_info.value)

    def test_runtime_error_is_error_true_via_harness(self):
        result = _run_interpreter_via_harness("raise ValueError('bad')")
        assert result.is_error is True

    def test_syntax_error_raises(self):
        cache = SessionCache()
        interp = PythonInterpreter(cache=cache)
        with pytest.raises(PythonInterpreterError, match="SyntaxError"):
            interp.run(code="def f(:\n    pass")

    def test_syntax_error_is_error_true_via_harness(self):
        result = _run_interpreter_via_harness("def f(:\n    pass")
        assert result.is_error is True

    def test_security_error_is_error_true_via_harness(self):
        result = _run_interpreter_via_harness("import os")
        assert result.is_error is True

    def test_error_repr_is_message_not_class_name(self):
        """PythonInterpreterError repr returns the message, not the class wrapper."""
        err = PythonInterpreterError("SyntaxError: bad syntax")
        assert repr(err) == "SyntaxError: bad syntax"


# ---------------------------------------------------------------------------
# locals() targeted error
# ---------------------------------------------------------------------------


class TestLocalsGuard:
    def test_locals_call_raises_targeted_error(self):
        cache = SessionCache()
        cache.put("df", pd.DataFrame({"v": [1]}))
        interp = PythonInterpreter(cache=cache)
        with pytest.raises(PythonInterpreterError) as exc_info:
            interp.run(code='df = locals()["df"]')
        assert "locals()" in str(exc_info.value)
        assert "list_variables" in str(exc_info.value)

    def test_locals_error_message_matches_constant(self):
        cache = SessionCache()
        interp = PythonInterpreter(cache=cache)
        with pytest.raises(PythonInterpreterError) as exc_info:
            interp.run(code="x = locals()")
        assert str(exc_info.value) == _LOCALS_ERROR

    def test_locals_is_error_true_via_harness(self):
        cache = SessionCache()
        cache.put("df", pd.DataFrame({"v": [1]}))
        result = _run_interpreter_via_harness('df = locals()["df"]', cache=cache)
        assert result.is_error is True


# ---------------------------------------------------------------------------
# Empty output guidance
# ---------------------------------------------------------------------------


class TestEmptyOutputGuidance:
    def test_assignment_returns_guidance(self):
        cache = SessionCache()
        interp = PythonInterpreter(cache=cache)
        result = interp.run(code="x = 42")
        assert result == _EMPTY_OUTPUT_GUIDANCE

    def test_empty_output_guidance_text(self):
        assert "print(" in _EMPTY_OUTPUT_GUIDANCE
        assert "save(" in _EMPTY_OUTPUT_GUIDANCE

    def test_empty_output_not_is_error_via_harness(self):
        result = _run_interpreter_via_harness("stats = 1 + 1")
        assert result.is_error is False
        assert "print(" in result.content or "save(" in result.content


# ---------------------------------------------------------------------------
# Final-expression capture (notebook-like behaviour)
# ---------------------------------------------------------------------------


class TestFinalExpressionCapture:
    def test_bare_expression_returns_repr(self):
        cache = SessionCache()
        interp = PythonInterpreter(cache=cache)
        result = interp.run(code="1 + 1")
        assert result == "2"

    def test_dataframe_describe_captured(self):
        cache = SessionCache()
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]})
        cache.put("fred_unrate", df)
        interp = PythonInterpreter(cache=cache)
        result = interp.run(code='fred_unrate["value"].describe()')
        assert "mean" in result or "count" in result or "dtype" in result

    def test_dataframe_head_captured(self):
        cache = SessionCache()
        df = pd.DataFrame({"x": [10, 20, 30]})
        cache.put("ds", df)
        interp = PythonInterpreter(cache=cache)
        result = interp.run(code="ds.head()")
        assert "10" in result or "x" in result

    def test_stdout_takes_precedence_over_expr(self):
        cache = SessionCache()
        interp = PythonInterpreter(cache=cache)
        result = interp.run(code="print('printed')\n99")
        assert "printed" in result
        assert "99" not in result

    def test_expr_returning_none_yields_guidance(self):
        """Expressions that evaluate to None (e.g. inplace ops) return guidance."""
        cache = SessionCache()
        df = pd.DataFrame({"x": [3, 1, 2]})
        cache.put("ds", df)
        interp = PythonInterpreter(cache=cache)
        result = interp.run(code="ds.sort_values('x', inplace=True)")
        assert result == _EMPTY_OUTPUT_GUIDANCE

    def test_string_expression_captured(self):
        cache = SessionCache()
        interp = PythonInterpreter(cache=cache)
        result = interp.run(code="'hello from expr'")
        assert "hello from expr" in result

    def test_multiline_with_final_expr(self):
        cache = SessionCache()
        interp = PythonInterpreter(cache=cache)
        result = interp.run(code="x = 10\ny = 20\nx + y")
        assert result == "30"
