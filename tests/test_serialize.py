import dataclasses
from datetime import datetime
from enum import Enum

import pytest

from data_harness.core.serialize import to_jsonable
from data_harness.llm.types import TextBlock, ToolResultBlock, ToolUseBlock


class Color(Enum):
    RED = "red"
    BLUE = "blue"


@dataclasses.dataclass
class Simple:
    x: int
    y: str


def test_none():
    assert to_jsonable(None) is None


def test_primitive_types():
    assert to_jsonable(42) == 42
    assert to_jsonable(3.14) == 3.14
    assert to_jsonable(True) is True
    assert to_jsonable("hello") == "hello"


def test_enum():
    assert to_jsonable(Color.RED) == "red"
    assert to_jsonable(Color.BLUE) == "blue"


def test_exception():
    exc = ValueError("oops")
    result = to_jsonable(exc)
    assert result["error_type"] == "ValueError"
    assert result["error_message"] == "oops"


def test_datetime():
    dt = datetime(2024, 1, 15, 12, 0, 0)
    result = to_jsonable(dt)
    assert "2024-01-15" in result


def test_dataframe():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = to_jsonable(df)
    assert result["type"] == "dataframe_snapshot"
    assert result["shape"] == [3, 2]
    assert "a" in result["columns"]


def test_ndarray():
    np = pytest.importorskip("numpy")
    arr = np.array([1, 2, 3, 4, 5])
    result = to_jsonable(arr)
    assert result["type"] == "ndarray_snapshot"
    assert result["shape"] == [5]


def test_nested():
    data = {"key": [1, "two", Color.RED, datetime(2024, 1, 1)]}
    result = to_jsonable(data)
    assert result["key"][0] == 1
    assert result["key"][1] == "two"
    assert result["key"][2] == "red"
    assert "2024-01-01" in result["key"][3]


def test_generic_dataclass():
    s = Simple(x=10, y="abc")
    result = to_jsonable(s)
    assert result["x"] == 10
    assert result["y"] == "abc"


def test_typed_blocks():
    tb = TextBlock(text="hello")
    assert to_jsonable(tb) == {"type": "text", "text": "hello"}

    tub = ToolUseBlock(tool_use_id="id1", tool_name="tool", tool_input={"a": 1})
    j = to_jsonable(tub)
    assert j["type"] == "tool_use"
    assert j["id"] == "id1"

    trb = ToolResultBlock(tool_use_id="id1", content="res", is_error=False)
    j2 = to_jsonable(trb)
    assert j2["type"] == "tool_result"
    assert j2["tool_use_id"] == "id1"


def test_unknown_object():
    class Weird:
        def __repr__(self):
            return "Weird()"

    result = to_jsonable(Weird())
    assert "Weird" in result
