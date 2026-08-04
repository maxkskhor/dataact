import pytest

from data_harness.core.serialize import to_jsonable
from data_harness.llm.providers.base import ProviderAdapter
from data_harness.llm.types import (
    Message,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)


def test_text_block_roundtrip():
    b = TextBlock(text="hello")
    j = to_jsonable(b)
    assert j == {"type": "text", "text": "hello"}


def test_tool_use_block_roundtrip():
    b = ToolUseBlock(tool_use_id="tu_1", tool_name="my_tool", tool_input={"x": 1})
    j = to_jsonable(b)
    assert j == {"type": "tool_use", "id": "tu_1", "name": "my_tool", "input": {"x": 1}}


def test_tool_result_block_roundtrip():
    b = ToolResultBlock(tool_use_id="tu_1", content="ok", is_error=False)
    j = to_jsonable(b)
    assert j == {
        "type": "tool_result",
        "tool_use_id": "tu_1",
        "content": "ok",
        "is_error": False,
    }


def test_tool_result_block_error():
    b = ToolResultBlock(tool_use_id="tu_2", content="fail", is_error=True)
    j = to_jsonable(b)
    assert j["is_error"] is True


def test_message_valid_roles():
    m_user = Message(role="user", content=[TextBlock(text="hi")])
    assert m_user.role == "user"
    m_asst = Message(role="assistant", content=[TextBlock(text="yo")])
    assert m_asst.role == "assistant"


def test_message_invalid_role():
    with pytest.raises(ValueError):
        Message(role="system", content=[])  # type: ignore


def test_tool_spec_to_provider_dict_excludes_handler_and_visible():
    spec = ToolSpec(
        name="foo",
        description="bar",
        input_schema={"type": "object"},
        handler=lambda: None,
        visible=False,
    )
    d = spec.to_provider_dict()
    assert "handler" not in d
    assert "visible" not in d
    assert d["name"] == "foo"
    assert d["description"] == "bar"
    assert d["input_schema"] == {"type": "object"}


def test_provider_adapter_chat_signature():
    import inspect

    sig = inspect.signature(ProviderAdapter.chat)
    params = list(sig.parameters.keys())
    assert "system" in params
    assert "messages" in params
    assert "tools" in params
