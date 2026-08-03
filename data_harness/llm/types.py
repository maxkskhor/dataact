from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal


@dataclass(frozen=True)
class ToolAnnotations:
    """Model-visible hints about a tool's side-effect profile.

    All fields are optional. Omitted fields are treated as unknown by the
    harness; set them explicitly when the information is relevant to how the
    model should reason about the tool.

    Attributes:
        title: Human-readable display name for the tool.
        read_only: True if the tool never mutates external state.
        cache_mutating: True if the tool writes to the session cache.
        destructive: True if the tool performs irreversible side effects.
        open_world: True if the tool can reach external systems at runtime.
    """

    title: str | None = None
    read_only: bool | None = None
    cache_mutating: bool | None = None
    destructive: bool | None = None
    open_world: bool | None = None


@dataclass
class TextBlock:
    """A plain-text content block inside a `Message`.

    Attributes:
        text: The raw text content.
    """

    text: str


@dataclass
class ToolUseBlock:
    """A tool-invocation block emitted by the model.

    Attributes:
        tool_use_id: Unique identifier for this invocation, echoed back in the
            matching `ToolResultBlock`.
        tool_name: Name of the tool to call.
        tool_input: Parsed JSON arguments for the tool.
    """

    tool_use_id: str
    tool_name: str
    tool_input: dict


@dataclass
class ToolResultBlock:
    """The harness-side result of a tool invocation.

    Attributes:
        tool_use_id: Must match the `tool_use_id` of the originating
            `ToolUseBlock`.
        content: Serialised tool output, or an error message when `is_error`
            is `True`.
        is_error: Whether the tool call raised an exception.
    """

    tool_use_id: str
    content: str
    is_error: bool = False


ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock
"""Union of the three block types that can appear in a `Message`."""


@dataclass
class Message:
    """A single turn in the conversation history.

    Attributes:
        role: Either ``"user"`` or ``"assistant"``.
        content: Ordered list of content blocks for this turn.
    """

    role: Literal["user", "assistant"]
    content: list[ContentBlock]

    def __post_init__(self) -> None:
        if self.role not in ("user", "assistant"):
            raise ValueError(
                f"Invalid role: {self.role!r}. Must be 'user' or 'assistant'."
            )


@dataclass
class ToolSpec:
    """Everything the harness needs to register and dispatch one tool.

    Attributes:
        name: Unique tool name exposed to the model.
        description: Natural-language description shown to the model.
        input_schema: JSON Schema object describing the tool's parameters.
        handler: Callable invoked with ``**tool_input`` when the model calls
            the tool. ``None`` means the tool is schema-only (not dispatchable).
        visible: Whether the tool appears in the model's tool list. Hidden tools
            can still be called if the model somehow names them; set
            ``handler=None`` to block dispatch entirely.
        annotations: Optional side-effect hints passed to `ToolAnnotations`.
    """

    name: str
    description: str
    input_schema: dict
    handler: Callable[..., Any] | None = None
    visible: bool = True
    annotations: ToolAnnotations | None = None

    def to_provider_dict(self) -> dict:
        """Return the provider-facing tool dict (name, description, input_schema)."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
