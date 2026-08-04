"""Input-schema inference for small connector functions."""

from __future__ import annotations

import dataclasses
import inspect
import types
from collections.abc import Callable
from typing import Any, get_args, get_origin, get_type_hints

_OVERRIDE_HINT = "pass input_schema=... to override"


def infer_input_schema(fn: Callable[..., Any]) -> dict:
    """Infer a small JSON schema from a connector function signature."""
    signature = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception as exc:
        raise _unsupported(fn, "could not resolve type annotations") from exc

    properties: dict[str, dict] = {}
    required: list[str] = []

    for name, parameter in signature.parameters.items():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise _unsupported(fn, f"unsupported variadic parameter {name!r}")
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            raise _unsupported(fn, f"unsupported positional-only parameter {name!r}")
        if parameter.annotation is inspect.Parameter.empty or name not in hints:
            raise _unsupported(fn, f"missing annotation for parameter {name!r}")

        properties[name] = _schema_for_annotation(fn, hints[name])
        if parameter.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _schema_for_annotation(fn: Callable[..., Any], annotation: Any) -> dict:
    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}

    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is list and args == (str,):
        return {"type": "array", "items": {"type": "string"}}

    if annotation is Any:
        raise _unsupported(fn, "Any is not supported")
    if annotation is dict or origin is dict:
        raise _unsupported(fn, "dict is not supported")
    if dataclasses.is_dataclass(annotation):
        raise _unsupported(fn, "dataclass annotations are not supported")
    if origin in (types.UnionType, getattr(types, "UnionType", object)):
        raise _unsupported(fn, "union annotations are not supported")
    if str(origin) == "typing.Union":
        raise _unsupported(fn, "union annotations are not supported")

    raise _unsupported(fn, f"unsupported annotation {annotation!r}")


def _unsupported(fn: Callable[..., Any], reason: str) -> TypeError:
    return TypeError(
        f"Cannot infer input schema for {fn.__name__}: {reason}; {_OVERRIDE_HINT}"
    )
