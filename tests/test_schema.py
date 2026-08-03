"""Tests for connector function schema inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pytest

from data_harness.core.schema import infer_input_schema

OVERRIDE_HINT = "pass input_schema=... to override"


class TestInferInputSchema:
    def test_supported_annotations(self):
        def fn(
            symbol: str,
            count: int,
            price: float,
            active: bool,
            tags: list[str],
        ) -> None:
            return None

        schema = infer_input_schema(fn)

        assert schema == {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "count": {"type": "integer"},
                "price": {"type": "number"},
                "active": {"type": "boolean"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["symbol", "count", "price", "active", "tags"],
        }

    def test_defaults_make_parameters_optional(self):
        def fn(symbol: str, count: int = 10, active: bool = True) -> None:
            return None

        schema = infer_input_schema(fn)

        assert schema["required"] == ["symbol"]

    @pytest.mark.parametrize(
        "fn",
        [
            lambda *args: None,
            lambda **kwargs: None,
        ],
    )
    def test_rejects_varargs_and_kwargs(self, fn):
        with pytest.raises(TypeError, match=OVERRIDE_HINT):
            infer_input_schema(fn)

    @pytest.mark.parametrize(
        "annotation",
        [
            dict,
            Any,
            Optional[str],
        ],
    )
    def test_rejects_unsupported_annotations(self, annotation):
        def fn(payload: annotation) -> None:  # type: ignore[valid-type]
            return None

        with pytest.raises(TypeError, match=OVERRIDE_HINT):
            infer_input_schema(fn)

    def test_rejects_dataclass_annotation(self):
        @dataclass
        class Payload:
            symbol: str

        def fn(payload: Payload) -> None:
            return None

        with pytest.raises(TypeError, match=OVERRIDE_HINT):
            infer_input_schema(fn)

    def test_rejects_positional_only_parameter(self):
        def fn(symbol: str, /) -> None:
            return None

        with pytest.raises(TypeError, match=OVERRIDE_HINT):
            infer_input_schema(fn)

    def test_rejects_unannotated_parameter(self):
        def fn(symbol, interval: str) -> None:
            return None

        with pytest.raises(TypeError, match=OVERRIDE_HINT):
            infer_input_schema(fn)
