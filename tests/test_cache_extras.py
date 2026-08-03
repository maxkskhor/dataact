"""Tier 1/2: cache answer slot and the semantic layer."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from data_harness.data.cache import SessionCache


def test_answer_slot_roundtrip():
    cache = SessionCache()
    assert cache.has_answer is False
    assert cache.get_answer() is None
    cache.set_answer(123)
    assert cache.has_answer is True
    assert cache.get_answer() == 123
    cache.clear_answer()
    assert cache.has_answer is False


def test_semantics_fold_into_dataframe_snapshot():
    cache = SessionCache()
    sem = {"columns": {"revenue": "Gross monthly revenue in USD"}}
    handle = cache.put("sales", pd.DataFrame({"revenue": [1, 2]}), semantics=sem)
    snap = json.loads(cache.snapshot(handle))
    assert snap["semantics"] == sem


def test_describe_updates_existing_handle():
    cache = SessionCache()
    handle = cache.put("sales", pd.DataFrame({"a": [1]}))
    cache.describe(handle, {"note": "test"})
    assert cache.get_semantics(handle) == {"note": "test"}
    assert "semantics" in cache.snapshot(handle)


def test_describe_missing_handle_raises():
    cache = SessionCache()
    with pytest.raises(KeyError):
        cache.describe("nope", {"x": 1})


def test_semantics_on_scalar_snapshot():
    cache = SessionCache()
    handle = cache.put("k", 5, semantics={"unit": "count"})
    snap = cache.snapshot(handle)
    assert "semantics" in snap and "count" in snap


def test_delete_clears_semantics_and_charts():
    cache = SessionCache()
    handle = cache.put("sales", pd.DataFrame({"a": [1]}), semantics={"x": 1})
    cache.delete(handle)
    assert cache.get_semantics(handle) is None
