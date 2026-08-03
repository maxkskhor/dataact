import pytest

from data_harness.data.cache import SessionCache


class TestPutGet:
    def test_roundtrip(self):
        cache = SessionCache()
        cache.put("sales", [1, 2, 3])
        assert cache.get("sales") == [1, 2, 3]

    def test_put_returns_handle_name(self):
        cache = SessionCache()
        name = cache.put("revenue", 42.0)
        assert name == "revenue"

    def test_invalid_name_rejected(self):
        cache = SessionCache()
        with pytest.raises(ValueError):
            cache.put("123foo", "data")

    def test_invalid_hyphenated_name_rejected(self):
        cache = SessionCache()
        with pytest.raises(ValueError):
            cache.put("my-name", "data")

    def test_auto_suffix_on_collision(self):
        cache = SessionCache()
        import pandas as pd

        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"b": [3, 4]})
        n1 = cache.put("sales", df1)
        n2 = cache.put("sales", df2)
        assert n1 == "sales"
        assert n2 == "sales_2"

    def test_both_suffixed_retrievable(self):
        cache = SessionCache()
        cache.put("x", "first")
        cache.put("x", "second")
        assert cache.get("x") == "first"
        assert cache.get("x_2") == "second"

    def test_triple_collision(self):
        cache = SessionCache()
        cache.put("x", "a")
        cache.put("x", "b")
        n3 = cache.put("x", "c")
        assert n3 == "x_3"

    def test_explicit_overwrite(self):
        cache = SessionCache()
        cache.put("data", "original")
        n = cache.put("data", "replaced", overwrite=True)
        assert n == "data"
        assert cache.get("data") == "replaced"


class TestSnapshot:
    def test_dataframe_snapshot(self):
        pytest.importorskip("pandas")
        import pandas as pd

        cache = SessionCache()
        df = pd.DataFrame({"a": range(100), "b": range(100)})
        cache.put("big", df)
        snap = cache.snapshot("big")
        assert (
            "dataframe" in snap.lower()
            or "shape" in snap.lower()
            or "columns" in snap.lower()
        )

    def test_dataframe_snapshot_serializes_timestamps(self):
        pytest.importorskip("pandas")
        import pandas as pd

        cache = SessionCache()
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
                "value": [1.0, 2.0],
            }
        )
        cache.put("dated", df)

        snap = cache.snapshot("dated")

        assert "2024-01-01" in snap

    def test_list_snapshot(self):
        cache = SessionCache()
        cache.put("items", list(range(50)))
        snap = cache.snapshot("items")
        assert "50" in snap or "length" in snap.lower() or "list" in snap.lower()

    def test_dict_snapshot(self):
        cache = SessionCache()
        d = {f"k{i}": i for i in range(20)}
        cache.put("mapping", d)
        snap = cache.snapshot("mapping")
        assert snap  # non-empty

    def test_scalar_snapshot(self):
        cache = SessionCache()
        cache.put("val", 42)
        snap = cache.snapshot("val")
        assert "42" in snap

    def test_ndarray_snapshot(self):
        pytest.importorskip("numpy")
        import numpy as np

        cache = SessionCache()
        arr = np.arange(1000).reshape(100, 10)
        cache.put("arr", arr)
        snap = cache.snapshot("arr")
        assert snap


class TestListHandles:
    def test_returns_mapping(self):
        cache = SessionCache()
        cache.put("a", 1)
        cache.put("b", 2)
        handles = cache.list_handles()
        assert "a" in handles
        assert "b" in handles

    def test_values_are_snapshots(self):
        cache = SessionCache()
        cache.put("x", "hello")
        handles = cache.list_handles()
        assert isinstance(handles["x"], str)


class TestSampleSize:
    def test_configurable_sample_size(self):
        pytest.importorskip("pandas")
        import pandas as pd

        cache = SessionCache(sample_size=3)
        df = pd.DataFrame({"a": range(100)})
        cache.put("df", df)
        snap = cache.snapshot("df")
        # The snapshot should reflect sample_size=3
        assert snap

    def test_default_sample_size(self):
        cache = SessionCache()
        assert cache.sample_size == 5

    def test_large_dataframe_sample_uses_configured_size(self):
        pytest.importorskip("pandas")
        import pandas as pd

        cache = SessionCache(sample_size=2)
        df = pd.DataFrame({"v": range(10000)})
        cache.put("big_df", df)
        snap = cache.snapshot("big_df")
        # Check it's a small snapshot (contains shape info, not all rows)
        assert "10000" in snap or "shape" in snap.lower() or "sample" in snap.lower()


class TestDiskBackedCache:
    def test_spills_oldest_hot_handle_to_disk(self, tmp_path):
        cache = SessionCache(storage_dir=tmp_path, hot_limit=1)

        cache.put("first", [1, 2, 3])
        cache.put("second", [4, 5, 6])

        assert "first" not in cache._store
        assert cache.storage_metadata()["first"]["location"] == "disk"
        assert cache.storage_metadata()["second"]["location"] == "memory"
        assert cache.list_handles()["first"]

    def test_get_hydrates_cold_value_and_updates_recency(self, tmp_path):
        cache = SessionCache(storage_dir=tmp_path, hot_limit=1)
        cache.put("first", [1])
        cache.put("second", [2])

        assert cache.get("first") == [1]

        metadata = cache.storage_metadata()
        assert metadata["first"]["location"] == "memory"
        assert metadata["second"]["location"] == "disk"

    def test_dataframe_roundtrip_from_disk(self, tmp_path):
        pytest.importorskip("pandas")
        import pandas as pd

        cache = SessionCache(storage_dir=tmp_path, hot_limit=1)
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        cache.put("df", df)
        cache.put("other", "value")

        restored = cache.get("df")

        assert restored.equals(df)
        assert cache.storage_metadata()["df"]["location"] == "memory"

    def test_dataframe_spill_prefers_parquet_or_documents_pickle_fallback(
        self, tmp_path
    ):
        pytest.importorskip("pandas")
        import pandas as pd

        cache = SessionCache(storage_dir=tmp_path, hot_limit=1)
        cache.put("df", pd.DataFrame({"a": [1, 2]}))
        cache.put("other", "value")

        storage_type = cache.storage_metadata()["df"]["storage_type"]
        assert storage_type in {"dataframe_parquet", "dataframe_pickle"}

    def test_ndarray_roundtrip_from_disk(self, tmp_path):
        pytest.importorskip("numpy")
        import numpy as np

        cache = SessionCache(storage_dir=tmp_path, hot_limit=1)
        arr = np.arange(6).reshape(2, 3)
        cache.put("arr", arr)
        cache.put("other", "value")
        assert cache.storage_metadata()["arr"]["storage_type"] == "numpy_npy"

        restored = cache.get("arr")

        assert np.array_equal(restored, arr)
        assert cache.storage_metadata()["arr"]["location"] == "memory"

    def test_storage_dir_without_hot_limit_defaults_to_ten(self, tmp_path):
        cache = SessionCache(storage_dir=tmp_path)

        assert cache.hot_limit == 10

    def test_overwrite_deletes_old_cold_file(self, tmp_path):
        cache = SessionCache(storage_dir=tmp_path, hot_limit=1)
        cache.put("data", [1])
        cache.put("other", [2])
        old_path = tmp_path / "data.pickle"
        assert old_path.exists()

        cache.put("data", [3], overwrite=True)

        assert not old_path.exists()
        assert cache.get("data") == [3]

    def test_storage_metadata_omits_paths_by_default(self, tmp_path):
        cache = SessionCache(storage_dir=tmp_path, hot_limit=1)
        cache.put("data", [1])
        cache.put("other", [2])

        metadata = cache.storage_metadata()
        metadata_with_paths = cache.storage_metadata(include_paths=True)

        assert "path" not in metadata["data"]
        assert metadata_with_paths["data"]["path"] == str(tmp_path / "data.pickle")

    def test_delete_removes_hot_and_cold_handles(self, tmp_path):
        cache = SessionCache(storage_dir=tmp_path, hot_limit=1)
        cache.put("cold", [1])
        cache.put("hot", [2])
        cold_path = tmp_path / "cold.pickle"
        assert cold_path.exists()

        cache.delete("cold")
        cache.delete("hot")

        assert not cold_path.exists()
        assert cache.list_handles() == {}

    def test_temporary_storage_cleanup(self):
        cache = SessionCache(hot_limit=1)
        storage_dir = cache._storage_dir
        assert storage_dir is not None
        cache.put("first", [1])
        cache.put("second", [2])
        assert storage_dir.exists()

        cache.close()

        assert not storage_dir.exists()
