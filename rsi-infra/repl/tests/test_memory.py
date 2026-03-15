"""Tests for REPL memory persistence."""

from __future__ import annotations

import pytest

from repl.src.memory import REPLMemory


class TestRoundTrip:
    def test_basic_types(self) -> None:
        mem = REPLMemory()
        mem.save_variable("an_int", 42)
        mem.save_variable("a_str", "hello")
        mem.save_variable("a_list", [1, 2, 3])
        assert mem.load_variable("an_int") == 42
        assert mem.load_variable("a_str") == "hello"
        assert mem.load_variable("a_list") == [1, 2, 3]

    def test_numpy_array(self) -> None:
        np = pytest.importorskip("numpy")
        mem = REPLMemory()
        arr = np.array([1.0, 2.0, 3.0])
        mem.save_variable("arr", arr)
        loaded = mem.load_variable("arr")
        np.testing.assert_array_equal(loaded, arr)

    def test_pandas_dataframe(self) -> None:
        pd = pytest.importorskip("pandas")
        mem = REPLMemory()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        mem.save_variable("df", df)
        loaded = mem.load_variable("df")
        pd.testing.assert_frame_equal(loaded, df)

    def test_bulk_save_load(self) -> None:
        mem = REPLMemory()
        ns = {"x": 1, "y": "hello", "z": [1, 2]}
        mem.save(ns)
        loaded = mem.load()
        assert loaded == ns


class TestClone:
    def test_clone_independence(self) -> None:
        mem = REPLMemory()
        mem.save_variable("x", 42)
        clone = mem.clone()
        clone.save_variable("x", 99)
        assert mem.load_variable("x") == 42
        assert clone.load_variable("x") == 99

    def test_clone_has_same_data(self) -> None:
        mem = REPLMemory()
        mem.save_variable("a", [1, 2, 3])
        clone = mem.clone()
        assert clone.load_variable("a") == [1, 2, 3]


class TestSizeTracking:
    def test_size_increases(self) -> None:
        mem = REPLMemory()
        assert mem.size_bytes() == 0
        mem.save_variable("x", 42)
        assert mem.size_bytes() > 0

    def test_size_after_eviction(self) -> None:
        mem = REPLMemory()
        mem.save_variable("small", 1)
        mem.save_variable("big", list(range(10000)))
        size_before = mem.size_bytes()
        evicted = mem.evict_largest()
        assert evicted == "big"
        assert mem.size_bytes() < size_before


class TestEviction:
    def test_evict_largest(self) -> None:
        mem = REPLMemory()
        mem.save_variable("small", 1)
        mem.save_variable("big", list(range(10000)))
        evicted = mem.evict_largest()
        assert evicted == "big"
        assert "big" not in mem
        assert "small" in mem

    def test_evict_empty_returns_none(self) -> None:
        mem = REPLMemory()
        assert mem.evict_largest() is None

    def test_auto_eviction_on_save(self) -> None:
        # Create memory with very small budget
        mem = REPLMemory(max_size_bytes=200)
        mem.save_variable("a", list(range(100)))  # will be large
        # The variable should still be saved (it's the only one, we evict
        # others to make room, but if there is nothing to evict and the
        # single value exceeds the budget, it is still stored after
        # evicting everything else).
        assert "a" in mem

    def test_summary(self) -> None:
        mem = REPLMemory()
        mem.save_variable("x", 42)
        mem.save_variable("y", "hello")
        summary = mem.summary()
        assert "x" in summary
        assert "y" in summary
        assert all(isinstance(v, int) for v in summary.values())


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------

class TestLoadVariableMissing:
    def test_load_missing_raises_keyerror(self) -> None:
        mem = REPLMemory()
        with pytest.raises(KeyError, match="missing"):
            mem.load_variable("missing")


class TestContainsAndLen:
    def test_contains(self) -> None:
        mem = REPLMemory()
        assert "x" not in mem
        mem.save_variable("x", 42)
        assert "x" in mem

    def test_len(self) -> None:
        mem = REPLMemory()
        assert len(mem) == 0
        mem.save_variable("x", 42)
        assert len(mem) == 1
        mem.save_variable("y", "hello")
        assert len(mem) == 2


class TestSummaryWithTypes:
    """Test summary produces correct types for various data."""

    def test_summary_values_are_sizes(self) -> None:
        mem = REPLMemory()
        mem.save_variable("small", 1)
        mem.save_variable("medium", list(range(100)))
        mem.save_variable("string", "hello world")
        summary = mem.summary()
        assert len(summary) == 3
        # All values should be positive integers representing byte sizes
        for name, size in summary.items():
            assert isinstance(size, int)
            assert size > 0

    def test_summary_empty(self) -> None:
        mem = REPLMemory()
        summary = mem.summary()
        assert summary == {}


class TestSerialiseDeserialise:
    """Test serialisation/deserialisation edge cases."""

    def test_pickle_roundtrip(self) -> None:
        """Plain Python objects use pickle serialisation."""
        mem = REPLMemory()
        value = {"nested": [1, 2, {"a": True}]}
        mem.save_variable("data", value)
        loaded = mem.load_variable("data")
        assert loaded == value

    def test_numpy_detection(self) -> None:
        """Numpy arrays are detected by magic bytes on deserialise."""
        np = pytest.importorskip("numpy")
        mem = REPLMemory()
        arr = np.zeros((3, 3))
        mem.save_variable("arr", arr)
        # Check the raw bytes start with numpy magic
        raw = mem._store["arr"]
        assert raw[:6] == b"\x93NUMPY"
        loaded = mem.load_variable("arr")
        np.testing.assert_array_equal(loaded, arr)


class TestEvictionUnderPressure:
    """Test eviction when memory budget is tight."""

    def test_evict_to_fit_new_variable(self) -> None:
        """When a new variable doesn't fit, existing ones are evicted."""
        mem = REPLMemory(max_size_bytes=500)
        mem.save_variable("a", list(range(50)))
        mem.save_variable("b", list(range(50)))
        # Now save something that may require eviction
        mem.save_variable("c", list(range(200)))
        # At least c should be stored
        assert "c" in mem

    def test_evict_largest_correctly_identified(self) -> None:
        mem = REPLMemory()
        mem.save_variable("tiny", 1)
        mem.save_variable("huge", list(range(10000)))
        mem.save_variable("medium", list(range(100)))
        evicted = mem.evict_largest()
        assert evicted == "huge"


class TestSizeBytes:
    def test_size_bytes_accurate(self) -> None:
        mem = REPLMemory()
        mem.save_variable("x", 42)
        size1 = mem.size_bytes()
        mem.save_variable("y", list(range(1000)))
        size2 = mem.size_bytes()
        assert size2 > size1

    def test_total_bytes_empty(self) -> None:
        mem = REPLMemory()
        assert mem._total_bytes() == 0
