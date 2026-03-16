"""Tests for the VariableStore."""

import pytest
from src.memory.variable_store import VariableStore, VariableDiff


class TestVariableStoreCRUD:
    """Test basic CRUD operations."""

    def setup_method(self):
        self.store = VariableStore()

    def test_set_and_get(self):
        self.store.set("x", 42)
        assert self.store.get("x") == 42

    def test_get_nonexistent(self):
        with pytest.raises(KeyError):
            self.store.get("nonexistent")

    def test_delete(self):
        self.store.set("x", 42)
        self.store.delete("x")
        with pytest.raises(KeyError):
            self.store.get("x")

    def test_delete_nonexistent(self):
        with pytest.raises(KeyError):
            self.store.delete("nonexistent")

    def test_list_all(self):
        self.store.set("a", 1)
        self.store.set("b", 2)
        self.store.set("c", 3)
        names = self.store.list_all()
        assert names == ["a", "b", "c"]

    def test_list_excludes_internals(self):
        self.store.set("__builtins__", {})
        self.store.set("FINAL", lambda: None)
        self.store.set("x", 42)
        names = self.store.list_all()
        assert "x" in names
        assert "__builtins__" not in names
        assert "FINAL" not in names

    def test_list_excludes_underscore_prefix(self):
        self.store.set("_internal", 1)
        self.store.set("public", 2)
        names = self.store.list_all()
        assert "public" in names
        assert "_internal" not in names

    def test_clear(self):
        self.store.set("x", 1)
        self.store.set("y", 2)
        self.store.clear()
        assert self.store.list_all() == []

    def test_overwrite(self):
        self.store.set("x", 1)
        self.store.set("x", 2)
        assert self.store.get("x") == 2

    def test_various_types(self):
        self.store.set("int_val", 42)
        self.store.set("str_val", "hello")
        self.store.set("list_val", [1, 2, 3])
        self.store.set("dict_val", {"a": 1})
        assert self.store.get("int_val") == 42
        assert self.store.get("str_val") == "hello"
        assert self.store.get("list_val") == [1, 2, 3]
        assert self.store.get("dict_val") == {"a": 1}


class TestVariableStoreDiff:
    """Test diff tracking."""

    def setup_method(self):
        self.store = VariableStore()

    def test_diff_added(self):
        previous = {}
        self.store.set("x", 42)
        diff = self.store.diff(previous)
        assert "x" in diff.added
        assert diff.has_changes

    def test_diff_modified(self):
        previous = {"x": 1}
        self.store.set("x", 2)
        diff = self.store.diff(previous)
        assert "x" in diff.modified
        assert diff.has_changes

    def test_diff_removed(self):
        previous = {"x": 1}
        diff = self.store.diff(previous)
        assert "x" in diff.removed
        assert diff.has_changes

    def test_diff_no_changes(self):
        self.store.set("x", 42)
        previous = {"x": 42}
        diff = self.store.diff(previous)
        assert not diff.has_changes

    def test_diff_changed_property(self):
        previous = {}
        self.store.set("a", 1)
        self.store.set("b", 2)
        diff = self.store.diff(previous)
        assert set(diff.changed) == {"a", "b"}


class TestVariableStoreSize:
    """Test size tracking."""

    def setup_method(self):
        self.store = VariableStore()

    def test_total_size_empty(self):
        assert self.store.total_size_bytes() == 0

    def test_total_size_with_data(self):
        self.store.set("x", 42)
        self.store.set("y", "hello" * 100)
        size = self.store.total_size_bytes()
        assert size > 0

    def test_size_grows_with_data(self):
        self.store.set("x", 1)
        size1 = self.store.total_size_bytes()
        self.store.set("y", "a" * 10000)
        size2 = self.store.total_size_bytes()
        assert size2 > size1

    def test_namespace_operations(self):
        self.store.set("x", 1)
        ns = self.store.get_namespace()
        assert "x" in ns

        self.store.set_namespace({"a": 10, "b": 20})
        assert self.store.get("a") == 10
        assert self.store.get("b") == 20

    def test_snapshot(self):
        self.store.set("x", 42)
        snap = self.store.snapshot()
        self.store.set("x", 99)
        assert snap["x"] == 42
        assert self.store.get("x") == 99
