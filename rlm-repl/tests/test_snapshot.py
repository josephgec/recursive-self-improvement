"""Tests for the SnapshotManager."""

import pytest
from src.memory.snapshot import SnapshotManager, REPLSnapshot


class TestSnapshotManager:
    """Test snapshot take/restore functionality."""

    def setup_method(self):
        self.mgr = SnapshotManager()

    def test_take_snapshot(self):
        namespace = {"x": 42, "y": "hello"}
        snap_id = self.mgr.take(namespace)
        assert snap_id is not None
        assert len(snap_id) > 0

    def test_restore_snapshot(self):
        namespace = {"x": 42, "y": "hello"}
        snap_id = self.mgr.take(namespace)

        restored = self.mgr.restore(snap_id)
        assert restored["x"] == 42
        assert restored["y"] == "hello"

    def test_restore_nonexistent(self):
        with pytest.raises(KeyError):
            self.mgr.restore("nonexistent")

    def test_rollback_verification(self):
        """Verify that modifying after snapshot doesn't affect the snapshot."""
        namespace = {"x": 42}
        snap_id = self.mgr.take(namespace)

        # Modify original
        namespace["x"] = 99
        namespace["y"] = "new"

        # Restore should have original values
        restored = self.mgr.restore(snap_id)
        assert restored["x"] == 42
        assert "y" not in restored

    def test_multiple_snapshots(self):
        snap1 = self.mgr.take({"x": 1})
        snap2 = self.mgr.take({"x": 2})
        snap3 = self.mgr.take({"x": 3})

        assert self.mgr.restore(snap1)["x"] == 1
        assert self.mgr.restore(snap2)["x"] == 2
        assert self.mgr.restore(snap3)["x"] == 3

    def test_size_bytes(self):
        namespace = {"x": 42, "data": "hello" * 100}
        snap_id = self.mgr.take(namespace)
        size = self.mgr.size_bytes(snap_id)
        assert size > 0

    def test_size_bytes_nonexistent(self):
        with pytest.raises(KeyError):
            self.mgr.size_bytes("nonexistent")

    def test_list_snapshots(self):
        snap1 = self.mgr.take({"x": 1})
        snap2 = self.mgr.take({"x": 2})
        snapshots = self.mgr.list_snapshots()
        assert snap1 in snapshots
        assert snap2 in snapshots

    def test_delete_snapshot(self):
        snap_id = self.mgr.take({"x": 1})
        self.mgr.delete(snap_id)
        with pytest.raises(KeyError):
            self.mgr.restore(snap_id)

    def test_delete_nonexistent(self):
        with pytest.raises(KeyError):
            self.mgr.delete("nonexistent")

    def test_skips_dunder_variables(self):
        namespace = {"x": 42, "__builtins__": {}, "__name__": "__main__"}
        snap_id = self.mgr.take(namespace)
        restored = self.mgr.restore(snap_id)
        assert "x" in restored
        assert "__builtins__" not in restored
        assert "__name__" not in restored

    def test_snapshot_with_metadata(self):
        snap_id = self.mgr.take({"x": 1}, metadata={"note": "test"})
        assert snap_id is not None

    def test_snapshot_dataclass(self):
        snap = REPLSnapshot(snapshot_id="test", timestamp=1.0)
        assert snap.snapshot_id == "test"
        assert snap.timestamp == 1.0
