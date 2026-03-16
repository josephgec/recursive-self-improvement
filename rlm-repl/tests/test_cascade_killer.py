"""Tests for the CascadeKiller."""

import pytest
from src.safety.cascade_killer import CascadeKiller
from src.interface.errors import CascadeKillError


class TestCascadeKiller:
    """Test cascade killing of REPL trees."""

    def setup_method(self):
        self.killer = CascadeKiller()

    def test_register_root(self):
        self.killer.register("root")
        assert self.killer.is_alive("root")

    def test_register_child(self):
        self.killer.register("root")
        self.killer.register("child1", parent_id="root")
        assert self.killer.is_alive("child1")

    def test_kill_single(self):
        killed_list = []
        self.killer.register("root", kill_callback=lambda: killed_list.append("root"))
        killed = self.killer.kill("root")
        assert "root" in killed
        assert not self.killer.is_alive("root")

    def test_kill_with_descendants(self):
        killed_list = []
        self.killer.register("root", kill_callback=lambda: killed_list.append("root"))
        self.killer.register("child1", parent_id="root",
                             kill_callback=lambda: killed_list.append("child1"))
        self.killer.register("child2", parent_id="root",
                             kill_callback=lambda: killed_list.append("child2"))

        killed = self.killer.kill("root")
        assert "root" in killed
        assert "child1" in killed
        assert "child2" in killed
        assert len(killed_list) == 3

    def test_kill_subtree(self):
        self.killer.register("root")
        self.killer.register("child1", parent_id="root")
        self.killer.register("child2", parent_id="root")

        killed = self.killer.kill_subtree("root")
        assert "child1" in killed
        assert "child2" in killed
        assert self.killer.is_alive("root")  # root should still be alive

    def test_get_descendants(self):
        self.killer.register("root")
        self.killer.register("child1", parent_id="root")
        self.killer.register("grandchild1", parent_id="child1")

        descendants = self.killer.get_descendants("root")
        assert "child1" in descendants
        assert "grandchild1" in descendants

    def test_get_depth(self):
        self.killer.register("root")
        self.killer.register("child1", parent_id="root")
        self.killer.register("grandchild1", parent_id="child1")

        assert self.killer.get_depth("root") == 0
        assert self.killer.get_depth("child1") == 1
        assert self.killer.get_depth("grandchild1") == 2

    def test_kill_unknown_repl(self):
        with pytest.raises(CascadeKillError):
            self.killer.kill("nonexistent")

    def test_kill_already_dead(self):
        self.killer.register("root")
        self.killer.kill("root")
        # Killing again should return empty (already dead)
        killed = self.killer.kill("root")
        assert len(killed) == 0

    def test_deep_tree(self):
        self.killer.register("r")
        self.killer.register("c1", parent_id="r")
        self.killer.register("c2", parent_id="c1")
        self.killer.register("c3", parent_id="c2")

        killed = self.killer.kill("r")
        assert len(killed) == 4

    def test_unregister(self):
        self.killer.register("root")
        self.killer.register("child", parent_id="root")
        self.killer.unregister("child")
        descendants = self.killer.get_descendants("root")
        assert "child" not in descendants

    def test_is_alive_unknown(self):
        assert not self.killer.is_alive("nonexistent")

    def test_callback_exception_handled(self):
        """Callbacks that raise should not prevent killing."""
        def bad_callback():
            raise RuntimeError("callback error")

        self.killer.register("root", kill_callback=bad_callback)
        killed = self.killer.kill("root")
        assert "root" in killed
