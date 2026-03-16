"""Tests for the ModalREPL backend."""

import pytest
from src.backends.modal_repl import ModalREPL


class TestModalREPL:
    """All methods should raise NotImplementedError."""

    def setup_method(self):
        self.repl = ModalREPL()

    def test_execute_raises(self):
        with pytest.raises(NotImplementedError):
            self.repl.execute("x = 1")

    def test_get_variable_raises(self):
        with pytest.raises(NotImplementedError):
            self.repl.get_variable("x")

    def test_set_variable_raises(self):
        with pytest.raises(NotImplementedError):
            self.repl.set_variable("x", 1)

    def test_list_variables_raises(self):
        with pytest.raises(NotImplementedError):
            self.repl.list_variables()

    def test_spawn_child_raises(self):
        with pytest.raises(NotImplementedError):
            self.repl.spawn_child()

    def test_snapshot_raises(self):
        with pytest.raises(NotImplementedError):
            self.repl.snapshot()

    def test_restore_raises(self):
        with pytest.raises(NotImplementedError):
            self.repl.restore("snap")

    def test_reset_raises(self):
        with pytest.raises(NotImplementedError):
            self.repl.reset()

    def test_shutdown_raises(self):
        with pytest.raises(NotImplementedError):
            self.repl.shutdown()

    def test_is_alive_raises(self):
        with pytest.raises(NotImplementedError):
            self.repl.is_alive()
