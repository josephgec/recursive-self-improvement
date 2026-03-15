"""Tests for ModalREPL stub — all methods should raise NotImplementedError."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestModalREPLWithoutModal:
    """When modal is not installed, __init__ raises NotImplementedError."""

    def test_init_raises_without_modal(self) -> None:
        """ModalREPL raises NotImplementedError if modal is not installed."""
        # Ensure modal import fails inside _check_modal
        with patch.dict(sys.modules, {"modal": None}):
            from repl.src.modal_repl import ModalREPL
            with pytest.raises(NotImplementedError, match="Modal is not installed"):
                ModalREPL()


class TestModalREPLWithModal:
    """When modal IS installed, __init__ still raises NotImplementedError (stub)."""

    def test_init_raises_stub_message(self) -> None:
        """ModalREPL raises NotImplementedError with a stub message."""
        mock_modal = MagicMock()
        with patch.dict(sys.modules, {"modal": mock_modal}):
            from repl.src.modal_repl import ModalREPL
            with pytest.raises(NotImplementedError, match="stub"):
                ModalREPL()


class TestModalREPLMethods:
    """All ModalREPL methods raise NotImplementedError.

    We bypass __init__ to test individual methods by creating the object
    via __new__ and manually setting the minimum required state.
    """

    @pytest.fixture
    def modal_repl(self):
        """Create a ModalREPL bypassing __init__."""
        from repl.src.modal_repl import ModalREPL
        from repl.src.sandbox import REPLConfig
        instance = object.__new__(ModalREPL)
        instance._config = REPLConfig()
        instance._depth = 0
        return instance

    def test_execute_raises(self, modal_repl) -> None:
        with pytest.raises(NotImplementedError, match="execute"):
            modal_repl.execute("x = 1")

    def test_get_variable_raises(self, modal_repl) -> None:
        with pytest.raises(NotImplementedError, match="get_variable"):
            modal_repl.get_variable("x")

    def test_set_variable_raises(self, modal_repl) -> None:
        with pytest.raises(NotImplementedError, match="set_variable"):
            modal_repl.set_variable("x", 42)

    def test_list_variables_raises(self, modal_repl) -> None:
        with pytest.raises(NotImplementedError, match="list_variables"):
            modal_repl.list_variables()

    def test_spawn_child_raises(self, modal_repl) -> None:
        with pytest.raises(NotImplementedError, match="spawn_child"):
            modal_repl.spawn_child()

    def test_reset_raises(self, modal_repl) -> None:
        with pytest.raises(NotImplementedError, match="reset"):
            modal_repl.reset()

    def test_shutdown_raises(self, modal_repl) -> None:
        with pytest.raises(NotImplementedError, match="shutdown"):
            modal_repl.shutdown()
