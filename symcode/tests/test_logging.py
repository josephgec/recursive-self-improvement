"""Tests for logging setup and configuration."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from src.utils.logging import get_logger, setup_logging


class TestSetupLogging:
    """Test setup_logging with and without rich."""

    def teardown_method(self):
        """Clean up loggers between tests."""
        logger = logging.getLogger("symcode")
        logger.handlers.clear()
        logger.setLevel(logging.WARNING)

    def test_setup_logging_returns_logger(self):
        logger = setup_logging()
        assert isinstance(logger, logging.Logger)
        assert logger.name == "symcode"

    def test_setup_logging_sets_level(self):
        logger = setup_logging(level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_setup_logging_default_info(self):
        logger = setup_logging()
        assert logger.level == logging.INFO

    def test_setup_logging_adds_handler(self):
        logger = setup_logging()
        assert len(logger.handlers) >= 1

    def test_setup_logging_idempotent(self):
        """Calling setup_logging twice should not duplicate handlers."""
        logger1 = setup_logging()
        handler_count = len(logger1.handlers)
        logger2 = setup_logging()
        assert logger2 is logger1
        assert len(logger2.handlers) == handler_count

    def test_setup_logging_without_rich(self):
        """Fallback to StreamHandler when rich is not available."""
        logger = logging.getLogger("symcode")
        logger.handlers.clear()

        with patch.dict("sys.modules", {"rich": None, "rich.logging": None}):
            # Need to force reimport of the handler creation
            import importlib
            import src.utils.logging as log_mod
            # Manually clear handlers to force re-creation
            logger.handlers.clear()
            result = log_mod.setup_logging()
            assert isinstance(result, logging.Logger)
            assert len(result.handlers) >= 1
            # Should have a StreamHandler (not RichHandler) as fallback
            handler = result.handlers[0]
            assert isinstance(handler, logging.StreamHandler)

    def test_setup_logging_with_rich_available(self):
        """When rich is available, should use RichHandler."""
        logger = logging.getLogger("symcode")
        logger.handlers.clear()

        try:
            from rich.logging import RichHandler
            rich_available = True
        except ImportError:
            rich_available = False

        # Call setup_logging fresh (handlers were cleared above)
        result = setup_logging()
        assert len(result.handlers) == 1
        if rich_available:
            assert type(result.handlers[0]).__name__ == "RichHandler"
        else:
            assert isinstance(result.handlers[0], logging.StreamHandler)


class TestGetLogger:
    """Test get_logger child logger creation."""

    def test_get_logger_returns_child(self):
        logger = get_logger("test_module")
        assert logger.name == "symcode.test_module"
        assert isinstance(logger, logging.Logger)

    def test_get_logger_different_names(self):
        logger1 = get_logger("module_a")
        logger2 = get_logger("module_b")
        assert logger1.name != logger2.name
        assert logger1.name == "symcode.module_a"
        assert logger2.name == "symcode.module_b"

    def test_get_logger_nested_name(self):
        logger = get_logger("analysis.retry")
        assert logger.name == "symcode.analysis.retry"
