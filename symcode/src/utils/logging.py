"""Logging setup with rich formatting."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure structured logging with rich console output.

    Returns the root 'symcode' logger.
    """
    logger = logging.getLogger("symcode")
    logger.setLevel(level)

    if logger.handlers:
        return logger

    try:
        from rich.logging import RichHandler

        handler = RichHandler(
            level=level,
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        fmt = "%(message)s"
    except ImportError:
        handler = logging.StreamHandler(sys.stderr)
        fmt = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the symcode namespace."""
    return logging.getLogger(f"symcode.{name}")
