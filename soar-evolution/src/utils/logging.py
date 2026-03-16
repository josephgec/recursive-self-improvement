"""Logging configuration and utilities."""

from __future__ import annotations

import logging
import sys
from typing import Optional


_configured = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> None:
    """Configure logging for the application."""
    global _configured

    if format_string is None:
        format_string = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    root_logger = logging.getLogger("soar")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(format_string))
    root_logger.addHandler(console)

    # File handler (optional)
    if log_file:
        try:
            from pathlib import Path
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(format_string))
            root_logger.addHandler(file_handler)
        except Exception:
            pass

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the given module name."""
    global _configured
    if not _configured:
        setup_logging(level="WARNING")
    return logging.getLogger(f"soar.{name}")
