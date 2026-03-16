"""Helper: peek — preview a slice of the context."""

from __future__ import annotations

from typing import Dict, Any, Optional


def peek(
    repl: Dict[str, Any],
    start: int = 0,
    length: int = 500,
) -> str:
    """Return a substring of CONTEXT starting at *start* of given *length*."""
    ctx = repl.get("CONTEXT", "")
    return ctx[start: start + length]


def make_peek(repl: Dict[str, Any]):
    """Return a peek function bound to *repl*."""

    def _peek(start: int = 0, length: int = 500) -> str:
        return peek(repl, start, length)

    return _peek
