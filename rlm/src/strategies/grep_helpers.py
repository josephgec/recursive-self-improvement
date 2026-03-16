"""Helper: grep — search context lines for a pattern."""

from __future__ import annotations

import re
from typing import Any, Dict, List


def grep(
    repl: Dict[str, Any],
    pattern: str,
    context_lines: int = 0,
    max_results: int = 50,
) -> List[str]:
    """Search CONTEXT line-by-line for *pattern* (regex).

    Returns matching lines (with optional surrounding context).
    """
    ctx: str = repl.get("CONTEXT", "")
    lines = ctx.split("\n")
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error:
        # Fall back to literal match
        regex = re.compile(re.escape(pattern), re.IGNORECASE)

    results: List[str] = []
    for i, line in enumerate(lines):
        if regex.search(line):
            if context_lines > 0:
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                snippet = "\n".join(f"{j + 1}: {lines[j]}" for j in range(start, end))
                results.append(snippet)
            else:
                results.append(f"{i + 1}: {line}")
            if len(results) >= max_results:
                break
    return results


def search(
    repl: Dict[str, Any],
    query: str,
    max_results: int = 20,
) -> List[str]:
    """Simple keyword search in CONTEXT — finds lines containing *query*."""
    ctx: str = repl.get("CONTEXT", "")
    lines = ctx.split("\n")
    results: List[str] = []
    q_lower = query.lower()
    for i, line in enumerate(lines):
        if q_lower in line.lower():
            results.append(f"{i + 1}: {line}")
            if len(results) >= max_results:
                break
    return results


def make_grep(repl: Dict[str, Any]):
    """Return grep/search functions bound to *repl*."""

    def _grep(pattern: str, context_lines: int = 0, max_results: int = 50) -> List[str]:
        return grep(repl, pattern, context_lines, max_results)

    def _search(query: str, max_results: int = 20) -> List[str]:
        return search(repl, query, max_results)

    return _grep, _search
