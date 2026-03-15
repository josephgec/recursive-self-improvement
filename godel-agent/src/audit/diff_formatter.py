"""Diff formatting utilities."""

from __future__ import annotations

import difflib
from typing import Any

from src.modification.diff_engine import CodeDiff


class DiffFormatter:
    """Formats code diffs for display and logging."""

    def format_unified(self, diff: CodeDiff, context: int = 3) -> str:
        """Format a diff in unified diff format."""
        if diff.unified_diff:
            return diff.unified_diff

        before_lines = diff.before.splitlines(keepends=True)
        after_lines = diff.after.splitlines(keepends=True)

        return "".join(difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile="before",
            tofile="after",
            n=context,
        ))

    def format_summary(self, diff: CodeDiff) -> str:
        """Format a short summary of the diff."""
        parts: list[str] = []
        parts.append(f"Lines added: {diff.lines_added}")
        parts.append(f"Lines removed: {diff.lines_removed}")
        parts.append(f"Lines changed: {diff.lines_changed}")
        return "\n".join(parts)

    def format_side_by_side(self, before: str, after: str, width: int = 80) -> str:
        """Format a side-by-side comparison."""
        before_lines = before.splitlines()
        after_lines = after.splitlines()

        half = width // 2 - 2
        lines: list[str] = []
        lines.append(f"{'BEFORE':<{half}} | {'AFTER':<{half}}")
        lines.append("-" * width)

        max_lines = max(len(before_lines), len(after_lines))
        for i in range(max_lines):
            left = before_lines[i] if i < len(before_lines) else ""
            right = after_lines[i] if i < len(after_lines) else ""
            lines.append(f"{left:<{half}} | {right:<{half}}")

        return "\n".join(lines)
