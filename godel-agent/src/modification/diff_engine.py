"""Diff computation for code and behavioral changes."""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CodeDiff:
    """Diff between two code versions."""

    before: str = ""
    after: str = ""
    unified_diff: str = ""
    lines_added: int = 0
    lines_removed: int = 0
    lines_changed: int = 0


@dataclass
class BehavioralDiff:
    """Diff in behavioral outcomes between two states."""

    accuracy_before: float = 0.0
    accuracy_after: float = 0.0
    accuracy_delta: float = 0.0
    tasks_improved: list[str] = field(default_factory=list)
    tasks_degraded: list[str] = field(default_factory=list)
    tasks_unchanged: list[str] = field(default_factory=list)


class DiffEngine:
    """Computes code and behavioral diffs."""

    def compute_diff(self, before: str, after: str, context: int = 3) -> CodeDiff:
        """Compute a unified diff between two code strings."""
        before_lines = before.splitlines(keepends=True)
        after_lines = after.splitlines(keepends=True)

        diff_lines = list(difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile="before",
            tofile="after",
            n=context,
        ))

        unified = "".join(diff_lines)
        added = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))

        return CodeDiff(
            before=before,
            after=after,
            unified_diff=unified,
            lines_added=added,
            lines_removed=removed,
            lines_changed=min(added, removed),
        )

    def compute_behavioral_diff(
        self,
        results_before: list[dict[str, Any]],
        results_after: list[dict[str, Any]],
    ) -> BehavioralDiff:
        """Compare behavioral outcomes before and after a change."""
        before_map = {r.get("task_id", str(i)): r.get("correct", False) for i, r in enumerate(results_before)}
        after_map = {r.get("task_id", str(i)): r.get("correct", False) for i, r in enumerate(results_after)}

        correct_before = sum(1 for v in before_map.values() if v)
        correct_after = sum(1 for v in after_map.values() if v)

        acc_before = correct_before / len(before_map) if before_map else 0.0
        acc_after = correct_after / len(after_map) if after_map else 0.0

        improved: list[str] = []
        degraded: list[str] = []
        unchanged: list[str] = []

        all_ids = set(list(before_map.keys()) + list(after_map.keys()))
        for tid in all_ids:
            b = before_map.get(tid, False)
            a = after_map.get(tid, False)
            if a and not b:
                improved.append(tid)
            elif b and not a:
                degraded.append(tid)
            else:
                unchanged.append(tid)

        return BehavioralDiff(
            accuracy_before=acc_before,
            accuracy_after=acc_after,
            accuracy_delta=acc_after - acc_before,
            tasks_improved=improved,
            tasks_degraded=degraded,
            tasks_unchanged=unchanged,
        )
