"""RejectionAnalyzer: analyze patterns in gate rejections."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List


class RejectionAnalyzer:
    """Analyze rejection patterns from audit log entries."""

    def __init__(self, audit_entries: List[Dict[str, Any]]) -> None:
        self._entries = audit_entries

    def rejection_rate(self) -> float:
        """Compute overall rejection rate."""
        if not self._entries:
            return 0.0
        rejections = sum(1 for e in self._entries if not e.get("passed", True))
        return rejections / len(self._entries)

    def rejection_by_constraint(self) -> Dict[str, int]:
        """Count rejections grouped by violated constraint."""
        counts: Counter = Counter()
        for entry in self._entries:
            if not entry.get("passed", True):
                for violation in entry.get("violations", []):
                    counts[violation] += 1
        return dict(counts)

    def rejection_by_modification_type(self) -> Dict[str, Dict[str, int]]:
        """Count rejections and total checks grouped by modification type."""
        type_stats: Dict[str, Dict[str, int]] = {}
        for entry in self._entries:
            mod_type = entry.get("modification_type", "unknown")
            if mod_type not in type_stats:
                type_stats[mod_type] = {"total": 0, "rejected": 0}
            type_stats[mod_type]["total"] += 1
            if not entry.get("passed", True):
                type_stats[mod_type]["rejected"] += 1
        return type_stats
