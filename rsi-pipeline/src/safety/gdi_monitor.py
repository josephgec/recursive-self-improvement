"""GDI monitor: computes Generalized Drift Index between current and original agent."""
from __future__ import annotations

from typing import Optional


class GDIMonitor:
    """Monitors Generalized Drift Index — how far the agent has drifted from original."""

    def __init__(self, threshold: float = 0.3):
        self._threshold = threshold
        self._history: list = []

    def compute(self, current: str, original: str) -> float:
        """Compute GDI between current and original code.

        Uses normalized edit distance as a proxy for drift.
        Returns a value between 0.0 (identical) and 1.0 (completely different).
        """
        if not original and not current:
            return 0.0
        if not original or not current:
            return 1.0

        # Normalized Levenshtein-like distance using set-based token comparison
        current_tokens = set(current.split())
        original_tokens = set(original.split())

        if not current_tokens and not original_tokens:
            return 0.0

        union = current_tokens | original_tokens
        if not union:
            return 0.0

        intersection = current_tokens & original_tokens
        jaccard_distance = 1.0 - len(intersection) / len(union)

        # Also consider length ratio
        len_ratio = abs(len(current) - len(original)) / max(len(current), len(original), 1)

        gdi = 0.7 * jaccard_distance + 0.3 * len_ratio
        gdi = min(max(gdi, 0.0), 1.0)

        self._history.append(gdi)
        return round(gdi, 4)

    def check_threshold(self, gdi: float) -> bool:
        """Check if GDI exceeds the threshold. Returns True if violation."""
        return gdi > self._threshold

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def history(self) -> list:
        return list(self._history)
