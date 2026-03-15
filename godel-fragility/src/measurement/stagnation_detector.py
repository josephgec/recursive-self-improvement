"""Detect stagnation and oscillation in agent performance."""

from __future__ import annotations

from typing import List, Optional, Sequence

from src.utils.metrics import detect_trend, moving_average


class StagnationDetector:
    """Detect patterns in performance history: stagnation, oscillation, improvement."""

    def __init__(
        self,
        stagnation_threshold: float = 0.01,
        oscillation_threshold: float = 0.6,
        window: int = 5,
    ) -> None:
        """
        Args:
            stagnation_threshold: Max variance to consider "flat".
            oscillation_threshold: Min fraction of sign changes to consider oscillating.
            window: Moving average window size.
        """
        self._threshold = stagnation_threshold
        self._osc_threshold = oscillation_threshold
        self._window = window

    def is_stagnant(self, values: Sequence[float]) -> bool:
        """Check if performance is flat (not changing)."""
        if len(values) < self._window:
            return False

        recent = values[-self._window:]
        spread = max(recent) - min(recent)
        return spread <= self._threshold

    def is_oscillating(self, values: Sequence[float]) -> bool:
        """Check if performance is alternating up/down."""
        if len(values) < 4:
            return False

        diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        sign_changes = sum(
            1 for i in range(len(diffs) - 1)
            if (diffs[i] > 0) != (diffs[i + 1] > 0) and abs(diffs[i]) > 0.001
        )
        total = max(len(diffs) - 1, 1)
        return (sign_changes / total) >= self._osc_threshold

    def is_improving(self, values: Sequence[float]) -> bool:
        """Check if performance is trending upward."""
        if len(values) < 3:
            return False
        trend = detect_trend(values, self._window)
        return trend == "improving"

    def get_trend(self, values: Sequence[float]) -> str:
        """Get the overall trend.

        Returns:
            One of: 'improving', 'declining', 'flat', 'oscillating', 'insufficient_data'.
        """
        if len(values) < 3:
            return "insufficient_data"

        if self.is_oscillating(values):
            return "oscillating"

        return detect_trend(values, self._window)
