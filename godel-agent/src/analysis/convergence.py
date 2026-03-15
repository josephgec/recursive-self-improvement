"""Convergence detection for agent performance."""

from __future__ import annotations

import numpy as np


class ConvergenceDetector:
    """Detects stagnation and divergence in performance."""

    def __init__(self, window: int = 5, stagnation_threshold: float = 0.01) -> None:
        self.window = window
        self.stagnation_threshold = stagnation_threshold

    def is_stagnant(self, accuracy_history: list[float]) -> bool:
        """Check if performance has stagnated (low variance in recent window)."""
        if len(accuracy_history) < self.window:
            return False

        recent = accuracy_history[-self.window:]
        return (max(recent) - min(recent)) < self.stagnation_threshold

    def is_diverging(self, accuracy_history: list[float], threshold: float = -0.02) -> bool:
        """Check if performance is diverging (negative trend)."""
        if len(accuracy_history) < self.window:
            return False

        recent = accuracy_history[-self.window:]
        x = np.arange(len(recent), dtype=float)
        y = np.array(recent, dtype=float)
        coeffs = np.polyfit(x, y, 1)
        return float(coeffs[0]) < threshold

    def compute_trend(self, accuracy_history: list[float], n: int | None = None) -> float:
        """Compute the linear trend over the last n data points."""
        if not accuracy_history:
            return 0.0

        n = n or self.window
        recent = accuracy_history[-n:]
        if len(recent) < 2:
            return 0.0

        x = np.arange(len(recent), dtype=float)
        y = np.array(recent, dtype=float)
        coeffs = np.polyfit(x, y, 1)
        return float(coeffs[0])

    def should_modify(self, accuracy_history: list[float]) -> bool:
        """Suggest whether self-modification is warranted."""
        if len(accuracy_history) < self.window:
            return False
        return self.is_stagnant(accuracy_history) or self.is_diverging(accuracy_history)
