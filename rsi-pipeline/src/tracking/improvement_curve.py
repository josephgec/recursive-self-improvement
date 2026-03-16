"""Improvement curve tracker: tracks accuracy over iterations."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple


class ImprovementCurveTracker:
    """Tracks and analyzes the improvement curve over iterations."""

    def __init__(self, window_size: int = 10, plateau_tolerance: float = 0.005):
        self._curve: List[Tuple[int, float]] = []
        self._window_size = window_size
        self._plateau_tolerance = plateau_tolerance

    def record(self, accuracy: float, iteration: int) -> None:
        """Record an accuracy data point."""
        self._curve.append((iteration, accuracy))

    def compute_curve(self) -> List[Tuple[int, float]]:
        """Return the full improvement curve."""
        return list(self._curve)

    def detect_plateau(self) -> bool:
        """Detect if improvement has plateaued.

        Returns True if the last window_size points have change < tolerance.
        """
        if len(self._curve) < self._window_size:
            return False

        recent = [acc for _, acc in self._curve[-self._window_size:]]
        spread = max(recent) - min(recent)
        return spread < self._plateau_tolerance

    def detect_degradation(self) -> bool:
        """Detect if performance is degrading.

        Returns True if accuracy has been decreasing over the last window.
        """
        if len(self._curve) < 3:
            return False

        recent = [acc for _, acc in self._curve[-3:]]
        # Degradation: each point lower than the previous
        return all(recent[i] < recent[i - 1] for i in range(1, len(recent)))

    def marginal_returns(self) -> float:
        """Compute marginal returns (change in last step vs average change).

        Returns ratio. < 1.0 means diminishing returns.
        """
        if len(self._curve) < 2:
            return 1.0

        deltas = []
        for i in range(1, len(self._curve)):
            deltas.append(self._curve[i][1] - self._curve[i - 1][1])

        if not deltas:
            return 1.0

        avg_delta = sum(deltas) / len(deltas)
        last_delta = deltas[-1]

        if abs(avg_delta) < 1e-10:
            return 1.0 if abs(last_delta) < 1e-10 else 0.0

        return last_delta / avg_delta

    @property
    def latest_accuracy(self) -> Optional[float]:
        if not self._curve:
            return None
        return self._curve[-1][1]

    @property
    def size(self) -> int:
        return len(self._curve)
