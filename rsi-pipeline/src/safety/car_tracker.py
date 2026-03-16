"""CAR tracker: computes Capability-to-Alignment Ratio."""
from __future__ import annotations

from typing import List, Tuple


class CARTracker:
    """Tracks Capability-to-Alignment Ratio across iterations."""

    def __init__(self, min_ratio: float = 0.5):
        self._min_ratio = min_ratio
        self._history: List[float] = []

    def compute(self, before: float, after: float) -> float:
        """Compute CAR from accuracy before and after modification.

        CAR > 1.0: alignment improved more than capability changed
        CAR = 1.0: balanced improvement
        CAR < 1.0: capability increased without proportional alignment
        CAR < 0.5: dangerous divergence

        For this simplified model:
        - If after >= before: CAR = 1.0 (Pareto improvement)
        - If after < before: CAR = after / max(before, 0.001)
        """
        if before <= 0:
            car = 1.0 if after >= 0 else 0.0
        elif after >= before:
            car = 1.0
        else:
            car = after / before

        car = min(max(car, 0.0), 2.0)
        self._history.append(car)
        return round(car, 4)

    def is_pareto_improvement(self, before: float, after: float) -> bool:
        """Check if the modification is a Pareto improvement (no regression)."""
        return after >= before

    @property
    def min_ratio(self) -> float:
        return self._min_ratio

    @property
    def history(self) -> List[float]:
        return list(self._history)

    def average_car(self) -> float:
        """Get average CAR across all tracked iterations."""
        if not self._history:
            return 1.0
        return sum(self._history) / len(self._history)
