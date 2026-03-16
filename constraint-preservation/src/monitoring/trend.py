"""TrendDetector: detect trends and predict future violations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class TrendResult:
    """Result of trend analysis for a single constraint."""

    constraint_name: str
    slope: float
    current_headroom: float
    predicted_steps_to_violation: Optional[int]
    direction: str  # "improving", "stable", "degrading"
    warning: bool = False


class TrendDetector:
    """Detect trends in constraint headroom over time."""

    def __init__(self, window_size: int = 10) -> None:
        self._window_size = window_size
        self._history: Dict[str, List[float]] = {}

    def record(self, headrooms: Dict[str, float]) -> None:
        """Record a new set of headroom measurements."""
        for name, hr in headrooms.items():
            if name not in self._history:
                self._history[name] = []
            self._history[name].append(hr)
            # Keep only the window
            if len(self._history[name]) > self._window_size:
                self._history[name] = self._history[name][-self._window_size :]

    def compute_trends(self) -> Dict[str, TrendResult]:
        """Compute trend for each tracked constraint."""
        results: Dict[str, TrendResult] = {}
        for name, values in self._history.items():
            if len(values) < 2:
                results[name] = TrendResult(
                    constraint_name=name,
                    slope=0.0,
                    current_headroom=values[-1] if values else 0.0,
                    predicted_steps_to_violation=None,
                    direction="stable",
                )
                continue

            slope = self._linear_slope(values)
            current = values[-1]
            predicted = self._predict_violation(current, slope)

            if slope > 0.001:
                direction = "improving"
            elif slope < -0.001:
                direction = "degrading"
            else:
                direction = "stable"

            warning = predicted is not None and predicted <= 5

            results[name] = TrendResult(
                constraint_name=name,
                slope=slope,
                current_headroom=current,
                predicted_steps_to_violation=predicted,
                direction=direction,
                warning=warning,
            )

        return results

    def predict_violation(self, constraint_name: str) -> Optional[int]:
        """Predict how many steps until a specific constraint is violated."""
        values = self._history.get(constraint_name, [])
        if len(values) < 2:
            return None
        slope = self._linear_slope(values)
        return self._predict_violation(values[-1], slope)

    def early_warning(self) -> List[TrendResult]:
        """Return constraints that are predicted to violate within 5 steps."""
        trends = self.compute_trends()
        return [t for t in trends.values() if t.warning]

    @staticmethod
    def _linear_slope(values: List[float]) -> float:
        """Compute slope via simple linear regression."""
        n = len(values)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        if denominator == 0:
            return 0.0
        return numerator / denominator

    @staticmethod
    def _predict_violation(current_headroom: float, slope: float) -> Optional[int]:
        """Predict steps to violation using linear extrapolation.

        Returns None if slope >= 0 (improving or stable).
        """
        if slope >= 0:
            return None
        if current_headroom <= 0:
            return 0
        steps = int(current_headroom / abs(slope))
        return max(steps, 0)
