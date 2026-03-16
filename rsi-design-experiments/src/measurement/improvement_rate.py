"""Tracks improvement rate over time."""

from typing import List, Optional


class ImprovementRateTracker:
    """Tracks accuracy improvement over iterations."""

    def __init__(self):
        self._trajectory: List[float] = []

    def record(self, accuracy: float):
        """Record an accuracy measurement."""
        self._trajectory.append(accuracy)

    def compute_rolling_delta(self, window: int = 5) -> float:
        """Compute the mean improvement rate over the last `window` steps."""
        if len(self._trajectory) < 2:
            return 0.0
        deltas = []
        start = max(0, len(self._trajectory) - window - 1)
        for i in range(start + 1, len(self._trajectory)):
            deltas.append(self._trajectory[i] - self._trajectory[i - 1])
        if not deltas:
            return 0.0
        return sum(deltas) / len(deltas)

    def detect_plateau(self, window: int = 5, threshold: float = 0.001) -> bool:
        """Detect if improvement has plateaued (mean delta below threshold)."""
        if len(self._trajectory) < window + 1:
            return False
        recent = self._trajectory[-window:]
        deltas = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
        mean_delta = sum(abs(d) for d in deltas) / len(deltas)
        return mean_delta < threshold

    def get_trajectory(self) -> List[float]:
        """Return the full accuracy trajectory."""
        return list(self._trajectory)

    def marginal_improvement(self) -> float:
        """Improvement from first to last recorded accuracy."""
        if len(self._trajectory) < 2:
            return 0.0
        return self._trajectory[-1] - self._trajectory[0]

    def get_improvement_at(self, index: int) -> Optional[float]:
        """Get improvement at a specific index relative to previous."""
        if index < 1 or index >= len(self._trajectory):
            return None
        return self._trajectory[index] - self._trajectory[index - 1]
