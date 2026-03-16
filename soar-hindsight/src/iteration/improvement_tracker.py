"""Track solve rate and other metrics across SOAR iterations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class ImprovementTracker:
    """Track metric values across iterations to monitor improvement."""

    def __init__(self) -> None:
        self._records: List[Tuple[int, float]] = []

    def record(self, iteration: int, value: float) -> None:
        """Record a metric value for an iteration."""
        self._records.append((iteration, value))

    @property
    def history(self) -> List[Tuple[int, float]]:
        return list(self._records)

    @property
    def values(self) -> List[float]:
        return [v for _, v in self._records]

    @property
    def latest_value(self) -> Optional[float]:
        if not self._records:
            return None
        return self._records[-1][1]

    @property
    def best_value(self) -> Optional[float]:
        if not self._records:
            return None
        return max(v for _, v in self._records)

    @property
    def best_iteration(self) -> Optional[int]:
        if not self._records:
            return None
        return max(self._records, key=lambda x: x[1])[0]

    def improvement_from_start(self) -> float:
        """Total improvement from first to last recorded value."""
        if len(self._records) < 2:
            return 0.0
        return self._records[-1][1] - self._records[0][1]

    def recent_improvement(self, window: int = 3) -> float:
        """Average improvement per iteration over the last `window` iterations."""
        if len(self._records) < 2:
            return 0.0
        recent = self._records[-window:]
        if len(recent) < 2:
            return 0.0
        return (recent[-1][1] - recent[0][1]) / (len(recent) - 1)

    def is_improving(self, window: int = 3, threshold: float = 0.01) -> bool:
        """Check if the metric is still improving."""
        return self.recent_improvement(window) > threshold

    def summary(self) -> Dict[str, Any]:
        """Return summary of tracked improvements."""
        return {
            "n_records": len(self._records),
            "latest_value": self.latest_value,
            "best_value": self.best_value,
            "best_iteration": self.best_iteration,
            "total_improvement": round(self.improvement_from_start(), 4),
        }

    def clear(self) -> None:
        self._records.clear()
