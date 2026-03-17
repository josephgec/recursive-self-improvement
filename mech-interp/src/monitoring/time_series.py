"""Time series recording for interpretability metrics."""

from typing import Dict, List, Optional, Any
import numpy as np


class InterpretabilityTimeSeries:
    """Record and query interpretability metrics over time."""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._data: List[Dict[str, Any]] = []
        self._iterations: List[int] = []

    def record(self, iteration: int, metrics: Dict[str, float]) -> None:
        """Record metrics for an iteration."""
        self._iterations.append(iteration)
        self._data.append(dict(metrics))

        # Trim if needed
        if len(self._data) > self.max_history:
            self._data = self._data[-self.max_history:]
            self._iterations = self._iterations[-self.max_history:]

    def get_history(self, metric_name: Optional[str] = None) -> List[Dict]:
        """Get full history, optionally filtered to a specific metric."""
        if metric_name is None:
            return [
                {"iteration": it, **data}
                for it, data in zip(self._iterations, self._data)
            ]
        result = []
        for it, data in zip(self._iterations, self._data):
            if metric_name in data:
                result.append({"iteration": it, metric_name: data[metric_name]})
        return result

    def get_window(self, size: int, metric_name: Optional[str] = None) -> List[Dict]:
        """Get the most recent window of data."""
        full = self.get_history(metric_name)
        return full[-size:]

    def get_metric_values(self, metric_name: str) -> List[float]:
        """Get all values for a specific metric."""
        return [
            d[metric_name] for d in self._data
            if metric_name in d
        ]

    def get_metric_stats(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a specific metric."""
        values = self.get_metric_values(metric_name)
        if not values:
            return {"count": 0, "mean": 0, "std": 0, "min": 0, "max": 0}
        arr = np.array(values)
        return {
            "count": len(values),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    def __len__(self) -> int:
        return len(self._data)
