"""Monitor divergence ratio history for anomaly detection."""

from typing import List, Optional
import numpy as np


class RatioMonitor:
    """Track ratio history and detect anomalous spikes."""

    def __init__(self, window_size: int = 20, z_score_threshold: float = 2.5):
        self.window_size = window_size
        self.z_score_threshold = z_score_threshold
        self.history: List[float] = []

    def record(self, ratio: float) -> None:
        """Record a new ratio value."""
        self.history.append(ratio)

    def get_history(self) -> List[float]:
        """Return full ratio history."""
        return list(self.history)

    def get_window(self, size: Optional[int] = None) -> List[float]:
        """Return the most recent window of ratios."""
        w = size or self.window_size
        return self.history[-w:]

    def get_mean(self) -> float:
        """Return mean of history."""
        if not self.history:
            return 0.0
        return float(np.mean(self.history))

    def get_std(self) -> float:
        """Return std of history."""
        if len(self.history) < 2:
            return 0.0
        return float(np.std(self.history))

    def compute_z_score(self, value: Optional[float] = None) -> float:
        """Compute z-score for a value (default: latest) against history."""
        if len(self.history) < 2:
            return 0.0

        if value is None:
            value = self.history[-1]
            # Use all except latest for baseline
            baseline = self.history[:-1]
        else:
            baseline = self.history

        mean = float(np.mean(baseline))
        std = float(np.std(baseline))
        if std < 1e-10:
            return 0.0 if abs(value - mean) < 1e-10 else float('inf')

        return float((value - mean) / std)

    def is_anomalous(self, value: Optional[float] = None) -> bool:
        """Check if a value is anomalous based on z-score."""
        z = self.compute_z_score(value)
        return abs(z) > self.z_score_threshold

    def detect_spike(self) -> bool:
        """Detect if the latest value is a spike."""
        if len(self.history) < 3:
            return False
        return self.is_anomalous()

    def get_trend(self, window: Optional[int] = None) -> str:
        """Get trend direction over recent window."""
        w = window or self.window_size
        recent = self.history[-w:]
        if len(recent) < 3:
            return "stable"

        x = np.arange(len(recent), dtype=float)
        y = np.array(recent)
        mean_x, mean_y = x.mean(), y.mean()
        num = np.sum((x - mean_x) * (y - mean_y))
        den = np.sum((x - mean_x)**2)
        if abs(den) < 1e-10:
            return "stable"

        slope = num / den
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        return "stable"
