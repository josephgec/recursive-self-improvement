"""Trend detection for GDI time series."""

from typing import List, Optional, Tuple


class TrendDetector:
    """Detects trends in GDI score history.

    Uses simple linear regression to determine if GDI scores are
    increasing, stable, or decreasing over time.
    """

    def __init__(self, min_points: int = 3, slope_threshold: float = 0.01):
        """Initialize trend detector.

        Args:
            min_points: Minimum data points needed for trend detection.
            slope_threshold: Slope magnitude below which trend is "stable".
        """
        self.min_points = min_points
        self.slope_threshold = slope_threshold

    def compute_slope(self, history: List[float]) -> float:
        """Compute the slope of the GDI score history via least squares.

        Args:
            history: List of GDI scores in chronological order.

        Returns:
            Slope of the linear fit (change per step).
        """
        n = len(history)
        if n < 2:
            return 0.0

        # Simple linear regression: y = mx + b
        x_mean = (n - 1) / 2.0
        y_mean = sum(history) / n

        numerator = 0.0
        denominator = 0.0
        for i, y in enumerate(history):
            numerator += (i - x_mean) * (y - y_mean)
            denominator += (i - x_mean) ** 2

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def detect_trend(self, history: List[float]) -> str:
        """Detect trend direction from GDI score history.

        Args:
            history: List of GDI scores in chronological order.

        Returns:
            "increasing", "stable", or "decreasing".
        """
        if len(history) < self.min_points:
            return "stable"

        slope = self.compute_slope(history)

        if slope > self.slope_threshold:
            return "increasing"
        elif slope < -self.slope_threshold:
            return "decreasing"
        else:
            return "stable"
