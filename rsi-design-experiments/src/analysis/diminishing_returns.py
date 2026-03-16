"""Diminishing returns analysis using Kneedle-like algorithm."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class KneeResult:
    """Result of knee point detection."""

    knee_index: int
    knee_x: float
    knee_y: float
    marginal_gains: List[float]


class DiminishingReturnsAnalyzer:
    """Analyzes diminishing returns to find optimal points."""

    def find_knee(self, x: List[float], y: List[float]) -> Optional[KneeResult]:
        """Find the knee point using the Kneedle algorithm.

        The knee is where the curve transitions from steep to flat,
        indicating diminishing returns.

        Args:
            x: independent variable (e.g., depth levels)
            y: dependent variable (e.g., accuracy)

        Returns:
            KneeResult with the knee point information, or None if not found.
        """
        if len(x) < 3 or len(y) < 3:
            return None

        n = min(len(x), len(y))
        x = x[:n]
        y = y[:n]

        # Normalize to [0, 1]
        x_min, x_max = min(x), max(x)
        y_min, y_max = min(y), max(y)

        x_range = x_max - x_min
        y_range = y_max - y_min

        if x_range == 0 or y_range == 0:
            return None

        x_norm = [(xi - x_min) / x_range for xi in x]
        y_norm = [(yi - y_min) / y_range for yi in y]

        # Compute the difference between the curve and the diagonal line
        # from (x_norm[0], y_norm[0]) to (x_norm[-1], y_norm[-1])
        differences = []
        for i in range(n):
            # Expected y on the diagonal
            y_expected = y_norm[0] + (y_norm[-1] - y_norm[0]) * (
                (x_norm[i] - x_norm[0]) / (x_norm[-1] - x_norm[0])
                if x_norm[-1] != x_norm[0]
                else 0
            )
            diff = y_norm[i] - y_expected
            differences.append(diff)

        # The knee is where the difference is maximized (for concave curves)
        knee_idx = 0
        max_diff = differences[0]
        for i in range(1, n):
            if differences[i] > max_diff:
                max_diff = differences[i]
                knee_idx = i

        marginal_gains = self.compute_marginal_gains(x, y)

        return KneeResult(
            knee_index=knee_idx,
            knee_x=x[knee_idx],
            knee_y=y[knee_idx],
            marginal_gains=marginal_gains,
        )

    def compute_marginal_gains(self, x: List[float], y: List[float]) -> List[float]:
        """Compute marginal gain at each step (dy/dx)."""
        n = min(len(x), len(y))
        gains = []
        for i in range(1, n):
            dx = x[i] - x[i - 1]
            dy = y[i] - y[i - 1]
            gain = dy / dx if dx != 0 else 0.0
            gains.append(gain)
        return gains

    def optimal_depth(
        self,
        depths: List[int],
        accuracies: List[float],
        costs: List[float],
        cost_budget: Optional[float] = None,
    ) -> int:
        """Find the optimal depth balancing accuracy and cost.

        If cost_budget is given, returns the deepest depth within budget.
        Otherwise, uses the knee point of the accuracy-per-cost curve.
        """
        if not depths:
            return 0

        if cost_budget is not None:
            best_depth = depths[0]
            for d, c in zip(depths, costs):
                if c <= cost_budget:
                    best_depth = d
            return best_depth

        # Find knee in accuracy curve
        x = [float(d) for d in depths]
        knee = self.find_knee(x, accuracies)
        if knee is not None:
            return depths[knee.knee_index]
        return depths[0]
