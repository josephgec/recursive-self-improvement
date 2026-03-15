"""Track and analyze how performance changes with code complexity."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.utils.metrics import safe_division


@dataclass
class ComplexityDataPoint:
    """A single data point on the complexity-performance curve."""

    complexity: int  # e.g., AST node count
    accuracy: float
    comprehension_score: Optional[float] = None
    modification_success: Optional[bool] = None
    iteration: int = 0


class ComplexityCurveTracker:
    """Track performance as a function of code complexity."""

    def __init__(self) -> None:
        self._data: List[ComplexityDataPoint] = []

    def record(self, point: ComplexityDataPoint) -> None:
        """Record a data point."""
        self._data.append(point)

    @property
    def data(self) -> List[ComplexityDataPoint]:
        return list(self._data)

    def compute_success_curve(
        self, bins: int = 10
    ) -> List[Tuple[float, float]]:
        """Compute binned success rate vs complexity.

        Returns:
            List of (avg_complexity, success_rate) tuples.
        """
        if not self._data:
            return []

        complexities = [d.complexity for d in self._data]
        min_c, max_c = min(complexities), max(complexities)

        if min_c == max_c:
            rate = safe_division(
                sum(1 for d in self._data if d.accuracy >= 0.5),
                len(self._data),
            )
            return [(float(min_c), rate)]

        bin_size = (max_c - min_c) / bins
        curve = []

        for b in range(bins):
            low = min_c + b * bin_size
            high = low + bin_size
            points = [
                d for d in self._data
                if low <= d.complexity < high
                or (b == bins - 1 and d.complexity == high)
            ]
            if points:
                avg_c = sum(d.complexity for d in points) / len(points)
                rate = safe_division(
                    sum(1 for d in points if d.accuracy >= 0.5),
                    len(points),
                )
                curve.append((avg_c, rate))

        return curve

    def find_complexity_ceiling(self) -> Optional[float]:
        """Find the complexity level where performance drops below 50%.

        Uses logistic regression: P(success) = 1 / (1 + exp(a*(x - c)))
        where c is the ceiling.

        Returns:
            Estimated complexity ceiling, or None if insufficient data.
        """
        if len(self._data) < 5:
            return None

        curve = self.compute_success_curve(bins=min(20, len(self._data)))
        if len(curve) < 3:
            return None

        # Find where success rate crosses 0.5
        for i in range(len(curve) - 1):
            c1, r1 = curve[i]
            c2, r2 = curve[i + 1]
            if r1 >= 0.5 and r2 < 0.5:
                # Linear interpolation
                if r1 == r2:
                    return c1
                fraction = (0.5 - r1) / (r2 - r1)
                return c1 + fraction * (c2 - c1)

        # If always above 0.5, ceiling is beyond our data
        if all(r >= 0.5 for _, r in curve):
            return float(max(d.complexity for d in self._data))

        # If always below 0.5, ceiling is below our data
        if all(r < 0.5 for _, r in curve):
            return float(min(d.complexity for d in self._data))

        return None

    def compute_comprehension_curve(
        self, bins: int = 10
    ) -> List[Tuple[float, float]]:
        """Compute binned comprehension score vs complexity.

        Returns:
            List of (avg_complexity, avg_comprehension) tuples.
        """
        data_with_comp = [
            d for d in self._data if d.comprehension_score is not None
        ]
        if not data_with_comp:
            return []

        complexities = [d.complexity for d in data_with_comp]
        min_c, max_c = min(complexities), max(complexities)

        if min_c == max_c:
            avg_score = sum(d.comprehension_score for d in data_with_comp) / len(data_with_comp)  # type: ignore[arg-type]
            return [(float(min_c), avg_score)]

        bin_size = (max_c - min_c) / bins
        curve = []

        for b in range(bins):
            low = min_c + b * bin_size
            high = low + bin_size
            points = [
                d for d in data_with_comp
                if low <= d.complexity < high
                or (b == bins - 1 and d.complexity == high)
            ]
            if points:
                avg_c = sum(d.complexity for d in points) / len(points)
                avg_s = sum(d.comprehension_score for d in points) / len(points)  # type: ignore[arg-type]
                curve.append((avg_c, avg_s))

        return curve

    def test_correlation(self) -> Optional[float]:
        """Compute Pearson correlation between complexity and accuracy.

        Returns:
            Correlation coefficient, or None if insufficient data.
        """
        if len(self._data) < 3:
            return None

        x = np.array([d.complexity for d in self._data], dtype=float)
        y = np.array([d.accuracy for d in self._data], dtype=float)

        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0

        correlation = np.corrcoef(x, y)[0, 1]
        return float(correlation)
