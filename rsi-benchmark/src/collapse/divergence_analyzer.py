"""Divergence analyzer: compare RSI curves against collapse baselines."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DivergenceResult:
    """Result of divergence analysis between RSI and collapse."""
    rsi_values: List[float]
    collapse_values: List[float]
    divergence_values: List[float]
    mean_divergence: float
    max_divergence: float
    divergence_trend: str  # "increasing", "decreasing", "stable"
    collapse_prevention_score: float  # 0-1, higher is better
    metadata: Dict[str, Any] = field(default_factory=dict)


class DivergenceAnalyzer:
    """Analyze divergence between RSI improvement and collapse baselines."""

    def compute_divergence(
        self,
        rsi_curve: List[Tuple[int, float]],
        collapse_curve: List[float],
    ) -> DivergenceResult:
        """Compute divergence between RSI and collapse curves.

        Args:
            rsi_curve: List of (iteration, accuracy) tuples.
            collapse_curve: List of accuracy values from collapse baseline.
        """
        # Align curves to same length
        min_len = min(len(rsi_curve), len(collapse_curve))
        rsi_values = [v for _, v in rsi_curve[:min_len]]
        collapse_values = collapse_curve[:min_len]

        # Compute point-wise divergence
        divergence_values = [
            rsi - collapse for rsi, collapse in zip(rsi_values, collapse_values)
        ]

        mean_div = sum(divergence_values) / len(divergence_values) if divergence_values else 0.0
        max_div = max(divergence_values) if divergence_values else 0.0

        # Determine trend
        trend = self._compute_trend(divergence_values)

        # Prevention score: how much RSI avoids collapse
        prevention_score = self._compute_prevention_score(rsi_values, collapse_values)

        return DivergenceResult(
            rsi_values=rsi_values,
            collapse_values=collapse_values,
            divergence_values=divergence_values,
            mean_divergence=mean_div,
            max_divergence=max_div,
            divergence_trend=trend,
            collapse_prevention_score=prevention_score,
        )

    def _compute_trend(self, values: List[float]) -> str:
        """Determine if divergence is increasing, decreasing, or stable."""
        if len(values) < 2:
            return "stable"

        # Simple linear trend
        n = len(values)
        xs = list(range(n))
        x_mean = sum(xs) / n
        y_mean = sum(values) / n
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values))
        den = sum((x - x_mean) ** 2 for x in xs)

        if abs(den) < 1e-10:
            return "stable"

        slope = num / den
        if slope > 0.005:
            return "increasing"
        elif slope < -0.005:
            return "decreasing"
        return "stable"

    def _compute_prevention_score(
        self,
        rsi_values: List[float],
        collapse_values: List[float],
    ) -> float:
        """Compute collapse prevention score (0-1).

        Higher score means RSI better avoids collapse.
        """
        if not rsi_values or not collapse_values:
            return 0.0

        # Score based on how much RSI stays above collapse
        above_count = sum(
            1 for r, c in zip(rsi_values, collapse_values) if r >= c
        )
        above_fraction = above_count / len(rsi_values)

        # Bonus for increasing divergence
        if len(rsi_values) >= 2:
            start_gap = rsi_values[0] - collapse_values[0]
            end_gap = rsi_values[-1] - collapse_values[-1]
            gap_growth = max(0, end_gap - start_gap)
            gap_bonus = min(gap_growth, 0.3)
        else:
            gap_bonus = 0.0

        return min(1.0, above_fraction * 0.7 + gap_bonus)

    def compute_collapse_prevention_score(
        self,
        rsi_curve: List[Tuple[int, float]],
        collapse_curve: List[float],
    ) -> float:
        """Convenience method to compute just the prevention score."""
        result = self.compute_divergence(rsi_curve, collapse_curve)
        return result.collapse_prevention_score

    def plot_divergence(
        self,
        result: DivergenceResult,
    ) -> Dict[str, Any]:
        """Generate plot data for divergence visualization."""
        return {
            "rsi": result.rsi_values,
            "collapse": result.collapse_values,
            "divergence": result.divergence_values,
            "mean_divergence": result.mean_divergence,
            "prevention_score": result.collapse_prevention_score,
        }
