"""Convergence analyzer: detects convergence and estimates ceilings."""
from __future__ import annotations

from typing import List, Optional, Tuple


class ConvergenceAnalyzer:
    """Analyzes convergence behavior of the improvement process."""

    def __init__(self, window_size: int = 10, threshold: float = 0.01):
        self._window_size = window_size
        self._threshold = threshold

    def is_converged(self, curve: List[Tuple[int, float]]) -> bool:
        """Check if the improvement curve has converged.

        Converged means the last window_size points have
        total variation < threshold.
        """
        if len(curve) < self._window_size:
            return False

        recent = [acc for _, acc in curve[-self._window_size:]]
        variation = max(recent) - min(recent)
        return variation < self._threshold

    def estimate_ceiling(self, curve: List[Tuple[int, float]]) -> float:
        """Estimate the accuracy ceiling based on the curve trend.

        Uses the maximum observed accuracy plus diminishing trend extrapolation.
        """
        if not curve:
            return 0.0
        if len(curve) < 2:
            return curve[0][1]

        max_acc = max(acc for _, acc in curve)
        # Extrapolate from recent trend
        recent = [acc for _, acc in curve[-min(5, len(curve)):]]
        if len(recent) >= 2:
            trend = recent[-1] - recent[0]
            # Ceiling is max + diminishing projection
            ceiling = max_acc + max(trend * 0.5, 0)
        else:
            ceiling = max_acc

        return min(ceiling, 1.0)

    def marginal_returns(self, curve: List[Tuple[int, float]]) -> float:
        """Compute marginal returns ratio.

        Compares recent improvement rate to overall improvement rate.
        < 1.0 means diminishing returns.
        """
        if len(curve) < 3:
            return 1.0

        # Overall rate
        total_change = curve[-1][1] - curve[0][1]
        total_steps = len(curve) - 1
        overall_rate = total_change / total_steps if total_steps > 0 else 0

        # Recent rate (last 3 points)
        recent_change = curve[-1][1] - curve[-3][1]
        recent_rate = recent_change / 2  # 2 steps

        if abs(overall_rate) < 1e-10:
            return 1.0 if abs(recent_rate) < 1e-10 else 0.0

        return recent_rate / overall_rate
