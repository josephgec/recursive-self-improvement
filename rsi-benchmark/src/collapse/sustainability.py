"""Sustainability analyzer: analyze improvement sustainability and resilience."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SustainabilityReport:
    """Report on sustainability of improvement."""
    longest_improvement_streak: int
    recovery_after_dip_count: int
    monotonicity_score: float  # 0-1, 1 = perfectly monotonic improvement
    average_improvement_per_step: float
    max_drawdown: float
    resilience_score: float  # 0-1, how well it recovers from dips
    overall_sustainability_score: float  # 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)


class SustainabilityAnalyzer:
    """Analyze sustainability of RSI improvement trajectories."""

    def analyze(self, accuracy_curve: List[float]) -> SustainabilityReport:
        """Run full sustainability analysis on an accuracy curve."""
        streak = self.longest_improvement_streak(accuracy_curve)
        recovery = self.recovery_after_dip(accuracy_curve)
        monotonicity = self.monotonicity_score(accuracy_curve)
        avg_improvement = self._average_improvement(accuracy_curve)
        drawdown = self._max_drawdown(accuracy_curve)
        resilience = self._resilience_score(accuracy_curve)

        # Overall score: weighted combination
        overall = (
            0.3 * monotonicity
            + 0.3 * resilience
            + 0.2 * min(streak / max(len(accuracy_curve) - 1, 1), 1.0)
            + 0.2 * (1.0 - min(drawdown * 5, 1.0))
        )

        return SustainabilityReport(
            longest_improvement_streak=streak,
            recovery_after_dip_count=recovery,
            monotonicity_score=monotonicity,
            average_improvement_per_step=avg_improvement,
            max_drawdown=drawdown,
            resilience_score=resilience,
            overall_sustainability_score=max(0.0, overall),
        )

    def longest_improvement_streak(self, accuracy_curve: List[float]) -> int:
        """Find the longest consecutive improvement streak."""
        if len(accuracy_curve) < 2:
            return 0
        max_streak = 0
        current = 0
        for i in range(1, len(accuracy_curve)):
            if accuracy_curve[i] > accuracy_curve[i - 1]:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak

    def recovery_after_dip(self, accuracy_curve: List[float]) -> int:
        """Count how many times accuracy recovered after a dip."""
        if len(accuracy_curve) < 3:
            return 0
        recoveries = 0
        for i in range(2, len(accuracy_curve)):
            if (
                accuracy_curve[i - 1] < accuracy_curve[i - 2]
                and accuracy_curve[i] > accuracy_curve[i - 1]
            ):
                recoveries += 1
        return recoveries

    def monotonicity_score(self, accuracy_curve: List[float]) -> float:
        """Compute monotonicity: fraction of steps that are improvements."""
        if len(accuracy_curve) < 2:
            return 0.0
        improvements = sum(
            1 for i in range(1, len(accuracy_curve))
            if accuracy_curve[i] >= accuracy_curve[i - 1]
        )
        return improvements / (len(accuracy_curve) - 1)

    def _average_improvement(self, accuracy_curve: List[float]) -> float:
        """Compute average improvement per step."""
        if len(accuracy_curve) < 2:
            return 0.0
        total = accuracy_curve[-1] - accuracy_curve[0]
        return total / (len(accuracy_curve) - 1)

    def _max_drawdown(self, accuracy_curve: List[float]) -> float:
        """Compute maximum drawdown from peak."""
        if len(accuracy_curve) < 2:
            return 0.0
        peak = accuracy_curve[0]
        max_dd = 0.0
        for val in accuracy_curve:
            if val > peak:
                peak = val
            dd = peak - val
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def _resilience_score(self, accuracy_curve: List[float]) -> float:
        """Score how well the system recovers from dips."""
        if len(accuracy_curve) < 3:
            return 0.5

        dips = 0
        recoveries = 0
        for i in range(1, len(accuracy_curve)):
            if accuracy_curve[i] < accuracy_curve[i - 1]:
                dips += 1
                # Check if it recovers within next 2 steps
                for j in range(i + 1, min(i + 3, len(accuracy_curve))):
                    if accuracy_curve[j] >= accuracy_curve[i - 1]:
                        recoveries += 1
                        break
        if dips == 0:
            return 1.0
        return recoveries / dips
