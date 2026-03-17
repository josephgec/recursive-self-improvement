"""Complexity budget for self-modifications.

Ensures modifications do not increase code complexity beyond safe bounds.
Max 5x ratio over baseline, max 50 cyclomatic complexity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class BudgetStatus:
    """Status of the complexity budget."""
    baseline_complexity: float
    current_complexity: float
    ratio: float
    max_ratio: float
    max_cyclomatic: int
    current_cyclomatic: int
    within_budget: bool
    remaining_ratio: float
    remaining_cyclomatic: int

    @property
    def budget_utilization(self) -> float:
        """Fraction of budget used (0 to 1+)."""
        ratio_util = self.ratio / self.max_ratio if self.max_ratio > 0 else 0.0
        cyclo_util = (
            self.current_cyclomatic / self.max_cyclomatic
            if self.max_cyclomatic > 0
            else 0.0
        )
        return max(ratio_util, cyclo_util)


class ComplexityBudget:
    """Manages complexity budget for code modifications.

    Constraints:
    - Max 5x ratio over baseline complexity
    - Max 50 cyclomatic complexity
    """

    def __init__(
        self,
        max_ratio: float = 5.0,
        max_cyclomatic: int = 50,
    ):
        self.max_ratio = max_ratio
        self.max_cyclomatic = max_cyclomatic
        self._baseline: Optional[float] = None
        self._current: float = 0.0
        self._current_cyclomatic: int = 0

    def set_baseline(self, complexity: float, cyclomatic: int = 0) -> None:
        """Set the baseline complexity measurement.

        Args:
            complexity: Baseline complexity score.
            cyclomatic: Baseline cyclomatic complexity.
        """
        if complexity < 0:
            raise ValueError("Complexity must be non-negative")
        self._baseline = complexity
        self._current = complexity
        self._current_cyclomatic = cyclomatic

    def check(self, code_metrics: Dict[str, Any]) -> BudgetStatus:
        """Check if code metrics are within budget.

        Args:
            code_metrics: Dict with 'complexity' and/or 'cyclomatic' keys.

        Returns:
            BudgetStatus with current standing.
        """
        if self._baseline is None:
            raise RuntimeError("Baseline not set. Call set_baseline() first.")

        complexity = code_metrics.get("complexity", self._current)
        cyclomatic = code_metrics.get("cyclomatic", self._current_cyclomatic)

        self._current = complexity
        self._current_cyclomatic = cyclomatic

        ratio = complexity / self._baseline if self._baseline > 0 else 0.0
        within_budget = ratio <= self.max_ratio and cyclomatic <= self.max_cyclomatic

        return BudgetStatus(
            baseline_complexity=self._baseline,
            current_complexity=complexity,
            ratio=ratio,
            max_ratio=self.max_ratio,
            max_cyclomatic=self.max_cyclomatic,
            current_cyclomatic=cyclomatic,
            within_budget=within_budget,
            remaining_ratio=max(0.0, self.max_ratio - ratio),
            remaining_cyclomatic=max(0, self.max_cyclomatic - cyclomatic),
        )

    def would_exceed(self, proposed_metrics: Dict[str, Any]) -> bool:
        """Check if proposed metrics would exceed budget.

        Args:
            proposed_metrics: Dict with 'complexity' and/or 'cyclomatic'.

        Returns:
            True if the proposed change would exceed budget.
        """
        if self._baseline is None:
            raise RuntimeError("Baseline not set. Call set_baseline() first.")

        complexity = proposed_metrics.get("complexity", self._current)
        cyclomatic = proposed_metrics.get("cyclomatic", self._current_cyclomatic)

        ratio = complexity / self._baseline if self._baseline > 0 else 0.0
        return ratio > self.max_ratio or cyclomatic > self.max_cyclomatic

    def remaining_budget(self) -> Dict[str, float]:
        """Return remaining budget headroom.

        Returns:
            Dict with 'complexity_headroom' and 'cyclomatic_headroom'.
        """
        if self._baseline is None:
            raise RuntimeError("Baseline not set. Call set_baseline() first.")

        max_allowed_complexity = self._baseline * self.max_ratio
        return {
            "complexity_headroom": max(0.0, max_allowed_complexity - self._current),
            "cyclomatic_headroom": max(0, self.max_cyclomatic - self._current_cyclomatic),
            "max_allowed_complexity": max_allowed_complexity,
            "max_allowed_cyclomatic": self.max_cyclomatic,
        }
