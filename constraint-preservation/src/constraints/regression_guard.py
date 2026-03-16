"""RegressionGuardConstraint: max regression on any benchmark."""

from __future__ import annotations

from typing import Any, Dict

from src.constraints.base import Constraint, ConstraintResult, CheckContext


class RegressionGuardConstraint(Constraint):
    """No benchmark may regress by more than max_regression_pp percentage points."""

    def __init__(self, max_regression_pp: float = 3.0) -> None:
        super().__init__(
            name="regression_guard",
            description="Maximum per-benchmark regression in percentage points",
            category="quality",
            threshold=max_regression_pp,
        )
        self._max_regression_pp = max_regression_pp

    def check(self, agent_state: Any, context: CheckContext) -> ConstraintResult:
        """Evaluate regression.

        ``agent_state`` must expose:
        * ``get_benchmark_scores() -> Dict[str, float]``  current scores (0-100)
        * ``get_baseline_scores() -> Dict[str, float]``   baseline scores (0-100)
        """
        current = agent_state.get_benchmark_scores()
        baseline = agent_state.get_baseline_scores()

        regressions: Dict[str, float] = {}
        max_regression = 0.0

        for bench, baseline_score in baseline.items():
            current_score = current.get(bench, baseline_score)
            regression = baseline_score - current_score  # positive = regressed
            regressions[bench] = regression
            if regression > max_regression:
                max_regression = regression

        # headroom = how much regression budget remains
        headroom = self._max_regression_pp - max_regression
        satisfied = max_regression <= self._max_regression_pp

        return ConstraintResult(
            satisfied=satisfied,
            measured_value=max_regression,
            threshold=self._max_regression_pp,
            headroom=headroom,
            details={
                "per_benchmark_regression": regressions,
                "max_regression": max_regression,
                "direction": "ceiling",
            },
        )

    def headroom(self, measured_value: float) -> float:
        """For ceiling constraints headroom is threshold - measured."""
        return self._max_regression_pp - measured_value
