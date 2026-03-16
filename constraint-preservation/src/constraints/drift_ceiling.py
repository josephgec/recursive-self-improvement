"""DriftCeilingConstraint: maximum allowable goal drift index."""

from __future__ import annotations

from typing import Any

from src.constraints.base import Constraint, ConstraintResult, CheckContext


class DriftCeilingConstraint(Constraint):
    """Goal drift index must remain below threshold.

    Bridges to a mock Goal Drift Index (GDI) computation.
    The constraint is satisfied when measured drift <= threshold.
    """

    def __init__(self, threshold: float = 0.40) -> None:
        super().__init__(
            name="drift_ceiling",
            description="Maximum allowable goal drift index",
            category="quality",
            threshold=threshold,
        )

    def check(self, agent_state: Any, context: CheckContext) -> ConstraintResult:
        """Evaluate drift.

        ``agent_state`` must expose:
        * ``compute_drift() -> float`` returning the GDI in [0, 1].
        """
        drift_index = agent_state.compute_drift()

        # For ceiling constraints, headroom = threshold - measured
        headroom = self._threshold - drift_index
        satisfied = drift_index <= self._threshold

        return ConstraintResult(
            satisfied=satisfied,
            measured_value=drift_index,
            threshold=self._threshold,
            headroom=headroom,
            details={
                "drift_index": drift_index,
                "direction": "ceiling",
            },
        )

    def headroom(self, measured_value: float) -> float:
        """For ceiling constraints headroom is threshold - measured."""
        return self._threshold - measured_value
