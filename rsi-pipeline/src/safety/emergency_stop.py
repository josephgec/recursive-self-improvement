"""Emergency stop: triggers on dangerous conditions."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.pipeline.state import PipelineState


class EmergencyStop:
    """Emergency stop mechanism for the RSI pipeline.

    Triggers on:
    - CAR < 0.5
    - 3 or more consecutive rollbacks
    - Any hard constraint violation
    """

    def __init__(
        self,
        car_threshold: float = 0.5,
        max_consecutive_rollbacks: int = 3,
    ):
        self._car_threshold = car_threshold
        self._max_rollbacks = max_consecutive_rollbacks
        self._triggered = False
        self._reason: str = ""

    def check(self, state: PipelineState, recent_results: List[Any] = None) -> bool:
        """Check if emergency stop should be triggered.

        Returns True if emergency stop conditions are met.
        """
        # Check CAR
        if state.safety.car_score < self._car_threshold:
            self._triggered = True
            self._reason = f"car_below_threshold: {state.safety.car_score:.3f} < {self._car_threshold}"
            return True

        # Check consecutive rollbacks
        if state.safety.consecutive_rollbacks >= self._max_rollbacks:
            self._triggered = True
            self._reason = f"consecutive_rollbacks: {state.safety.consecutive_rollbacks} >= {self._max_rollbacks}"
            return True

        # Check constraint violations
        if not state.safety.constraints_satisfied:
            self._triggered = True
            self._reason = f"constraint_violation: {state.safety.violations}"
            return True

        return False

    def execute(self, state: PipelineState, reason: str = "") -> None:
        """Execute emergency stop."""
        state.status = "emergency"
        state.safety.emergency_stop = True
        self._triggered = True
        self._reason = reason or self._reason

    @property
    def triggered(self) -> bool:
        return self._triggered

    @property
    def reason(self) -> str:
        return self._reason

    def reset(self) -> None:
        """Reset emergency stop (for testing)."""
        self._triggered = False
        self._reason = ""
