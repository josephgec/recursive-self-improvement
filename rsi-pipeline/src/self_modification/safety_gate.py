"""Safety gate: pre-modification safety checks."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.pipeline.state import PipelineState
from src.outer_loop.strategy_evolver import Candidate


class SafetyGate:
    """Pre-modification safety checks."""

    def __init__(self, max_complexity: float = 1000, min_accuracy: float = 0.6):
        self._max_complexity = max_complexity
        self._min_accuracy = min_accuracy

    def check_pre_modification(self, candidate: Candidate, state: PipelineState) -> Dict[str, Any]:
        """Check if it is safe to apply a modification.

        Returns dict with 'allowed' (bool) and 'reasons' (list of strings).
        """
        reasons: List[str] = []
        allowed = True

        # Check code complexity
        code_length = len(candidate.proposed_code)
        if code_length > self._max_complexity:
            reasons.append(f"code_too_complex: length={code_length} > {self._max_complexity}")
            allowed = False

        # Check current accuracy is stable enough
        if state.performance.accuracy < self._min_accuracy:
            reasons.append(
                f"accuracy_too_low: {state.performance.accuracy:.3f} < {self._min_accuracy}"
            )
            # Allow but warn — don't block modifications when accuracy is low
            # since the modification might improve it

        # Check for emergency state
        if state.status == "emergency":
            reasons.append("pipeline_in_emergency_state")
            allowed = False

        # Check consecutive rollbacks
        if state.safety.consecutive_rollbacks >= 3:
            reasons.append(
                f"too_many_rollbacks: {state.safety.consecutive_rollbacks}"
            )
            allowed = False

        return {"allowed": allowed, "reasons": reasons}
