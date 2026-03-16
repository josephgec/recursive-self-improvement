"""Constraint enforcer: checks hard constraints on pipeline state."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.pipeline.state import PipelineState


@dataclass
class ConstraintVerdict:
    """Result of constraint enforcement."""
    satisfied: bool = True
    violations: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "satisfied": self.satisfied,
            "violations": self.violations,
            "details": self.details,
        }


class ConstraintEnforcer:
    """Enforces hard constraints on pipeline state.

    Hard constraints:
    - accuracy_floor: minimum accuracy
    - entropy_floor: minimum code diversity
    - drift_ceiling: maximum GDI drift
    """

    def __init__(
        self,
        accuracy_floor: float = 0.6,
        entropy_floor: float = 0.1,
        drift_ceiling: float = 0.5,
    ):
        self._accuracy_floor = accuracy_floor
        self._entropy_floor = entropy_floor
        self._drift_ceiling = drift_ceiling

    def check_all(self, state: PipelineState) -> ConstraintVerdict:
        """Check all hard constraints against the current state."""
        violations: List[str] = []
        details: Dict[str, Any] = {}

        # Accuracy floor
        acc = state.performance.accuracy
        details["accuracy"] = {"value": acc, "floor": self._accuracy_floor}
        if acc < self._accuracy_floor:
            violations.append(
                f"accuracy_below_floor: {acc:.3f} < {self._accuracy_floor}"
            )

        # Entropy floor
        entropy = state.performance.entropy
        details["entropy"] = {"value": entropy, "floor": self._entropy_floor}
        if entropy < self._entropy_floor:
            violations.append(
                f"entropy_below_floor: {entropy:.3f} < {self._entropy_floor}"
            )

        # Drift ceiling (using safety.gdi_score)
        drift = state.safety.gdi_score
        details["drift"] = {"value": drift, "ceiling": self._drift_ceiling}
        if drift > self._drift_ceiling:
            violations.append(
                f"drift_above_ceiling: {drift:.3f} > {self._drift_ceiling}"
            )

        return ConstraintVerdict(
            satisfied=len(violations) == 0,
            violations=violations,
            details=details,
        )

    @property
    def accuracy_floor(self) -> float:
        return self._accuracy_floor

    @property
    def entropy_floor(self) -> float:
        return self._entropy_floor

    @property
    def drift_ceiling(self) -> float:
        return self._drift_ceiling
