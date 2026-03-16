"""ConstraintSuite: builds and holds the immutable set of enabled constraints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.constraints.base import Constraint
from src.constraints.accuracy_floor import AccuracyFloorConstraint
from src.constraints.entropy_floor import EntropyFloorConstraint
from src.constraints.safety_eval import SafetyEvalConstraint
from src.constraints.drift_ceiling import DriftCeilingConstraint
from src.constraints.regression_guard import RegressionGuardConstraint
from src.constraints.consistency import ConsistencyConstraint
from src.constraints.latency_ceiling import LatencyCeilingConstraint


# Map from config key to factory
_BUILTIN_FACTORIES: Dict[str, Any] = {
    "accuracy_floor": lambda cfg: AccuracyFloorConstraint(
        threshold=cfg.get("threshold", 0.80)
    ),
    "entropy_floor": lambda cfg: EntropyFloorConstraint(
        threshold=cfg.get("threshold", 3.5)
    ),
    "safety_eval": lambda cfg: SafetyEvalConstraint(
        threshold=cfg.get("threshold", 1.0)
    ),
    "drift_ceiling": lambda cfg: DriftCeilingConstraint(
        threshold=cfg.get("threshold", 0.40)
    ),
    "regression_guard": lambda cfg: RegressionGuardConstraint(
        max_regression_pp=cfg.get("max_regression_pp", 3.0)
    ),
    "consistency": lambda cfg: ConsistencyConstraint(
        threshold=cfg.get("threshold", 0.85)
    ),
    "latency_ceiling": lambda cfg: LatencyCeilingConstraint(
        p95_threshold_ms=cfg.get("p95_threshold_ms", 30000)
    ),
}


class ConstraintSuite:
    """Immutable collection of enabled constraints built from configuration."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        constraints_cfg = (config or {}).get("constraints", {})
        built: List[Constraint] = []

        for name, factory in _BUILTIN_FACTORIES.items():
            entry = constraints_cfg.get(name, {})
            if not entry.get("enabled", True):
                continue
            built.append(factory(entry))

        # Freeze
        self._constraints: Tuple[Constraint, ...] = tuple(built)

    # -- accessors -----------------------------------------------------------

    @property
    def constraints(self) -> Tuple[Constraint, ...]:
        return self._constraints

    def get_by_name(self, name: str) -> Optional[Constraint]:
        for c in self._constraints:
            if c.name == name:
                return c
        return None

    def get_by_category(self, category: str) -> Tuple[Constraint, ...]:
        return tuple(c for c in self._constraints if c.category == category)

    def __len__(self) -> int:
        return len(self._constraints)

    def __iter__(self):
        return iter(self._constraints)

    def add_custom(self, constraint: Constraint) -> "ConstraintSuite":
        """Return a new suite with an additional custom constraint appended."""
        new = ConstraintSuite.__new__(ConstraintSuite)
        new._constraints = self._constraints + (constraint,)
        return new
