"""Constraint preservation — ensures hard safety invariants hold across
generations and recommends proceed / revert / halt."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ConstraintDef:
    """Definition of a single constraint."""

    name: str
    metric_key: str
    threshold: float
    direction: str = ">="  # one of >=, <=, >, <
    severity: str = "revert"  # "revert" or "halt"


@dataclass
class ConstraintResult:
    """Outcome of checking one constraint."""

    name: str
    passed: bool
    value: float
    threshold: float
    violation_severity: str | None = None  # None when passed


@dataclass
class PreservationReport:
    """Aggregate outcome for all constraints at a given generation."""

    all_passed: bool
    results: list[ConstraintResult]
    generation: int
    recommendation: str  # "proceed" | "halt" | "revert"


# ---------------------------------------------------------------------------
# Default constraints
# ---------------------------------------------------------------------------

DEFAULT_CONSTRAINTS: list[ConstraintDef] = [
    ConstraintDef(
        name="accuracy_floor",
        metric_key="accuracy",
        threshold=0.6,
        direction=">=",
        severity="revert",
    ),
    ConstraintDef(
        name="safety_eval_pass",
        metric_key="safety_score",
        threshold=0.9,
        direction=">=",
        severity="halt",
    ),
    ConstraintDef(
        name="loss_ceiling",
        metric_key="loss",
        threshold=5.0,
        direction="<=",
        severity="revert",
    ),
    ConstraintDef(
        name="perplexity_ceiling",
        metric_key="perplexity",
        threshold=100.0,
        direction="<=",
        severity="revert",
    ),
]


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

_CMP = {
    ">=": lambda val, thr: val >= thr,
    "<=": lambda val, thr: val <= thr,
    ">": lambda val, thr: val > thr,
    "<": lambda val, thr: val < thr,
}


# ---------------------------------------------------------------------------
# ConstraintPreserver
# ---------------------------------------------------------------------------

class ConstraintPreserver:
    """Check a set of constraints against observed metrics.

    Parameters
    ----------
    constraints : list[ConstraintDef] | None
        Constraint definitions.  Defaults to ``DEFAULT_CONSTRAINTS``.
    """

    def __init__(self, constraints: list[ConstraintDef] | None = None) -> None:
        self._constraints: list[ConstraintDef] = list(
            constraints if constraints is not None else DEFAULT_CONSTRAINTS
        )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def add_constraint(
        self,
        name: str,
        metric_key: str,
        threshold: float,
        direction: str = ">=",
        severity: str = "revert",
    ) -> None:
        """Add a new constraint at runtime."""
        self._constraints.append(
            ConstraintDef(
                name=name,
                metric_key=metric_key,
                threshold=threshold,
                direction=direction,
                severity=severity,
            )
        )

    def check(self, generation: int, metrics: dict[str, Any]) -> PreservationReport:
        """Evaluate every constraint against *metrics*.

        Returns a ``PreservationReport`` whose ``recommendation`` is:
        * ``"proceed"`` — all constraints pass
        * ``"revert"``  — at least one *revert*-severity constraint fails
        * ``"halt"``    — at least one *halt*-severity constraint fails
        """
        results: list[ConstraintResult] = []
        worst_severity: str | None = None

        for c in self._constraints:
            value = metrics.get(c.metric_key)
            if value is None:
                # Metric not reported — skip (assume OK)
                continue

            cmp_fn = _CMP.get(c.direction)
            if cmp_fn is None:
                raise ValueError(f"Unknown direction {c.direction!r} in constraint {c.name!r}")

            passed = cmp_fn(float(value), c.threshold)
            result = ConstraintResult(
                name=c.name,
                passed=passed,
                value=float(value),
                threshold=c.threshold,
                violation_severity=None if passed else c.severity,
            )
            results.append(result)

            if not passed:
                if worst_severity is None:
                    worst_severity = c.severity
                elif c.severity == "halt":
                    worst_severity = "halt"

        all_passed = worst_severity is None
        if worst_severity == "halt":
            recommendation = "halt"
        elif worst_severity == "revert":
            recommendation = "revert"
        else:
            recommendation = "proceed"

        return PreservationReport(
            all_passed=all_passed,
            results=results,
            generation=generation,
            recommendation=recommendation,
        )
