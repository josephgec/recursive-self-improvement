"""SuiteVerdict: result of running all constraints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.constraints.base import ConstraintResult


@dataclass(frozen=True)
class SuiteVerdict:
    """Aggregated result of running an entire constraint suite."""

    passed: bool
    results: Dict[str, ConstraintResult] = field(default_factory=dict)

    @property
    def violations(self) -> Dict[str, ConstraintResult]:
        """Return only the constraints that were violated."""
        return {
            name: result
            for name, result in self.results.items()
            if not result.satisfied
        }

    @property
    def closest_to_violation(self) -> Optional[Tuple[str, ConstraintResult]]:
        """Return the satisfied constraint with the smallest headroom."""
        satisfied = [
            (name, result)
            for name, result in self.results.items()
            if result.satisfied
        ]
        if not satisfied:
            return None
        return min(satisfied, key=lambda x: x[1].headroom)

    def summary(self) -> str:
        """Human-readable summary."""
        total = len(self.results)
        violated = len(self.violations)
        lines = [
            f"Suite verdict: {'PASSED' if self.passed else 'FAILED'}",
            f"  Constraints checked: {total}",
            f"  Violations: {violated}",
        ]
        if self.violations:
            lines.append("  Failed constraints:")
            for name, result in self.violations.items():
                lines.append(
                    f"    - {name}: measured={result.measured_value:.4f}, "
                    f"threshold={result.threshold:.4f}"
                )
        closest = self.closest_to_violation
        if closest:
            name, result = closest
            lines.append(
                f"  Closest to violation: {name} "
                f"(headroom={result.headroom:.4f})"
            )
        return "\n".join(lines)
