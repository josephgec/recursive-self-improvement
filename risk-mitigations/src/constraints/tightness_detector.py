"""Tightness detector for constraint analysis.

Detects when constraints are too tight (blocking progress) or too loose
(not providing safety).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TightnessReport:
    """Report on constraint tightness analysis."""
    constraint_name: str
    binding_fraction: float  # Fraction of time the constraint is binding
    is_too_tight: bool
    is_too_loose: bool
    recommendation: str
    details: Dict[str, Any] = field(default_factory=dict)


class TightnessDetector:
    """Detects when constraints are too tight or too loose.

    A constraint is:
    - Too tight if it binds > 50% of the time (blocking progress)
    - Too loose if it never binds (not providing safety)
    """

    def __init__(
        self,
        too_tight_threshold: float = 0.50,
        too_loose_threshold: float = 0.10,
    ):
        self.too_tight_threshold = too_tight_threshold
        self.too_loose_threshold = too_loose_threshold

    def detect(
        self,
        constraint_name: str,
        history: List[Dict[str, Any]],
    ) -> TightnessReport:
        """Analyze constraint tightness from history.

        Args:
            constraint_name: Name of the constraint.
            history: List of dicts with 'value' and 'threshold' keys.

        Returns:
            TightnessReport with analysis.
        """
        if not history:
            return TightnessReport(
                constraint_name=constraint_name,
                binding_fraction=0.0,
                is_too_tight=False,
                is_too_loose=True,
                recommendation="Insufficient data to assess tightness",
            )

        # Count how often the constraint is binding
        binding_count = 0
        for entry in history:
            value = entry.get("value", 0.0)
            threshold = entry.get("threshold", 1.0)
            # Constraint is binding if value is within 5% of threshold
            if threshold > 0 and abs(value - threshold) / threshold <= 0.05:
                binding_count += 1
            elif value >= threshold:
                binding_count += 1

        binding_fraction = binding_count / len(history)

        is_too_tight = binding_fraction > self.too_tight_threshold
        is_too_loose = binding_fraction < self.too_loose_threshold

        if is_too_tight:
            recommendation = (
                f"Constraint '{constraint_name}' is binding {binding_fraction:.0%} of the time. "
                "Consider graduated relaxation."
            )
        elif is_too_loose:
            recommendation = (
                f"Constraint '{constraint_name}' is binding only {binding_fraction:.0%} of the time. "
                "Consider tightening or removing."
            )
        else:
            recommendation = (
                f"Constraint '{constraint_name}' tightness is appropriate "
                f"(binding {binding_fraction:.0%})."
            )

        return TightnessReport(
            constraint_name=constraint_name,
            binding_fraction=binding_fraction,
            is_too_tight=is_too_tight,
            is_too_loose=is_too_loose,
            recommendation=recommendation,
            details={
                "total_entries": len(history),
                "binding_count": binding_count,
            },
        )

    def too_tight(self, binding_fraction: float) -> bool:
        """Check if a binding fraction indicates too-tight constraint."""
        return binding_fraction > self.too_tight_threshold

    def too_loose(self, binding_fraction: float) -> bool:
        """Check if a binding fraction indicates too-loose constraint."""
        return binding_fraction < self.too_loose_threshold
