"""Adaptive thresholds for constraint tuning.

Suggests threshold adjustments based on historical performance data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ThresholdAdjustment:
    """A suggested threshold adjustment."""
    constraint_name: str
    current_threshold: float
    suggested_threshold: float
    direction: str  # "tighten", "loosen", "maintain"
    confidence: float
    rationale: str


class AdaptiveThresholds:
    """Suggests adaptive threshold adjustments for constraints.

    Analyzes historical constraint binding data and performance
    to suggest optimal thresholds.
    """

    def __init__(self, adjustment_rate: float = 0.1):
        """
        Args:
            adjustment_rate: Maximum fraction to adjust thresholds by.
        """
        self.adjustment_rate = adjustment_rate
        self._adjustment_history: List[ThresholdAdjustment] = []

    def suggest_adjustment(
        self,
        constraint_name: str,
        history: List[Dict[str, Any]],
        current_threshold: float = 1.0,
    ) -> ThresholdAdjustment:
        """Suggest a threshold adjustment based on history.

        Args:
            constraint_name: Name of the constraint.
            history: List of dicts with 'value', 'threshold', and 'performance' keys.
            current_threshold: Current threshold value.

        Returns:
            ThresholdAdjustment suggestion.
        """
        if not history:
            adj = ThresholdAdjustment(
                constraint_name=constraint_name,
                current_threshold=current_threshold,
                suggested_threshold=current_threshold,
                direction="maintain",
                confidence=0.0,
                rationale="Insufficient data for adjustment",
            )
            self._adjustment_history.append(adj)
            return adj

        # Analyze binding frequency
        binding_count = 0
        total_performance = 0.0
        binding_performance = 0.0
        non_binding_performance = 0.0
        n_binding = 0
        n_non_binding = 0

        for entry in history:
            value = entry.get("value", 0.0)
            threshold = entry.get("threshold", current_threshold)
            perf = entry.get("performance", 1.0)
            total_performance += perf

            is_binding = value >= threshold * 0.95
            if is_binding:
                binding_count += 1
                binding_performance += perf
                n_binding += 1
            else:
                non_binding_performance += perf
                n_non_binding += 1

        binding_fraction = binding_count / len(history)
        avg_performance = total_performance / len(history)
        avg_binding_perf = binding_performance / n_binding if n_binding > 0 else avg_performance
        avg_non_binding_perf = non_binding_performance / n_non_binding if n_non_binding > 0 else avg_performance

        # Decide direction
        if binding_fraction > 0.5 and avg_binding_perf < avg_non_binding_perf:
            # Too tight: constraint is binding often and hurting performance
            direction = "loosen"
            adjustment = current_threshold * self.adjustment_rate
            suggested = current_threshold + adjustment
            rationale = (
                f"Binding {binding_fraction:.0%} with lower performance when binding "
                f"({avg_binding_perf:.2f} vs {avg_non_binding_perf:.2f})"
            )
        elif binding_fraction < 0.1 and avg_performance < 0.8:
            # Too loose: constraint rarely binding but performance is poor
            direction = "tighten"
            adjustment = current_threshold * self.adjustment_rate
            suggested = current_threshold - adjustment
            rationale = (
                f"Binding only {binding_fraction:.0%} but performance is low "
                f"({avg_performance:.2f}). Tightening may improve quality."
            )
        else:
            direction = "maintain"
            suggested = current_threshold
            rationale = f"Current threshold appears appropriate (binding {binding_fraction:.0%})"

        confidence = min(len(history) / 50.0, 1.0)

        adj = ThresholdAdjustment(
            constraint_name=constraint_name,
            current_threshold=current_threshold,
            suggested_threshold=suggested,
            direction=direction,
            confidence=confidence,
            rationale=rationale,
        )
        self._adjustment_history.append(adj)
        return adj

    def get_history(self) -> List[ThresholdAdjustment]:
        """Return adjustment history."""
        return list(self._adjustment_history)
