"""Capability-Alignment Ratio (CAR) — tracks whether capability gains
outpace alignment costs (or vice-versa) across generations.

CAR = capability_gain / alignment_cost

Where *capability_gain* is the mean improvement of capability metrics and
*alignment_cost* is the mean degradation of alignment metrics.  Both are
computed relative to the previous generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CARMeasurement:
    """Single CAR measurement for one generation."""

    generation: int
    capability_gain: float
    alignment_cost: float
    car: float
    pareto_improving: bool  # True iff capability_gain > 0 and alignment_cost <= 0


class CapabilityAlignmentTracker:
    """Track Capability-Alignment Ratio across generations.

    Parameters
    ----------
    capability_metrics : list[str]
        Keys in the metrics dict that represent capabilities (higher is better).
    alignment_metrics : list[str]
        Keys in the metrics dict that represent alignment / safety (higher is
        better — a *drop* means alignment cost).
    """

    def __init__(
        self,
        capability_metrics: list[str],
        alignment_metrics: list[str],
    ) -> None:
        self._cap_keys = list(capability_metrics)
        self._align_keys = list(alignment_metrics)
        self._trajectory: list[CARMeasurement] = []

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _mean_delta(keys: list[str], current: dict[str, Any], previous: dict[str, Any]) -> float:
        """Mean change (current − previous) for the given metric keys."""
        deltas: list[float] = []
        for k in keys:
            cur_val = current.get(k)
            prev_val = previous.get(k)
            if cur_val is not None and prev_val is not None:
                deltas.append(float(cur_val) - float(prev_val))
        if not deltas:
            return 0.0
        return sum(deltas) / len(deltas)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def compute(
        self,
        generation: int,
        current_metrics: dict[str, Any],
        previous_metrics: dict[str, Any],
    ) -> float:
        """Compute CAR for *generation* and append to trajectory.

        Returns the CAR value.
        """
        capability_gain = self._mean_delta(self._cap_keys, current_metrics, previous_metrics)

        # alignment_cost = how much alignment *dropped* (positive means degradation)
        alignment_delta = self._mean_delta(self._align_keys, current_metrics, previous_metrics)
        alignment_cost = -alignment_delta  # positive when alignment gets worse

        # CAR = capability_gain / alignment_cost (handle division by zero)
        if alignment_cost == 0.0:
            if capability_gain > 0.0:
                car = float("inf")
            elif capability_gain < 0.0:
                car = float("-inf")
            else:
                car = 0.0
        else:
            car = capability_gain / alignment_cost

        pareto = capability_gain > 0 and alignment_cost <= 0

        measurement = CARMeasurement(
            generation=generation,
            capability_gain=capability_gain,
            alignment_cost=alignment_cost,
            car=car,
            pareto_improving=pareto,
        )
        self._trajectory.append(measurement)
        return car

    def get_trajectory(self) -> list[CARMeasurement]:
        """Return all measurements."""
        return list(self._trajectory)

    def is_pareto_improving(self, generation: int) -> bool:
        """Whether the given *generation* was Pareto-improving.

        A generation is Pareto-improving if capability went up and alignment
        did not degrade.
        """
        for m in self._trajectory:
            if m.generation == generation:
                return m.pareto_improving
        raise ValueError(f"No measurement for generation {generation}")
