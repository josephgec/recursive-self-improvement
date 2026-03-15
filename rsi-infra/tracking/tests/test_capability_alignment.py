"""Tests for Capability-Alignment Ratio (CAR) tracker."""

from __future__ import annotations

import math

import pytest

from tracking.src.capability_alignment import CARMeasurement, CapabilityAlignmentTracker


class TestCARBasic:
    def test_compute_returns_car(self) -> None:
        tracker = CapabilityAlignmentTracker(
            capability_metrics=["accuracy"],
            alignment_metrics=["safety"],
        )
        car = tracker.compute(
            generation=1,
            current_metrics={"accuracy": 0.9, "safety": 0.95},
            previous_metrics={"accuracy": 0.8, "safety": 0.95},
        )
        # capability_gain = 0.1, alignment_cost = 0 (no degradation)
        # => car = inf
        assert car == float("inf")

    def test_trajectory_populated(self) -> None:
        tracker = CapabilityAlignmentTracker(
            capability_metrics=["acc"],
            alignment_metrics=["safe"],
        )
        tracker.compute(1, {"acc": 0.9, "safe": 0.9}, {"acc": 0.8, "safe": 0.95})
        trajectory = tracker.get_trajectory()
        assert len(trajectory) == 1
        m = trajectory[0]
        assert m.generation == 1
        assert isinstance(m, CARMeasurement)


class TestParetoImproving:
    """Test Pareto-improvement detection edge cases."""

    def test_pareto_improving_when_alignment_stable(self) -> None:
        tracker = CapabilityAlignmentTracker(
            capability_metrics=["acc"],
            alignment_metrics=["safe"],
        )
        tracker.compute(1, {"acc": 0.9, "safe": 0.95}, {"acc": 0.8, "safe": 0.95})
        assert tracker.is_pareto_improving(1) is True

    def test_pareto_improving_when_alignment_improves(self) -> None:
        """Cap up + alignment up => alignment_cost < 0 => pareto."""
        tracker = CapabilityAlignmentTracker(
            capability_metrics=["acc"],
            alignment_metrics=["safe"],
        )
        tracker.compute(1, {"acc": 0.9, "safe": 0.98}, {"acc": 0.8, "safe": 0.95})
        assert tracker.is_pareto_improving(1) is True

    def test_not_pareto_when_alignment_degrades(self) -> None:
        """Cap up + alignment down => not pareto."""
        tracker = CapabilityAlignmentTracker(
            capability_metrics=["acc"],
            alignment_metrics=["safe"],
        )
        tracker.compute(1, {"acc": 0.9, "safe": 0.85}, {"acc": 0.8, "safe": 0.95})
        assert tracker.is_pareto_improving(1) is False

    def test_not_pareto_when_capability_flat(self) -> None:
        """Cap flat (gain=0) => not pareto (requires cap > 0)."""
        tracker = CapabilityAlignmentTracker(
            capability_metrics=["acc"],
            alignment_metrics=["safe"],
        )
        tracker.compute(1, {"acc": 0.8, "safe": 0.95}, {"acc": 0.8, "safe": 0.95})
        assert tracker.is_pareto_improving(1) is False

    def test_not_pareto_when_capability_degrades(self) -> None:
        """Cap down => not pareto."""
        tracker = CapabilityAlignmentTracker(
            capability_metrics=["acc"],
            alignment_metrics=["safe"],
        )
        tracker.compute(1, {"acc": 0.7, "safe": 0.95}, {"acc": 0.8, "safe": 0.95})
        assert tracker.is_pareto_improving(1) is False

    def test_is_pareto_unknown_generation_raises(self) -> None:
        tracker = CapabilityAlignmentTracker(
            capability_metrics=["acc"],
            alignment_metrics=["safe"],
        )
        with pytest.raises(ValueError, match="No measurement"):
            tracker.is_pareto_improving(99)


class TestMeanDeltaEdgeCases:
    """Test _mean_delta edge cases."""

    def test_missing_keys_returns_zero(self) -> None:
        """When none of the keys are in the metrics, delta is 0."""
        tracker = CapabilityAlignmentTracker(
            capability_metrics=["nonexistent"],
            alignment_metrics=["also_nonexistent"],
        )
        car = tracker.compute(
            1,
            {"other": 0.9},
            {"other": 0.8},
        )
        # Both deltas are 0 => car = 0
        assert car == 0.0

    def test_partial_keys(self) -> None:
        """Only metrics present in both current and previous are considered."""
        tracker = CapabilityAlignmentTracker(
            capability_metrics=["acc", "speed"],
            alignment_metrics=["safe"],
        )
        # Only "acc" is present in both, "speed" only in current
        car = tracker.compute(
            1,
            {"acc": 0.9, "speed": 1.0, "safe": 0.95},
            {"acc": 0.8, "safe": 0.95},
        )
        # capability_gain = (0.9-0.8)/1 = 0.1 (speed ignored)
        # alignment_cost = 0
        assert car == float("inf")

    def test_none_values_skipped(self) -> None:
        """None values in metrics are skipped."""
        tracker = CapabilityAlignmentTracker(
            capability_metrics=["acc"],
            alignment_metrics=["safe"],
        )
        car = tracker.compute(
            1,
            {"acc": None, "safe": 0.9},
            {"acc": 0.8, "safe": 0.95},
        )
        # acc is None in current => skipped => capability_gain = 0
        # alignment_cost = -(0.9-0.95) = 0.05
        assert car == 0.0 / 0.05


class TestCARValues:
    """Test specific CAR values."""

    def test_negative_infinity(self) -> None:
        """Capability down + alignment stable => -inf."""
        tracker = CapabilityAlignmentTracker(
            capability_metrics=["acc"],
            alignment_metrics=["safe"],
        )
        car = tracker.compute(
            1,
            {"acc": 0.7, "safe": 0.95},
            {"acc": 0.8, "safe": 0.95},
        )
        assert car == float("-inf")

    def test_zero_car(self) -> None:
        """No change at all => car = 0."""
        tracker = CapabilityAlignmentTracker(
            capability_metrics=["acc"],
            alignment_metrics=["safe"],
        )
        car = tracker.compute(
            1,
            {"acc": 0.8, "safe": 0.95},
            {"acc": 0.8, "safe": 0.95},
        )
        assert car == 0.0

    def test_finite_positive_car(self) -> None:
        """Cap up + alignment down => finite positive car."""
        tracker = CapabilityAlignmentTracker(
            capability_metrics=["acc"],
            alignment_metrics=["safe"],
        )
        car = tracker.compute(
            1,
            {"acc": 0.9, "safe": 0.85},
            {"acc": 0.8, "safe": 0.95},
        )
        # capability_gain = 0.1, alignment_cost = 0.1
        assert car == pytest.approx(1.0)

    def test_multiple_generations_trajectory(self) -> None:
        """Multiple compute calls build up the trajectory."""
        tracker = CapabilityAlignmentTracker(
            capability_metrics=["acc"],
            alignment_metrics=["safe"],
        )
        tracker.compute(1, {"acc": 0.85, "safe": 0.94}, {"acc": 0.8, "safe": 0.95})
        tracker.compute(2, {"acc": 0.9, "safe": 0.93}, {"acc": 0.85, "safe": 0.94})
        tracker.compute(3, {"acc": 0.92, "safe": 0.92}, {"acc": 0.9, "safe": 0.93})

        trajectory = tracker.get_trajectory()
        assert len(trajectory) == 3
        assert trajectory[0].generation == 1
        assert trajectory[1].generation == 2
        assert trajectory[2].generation == 3

    def test_trajectory_is_copy(self) -> None:
        """get_trajectory returns a copy, not the internal list."""
        tracker = CapabilityAlignmentTracker(
            capability_metrics=["acc"],
            alignment_metrics=["safe"],
        )
        tracker.compute(1, {"acc": 0.9, "safe": 0.95}, {"acc": 0.8, "safe": 0.95})
        t1 = tracker.get_trajectory()
        t2 = tracker.get_trajectory()
        assert t1 is not t2
        assert t1 == t2
