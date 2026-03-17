"""Tests for delta bounder."""

import numpy as np
import pytest

from src.bounding.delta_bounder import DeltaBounder, DeltaBoundResult


class TestDeltaBounder:
    """Test delta bounding functionality."""

    def test_first_value_passes_through(self):
        """First value is never bounded."""
        bounder = DeltaBounder(max_delta=1.0)
        bounded, was_bounded = bounder.bound(10.0)
        assert bounded == 10.0
        assert was_bounded is False

    def test_smooth_spike(self):
        """Large spikes are smoothed to max_delta."""
        bounder = DeltaBounder(max_delta=1.0)
        bounder.bound(0.0)  # baseline
        bounded, was_bounded = bounder.bound(10.0)

        assert was_bounded is True
        assert bounded == 1.0  # 0.0 + max_delta

    def test_smooth_negative_spike(self):
        """Large negative spikes are smoothed."""
        bounder = DeltaBounder(max_delta=1.0)
        bounder.bound(5.0)
        bounded, was_bounded = bounder.bound(-10.0)

        assert was_bounded is True
        assert bounded == 4.0  # 5.0 - max_delta

    def test_no_bound_within_delta(self):
        """Values within max_delta are not bounded."""
        bounder = DeltaBounder(max_delta=2.0)
        bounder.bound(0.0)
        bounded, was_bounded = bounder.bound(1.5)
        assert was_bounded is False
        assert bounded == 1.5

    def test_sequential_bounding(self):
        """Sequential bounding uses last bounded value."""
        bounder = DeltaBounder(max_delta=1.0)
        bounder.bound(0.0)

        # First spike: 0 -> 10, bounded to 1
        bounder.bound(10.0)

        # Second spike: 1 -> 10, bounded to 2
        bounded, _ = bounder.bound(10.0)
        assert bounded == 2.0

    def test_batch_processing(self):
        """bound_batch processes sequences correctly."""
        bounder = DeltaBounder(max_delta=1.0)
        rewards = np.array([0.0, 5.0, 5.0, -5.0, 0.0])
        bounded, was_bounded = bounder.bound_batch(rewards)

        assert bounded[0] == 0.0
        assert bounded[1] == 1.0  # clamped
        assert was_bounded[1] == True
        assert bounded[3] == bounded[2] - 1.0  # clamped down

    def test_tracking(self):
        """Bound count and total count are tracked."""
        bounder = DeltaBounder(max_delta=1.0)
        bounder.bound(0.0)
        bounder.bound(0.5)  # within delta
        bounder.bound(5.0)  # bounded

        assert bounder.bound_count == 1
        assert bounder.total_count == 3

    def test_history(self):
        """History records all bound results."""
        bounder = DeltaBounder(max_delta=1.0)
        bounder.bound(0.0)
        bounder.bound(5.0)

        assert len(bounder.history) == 2
        assert isinstance(bounder.history[0], DeltaBoundResult)
        assert bounder.history[1].was_bounded is True

    def test_reset(self):
        """Reset clears all state."""
        bounder = DeltaBounder(max_delta=1.0)
        bounder.bound(0.0)
        bounder.bound(5.0)
        bounder.reset()

        assert bounder.bound_count == 0
        assert bounder.total_count == 0
        assert bounder.last_reward is None
        assert bounder.history == []

    def test_invalid_max_delta(self):
        """Raises ValueError for non-positive max_delta."""
        with pytest.raises(ValueError):
            DeltaBounder(max_delta=-1.0)
        with pytest.raises(ValueError):
            DeltaBounder(max_delta=0.0)

    def test_properties(self):
        """Properties return correct values."""
        bounder = DeltaBounder(max_delta=2.5)
        assert bounder.max_delta == 2.5
        assert bounder.last_reward is None
        bounder.bound(1.0)
        assert bounder.last_reward == 1.0
