"""Tests for reward clipper."""

import numpy as np
import pytest

from src.bounding.reward_clipper import RewardClipper, ClipStats


class TestRewardClipper:
    """Test reward clipping functionality."""

    def test_clip_extremes(self):
        """Clips values outside the range."""
        clipper = RewardClipper(clip_min=-2.0, clip_max=2.0)
        rewards = np.array([-10.0, -3.0, 0.0, 3.0, 10.0])
        clipped, stats = clipper.clip(rewards)

        assert clipped[0] == -2.0
        assert clipped[1] == -2.0
        assert clipped[2] == 0.0
        assert clipped[3] == 2.0
        assert clipped[4] == 2.0

    def test_no_op_in_range(self):
        """No clipping when all values are in range."""
        clipper = RewardClipper(clip_min=-5.0, clip_max=5.0)
        rewards = np.array([-1.0, 0.0, 1.0, 2.0])
        clipped, stats = clipper.clip(rewards)

        np.testing.assert_array_equal(clipped, rewards)
        assert stats.num_clipped_low == 0
        assert stats.num_clipped_high == 0
        assert stats.fraction_clipped == 0.0

    def test_stats_correct(self):
        """ClipStats accurately reports clipping statistics."""
        clipper = RewardClipper(clip_min=-1.0, clip_max=1.0)
        rewards = np.array([-5.0, -0.5, 0.0, 0.5, 5.0])
        clipped, stats = clipper.clip(rewards)

        assert stats.num_clipped_low == 1
        assert stats.num_clipped_high == 1
        assert stats.total == 5
        assert stats.fraction_clipped == 0.4
        assert stats.original_mean == pytest.approx(0.0)
        assert stats.clipped_mean == pytest.approx(0.0)

    def test_cumulative_tracking(self):
        """Total clipped and seen accumulate across calls."""
        clipper = RewardClipper(clip_min=-1.0, clip_max=1.0)

        clipper.clip(np.array([-5.0, 0.0, 5.0]))
        clipper.clip(np.array([-2.0, 0.5, 2.0]))

        assert clipper.total_clipped == 4
        assert clipper.total_seen == 6
        assert len(clipper.clip_history) == 2

    def test_clip_scalar(self):
        """clip_scalar handles single values."""
        clipper = RewardClipper(clip_min=-1.0, clip_max=1.0)
        assert clipper.clip_scalar(5.0) == 1.0
        assert clipper.clip_scalar(-5.0) == -1.0
        assert clipper.clip_scalar(0.5) == 0.5

    def test_invalid_range(self):
        """Raises ValueError for invalid clip range."""
        with pytest.raises(ValueError):
            RewardClipper(clip_min=5.0, clip_max=2.0)

    def test_properties(self):
        """Properties return correct values."""
        clipper = RewardClipper(clip_min=-3.0, clip_max=3.0)
        assert clipper.clip_min == -3.0
        assert clipper.clip_max == 3.0

    def test_original_stats_preserved(self):
        """Original mean and std computed before clipping."""
        clipper = RewardClipper(clip_min=-1.0, clip_max=1.0)
        rewards = np.array([10.0, 10.0, 10.0])
        _, stats = clipper.clip(rewards)
        assert stats.original_mean == pytest.approx(10.0)
        assert stats.clipped_mean == pytest.approx(1.0)
