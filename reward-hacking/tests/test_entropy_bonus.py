"""Tests for entropy bonus module."""

import pytest

from src.eppo.entropy_bonus import EntropyBonus, EntropyBonusState


class TestEntropyBonusCoefficient:
    """Test coefficient (decay) mode."""

    def test_initial_beta(self):
        """Initial beta matches configuration."""
        bonus = EntropyBonus(initial_beta=0.05, mode="coefficient")
        assert bonus.current_beta == 0.05

    def test_decay_step(self):
        """Beta decays after each step."""
        bonus = EntropyBonus(initial_beta=0.1, mode="coefficient", decay_rate=0.9)
        bonus.step(current_entropy=2.0)
        assert abs(bonus.current_beta - 0.09) < 1e-6

    def test_multi_step_decay(self):
        """Beta decays multiplicatively over multiple steps."""
        bonus = EntropyBonus(initial_beta=1.0, mode="coefficient", decay_rate=0.5)
        bonus.step(1.0)
        assert abs(bonus.current_beta - 0.5) < 1e-6
        bonus.step(1.0)
        assert abs(bonus.current_beta - 0.25) < 1e-6

    def test_min_beta_enforcement(self):
        """Beta does not go below min_beta."""
        bonus = EntropyBonus(
            initial_beta=0.01,
            mode="coefficient",
            decay_rate=0.1,
            min_beta=0.005,
        )
        for _ in range(20):
            bonus.step(1.0)
        assert bonus.current_beta >= 0.005

    def test_compute(self):
        """compute returns beta * entropy."""
        bonus = EntropyBonus(initial_beta=0.1)
        result = bonus.compute(2.5)
        assert abs(result - 0.25) < 1e-6

    def test_history_tracking(self):
        """History tracks beta values over steps."""
        bonus = EntropyBonus(initial_beta=0.1, mode="coefficient", decay_rate=0.9)
        for _ in range(5):
            bonus.step(1.0)
        assert len(bonus.history) == 5
        assert bonus.history[0] == 0.1  # First recorded is initial


class TestEntropyBonusTarget:
    """Test target (adaptive) mode."""

    def test_target_mode_increase(self):
        """Beta increases when entropy is below target."""
        bonus = EntropyBonus(
            initial_beta=0.01,
            mode="target",
            entropy_target=3.0,
            adaptive_alpha=0.1,
        )
        # Current entropy is well below target
        bonus.step(current_entropy=1.0)
        assert bonus.current_beta > 0.01

    def test_target_mode_decrease(self):
        """Beta decreases when entropy is above target."""
        bonus = EntropyBonus(
            initial_beta=0.5,
            mode="target",
            entropy_target=1.0,
            adaptive_alpha=0.1,
            min_beta=0.001,
        )
        # Current entropy is above target
        bonus.step(current_entropy=3.0)
        assert bonus.current_beta < 0.5

    def test_target_mode_min_beta(self):
        """Target mode respects min_beta."""
        bonus = EntropyBonus(
            initial_beta=0.01,
            mode="target",
            entropy_target=1.0,
            adaptive_alpha=1.0,
            min_beta=0.005,
        )
        # Push beta down hard
        for _ in range(50):
            bonus.step(current_entropy=10.0)
        assert bonus.current_beta >= 0.005

    def test_target_adaptation(self):
        """Beta adapts toward maintaining the target entropy."""
        bonus = EntropyBonus(
            initial_beta=0.01,
            mode="target",
            entropy_target=2.0,
            adaptive_alpha=0.05,
        )
        # Simulate low entropy driving beta up
        for _ in range(10):
            bonus.step(current_entropy=1.0)

        beta_after_low = bonus.current_beta

        # Now simulate high entropy driving beta down
        for _ in range(10):
            bonus.step(current_entropy=5.0)

        assert bonus.current_beta < beta_after_low


class TestEntropyBonusGeneral:
    """General entropy bonus tests."""

    def test_reset(self):
        """Reset restores initial state."""
        bonus = EntropyBonus(initial_beta=0.1, mode="coefficient", decay_rate=0.5)
        for _ in range(10):
            bonus.step(1.0)
        bonus.reset()
        assert bonus.current_beta == 0.1
        assert bonus.step_count == 0
        assert bonus.history == []

    def test_get_state(self):
        """get_state returns correct snapshot."""
        bonus = EntropyBonus(initial_beta=0.1, mode="target", entropy_target=2.0)
        state = bonus.get_state()
        assert isinstance(state, EntropyBonusState)
        assert state.beta == 0.1
        assert state.mode == "target"
        assert state.entropy_target == 2.0

    def test_invalid_mode(self):
        """Invalid mode raises ValueError on step."""
        bonus = EntropyBonus(initial_beta=0.1)
        bonus._mode = "invalid"
        with pytest.raises(ValueError, match="Unknown mode"):
            bonus.step(1.0)

    def test_step_count(self):
        """Step count increments correctly."""
        bonus = EntropyBonus(initial_beta=0.1)
        for i in range(5):
            bonus.step(1.0)
        assert bonus.step_count == 5
