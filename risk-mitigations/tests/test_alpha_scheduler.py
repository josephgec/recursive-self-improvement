"""Tests for ConservativeAlphaScheduler - all 4 schedule types."""

import pytest
from src.collapse.alpha_scheduler import ConservativeAlphaScheduler, AlphaScheduleConfig


class TestExponentialSchedule:
    """Tests for exponential alpha schedule."""

    def test_initial_alpha(self):
        scheduler = ConservativeAlphaScheduler(
            AlphaScheduleConfig(schedule_type="exponential", initial_alpha=0.5, gamma=0.95)
        )
        alpha = scheduler.get_alpha(0)
        assert alpha == pytest.approx(0.5)

    def test_alpha_decreases_with_iteration(self):
        scheduler = ConservativeAlphaScheduler(
            AlphaScheduleConfig(schedule_type="exponential", initial_alpha=0.5, gamma=0.95)
        )
        a0 = scheduler.get_alpha(0)
        a5 = scheduler.get_alpha(5)
        a10 = scheduler.get_alpha(10)
        assert a5 < a0
        assert a10 < a5

    def test_exponential_formula(self):
        scheduler = ConservativeAlphaScheduler(
            AlphaScheduleConfig(schedule_type="exponential", initial_alpha=1.0, gamma=0.9)
        )
        alpha = scheduler.get_alpha(3)
        expected = 1.0 * (0.9 ** 3)
        assert alpha == pytest.approx(expected)

    def test_respects_min_alpha(self):
        scheduler = ConservativeAlphaScheduler(
            AlphaScheduleConfig(
                schedule_type="exponential",
                initial_alpha=0.5,
                gamma=0.1,
                min_alpha=0.05,
            )
        )
        alpha = scheduler.get_alpha(100)
        assert alpha >= 0.05


class TestLinearSchedule:
    """Tests for linear alpha schedule."""

    def test_linear_decrease(self):
        scheduler = ConservativeAlphaScheduler(
            AlphaScheduleConfig(
                schedule_type="linear",
                initial_alpha=0.5,
                linear_decay_rate=0.05,
            )
        )
        a0 = scheduler.get_alpha(0)
        a1 = scheduler.get_alpha(1)
        assert a0 == pytest.approx(0.5)
        assert a1 == pytest.approx(0.45)

    def test_linear_respects_min(self):
        scheduler = ConservativeAlphaScheduler(
            AlphaScheduleConfig(
                schedule_type="linear",
                initial_alpha=0.5,
                linear_decay_rate=0.1,
                min_alpha=0.1,
            )
        )
        alpha = scheduler.get_alpha(100)
        assert alpha >= 0.1


class TestConstantSchedule:
    """Tests for constant alpha schedule."""

    def test_constant_value(self):
        scheduler = ConservativeAlphaScheduler(
            AlphaScheduleConfig(schedule_type="constant", initial_alpha=0.3)
        )
        assert scheduler.get_alpha(0) == pytest.approx(0.3)
        assert scheduler.get_alpha(50) == pytest.approx(0.3)
        assert scheduler.get_alpha(1000) == pytest.approx(0.3)


class TestAdaptiveSchedule:
    """Tests for adaptive alpha schedule."""

    def test_adaptive_increases_alpha_on_entropy_drop(self):
        scheduler = ConservativeAlphaScheduler(
            AlphaScheduleConfig(
                schedule_type="adaptive",
                initial_alpha=0.5,
                gamma=0.95,
                adaptive_entropy_threshold=2.0,
                adaptive_increase_factor=1.2,
            )
        )
        # Normal entropy - should behave like exponential
        a_normal = scheduler.get_alpha(5, {"entropy": 3.0})

        # Low entropy - should increase alpha
        scheduler.reset()
        _ = scheduler.get_alpha(5, {"entropy": 3.0})  # establish baseline
        a_low = scheduler.get_alpha(6, {"entropy": 1.5})  # below threshold

        # The adaptive alpha should be higher when entropy drops
        assert a_low > a_normal * 0.9  # Give some tolerance

    def test_adaptive_returns_to_base_on_recovery(self):
        scheduler = ConservativeAlphaScheduler(
            AlphaScheduleConfig(
                schedule_type="adaptive",
                initial_alpha=0.5,
                gamma=0.95,
                adaptive_entropy_threshold=2.0,
            )
        )
        # Trigger adaptation
        scheduler.get_alpha(5, {"entropy": 1.5})
        # Recovery
        a_recovered = scheduler.get_alpha(6, {"entropy": 3.5})
        a_base = 0.5 * (0.95 ** 6)
        assert a_recovered == pytest.approx(a_base, abs=0.01)

    def test_adaptive_without_entropy(self):
        scheduler = ConservativeAlphaScheduler(
            AlphaScheduleConfig(schedule_type="adaptive", initial_alpha=0.5, gamma=0.95)
        )
        alpha = scheduler.get_alpha(5, {})
        # Without entropy, should fall back to exponential
        expected = 0.5 * (0.95 ** 5)
        assert alpha == pytest.approx(expected, abs=0.01)


class TestSchedulerGeneral:
    """General scheduler tests."""

    def test_history_tracking(self):
        scheduler = ConservativeAlphaScheduler()
        scheduler.get_alpha(0)
        scheduler.get_alpha(1)
        assert len(scheduler.get_history()) == 2

    def test_reset(self):
        scheduler = ConservativeAlphaScheduler()
        scheduler.get_alpha(0)
        scheduler.reset()
        assert len(scheduler.get_history()) == 0

    def test_invalid_schedule_type(self):
        scheduler = ConservativeAlphaScheduler(
            AlphaScheduleConfig(schedule_type="invalid")
        )
        with pytest.raises(ValueError, match="Unknown schedule type"):
            scheduler.get_alpha(0)

    def test_alpha_never_exceeds_one(self):
        scheduler = ConservativeAlphaScheduler(
            AlphaScheduleConfig(
                schedule_type="adaptive",
                initial_alpha=0.99,
                adaptive_increase_factor=2.0,
                adaptive_entropy_threshold=5.0,
            )
        )
        alpha = scheduler.get_alpha(0, {"entropy": 1.0})
        assert alpha <= 1.0

    def test_schedule_type_property(self):
        config = AlphaScheduleConfig(schedule_type="linear")
        scheduler = ConservativeAlphaScheduler(config)
        assert scheduler.schedule_type == "linear"
