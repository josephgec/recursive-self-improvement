"""Tests for alpha schedules and schedule_from_config."""

from __future__ import annotations

import pytest

from src.training.schedules import (
    AlphaSchedule,
    ConstantAlpha,
    ExponentialDecay,
    LinearDecay,
    ZeroAlpha,
    schedule_from_config,
)


# ------------------------------------------------------------------
# ConstantAlpha
# ------------------------------------------------------------------


class TestConstantAlpha:
    def test_always_returns_alpha(self):
        s = ConstantAlpha(0.5)
        for gen in range(10):
            assert s(gen, 10) == 0.5

    def test_different_values(self):
        for val in (0.0, 0.3, 0.7, 1.0):
            s = ConstantAlpha(val)
            assert s(0, 5) == val
            assert s(4, 5) == val

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            ConstantAlpha(1.5)
        with pytest.raises(ValueError):
            ConstantAlpha(-0.1)

    def test_repr(self):
        s = ConstantAlpha(0.5)
        assert "0.5" in repr(s)


# ------------------------------------------------------------------
# LinearDecay
# ------------------------------------------------------------------


class TestLinearDecay:
    def test_endpoints(self):
        s = LinearDecay(alpha_0=1.0, alpha_min=0.0)
        # First generation -> alpha_0
        assert s(0, 10) == pytest.approx(1.0)
        # Last generation -> alpha_min
        assert s(9, 10) == pytest.approx(0.0)

    def test_midpoint(self):
        s = LinearDecay(alpha_0=1.0, alpha_min=0.0)
        mid = s(5, 11)  # t=5 out of 0..10 -> 0.5
        assert mid == pytest.approx(0.5)

    def test_monotonically_decreasing(self):
        s = LinearDecay(alpha_0=1.0, alpha_min=0.0)
        values = [s(g, 15) for g in range(15)]
        for i in range(1, len(values)):
            assert values[i] <= values[i - 1]

    def test_single_generation(self):
        s = LinearDecay(alpha_0=0.8, alpha_min=0.2)
        assert s(0, 1) == 0.8

    def test_custom_range(self):
        s = LinearDecay(alpha_0=0.8, alpha_min=0.2)
        assert s(0, 5) == pytest.approx(0.8)
        assert s(4, 5) == pytest.approx(0.2)


# ------------------------------------------------------------------
# ExponentialDecay
# ------------------------------------------------------------------


class TestExponentialDecay:
    def test_generation_zero(self):
        s = ExponentialDecay(alpha_0=1.0, gamma=0.8)
        assert s(0, 10) == pytest.approx(1.0)

    def test_known_values(self):
        s = ExponentialDecay(alpha_0=1.0, gamma=0.5)
        assert s(1, 10) == pytest.approx(0.5)
        assert s(2, 10) == pytest.approx(0.25)
        assert s(3, 10) == pytest.approx(0.125)

    def test_monotonically_decreasing(self):
        s = ExponentialDecay(alpha_0=1.0, gamma=0.8)
        values = [s(g, 15) for g in range(15)]
        for i in range(1, len(values)):
            assert values[i] < values[i - 1]

    def test_never_negative(self):
        s = ExponentialDecay(alpha_0=1.0, gamma=0.9)
        for g in range(100):
            assert s(g, 100) >= 0.0

    def test_custom_alpha_0(self):
        s = ExponentialDecay(alpha_0=0.5, gamma=0.5)
        assert s(0, 5) == pytest.approx(0.5)
        assert s(1, 5) == pytest.approx(0.25)


# ------------------------------------------------------------------
# ZeroAlpha
# ------------------------------------------------------------------


class TestZeroAlpha:
    def test_always_zero(self):
        s = ZeroAlpha()
        for gen in range(20):
            assert s(gen, 20) == 0.0

    def test_repr(self):
        assert "ZeroAlpha" in repr(ZeroAlpha())


# ------------------------------------------------------------------
# schedule_from_config
# ------------------------------------------------------------------


class TestScheduleFromConfig:
    def test_constant(self):
        s = schedule_from_config({"type": "constant", "alpha": 0.5})
        assert isinstance(s, ConstantAlpha)
        assert s(0, 10) == 0.5

    def test_linear(self):
        s = schedule_from_config(
            {"type": "linear", "alpha_0": 1.0, "alpha_min": 0.0}
        )
        assert isinstance(s, LinearDecay)
        assert s(0, 10) == pytest.approx(1.0)

    def test_exponential(self):
        s = schedule_from_config(
            {"type": "exponential", "alpha_0": 1.0, "gamma": 0.8}
        )
        assert isinstance(s, ExponentialDecay)
        assert s(0, 10) == pytest.approx(1.0)

    def test_zero(self):
        s = schedule_from_config({"type": "zero"})
        assert isinstance(s, ZeroAlpha)
        assert s(0, 10) == 0.0

    def test_missing_type_raises(self):
        with pytest.raises(ValueError, match="must contain a 'type' key"):
            schedule_from_config({"alpha": 0.5})

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown schedule type"):
            schedule_from_config({"type": "cosine_annealing"})

    def test_abc_cannot_instantiate(self):
        """AlphaSchedule is abstract and cannot be directly instantiated."""
        with pytest.raises(TypeError):
            AlphaSchedule()
