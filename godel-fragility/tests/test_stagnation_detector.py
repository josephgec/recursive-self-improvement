"""Tests for the stagnation detector."""

from __future__ import annotations

import pytest

from src.measurement.stagnation_detector import StagnationDetector


@pytest.fixture
def detector() -> StagnationDetector:
    return StagnationDetector(stagnation_threshold=0.01, oscillation_threshold=0.6, window=5)


# ------------------------------------------------------------------ #
# is_stagnant
# ------------------------------------------------------------------ #


class TestIsStagnant:
    def test_flat_values(self, detector: StagnationDetector) -> None:
        values = [0.50, 0.50, 0.50, 0.50, 0.50]
        assert detector.is_stagnant(values) is True

    def test_near_flat_values(self, detector: StagnationDetector) -> None:
        values = [0.500, 0.501, 0.502, 0.503, 0.504]
        assert detector.is_stagnant(values) is True

    def test_not_stagnant(self, detector: StagnationDetector) -> None:
        values = [0.50, 0.55, 0.60, 0.65, 0.70]
        assert detector.is_stagnant(values) is False

    def test_insufficient_data(self, detector: StagnationDetector) -> None:
        values = [0.50, 0.50]
        assert detector.is_stagnant(values) is False

    def test_only_uses_recent_window(self, detector: StagnationDetector) -> None:
        # Old values are variable, recent are flat
        values = [0.10, 0.90, 0.50, 0.50, 0.50, 0.50, 0.50]
        assert detector.is_stagnant(values) is True


# ------------------------------------------------------------------ #
# is_oscillating
# ------------------------------------------------------------------ #


class TestIsOscillating:
    def test_clear_oscillation(self, detector: StagnationDetector) -> None:
        values = [0.5, 0.8, 0.5, 0.8, 0.5, 0.8, 0.5, 0.8]
        assert detector.is_oscillating(values) is True

    def test_no_oscillation_monotonic(self, detector: StagnationDetector) -> None:
        values = [0.50, 0.55, 0.60, 0.65, 0.70]
        assert detector.is_oscillating(values) is False

    def test_insufficient_data(self, detector: StagnationDetector) -> None:
        values = [0.50, 0.80, 0.50]
        assert detector.is_oscillating(values) is False

    def test_small_oscillations_ignored(self, detector: StagnationDetector) -> None:
        # Oscillations smaller than 0.001 abs are ignored
        values = [0.500, 0.5005, 0.500, 0.5005, 0.500]
        assert detector.is_oscillating(values) is False


# ------------------------------------------------------------------ #
# is_improving
# ------------------------------------------------------------------ #


class TestIsImproving:
    def test_improving(self, detector: StagnationDetector) -> None:
        values = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
        assert detector.is_improving(values) is True

    def test_declining(self, detector: StagnationDetector) -> None:
        values = [0.80, 0.75, 0.70, 0.65, 0.60]
        assert detector.is_improving(values) is False

    def test_flat(self, detector: StagnationDetector) -> None:
        values = [0.50, 0.50, 0.50, 0.50, 0.50]
        assert detector.is_improving(values) is False

    def test_insufficient_data(self, detector: StagnationDetector) -> None:
        values = [0.50, 0.60]
        assert detector.is_improving(values) is False


# ------------------------------------------------------------------ #
# get_trend
# ------------------------------------------------------------------ #


class TestGetTrend:
    def test_improving(self, detector: StagnationDetector) -> None:
        values = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
        assert detector.get_trend(values) == "improving"

    def test_declining(self, detector: StagnationDetector) -> None:
        values = [0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
        assert detector.get_trend(values) == "declining"

    def test_flat(self, detector: StagnationDetector) -> None:
        values = [0.50, 0.50, 0.50, 0.50, 0.50, 0.50]
        assert detector.get_trend(values) == "flat"

    def test_oscillating(self, detector: StagnationDetector) -> None:
        values = [0.5, 0.8, 0.5, 0.8, 0.5, 0.8, 0.5, 0.8]
        assert detector.get_trend(values) == "oscillating"

    def test_insufficient_data(self, detector: StagnationDetector) -> None:
        values = [0.50, 0.60]
        assert detector.get_trend(values) == "insufficient_data"
