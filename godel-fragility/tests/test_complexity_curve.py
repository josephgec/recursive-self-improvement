"""Tests for the complexity curve tracker."""

from __future__ import annotations

import numpy as np
import pytest

from src.measurement.complexity_curve import ComplexityCurveTracker, ComplexityDataPoint


# ------------------------------------------------------------------ #
# ComplexityDataPoint
# ------------------------------------------------------------------ #


class TestComplexityDataPoint:
    def test_defaults(self) -> None:
        dp = ComplexityDataPoint(complexity=50, accuracy=0.8)
        assert dp.complexity == 50
        assert dp.accuracy == 0.8
        assert dp.comprehension_score is None
        assert dp.modification_success is None
        assert dp.iteration == 0

    def test_with_all_fields(self) -> None:
        dp = ComplexityDataPoint(
            complexity=100,
            accuracy=0.6,
            comprehension_score=0.7,
            modification_success=True,
            iteration=5,
        )
        assert dp.comprehension_score == 0.7
        assert dp.modification_success is True
        assert dp.iteration == 5


# ------------------------------------------------------------------ #
# ComplexityCurveTracker.record / .data
# ------------------------------------------------------------------ #


class TestRecord:
    def test_record_and_retrieve(self) -> None:
        tracker = ComplexityCurveTracker()
        dp = ComplexityDataPoint(complexity=10, accuracy=0.9)
        tracker.record(dp)
        assert len(tracker.data) == 1
        assert tracker.data[0].complexity == 10

    def test_data_returns_copy(self) -> None:
        tracker = ComplexityCurveTracker()
        tracker.record(ComplexityDataPoint(complexity=10, accuracy=0.9))
        data = tracker.data
        data.append(ComplexityDataPoint(complexity=99, accuracy=0.1))
        assert len(tracker.data) == 1  # original unaffected


# ------------------------------------------------------------------ #
# compute_success_curve
# ------------------------------------------------------------------ #


class TestComputeSuccessCurve:
    def test_empty(self) -> None:
        tracker = ComplexityCurveTracker()
        assert tracker.compute_success_curve() == []

    def test_single_complexity(self) -> None:
        tracker = ComplexityCurveTracker()
        for acc in [0.8, 0.3, 0.6]:
            tracker.record(ComplexityDataPoint(complexity=50, accuracy=acc))
        curve = tracker.compute_success_curve()
        assert len(curve) == 1
        # 0.8 >= 0.5 (yes), 0.3 < 0.5 (no), 0.6 >= 0.5 (yes) => 2/3
        assert curve[0][0] == 50.0
        assert curve[0][1] == pytest.approx(2 / 3)

    def test_decreasing_success(self) -> None:
        tracker = ComplexityCurveTracker()
        # Low complexity -> high accuracy, high complexity -> low accuracy
        for c in range(10, 110, 5):
            acc = max(0.0, 1.0 - c / 100.0)
            tracker.record(ComplexityDataPoint(complexity=c, accuracy=acc))
        curve = tracker.compute_success_curve(bins=5)
        assert len(curve) > 0
        # First bin should have higher success rate than last bin
        assert curve[0][1] >= curve[-1][1]

    def test_bins_parameter(self) -> None:
        tracker = ComplexityCurveTracker()
        for c in range(10, 110, 10):
            tracker.record(ComplexityDataPoint(complexity=c, accuracy=0.8))
        curve_5 = tracker.compute_success_curve(bins=5)
        curve_10 = tracker.compute_success_curve(bins=10)
        assert len(curve_5) <= 5
        assert len(curve_10) <= 10


# ------------------------------------------------------------------ #
# find_complexity_ceiling
# ------------------------------------------------------------------ #


class TestFindComplexityCeiling:
    def test_insufficient_data(self) -> None:
        tracker = ComplexityCurveTracker()
        tracker.record(ComplexityDataPoint(complexity=10, accuracy=0.9))
        assert tracker.find_complexity_ceiling() is None

    def test_ceiling_found(self) -> None:
        tracker = ComplexityCurveTracker()
        # Success rate crosses 0.5 somewhere around complexity 50-60
        for c in range(10, 110, 5):
            acc = max(0.0, 1.0 - c / 100.0)
            tracker.record(ComplexityDataPoint(complexity=c, accuracy=acc))
        ceiling = tracker.find_complexity_ceiling()
        assert ceiling is not None
        # Should be around 50-60
        assert 30 <= ceiling <= 80

    def test_always_above_50(self) -> None:
        tracker = ComplexityCurveTracker()
        for c in range(10, 60, 5):
            tracker.record(ComplexityDataPoint(complexity=c, accuracy=0.9))
        ceiling = tracker.find_complexity_ceiling()
        # Always above 0.5, ceiling is max complexity in data
        assert ceiling is not None
        assert ceiling == pytest.approx(55.0)

    def test_always_below_50(self) -> None:
        tracker = ComplexityCurveTracker()
        for c in range(10, 60, 5):
            tracker.record(ComplexityDataPoint(complexity=c, accuracy=0.2))
        ceiling = tracker.find_complexity_ceiling()
        # Always below 0.5, ceiling is min complexity in data
        assert ceiling is not None
        assert ceiling == pytest.approx(10.0)


# ------------------------------------------------------------------ #
# compute_comprehension_curve
# ------------------------------------------------------------------ #


class TestComputeComprehensionCurve:
    def test_no_comprehension_data(self) -> None:
        tracker = ComplexityCurveTracker()
        tracker.record(ComplexityDataPoint(complexity=10, accuracy=0.9))
        assert tracker.compute_comprehension_curve() == []

    def test_with_comprehension_data(self) -> None:
        tracker = ComplexityCurveTracker()
        for c in range(10, 60, 5):
            score = max(0.0, 1.0 - c / 100.0)
            tracker.record(
                ComplexityDataPoint(complexity=c, accuracy=0.8, comprehension_score=score)
            )
        curve = tracker.compute_comprehension_curve(bins=5)
        assert len(curve) > 0
        # Comprehension should decrease with complexity
        assert curve[0][1] >= curve[-1][1]

    def test_single_complexity_comprehension(self) -> None:
        tracker = ComplexityCurveTracker()
        for score in [0.8, 0.6, 0.7]:
            tracker.record(
                ComplexityDataPoint(complexity=50, accuracy=0.9, comprehension_score=score)
            )
        curve = tracker.compute_comprehension_curve()
        assert len(curve) == 1
        assert curve[0][0] == 50.0
        assert curve[0][1] == pytest.approx(0.7)


# ------------------------------------------------------------------ #
# test_correlation
# ------------------------------------------------------------------ #


class TestCorrelation:
    def test_insufficient_data(self) -> None:
        tracker = ComplexityCurveTracker()
        tracker.record(ComplexityDataPoint(complexity=10, accuracy=0.9))
        assert tracker.test_correlation() is None

    def test_negative_correlation(self) -> None:
        tracker = ComplexityCurveTracker()
        for c in range(10, 60, 5):
            acc = 1.0 - c / 100.0
            tracker.record(ComplexityDataPoint(complexity=c, accuracy=acc))
        corr = tracker.test_correlation()
        assert corr is not None
        assert corr < -0.5

    def test_no_correlation_constant(self) -> None:
        tracker = ComplexityCurveTracker()
        for c in range(10, 60, 5):
            tracker.record(ComplexityDataPoint(complexity=c, accuracy=0.5))
        corr = tracker.test_correlation()
        assert corr == 0.0

    def test_constant_complexity(self) -> None:
        tracker = ComplexityCurveTracker()
        for acc in [0.9, 0.8, 0.7]:
            tracker.record(ComplexityDataPoint(complexity=50, accuracy=acc))
        corr = tracker.test_correlation()
        assert corr == 0.0
