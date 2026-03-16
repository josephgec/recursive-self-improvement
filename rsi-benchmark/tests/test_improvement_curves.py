"""Test improvement curves: recording, curve properties, growth model fitting."""

import math

import pytest

from src.evaluation.improvement_curves import ImprovementCurveTracker, GrowthModel


class TestImprovementCurveTracker:
    """Test the improvement curve tracker."""

    def test_record_and_get(self):
        tracker = ImprovementCurveTracker()
        tracker.record("math500", 0, 0.60)
        tracker.record("math500", 1, 0.62)
        tracker.record("math500", 2, 0.64)

        curve = tracker.get_curve("math500")
        assert len(curve) == 3
        assert curve[0] == (0, 0.60)
        assert curve[2] == (2, 0.64)

    def test_get_empty_curve(self):
        tracker = ImprovementCurveTracker()
        curve = tracker.get_curve("nonexistent")
        assert curve == []

    def test_get_all_curves(self):
        tracker = ImprovementCurveTracker()
        tracker.record("math500", 0, 0.60)
        tracker.record("arc_agi", 0, 0.55)
        curves = tracker.get_all_curves()
        assert "math500" in curves
        assert "arc_agi" in curves

    def test_sorted_by_iteration(self):
        tracker = ImprovementCurveTracker()
        tracker.record("math500", 2, 0.64)
        tracker.record("math500", 0, 0.60)
        tracker.record("math500", 1, 0.62)
        curve = tracker.get_curve("math500")
        assert [x for x, _ in curve] == [0, 1, 2]


class TestImprovingCurve:
    """Test detection of improving curves."""

    def test_is_improving(self):
        tracker = ImprovementCurveTracker()
        for i in range(10):
            tracker.record("test", i, 0.60 + 0.015 * i)
        assert tracker.is_improving("test")

    def test_is_not_improving_when_declining(self):
        tracker = ImprovementCurveTracker()
        for i in range(10):
            tracker.record("test", i, 0.80 - 0.02 * i)
        assert not tracker.is_improving("test")

    def test_is_improving_insufficient_data(self):
        tracker = ImprovementCurveTracker()
        tracker.record("test", 0, 0.60)
        assert not tracker.is_improving("test")


class TestDegradingCurve:
    """Test detection of degrading curves."""

    def test_is_degrading(self):
        tracker = ImprovementCurveTracker()
        for i in range(10):
            tracker.record("test", i, 0.65 - 0.01 * i)
        assert tracker.is_degrading("test")

    def test_is_not_degrading_when_improving(self):
        tracker = ImprovementCurveTracker()
        for i in range(10):
            tracker.record("test", i, 0.60 + 0.015 * i)
        assert not tracker.is_degrading("test")

    def test_is_degrading_insufficient_data(self):
        tracker = ImprovementCurveTracker()
        tracker.record("test", 0, 0.65)
        assert not tracker.is_degrading("test")


class TestPlateauedCurve:
    """Test detection of plateaued curves."""

    def test_is_plateaued(self):
        tracker = ImprovementCurveTracker()
        for i in range(10):
            tracker.record("test", i, 0.75 + 0.001 * ((-1) ** i))
        assert tracker.is_plateaued("test", tolerance=0.005)

    def test_is_not_plateaued_when_improving(self):
        tracker = ImprovementCurveTracker()
        for i in range(10):
            tracker.record("test", i, 0.60 + 0.02 * i)
        assert not tracker.is_plateaued("test")

    def test_is_plateaued_insufficient_data(self):
        tracker = ImprovementCurveTracker()
        tracker.record("test", 0, 0.75)
        assert not tracker.is_plateaued("test")


class TestSustainedImprovement:
    """Test sustained improvement computation."""

    def test_sustained_improvement_monotonic(self):
        tracker = ImprovementCurveTracker()
        for i in range(10):
            tracker.record("test", i, 0.60 + 0.015 * i)
        streak = tracker.compute_sustained_improvement("test")
        assert streak == 9  # 10 points, 9 improvements

    def test_sustained_improvement_with_dip(self):
        tracker = ImprovementCurveTracker()
        values = [0.60, 0.62, 0.64, 0.63, 0.65, 0.67, 0.69, 0.71]
        for i, v in enumerate(values):
            tracker.record("test", i, v)
        streak = tracker.compute_sustained_improvement("test")
        assert streak == 4  # Longest streak: 0.63 -> 0.65 -> 0.67 -> 0.69 -> 0.71

    def test_sustained_improvement_empty(self):
        tracker = ImprovementCurveTracker()
        assert tracker.compute_sustained_improvement("test") == 0

    def test_sustained_improvement_single(self):
        tracker = ImprovementCurveTracker()
        tracker.record("test", 0, 0.60)
        assert tracker.compute_sustained_improvement("test") == 0


class TestTotalImprovement:
    """Test total improvement computation."""

    def test_total_improvement_positive(self):
        tracker = ImprovementCurveTracker()
        tracker.record("test", 0, 0.60)
        tracker.record("test", 14, 0.81)
        total = tracker.compute_total_improvement("test")
        assert abs(total - 0.21) < 0.001

    def test_total_improvement_negative(self):
        tracker = ImprovementCurveTracker()
        tracker.record("test", 0, 0.65)
        tracker.record("test", 14, 0.51)
        total = tracker.compute_total_improvement("test")
        assert total < 0

    def test_total_improvement_empty(self):
        tracker = ImprovementCurveTracker()
        assert tracker.compute_total_improvement("test") == 0.0


class TestGrowthModelFitting:
    """Test growth model fitting."""

    def test_fit_logarithmic(self):
        tracker = ImprovementCurveTracker()
        # Generate log-like data
        for i in range(1, 16):
            tracker.record("test", i, 0.5 + 0.1 * math.log(i))
        model = tracker.fit_growth_model("test", "logarithmic")
        assert model.model_type == "logarithmic"
        assert model.r_squared > 0.9
        assert "a" in model.params
        assert "b" in model.params

    def test_fit_power(self):
        tracker = ImprovementCurveTracker()
        for i in range(1, 16):
            tracker.record("test", i, 0.5 + 0.05 * (i ** 0.5))
        model = tracker.fit_growth_model("test", "power")
        assert model.model_type == "power"
        assert model.r_squared > 0.5

    def test_fit_sigmoid(self):
        tracker = ImprovementCurveTracker()
        for i in range(1, 16):
            v = 0.9 / (1 + math.exp(-0.5 * (i - 7)))
            tracker.record("test", i, v)
        model = tracker.fit_growth_model("test", "sigmoid")
        assert model.model_type == "sigmoid"
        assert "L" in model.params
        assert "k" in model.params

    def test_fit_empty(self):
        tracker = ImprovementCurveTracker()
        model = tracker.fit_growth_model("test", "logarithmic")
        assert model.r_squared == 0.0

    def test_unknown_model_raises(self):
        tracker = ImprovementCurveTracker()
        tracker.record("test", 1, 0.5)
        with pytest.raises(ValueError):
            tracker.fit_growth_model("test", "unknown")

    def test_growth_model_predict(self):
        model = GrowthModel(
            model_type="logarithmic",
            params={"a": 0.1, "b": 0.5},
            r_squared=0.95,
        )
        pred = model.predict(math.e)  # ln(e) = 1, so 0.1*1 + 0.5 = 0.6
        assert abs(pred - 0.6) < 0.01

    def test_growth_model_predict_power(self):
        model = GrowthModel(
            model_type="power",
            params={"a": 1.0, "b": 0.5, "c": 0.0},
            r_squared=0.95,
        )
        pred = model.predict(4.0)  # 1.0 * 4^0.5 + 0 = 2.0
        assert abs(pred - 2.0) < 0.01

    def test_growth_model_predict_sigmoid(self):
        model = GrowthModel(
            model_type="sigmoid",
            params={"L": 1.0, "k": 1.0, "x0": 0.0},
            r_squared=0.95,
        )
        pred = model.predict(0.0)  # 1/(1+e^0) = 0.5
        assert abs(pred - 0.5) < 0.01


class TestPlotData:
    """Test plot data generation."""

    def test_plot_improvement_curves(self):
        tracker = ImprovementCurveTracker()
        for i in range(10):
            tracker.record("math500", i, 0.60 + 0.015 * i)
            tracker.record("arc_agi", i, 0.55 + 0.01 * i)

        plot_data = tracker.plot_improvement_curves()
        assert "math500" in plot_data
        assert "arc_agi" in plot_data
        assert "iterations" in plot_data["math500"]
        assert "accuracy" in plot_data["math500"]
        assert "total_improvement" in plot_data["math500"]
        assert "sustained_improvement" in plot_data["math500"]
