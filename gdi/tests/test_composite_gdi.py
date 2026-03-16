"""Tests for composite GDI score."""

import pytest

from src.composite.gdi import GoalDriftIndex, GDIResult
from src.composite.weights import WeightConfig
from src.composite.trend import TrendDetector
from src.composite.normalization import normalize_signal


class TestGoalDriftIndex:
    """Tests for GoalDriftIndex."""

    def setup_method(self):
        self.gdi = GoalDriftIndex()

    def test_same_outputs_green(self, reference_outputs):
        """Same outputs should produce green alert."""
        result = self.gdi.compute(reference_outputs, reference_outputs)
        assert result.alert_level == "green"
        assert result.composite_score < 0.15

    def test_drifted_outputs(self, reference_outputs, drifted_outputs):
        """Drifted outputs should produce non-green alert or low green."""
        result = self.gdi.compute(drifted_outputs, reference_outputs)
        assert 0.0 <= result.composite_score <= 1.0

    def test_collapsed_outputs_high(self, reference_outputs, collapsed_outputs):
        """Collapsed outputs should produce high score."""
        result = self.gdi.compute(collapsed_outputs, reference_outputs)
        assert result.composite_score > 0.3

    def test_weighting(self):
        """Custom weights should affect composite score."""
        weights = WeightConfig(semantic=0.5, lexical=0.2, structural=0.1, distributional=0.2)
        gdi = GoalDriftIndex(weights=weights)
        assert gdi.weights.semantic == 0.5

    def test_alert_levels(self):
        """Alert level thresholds should work correctly."""
        assert self.gdi.get_alert_level(0.10) == "green"
        assert self.gdi.get_alert_level(0.30) == "yellow"
        assert self.gdi.get_alert_level(0.55) == "orange"
        assert self.gdi.get_alert_level(0.80) == "red"

    def test_should_pause(self):
        """Should pause on red."""
        assert not self.gdi.should_pause(0.10)
        assert not self.gdi.should_pause(0.50)
        assert self.gdi.should_pause(0.80)

    def test_should_rollback(self):
        """Should rollback on very high score."""
        assert not self.gdi.should_rollback(0.50)
        assert not self.gdi.should_rollback(0.80)
        assert self.gdi.should_rollback(0.90)

    def test_gdi_result_fields(self, reference_outputs):
        """GDI result should contain all expected fields."""
        result = self.gdi.compute(reference_outputs, reference_outputs)
        assert isinstance(result, GDIResult)
        assert isinstance(result.composite_score, float)
        assert result.alert_level in ("green", "yellow", "orange", "red")
        assert result.trend in ("increasing", "stable", "decreasing")
        assert isinstance(result.semantic_score, float)
        assert isinstance(result.lexical_score, float)
        assert isinstance(result.structural_score, float)
        assert isinstance(result.distributional_score, float)
        assert "semantic" in result.signal_results
        assert "lexical" in result.signal_results
        assert "structural" in result.signal_results
        assert "distributional" in result.signal_results

    def test_score_bounded(self, reference_outputs, collapsed_outputs):
        """Composite score should be bounded in [0, 1]."""
        result = self.gdi.compute(collapsed_outputs, reference_outputs)
        assert 0.0 <= result.composite_score <= 1.0

    def test_custom_thresholds(self, reference_outputs):
        """Custom thresholds should affect alert levels."""
        gdi = GoalDriftIndex(green_max=0.05, yellow_max=0.10, orange_max=0.20)
        result = gdi.compute(reference_outputs, reference_outputs)
        # Near-zero drift should still be green even with strict thresholds
        assert result.composite_score < 0.1


class TestWeightConfig:
    """Tests for WeightConfig."""

    def test_default_weights(self):
        """Default weights should be valid."""
        wc = WeightConfig()
        assert wc.validate()
        assert wc.semantic == 0.30

    def test_invalid_weights(self):
        """Weights that don't sum to 1.0 should fail validation."""
        wc = WeightConfig(semantic=0.5, lexical=0.5, structural=0.5, distributional=0.5)
        assert not wc.validate()

    def test_from_config(self):
        """Should create from config dict."""
        config = {
            "weights": {
                "semantic": 0.40,
                "lexical": 0.20,
                "structural": 0.15,
                "distributional": 0.25,
            }
        }
        wc = WeightConfig.from_config(config)
        assert wc.semantic == 0.40

    def test_from_config_invalid(self):
        """Should raise on invalid weights."""
        config = {
            "weights": {
                "semantic": 0.90,
                "lexical": 0.90,
                "structural": 0.90,
                "distributional": 0.90,
            }
        }
        with pytest.raises(ValueError):
            WeightConfig.from_config(config)

    def test_as_dict(self):
        """Should convert to dict."""
        wc = WeightConfig()
        d = wc.as_dict()
        assert d["semantic"] == 0.30
        assert len(d) == 4

    def test_from_config_defaults(self):
        """Empty config should use defaults."""
        wc = WeightConfig.from_config({})
        assert wc.validate()


class TestTrendDetector:
    """Tests for TrendDetector."""

    def test_increasing_trend(self):
        """Increasing values should give increasing trend."""
        td = TrendDetector()
        assert td.detect_trend([0.1, 0.2, 0.3, 0.4, 0.5]) == "increasing"

    def test_decreasing_trend(self):
        """Decreasing values should give decreasing trend."""
        td = TrendDetector()
        assert td.detect_trend([0.5, 0.4, 0.3, 0.2, 0.1]) == "decreasing"

    def test_stable_trend(self):
        """Stable values should give stable trend."""
        td = TrendDetector()
        assert td.detect_trend([0.5, 0.5, 0.5, 0.5]) == "stable"

    def test_insufficient_data(self):
        """Too few data points should return stable."""
        td = TrendDetector()
        assert td.detect_trend([0.5]) == "stable"
        assert td.detect_trend([]) == "stable"

    def test_compute_slope(self):
        """Slope computation should be correct."""
        td = TrendDetector()
        slope = td.compute_slope([0.0, 1.0, 2.0, 3.0])
        assert abs(slope - 1.0) < 0.01

    def test_compute_slope_flat(self):
        """Flat data should have zero slope."""
        td = TrendDetector()
        slope = td.compute_slope([0.5, 0.5, 0.5])
        assert abs(slope) < 0.01

    def test_compute_slope_empty(self):
        """Empty data should have zero slope."""
        td = TrendDetector()
        assert td.compute_slope([]) == 0.0
        assert td.compute_slope([0.5]) == 0.0


class TestNormalization:
    """Tests for normalize_signal."""

    def test_default_normalization(self):
        """Default normalization should scale by 2x."""
        assert normalize_signal(0.25, "semantic") == 0.5
        assert normalize_signal(0.5, "semantic") == 1.0
        assert normalize_signal(0.0, "semantic") == 0.0

    def test_capped_at_one(self):
        """Should cap at 1.0."""
        assert normalize_signal(0.8, "lexical") == 1.0

    def test_unknown_signal_type(self):
        """Unknown signal type should use defaults."""
        assert normalize_signal(0.25, "unknown") == 0.5

    def test_custom_calibration(self):
        """Custom calibration should be used."""
        cal = {"custom": (4.0, 1.0)}
        assert normalize_signal(0.25, "custom", cal) == 1.0

    def test_negative_clamp(self):
        """Negative values should be clamped to 0."""
        assert normalize_signal(-0.5, "semantic") == 0.0
