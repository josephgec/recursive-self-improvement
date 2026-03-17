"""Tests for Criterion 1: Sustained Improvement."""

import pytest

from src.criteria.base import Evidence
from src.criteria.sustained_improvement import (
    SustainedImprovementCriterion,
    _mann_kendall,
)
from tests.conftest import build_passing_evidence, build_failing_evidence


class TestMannKendall:
    """Test the Mann-Kendall trend test implementation."""

    def test_increasing_trend(self):
        """Monotonically increasing data should show significant trend."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        s, p, trend = _mann_kendall(data)
        assert s > 0
        assert p < 0.05
        assert trend == "increasing"

    def test_decreasing_trend(self):
        """Monotonically decreasing data should show decreasing trend."""
        data = [5.0, 4.0, 3.0, 2.0, 1.0]
        s, p, trend = _mann_kendall(data)
        assert s < 0
        assert trend == "decreasing"

    def test_flat_data(self):
        """Constant data should show no trend."""
        data = [3.0, 3.0, 3.0, 3.0, 3.0]
        s, p, trend = _mann_kendall(data)
        assert s == 0
        assert trend == "no trend"

    def test_short_data(self):
        """Data with fewer than 3 points returns no trend."""
        data = [1.0, 2.0]
        s, p, trend = _mann_kendall(data)
        assert p == 1.0
        assert trend == "no trend"

    def test_noisy_increasing(self):
        """Noisy but generally increasing data."""
        data = [50.0, 55.0, 53.0, 60.0, 65.0]
        s, p, trend = _mann_kendall(data)
        assert s > 0
        assert trend == "increasing"


class TestSustainedImprovement:
    """Test the Sustained Improvement criterion."""

    def test_passing_case(self, passing_evidence):
        """Increasing curve with sufficient gain and divergence passes."""
        criterion = SustainedImprovementCriterion()
        result = criterion.evaluate(passing_evidence)

        assert result.passed is True
        assert result.confidence > 0.5
        assert result.criterion_name == "Sustained Improvement"

        # Check sub-results
        sub = result.details["sub_results"]
        assert sub["trend"]["passed"] is True
        assert sub["total_gain"]["passed"] is True
        assert sub["collapse_divergence"]["passed"] is True

    def test_flat_curve_fails_gain(self):
        """Flat improvement curve fails the total gain sub-test."""
        evidence = Evidence(
            phase_0={"score": 50.0, "collapse_score": 50.0},
            phase_1={"score": 51.0, "collapse_score": 50.0},
            phase_2={"score": 50.5, "collapse_score": 50.0},
            phase_3={"score": 51.5, "collapse_score": 50.0},
            phase_4={"score": 52.0, "collapse_score": 50.0},
        )
        criterion = SustainedImprovementCriterion(min_total_gain_pp=5.0)
        result = criterion.evaluate(evidence)

        assert result.passed is False
        sub = result.details["sub_results"]
        assert sub["total_gain"]["passed"] is False
        assert sub["total_gain"]["gain_pp"] < 5.0

    def test_declining_curve_fails_trend(self):
        """Declining curve fails the trend sub-test."""
        evidence = Evidence(
            phase_0={"score": 60.0, "collapse_score": 50.0},
            phase_1={"score": 58.0, "collapse_score": 49.0},
            phase_2={"score": 55.0, "collapse_score": 48.0},
            phase_3={"score": 53.0, "collapse_score": 47.0},
            phase_4={"score": 50.0, "collapse_score": 46.0},
        )
        criterion = SustainedImprovementCriterion()
        result = criterion.evaluate(evidence)

        assert result.passed is False
        sub = result.details["sub_results"]
        assert sub["trend"]["direction"] == "decreasing"

    def test_low_divergence_fails(self):
        """Insufficient divergence from collapse baseline fails."""
        evidence = Evidence(
            phase_0={"score": 50.0, "collapse_score": 50.0},
            phase_1={"score": 55.0, "collapse_score": 54.0},
            phase_2={"score": 58.0, "collapse_score": 56.0},
            phase_3={"score": 62.0, "collapse_score": 60.0},
            phase_4={"score": 65.0, "collapse_score": 63.0},
        )
        criterion = SustainedImprovementCriterion(
            min_collapse_divergence_pp=10.0
        )
        result = criterion.evaluate(evidence)

        assert result.passed is False
        sub = result.details["sub_results"]
        assert sub["collapse_divergence"]["passed"] is False

    def test_custom_thresholds(self, passing_evidence):
        """Custom thresholds can make a passing case stricter."""
        criterion = SustainedImprovementCriterion(
            min_total_gain_pp=20.0,  # Very strict
        )
        result = criterion.evaluate(passing_evidence)
        assert result.passed is False

    def test_properties(self):
        """Test criterion properties."""
        criterion = SustainedImprovementCriterion()
        assert criterion.name == "Sustained Improvement"
        assert "improvement" in criterion.description.lower()
        assert "Mann-Kendall" in criterion.threshold_description
        assert len(criterion.required_evidence) >= 5

    def test_result_summary(self, passing_evidence):
        """Test that result summary is formatted correctly."""
        criterion = SustainedImprovementCriterion()
        result = criterion.evaluate(passing_evidence)
        summary = result.summary()
        assert "PASS" in summary or "FAIL" in summary
        assert "Sustained Improvement" in summary
