"""Tests for Criterion 2: Paradigm Improvement."""

import pytest

from src.criteria.base import Evidence
from src.criteria.paradigm_improvement import (
    ParadigmImprovementCriterion,
    _paired_t_test,
)
from tests.conftest import build_passing_evidence, build_partial_evidence


class TestPairedTTest:
    """Test the paired t-test implementation."""

    def test_significant_difference(self):
        """Clear difference should yield small p-value."""
        with_scores = [60.0, 65.0, 70.0, 75.0, 80.0]
        without_scores = [50.0, 55.0, 60.0, 65.0, 70.0]
        t_stat, p_value, mean_diff = _paired_t_test(with_scores, without_scores)

        assert mean_diff == 10.0
        assert t_stat > 0
        assert p_value < 0.05

    def test_no_difference(self):
        """Identical scores should yield p ~ 1."""
        scores = [50.0, 55.0, 60.0, 65.0, 70.0]
        t_stat, p_value, mean_diff = _paired_t_test(scores, scores)

        assert mean_diff == 0.0
        assert p_value == 1.0

    def test_single_observation(self):
        """Single observation is insufficient."""
        t_stat, p_value, mean_diff = _paired_t_test([60.0], [50.0])
        assert p_value == 1.0

    def test_perfect_difference(self):
        """Constant positive difference should be highly significant."""
        with_scores = [60.0, 65.0, 70.0, 75.0, 80.0]
        without_scores = [55.0, 60.0, 65.0, 70.0, 75.0]
        t_stat, p_value, mean_diff = _paired_t_test(with_scores, without_scores)

        assert mean_diff == 5.0
        assert t_stat == float("inf")
        assert p_value == 0.0


class TestParadigmImprovement:
    """Test the Paradigm Improvement criterion."""

    def test_all_paradigms_pass(self, passing_evidence):
        """All 4 paradigms exceeding thresholds should pass."""
        criterion = ParadigmImprovementCriterion()
        result = criterion.evaluate(passing_evidence)

        assert result.passed is True
        assert result.confidence > 0.5

        sub = result.details["sub_results"]
        for paradigm in ["symcode", "godel", "soar", "rlm"]:
            assert sub[paradigm]["passed"] is True, f"{paradigm} should pass"

    def test_one_paradigm_fails(self, partial_evidence):
        """One paradigm below threshold should fail overall."""
        criterion = ParadigmImprovementCriterion()
        result = criterion.evaluate(partial_evidence)

        assert result.passed is False
        sub = result.details["sub_results"]
        assert sub["symcode"]["passed"] is False

    def test_effect_sizes_correct(self, passing_evidence):
        """Verify effect sizes are computed correctly."""
        criterion = ParadigmImprovementCriterion()
        result = criterion.evaluate(passing_evidence)

        sub = result.details["sub_results"]
        # Each paradigm should have mean_diff > its threshold
        assert sub["symcode"]["mean_diff"] >= 5.0
        assert sub["godel"]["mean_diff"] >= 2.0
        assert sub["soar"]["mean_diff"] >= 5.0
        assert sub["rlm"]["mean_diff"] >= 10.0

    def test_missing_paradigm_data(self):
        """Missing paradigm data should fail that paradigm."""
        evidence = Evidence(
            phase_0={"score": 50.0, "ablations": {}},
            phase_1={"score": 55.0, "ablations": {}},
            phase_2={"score": 58.0, "ablations": {}},
            phase_3={"score": 62.0, "ablations": {}},
            phase_4={"score": 65.0, "ablations": {}},
        )
        criterion = ParadigmImprovementCriterion()
        result = criterion.evaluate(evidence)

        assert result.passed is False
        for paradigm in ["symcode", "godel", "soar", "rlm"]:
            assert result.details["sub_results"][paradigm]["passed"] is False

    def test_custom_min_effects(self, passing_evidence):
        """Custom effect sizes can make thresholds stricter."""
        criterion = ParadigmImprovementCriterion(
            min_effects={"symcode": 50.0, "godel": 50.0, "soar": 50.0, "rlm": 50.0}
        )
        result = criterion.evaluate(passing_evidence)
        assert result.passed is False

    def test_properties(self):
        """Test criterion properties."""
        criterion = ParadigmImprovementCriterion()
        assert criterion.name == "Paradigm Improvement"
        assert "paradigm" in criterion.description.lower()
        assert "paired" in criterion.threshold_description.lower()

    def test_margin_is_minimum(self, passing_evidence):
        """Margin should be the minimum across all paradigms."""
        criterion = ParadigmImprovementCriterion()
        result = criterion.evaluate(passing_evidence)

        sub = result.details["sub_results"]
        margins = [sub[p]["margin"] for p in ["symcode", "godel", "soar", "rlm"]]
        assert result.margin == min(margins)
