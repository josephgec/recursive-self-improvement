"""Tests for cost model."""

import pytest

from src.comparison.cost_model import CostModel, CostBreakdown, CostComparison
from src.benchmarks.task import EvalResult


class TestCostModel:
    """Test cost computation and efficiency metrics."""

    def test_compute_cost_basic(self):
        model = CostModel()
        cost = model.compute_cost(1000, 500, "rlm")
        # 1000 * 0.01/1000 + 500 * 0.03/1000 = 0.01 + 0.015 = 0.025
        assert abs(cost - 0.025) < 0.001

    def test_compute_cost_standard(self):
        model = CostModel()
        cost = model.compute_cost(1000, 500, "standard")
        # 1000 * 0.005/1000 + 500 * 0.015/1000 = 0.005 + 0.0075 = 0.0125
        assert abs(cost - 0.0125) < 0.001

    def test_cost_breakdown(self, sample_rlm_results):
        model = CostModel()
        breakdown = model.cost_breakdown(sample_rlm_results, "rlm")
        assert isinstance(breakdown, CostBreakdown)
        assert breakdown.system == "rlm"
        assert breakdown.num_tasks == len(sample_rlm_results)
        assert breakdown.total_cost > 0
        assert breakdown.cost_per_task > 0

    def test_cost_per_correct(self, sample_rlm_results):
        model = CostModel()
        cpc = model.cost_per_correct(sample_rlm_results, "rlm")
        assert cpc > 0

    def test_accuracy_per_dollar(self, sample_rlm_results):
        model = CostModel()
        apd = model.accuracy_per_dollar(sample_rlm_results, "rlm")
        assert apd > 0

    def test_compare_systems(self, sample_rlm_results, sample_std_results):
        model = CostModel()
        comparison = model.compare_systems(sample_rlm_results, sample_std_results)
        assert isinstance(comparison, CostComparison)
        assert "rlm" in comparison.systems
        assert "standard" in comparison.systems
        assert comparison.cost_ratio > 0
        assert comparison.efficiency_winner in ("rlm", "standard")

    def test_comparison_summary(self, sample_rlm_results, sample_std_results):
        model = CostModel()
        comparison = model.compare_systems(sample_rlm_results, sample_std_results)
        summary = comparison.summary()
        assert "Cost Comparison" in summary
        assert "rlm" in summary
        assert "standard" in summary

    def test_custom_pricing(self):
        pricing = {
            "custom": {"input": 0.02, "output": 0.06},
        }
        model = CostModel(pricing=pricing)
        cost = model.compute_cost(1000, 500, "custom")
        assert abs(cost - 0.05) < 0.001

    def test_cost_breakdown_fields(self):
        breakdown = CostBreakdown(
            system="test",
            total_cost=1.0,
            input_cost=0.6,
            output_cost=0.4,
            total_input_tokens=10000,
            total_output_tokens=5000,
            num_tasks=10,
            num_correct=8,
        )
        assert breakdown.cost_per_task == 0.1
        assert breakdown.cost_per_correct == 0.125

    def test_empty_results(self):
        model = CostModel()
        breakdown = model.cost_breakdown([], "rlm")
        assert breakdown.total_cost == 0
        assert breakdown.num_tasks == 0

    def test_accuracy_per_dollar_zero_cost(self):
        model = CostModel()
        result = model.accuracy_per_dollar([], "rlm")
        assert result == 0.0
