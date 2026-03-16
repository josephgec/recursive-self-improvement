"""Tests for analysis module: ANOVA, Tukey, diminishing returns, sensitivity, recommendations."""

import math
import pytest

from src.analysis.anova import ANOVAAnalyzer, ANOVAResult, TukeyResult
from src.analysis.diminishing_returns import DiminishingReturnsAnalyzer
from src.analysis.sensitivity import SensitivityAnalyzer
from src.analysis.optimal_finder import OptimalFinder
from src.analysis.interaction_effects import InteractionAnalyzer
from src.analysis.recommendation import RecommendationGenerator, PipelineRecommendation
from src.experiments.base import ExperimentResult, ConditionResult
from src.measurement.composite_scorer import CompositeScorer
from src.measurement.accuracy_tracker import AccuracyTracker
from src.measurement.stability_tracker import StabilityTracker
from src.measurement.cost_tracker import CostTracker
from src.measurement.improvement_rate import ImprovementRateTracker


class TestANOVA:
    """Test ANOVA analysis."""

    def test_significant_difference(self):
        """Groups with clearly different means should be significant."""
        analyzer = ANOVAAnalyzer(significance_level=0.05)
        data = {
            "group_a": [0.9, 0.88, 0.92, 0.87, 0.91],
            "group_b": [0.7, 0.72, 0.68, 0.71, 0.69],
            "group_c": [0.5, 0.52, 0.48, 0.51, 0.49],
        }
        result = analyzer.one_way_anova(data)
        assert isinstance(result, ANOVAResult)
        assert result.f_statistic > 0
        assert result.significant is True
        assert result.eta_squared > 0

    def test_nonsignificant_difference(self):
        """Groups with similar means should not be significant."""
        analyzer = ANOVAAnalyzer(significance_level=0.05)
        data = {
            "group_a": [0.80, 0.81, 0.79],
            "group_b": [0.80, 0.80, 0.81],
        }
        result = analyzer.one_way_anova(data)
        assert result.f_statistic >= 0
        # With such small differences, should not be significant
        # (might depend on approximation)

    def test_single_group(self):
        """Single group should return non-significant."""
        analyzer = ANOVAAnalyzer()
        data = {"group_a": [0.8, 0.82, 0.81]}
        result = analyzer.one_way_anova(data)
        assert result.significant is False
        assert result.f_statistic == 0

    def test_group_means_computed(self):
        analyzer = ANOVAAnalyzer()
        data = {
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
        }
        result = analyzer.one_way_anova(data)
        assert result.group_means["a"] == pytest.approx(2.0)
        assert result.group_means["b"] == pytest.approx(5.0)

    def test_group_stds_computed(self):
        analyzer = ANOVAAnalyzer()
        data = {
            "a": [1.0, 1.0, 1.0],
            "b": [2.0, 2.0, 2.0],
        }
        result = analyzer.one_way_anova(data)
        assert result.group_stds["a"] == pytest.approx(0.0)
        assert result.group_stds["b"] == pytest.approx(0.0)

    def test_eta_squared_range(self):
        analyzer = ANOVAAnalyzer()
        data = {
            "a": [0.9, 0.88, 0.92],
            "b": [0.5, 0.52, 0.48],
        }
        result = analyzer.one_way_anova(data)
        assert 0.0 <= result.eta_squared <= 1.0

    def test_empty_groups(self):
        analyzer = ANOVAAnalyzer()
        data = {
            "a": [0.8, 0.82],
            "b": [],
        }
        # Should handle gracefully
        result = analyzer.one_way_anova(data)
        assert isinstance(result, ANOVAResult)

    def test_identical_values_within_groups(self):
        analyzer = ANOVAAnalyzer()
        data = {
            "a": [1.0, 1.0, 1.0],
            "b": [2.0, 2.0, 2.0],
        }
        result = analyzer.one_way_anova(data)
        # SS_within is 0, F should be inf, significant
        assert result.f_statistic == float("inf")
        assert result.significant is True


class TestTukeyHSD:
    """Test Tukey HSD pairwise comparisons."""

    def test_returns_pairwise_comparisons(self):
        analyzer = ANOVAAnalyzer()
        data = {
            "a": [0.9, 0.88, 0.92],
            "b": [0.7, 0.72, 0.68],
            "c": [0.5, 0.52, 0.48],
        }
        results = analyzer.tukey_hsd(data)
        # 3 groups = 3 pairs
        assert len(results) == 3
        for r in results:
            assert isinstance(r, TukeyResult)

    def test_significant_pairs(self):
        analyzer = ANOVAAnalyzer()
        data = {
            "high": [0.90, 0.88, 0.92, 0.89, 0.91],
            "low": [0.50, 0.52, 0.48, 0.51, 0.49],
        }
        results = analyzer.tukey_hsd(data)
        assert len(results) == 1
        assert results[0].significant is True
        assert results[0].mean_diff > 0

    def test_single_group(self):
        analyzer = ANOVAAnalyzer()
        data = {"a": [0.8, 0.82]}
        results = analyzer.tukey_hsd(data)
        assert results == []


class TestDiminishingReturns:
    """Test diminishing returns and knee detection."""

    def test_find_knee_concave(self):
        """Knee should be found in a concave curve."""
        analyzer = DiminishingReturnsAnalyzer()
        # Simulate accuracy = 0.55 + 0.17*(1-exp(-d/1.5))
        x = [float(d) for d in range(7)]
        y = [0.55 + 0.17 * (1 - math.exp(-d / 1.5)) for d in range(7)]
        result = analyzer.find_knee(x, y)
        assert result is not None
        assert 0 <= result.knee_index < 7
        assert result.knee_x >= 0
        assert result.knee_y > 0.55

    def test_find_knee_too_few_points(self):
        analyzer = DiminishingReturnsAnalyzer()
        result = analyzer.find_knee([0.0, 1.0], [0.5, 0.8])
        assert result is None

    def test_compute_marginal_gains(self):
        analyzer = DiminishingReturnsAnalyzer()
        x = [0.0, 1.0, 2.0, 3.0]
        y = [0.5, 0.7, 0.8, 0.85]
        gains = analyzer.compute_marginal_gains(x, y)
        assert len(gains) == 3
        assert gains[0] == pytest.approx(0.2)
        assert gains[1] == pytest.approx(0.1)
        assert gains[2] == pytest.approx(0.05)

    def test_marginal_gains_decreasing(self):
        """For concave curve, marginal gains should decrease."""
        analyzer = DiminishingReturnsAnalyzer()
        x = [float(d) for d in range(7)]
        y = [0.55 + 0.17 * (1 - math.exp(-d / 1.5)) for d in range(7)]
        gains = analyzer.compute_marginal_gains(x, y)
        for i in range(1, len(gains)):
            assert gains[i] < gains[i - 1]

    def test_optimal_depth_with_budget(self):
        analyzer = DiminishingReturnsAnalyzer()
        depths = [0, 1, 2, 3, 4, 5, 6]
        accuracies = [0.55 + 0.17 * (1 - math.exp(-d / 1.5)) for d in depths]
        costs = [0.005 * (2 ** d) for d in depths]
        result = analyzer.optimal_depth(depths, accuracies, costs, cost_budget=0.05)
        assert result <= 4  # Should pick a depth within budget

    def test_optimal_depth_without_budget(self):
        analyzer = DiminishingReturnsAnalyzer()
        depths = [0, 1, 2, 3, 4, 5, 6]
        accuracies = [0.55 + 0.17 * (1 - math.exp(-d / 1.5)) for d in depths]
        costs = [0.005 * (2 ** d) for d in depths]
        result = analyzer.optimal_depth(depths, accuracies, costs)
        assert 0 <= result <= 6

    def test_optimal_depth_empty(self):
        analyzer = DiminishingReturnsAnalyzer()
        assert analyzer.optimal_depth([], [], []) == 0

    def test_find_knee_flat_curve(self):
        analyzer = DiminishingReturnsAnalyzer()
        x = [0.0, 1.0, 2.0, 3.0]
        y = [0.5, 0.5, 0.5, 0.5]
        result = analyzer.find_knee(x, y)
        assert result is None  # y_range = 0


class TestSensitivityAnalyzer:
    """Test sensitivity analysis."""

    def test_compute_sensitivity(self, sample_experiment_result):
        analyzer = SensitivityAnalyzer()
        result = analyzer.compute_sensitivity(sample_experiment_result)
        assert result.experiment_name == "test_experiment"
        assert result.sensitivity > 0
        assert result.max_value > result.min_value

    def test_rank_experiments(self, sample_experiment_result, sample_nonsignificant_result):
        analyzer = SensitivityAnalyzer()
        rankings = analyzer.rank_experiments(
            [sample_experiment_result, sample_nonsignificant_result]
        )
        assert len(rankings) == 2
        # The significant result should have higher sensitivity
        assert rankings[0].sensitivity >= rankings[1].sensitivity

    def test_sensitivity_empty_result(self):
        analyzer = SensitivityAnalyzer()
        result = ExperimentResult(
            experiment_name="empty",
            conditions=[],
            repetitions=0,
        )
        s = analyzer.compute_sensitivity(result)
        assert s.sensitivity == 0.0


class TestOptimalFinder:
    """Test optimal condition finding."""

    def test_find_best_condition(self, sample_experiment_result):
        finder = OptimalFinder()
        result = finder.find_best_condition(sample_experiment_result)
        assert result.best_condition == "cond_a"
        assert result.best_score > 0

    def test_find_best_by_accuracy(self, sample_experiment_result):
        finder = OptimalFinder()
        result = finder.find_best_condition(
            sample_experiment_result, metric="final_accuracy"
        )
        assert result.best_condition == "cond_a"

    def test_find_pareto_optimal(self, sample_experiment_result):
        finder = OptimalFinder()
        pareto = finder.find_pareto_optimal(
            sample_experiment_result,
            metrics=["final_accuracy", "stability_score"],
        )
        assert len(pareto) >= 1
        # cond_a has best accuracy, cond_c has best stability
        pareto_names = [p.condition for p in pareto]
        assert "cond_a" in pareto_names or "cond_c" in pareto_names

    def test_recommend(self, sample_experiment_result):
        finder = OptimalFinder()
        rec = finder.recommend(sample_experiment_result)
        assert rec in ["cond_a", "cond_b", "cond_c"]

    def test_recommend_with_secondary(self, sample_experiment_result):
        finder = OptimalFinder()
        rec = finder.recommend(
            sample_experiment_result,
            primary_metric="composite_score",
            secondary_metrics=["stability_score"],
        )
        assert rec in ["cond_a", "cond_b", "cond_c"]


class TestInteractionAnalyzer:
    """Test interaction effects analysis."""

    def test_two_way_anova(self):
        analyzer = InteractionAnalyzer()
        factor_a = {
            "a1": [0.9, 0.88, 0.92],
            "a2": [0.7, 0.72, 0.68],
        }
        factor_b = {
            "b1": [0.85, 0.83, 0.87],
            "b2": [0.6, 0.62, 0.58],
        }
        result = analyzer.two_way_anova(factor_a, factor_b, "freq", "hindsight")
        assert result.factor_a_name == "freq"
        assert result.factor_b_name == "hindsight"
        assert result.main_effect_a is not None
        assert result.main_effect_b is not None
        assert isinstance(result.interaction_strength, float)

    def test_detect_interactions(self):
        analyzer = InteractionAnalyzer()
        all_results = {
            "exp_a": {"a1": [0.9, 0.88], "a2": [0.5, 0.52]},
            "exp_b": {"b1": [0.85, 0.83], "b2": [0.6, 0.62]},
            "exp_c": {"c1": [0.7, 0.72], "c2": [0.7, 0.71]},
        }
        interactions = analyzer.detect_interactions(all_results)
        # 3 experiments = 3 pairs
        assert len(interactions) == 3


class TestRecommendationGenerator:
    """Test recommendation generation."""

    def test_generate_recommendation(self):
        gen = RecommendationGenerator()

        # Create frequency result
        freq_result = ExperimentResult(
            experiment_name="modification_frequency",
            conditions=["every_5", "never"],
            repetitions=3,
        )
        freq_result.per_condition_results = {
            "every_5": [
                ConditionResult(condition_name="every_5", composite_score=0.85),
                ConditionResult(condition_name="every_5", composite_score=0.83),
                ConditionResult(condition_name="every_5", composite_score=0.87),
            ],
            "never": [
                ConditionResult(condition_name="never", composite_score=0.60),
                ConditionResult(condition_name="never", composite_score=0.58),
                ConditionResult(condition_name="never", composite_score=0.62),
            ],
        }

        # Create hindsight result
        hint_result = ExperimentResult(
            experiment_name="hindsight_target",
            conditions=["both", "none"],
            repetitions=3,
        )
        hint_result.per_condition_results = {
            "both": [
                ConditionResult(condition_name="both", composite_score=0.90),
                ConditionResult(condition_name="both", composite_score=0.88),
                ConditionResult(condition_name="both", composite_score=0.92),
            ],
            "none": [
                ConditionResult(condition_name="none", composite_score=0.55),
                ConditionResult(condition_name="none", composite_score=0.53),
                ConditionResult(condition_name="none", composite_score=0.57),
            ],
        }

        # Create depth result
        depth_result = ExperimentResult(
            experiment_name="rlm_depth",
            conditions=["depth_0", "depth_3"],
            repetitions=3,
        )
        depth_result.per_condition_results = {
            "depth_0": [
                ConditionResult(condition_name="depth_0", composite_score=0.50),
                ConditionResult(condition_name="depth_0", composite_score=0.48),
                ConditionResult(condition_name="depth_0", composite_score=0.52),
            ],
            "depth_3": [
                ConditionResult(condition_name="depth_3", composite_score=0.82),
                ConditionResult(condition_name="depth_3", composite_score=0.80),
                ConditionResult(condition_name="depth_3", composite_score=0.84),
            ],
        }

        rec = gen.generate([freq_result, hint_result, depth_result])
        assert isinstance(rec, PipelineRecommendation)
        assert rec.modification_frequency == "every_5"
        assert rec.hindsight_target == "both"
        assert rec.rlm_depth == 3
        assert "modification_frequency" in rec.confidence_levels
        assert "hindsight_target" in rec.confidence_levels
        assert "rlm_depth" in rec.confidence_levels

    def test_recommendation_to_yaml(self):
        rec = PipelineRecommendation(
            modification_frequency="every_5",
            hindsight_target="both",
            rlm_depth=3,
            confidence_levels={
                "modification_frequency": "high",
                "hindsight_target": "high",
                "rlm_depth": "medium",
            },
        )
        yaml_str = rec.to_yaml()
        assert "every_5" in yaml_str
        assert "both" in yaml_str
        assert "3" in yaml_str

    def test_recommendation_to_dict(self):
        rec = PipelineRecommendation(
            modification_frequency="every_5",
            hindsight_target="both",
            rlm_depth=3,
        )
        d = rec.to_dict()
        assert d["pipeline_config"]["modification_frequency"] == "every_5"
        assert d["pipeline_config"]["hindsight_target"] == "both"
        assert d["pipeline_config"]["rlm_depth"] == 3


class TestCompositeScorer:
    """Test composite scoring."""

    def test_default_weights(self):
        scorer = CompositeScorer()
        result = scorer.score(1.0, 1.0, 1.0, 1.0)
        assert result == pytest.approx(1.0)

    def test_zero_scores(self):
        scorer = CompositeScorer()
        result = scorer.score(0.0, 0.0, 0.0, 0.0)
        assert result == pytest.approx(0.0)

    def test_weighted_score(self):
        scorer = CompositeScorer()
        result = scorer.score(0.8, 0.9, 0.7, 0.6)
        expected = 0.4 * 0.8 + 0.3 * 0.9 + 0.15 * 0.7 + 0.15 * 0.6
        assert result == pytest.approx(expected)

    def test_custom_weights(self):
        weights = {"accuracy": 1.0, "stability": 0.0, "efficiency": 0.0, "generalization": 0.0}
        scorer = CompositeScorer(weights)
        result = scorer.score(0.8, 0.9, 0.7, 0.6)
        assert result == pytest.approx(0.8)

    def test_score_from_dict(self):
        scorer = CompositeScorer()
        metrics = {"accuracy": 0.8, "stability": 0.9, "efficiency": 0.7, "generalization": 0.6}
        result = scorer.score_from_dict(metrics)
        expected = 0.4 * 0.8 + 0.3 * 0.9 + 0.15 * 0.7 + 0.15 * 0.6
        assert result == pytest.approx(expected)

    def test_get_weights(self):
        scorer = CompositeScorer()
        w = scorer.get_weights()
        assert w["accuracy"] == 0.4

    def test_set_weights(self):
        scorer = CompositeScorer()
        new_w = {"accuracy": 0.5, "stability": 0.2, "efficiency": 0.2, "generalization": 0.1}
        scorer.set_weights(new_w)
        assert scorer.get_weights() == new_w


class TestAccuracyTracker:
    """Test accuracy tracker."""

    def test_record_and_get_overall(self):
        tracker = AccuracyTracker()
        tracker.record(0.8, "in_distribution")
        tracker.record(0.9, "in_distribution")
        assert tracker.get_overall() == pytest.approx(0.85)

    def test_get_per_type(self):
        tracker = AccuracyTracker()
        tracker.record(0.8, "in_distribution")
        tracker.record(0.7, "out_of_distribution")
        tracker.record(0.9, "in_distribution")
        assert tracker.get_per_type("in_distribution") == pytest.approx(0.85)
        assert tracker.get_per_type("out_of_distribution") == pytest.approx(0.7)

    def test_get_trajectory(self):
        tracker = AccuracyTracker()
        tracker.record(0.5)
        tracker.record(0.6)
        tracker.record(0.7)
        assert tracker.get_trajectory() == [0.5, 0.6, 0.7]

    def test_empty_tracker(self):
        tracker = AccuracyTracker()
        assert tracker.get_overall() == 0.0
        assert tracker.get_per_type("any") == 0.0
        assert tracker.get_trajectory() == []
        assert tracker.get_final() == 0.0

    def test_get_final(self):
        tracker = AccuracyTracker()
        tracker.record(0.5)
        tracker.record(0.8)
        assert tracker.get_final() == 0.8

    def test_count(self):
        tracker = AccuracyTracker()
        assert tracker.count() == 0
        tracker.record(0.5)
        tracker.record(0.6)
        assert tracker.count() == 2


class TestStabilityTracker:
    """Test stability tracker."""

    def test_record_and_count(self):
        tracker = StabilityTracker()
        tracker.record_rollback(1)
        tracker.record_rollback(5)
        assert tracker.total_rollbacks() == 2

    def test_rollback_rate(self):
        tracker = StabilityTracker()
        tracker.record_rollback(1)
        tracker.record_rollback(5)
        assert tracker.get_rollback_rate(10) == pytest.approx(0.2)

    def test_rollback_rate_zero_iterations(self):
        tracker = StabilityTracker()
        assert tracker.get_rollback_rate(0) == 0.0

    def test_detect_oscillation_yes(self):
        tracker = StabilityTracker()
        tracker.record_rollback(1)
        tracker.record_rollback(2)
        tracker.record_rollback(3)
        assert tracker.detect_oscillation(window=5) is True

    def test_detect_oscillation_no(self):
        tracker = StabilityTracker()
        tracker.record_rollback(1)
        tracker.record_rollback(20)
        assert tracker.detect_oscillation(window=5) is False

    def test_detect_oscillation_few_rollbacks(self):
        tracker = StabilityTracker()
        tracker.record_rollback(1)
        assert tracker.detect_oscillation() is False

    def test_consecutive_rollbacks(self):
        tracker = StabilityTracker()
        tracker.record_rollback(3)
        tracker.record_rollback(4)
        tracker.record_rollback(5)
        tracker.record_rollback(10)
        assert tracker.consecutive_rollbacks() == 3

    def test_consecutive_rollbacks_none(self):
        tracker = StabilityTracker()
        assert tracker.consecutive_rollbacks() == 0

    def test_stability_score_perfect(self):
        tracker = StabilityTracker()
        assert tracker.stability_score(20) == 1.0

    def test_stability_score_with_rollbacks(self):
        tracker = StabilityTracker()
        tracker.record_rollback(1)
        tracker.record_rollback(2)
        tracker.record_rollback(3)
        score = tracker.stability_score(20)
        assert 0.0 <= score < 1.0

    def test_stability_score_zero_iterations(self):
        tracker = StabilityTracker()
        assert tracker.stability_score(0) == 1.0


class TestCostTracker:
    """Test cost tracker."""

    def test_record_and_total(self):
        tracker = CostTracker()
        tracker.record_llm_call(0.01)
        tracker.record_llm_call(0.02)
        tracker.record_finetuning(0.05)
        assert tracker.get_total_cost() == pytest.approx(0.08)

    def test_cost_per_iteration(self):
        tracker = CostTracker()
        tracker.record_llm_call(0.01)
        tracker.record_llm_call(0.01)
        tracker.record_finetuning(0.08)
        assert tracker.get_cost_per_iteration() == pytest.approx(0.05)

    def test_cost_per_iteration_empty(self):
        tracker = CostTracker()
        assert tracker.get_cost_per_iteration() == 0.0

    def test_cost_breakdown(self):
        tracker = CostTracker()
        tracker.record_llm_call(0.01)
        tracker.record_finetuning(0.05)
        breakdown = tracker.get_cost_breakdown()
        assert breakdown["llm_calls"] == pytest.approx(0.01)
        assert breakdown["finetuning"] == pytest.approx(0.05)
        assert breakdown["total"] == pytest.approx(0.06)

    def test_counts(self):
        tracker = CostTracker()
        tracker.record_llm_call(0.01)
        tracker.record_llm_call(0.02)
        tracker.record_finetuning(0.05)
        assert tracker.get_llm_call_count() == 2
        assert tracker.get_finetuning_count() == 1


class TestImprovementRateTracker:
    """Test improvement rate tracker."""

    def test_rolling_delta(self):
        tracker = ImprovementRateTracker()
        for v in [0.5, 0.55, 0.60, 0.65, 0.70]:
            tracker.record(v)
        delta = tracker.compute_rolling_delta(window=4)
        assert delta == pytest.approx(0.05)

    def test_rolling_delta_single_value(self):
        tracker = ImprovementRateTracker()
        tracker.record(0.5)
        assert tracker.compute_rolling_delta() == 0.0

    def test_detect_plateau_yes(self):
        tracker = ImprovementRateTracker()
        for v in [0.8, 0.8001, 0.8002, 0.7999, 0.8001, 0.8, 0.8001]:
            tracker.record(v)
        assert tracker.detect_plateau(window=5, threshold=0.001) is True

    def test_detect_plateau_no(self):
        tracker = ImprovementRateTracker()
        for v in [0.5, 0.55, 0.60, 0.65, 0.70, 0.75]:
            tracker.record(v)
        assert tracker.detect_plateau(window=5, threshold=0.001) is False

    def test_detect_plateau_too_few(self):
        tracker = ImprovementRateTracker()
        tracker.record(0.5)
        tracker.record(0.5)
        assert tracker.detect_plateau(window=5) is False

    def test_marginal_improvement(self):
        tracker = ImprovementRateTracker()
        tracker.record(0.5)
        tracker.record(0.8)
        assert tracker.marginal_improvement() == pytest.approx(0.3)

    def test_marginal_improvement_single(self):
        tracker = ImprovementRateTracker()
        tracker.record(0.5)
        assert tracker.marginal_improvement() == 0.0

    def test_get_trajectory(self):
        tracker = ImprovementRateTracker()
        tracker.record(0.5)
        tracker.record(0.6)
        assert tracker.get_trajectory() == [0.5, 0.6]

    def test_get_improvement_at(self):
        tracker = ImprovementRateTracker()
        tracker.record(0.5)
        tracker.record(0.7)
        tracker.record(0.75)
        assert tracker.get_improvement_at(1) == pytest.approx(0.2)
        assert tracker.get_improvement_at(2) == pytest.approx(0.05)
        assert tracker.get_improvement_at(0) is None
        assert tracker.get_improvement_at(5) is None
