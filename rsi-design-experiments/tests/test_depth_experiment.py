"""Tests for RLM depth experiment."""

import math
import pytest

from src.experiments.rlm_depth import RLMDepthExperiment
from src.conditions.depth_conditions import DepthCondition, build_depth_conditions
from src.harness.controlled_pipeline import MockPipeline


class TestDepthConditions:
    """Test building depth conditions."""

    def test_build_all_7_conditions(self):
        conditions = build_depth_conditions()
        assert len(conditions) == 7

    def test_condition_names(self):
        conditions = build_depth_conditions()
        names = [c.name for c in conditions]
        expected = [f"depth_{d}" for d in range(7)]
        assert names == expected

    def test_condition_depths(self):
        conditions = build_depth_conditions()
        for i, c in enumerate(conditions):
            assert c.depth == i

    def test_baseline_description(self):
        conditions = build_depth_conditions()
        assert "baseline" in conditions[0].description.lower() or "no recursion" in conditions[0].description.lower()


class TestTheoreticalModels:
    """Test the theoretical accuracy and cost models."""

    def test_theoretical_accuracy_depth_0(self):
        acc = RLMDepthExperiment.theoretical_accuracy(0)
        assert acc == pytest.approx(0.55, abs=0.01)

    def test_theoretical_accuracy_depth_6(self):
        acc = RLMDepthExperiment.theoretical_accuracy(6)
        # 0.55 + 0.17 * (1 - exp(-6/1.5)) = 0.55 + 0.17 * (1 - exp(-4))
        expected = 0.55 + 0.17 * (1.0 - math.exp(-4.0))
        assert acc == pytest.approx(expected, abs=0.001)

    def test_theoretical_accuracy_monotonically_increasing(self):
        accs = [RLMDepthExperiment.theoretical_accuracy(d) for d in range(7)]
        for i in range(1, len(accs)):
            assert accs[i] > accs[i - 1]

    def test_theoretical_accuracy_concave(self):
        """Marginal gains should decrease (concave curve)."""
        accs = [RLMDepthExperiment.theoretical_accuracy(d) for d in range(7)]
        gains = [accs[i] - accs[i - 1] for i in range(1, len(accs))]
        for i in range(1, len(gains)):
            assert gains[i] < gains[i - 1]

    def test_theoretical_cost_depth_0(self):
        cost = RLMDepthExperiment.theoretical_cost(0)
        assert cost == pytest.approx(0.005, abs=0.001)

    def test_theoretical_cost_depth_6(self):
        cost = RLMDepthExperiment.theoretical_cost(6)
        assert cost == pytest.approx(0.005 * 64, abs=0.01)

    def test_theoretical_cost_convex(self):
        """Cost should increase convexly (exponentially)."""
        costs = [RLMDepthExperiment.theoretical_cost(d) for d in range(7)]
        gains = [costs[i] - costs[i - 1] for i in range(1, len(costs))]
        for i in range(1, len(gains)):
            assert gains[i] > gains[i - 1]


class TestRLMDepthExperiment:
    """Test the full RLM depth experiment."""

    def test_get_conditions_returns_7(self, depth_experiment):
        conditions = depth_experiment.get_conditions()
        assert len(conditions) == 7

    def test_configure_pipeline(self, depth_experiment, mock_pipeline):
        conditions = depth_experiment.get_conditions()
        configured = depth_experiment.configure_pipeline(conditions[3], mock_pipeline)
        assert configured is mock_pipeline

    def test_measure_returns_condition_result(self, depth_experiment, mock_pipeline):
        conditions = depth_experiment.get_conditions()
        depth_experiment.configure_pipeline(conditions[3], mock_pipeline)  # depth_3
        mock_pipeline.set_seed(42)
        result = depth_experiment.measure(mock_pipeline, conditions[3])
        assert result.condition_name == "depth_3"
        assert result.metadata.get("depth") == 3
        assert 0.0 <= result.final_accuracy <= 1.0

    def test_run_full_experiment(self, depth_experiment, mock_pipeline):
        result = depth_experiment.run(mock_pipeline, repetitions=2, seed=42)
        assert result.experiment_name == "rlm_depth"
        assert len(result.conditions) == 7
        assert result.repetitions == 2

    def test_accuracy_increases_with_depth(self, depth_experiment, mock_pipeline):
        """Higher depths should produce higher accuracy on average."""
        result = depth_experiment.run(mock_pipeline, repetitions=3, seed=42)
        depth_0_acc = result.get_mean_accuracy("depth_0")
        depth_3_acc = result.get_mean_accuracy("depth_3")
        depth_6_acc = result.get_mean_accuracy("depth_6")
        assert depth_3_acc > depth_0_acc
        assert depth_6_acc > depth_3_acc

    def test_cost_increases_with_depth(self, depth_experiment, mock_pipeline):
        """Higher depths should cost more."""
        result = depth_experiment.run(mock_pipeline, repetitions=3, seed=42)
        costs = {}
        for d in range(7):
            cond = f"depth_{d}"
            cr = result.per_condition_results[cond]
            costs[d] = sum(r.total_cost for r in cr) / len(cr)
        # Check that cost increases with depth
        for d in range(1, 7):
            assert costs[d] > costs[d - 1]

    def test_marginal_accuracy_gain_decreasing(self, depth_experiment, mock_pipeline):
        """Marginal accuracy gain should decrease with depth."""
        result = depth_experiment.run(mock_pipeline, repetitions=3, seed=42)
        accs = []
        for d in range(7):
            accs.append(result.get_mean_accuracy(f"depth_{d}"))
        gains = [accs[i] - accs[i - 1] for i in range(1, len(accs))]
        # At least the first few gains should be decreasing
        assert gains[0] > gains[-1]
