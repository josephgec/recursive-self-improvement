"""Tests for scaling experiment."""

import pytest

from src.benchmarks.synthetic import SyntheticTaskGenerator
from src.benchmarks.task import EvalTask, EvalResult
from src.comparison.scaling_experiment import ContextScalingExperiment, ScalingResult
from src.execution.rlm_executor import RLMExecutor
from src.execution.standard_executor import StandardExecutor


class TestContextScalingExperiment:
    """Test scaling experiment and crossover detection."""

    @pytest.fixture
    def base_tasks(self):
        gen = SyntheticTaskGenerator()
        return gen.needle_in_haystack(context_tokens=1000, num_tasks=2)

    @pytest.fixture
    def experiment(self):
        rlm = RLMExecutor(seed=42)
        std = StandardExecutor(context_window=4096, seed=42)
        return ContextScalingExperiment(rlm.execute, std.execute)

    def test_run_scaling(self, experiment, base_tasks):
        sizes = [1000, 4000, 8000]
        results = experiment.run(base_tasks, sizes)
        assert len(results) == 3
        assert all(isinstance(r, ScalingResult) for r in results)

    def test_scaling_result_fields(self, experiment, base_tasks):
        results = experiment.run(base_tasks, [1000])
        r = results[0]
        assert r.context_size == 1000
        assert 0.0 <= r.rlm_accuracy <= 1.0
        assert 0.0 <= r.standard_accuracy <= 1.0
        assert r.rlm_cost >= 0
        assert r.standard_cost >= 0

    def test_degradation_curve(self, experiment, base_tasks):
        sizes = [1000, 4000, 16000]
        results = experiment.run(base_tasks, sizes)
        curves = experiment.compute_degradation_curve(results)
        assert "rlm" in curves
        assert "standard" in curves
        assert len(curves["rlm"]) == 3
        assert len(curves["standard"]) == 3
        # Each curve point is (size, accuracy) tuple
        for size, acc in curves["rlm"]:
            assert size in sizes
            assert 0.0 <= acc <= 1.0

    def test_crossover_detection(self, experiment, base_tasks):
        sizes = [1000, 2000, 4000, 8000, 16000]
        results = experiment.run(base_tasks, sizes)
        crossover = experiment.find_crossover_point(results)
        # Crossover may or may not exist depending on mock behavior
        if crossover is not None:
            assert crossover in sizes

    def test_plot_scaling_curves(self, experiment, base_tasks):
        sizes = [1000, 4000]
        results = experiment.run(base_tasks, sizes)
        plot = experiment.plot_scaling_curves(results)
        assert "Accuracy vs Context Size" in plot
        assert "RLM" in plot
        assert "STD" in plot

    def test_empty_results(self, experiment):
        curves = experiment.compute_degradation_curve([])
        assert curves["rlm"] == []
        assert curves["standard"] == []

    def test_empty_plot(self, experiment):
        plot = experiment.plot_scaling_curves([])
        assert "No data" in plot

    def test_no_crossover(self):
        """Test when standard always beats RLM."""
        # Use a very large context window for standard
        rlm = RLMExecutor(seed=42)
        std = StandardExecutor(context_window=1000000, seed=42)
        experiment = ContextScalingExperiment(rlm.execute, std.execute)

        gen = SyntheticTaskGenerator()
        tasks = gen.needle_in_haystack(context_tokens=500, num_tasks=1)
        results = experiment.run(tasks, [500])

        # Even if no crossover, the function should handle gracefully
        crossover = experiment.find_crossover_point(results)
        # Could be None or could exist, just ensure no crash


class TestScalingResult:
    """Test ScalingResult dataclass."""

    def test_scaling_result_creation(self):
        r = ScalingResult(
            context_size=1000,
            rlm_accuracy=0.8,
            standard_accuracy=0.6,
            rlm_cost=0.05,
            standard_cost=0.01,
        )
        assert r.context_size == 1000
        assert r.rlm_accuracy == 0.8
        assert r.standard_accuracy == 0.6

    def test_scaling_result_with_results(self):
        result = EvalResult(
            task_id="test",
            benchmark="test",
            answer="a",
            correct=True,
        )
        r = ScalingResult(
            context_size=1000,
            rlm_accuracy=0.8,
            standard_accuracy=0.6,
            rlm_cost=0.05,
            standard_cost=0.01,
            rlm_results=[result],
            standard_results=[result],
        )
        assert len(r.rlm_results) == 1
        assert len(r.standard_results) == 1
