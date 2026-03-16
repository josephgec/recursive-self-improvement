"""Tests for modification frequency experiment."""

import pytest

from src.experiments.modification_frequency import ModificationFrequencyExperiment
from src.conditions.frequency_conditions import (
    FrequencyCondition,
    ModificationFrequencyPolicy,
    build_frequency_conditions,
)
from src.harness.controlled_pipeline import MockPipeline


class TestModificationFrequencyPolicy:
    """Test the modification frequency policies."""

    def test_every_task_always_modifies(self):
        policy = ModificationFrequencyPolicy("every_task")
        for i in range(20):
            assert policy.should_modify(i, 0.5) is True

    def test_every_n_modifies_at_intervals(self):
        policy = ModificationFrequencyPolicy("every_n", param=5)
        results = [policy.should_modify(i, 0.5) for i in range(20)]
        # Should modify at iterations 5, 10, 15 (not 0)
        assert results[0] is False
        assert results[5] is True
        assert results[10] is True
        assert results[15] is True
        assert results[1] is False
        assert results[3] is False

    def test_every_10(self):
        policy = ModificationFrequencyPolicy("every_n", param=10)
        assert policy.should_modify(0, 0.5) is False
        assert policy.should_modify(5, 0.5) is False
        assert policy.should_modify(10, 0.5) is True
        assert policy.should_modify(20, 0.5) is True

    def test_threshold_below(self):
        policy = ModificationFrequencyPolicy("threshold", param=0.9)
        assert policy.should_modify(5, 0.5) is False
        assert policy.should_modify(5, 0.85) is False

    def test_threshold_above(self):
        policy = ModificationFrequencyPolicy("threshold", param=0.9)
        assert policy.should_modify(5, 0.91) is True
        assert policy.should_modify(5, 0.95) is True

    def test_plateau_no_improvement(self):
        policy = ModificationFrequencyPolicy("plateau", param=5)
        # Feed the same accuracy 6 times (no improvement for 5 steps)
        for i in range(5):
            result = policy.should_modify(i, 0.7)
        # After 5 iterations of no improvement, should trigger
        assert policy.should_modify(5, 0.7) is True

    def test_plateau_with_improvement(self):
        policy = ModificationFrequencyPolicy("plateau", param=5)
        # Feed increasing accuracy
        for i in range(10):
            result = policy.should_modify(i, 0.5 + i * 0.02)
        # Should not trigger since accuracy keeps improving
        assert result is False

    def test_never_modifies(self):
        policy = ModificationFrequencyPolicy("never")
        for i in range(20):
            assert policy.should_modify(i, 0.99) is False

    def test_policy_reset(self):
        policy = ModificationFrequencyPolicy("plateau", param=3)
        for i in range(5):
            policy.should_modify(i, 0.7)
        policy.reset()
        # After reset, should not trigger immediately
        assert policy.should_modify(0, 0.7) is False

    def test_unknown_policy(self):
        policy = ModificationFrequencyPolicy("unknown_type")
        assert policy.should_modify(0, 0.5) is False


class TestFrequencyConditions:
    """Test building frequency conditions."""

    def test_build_all_7_conditions(self):
        conditions = build_frequency_conditions()
        assert len(conditions) == 7

    def test_condition_names(self):
        conditions = build_frequency_conditions()
        names = [c.name for c in conditions]
        expected = [
            "every_task", "every_5", "every_10", "every_20",
            "threshold_90", "plateau_5", "never",
        ]
        assert names == expected

    def test_conditions_have_policies(self):
        conditions = build_frequency_conditions()
        for c in conditions:
            assert isinstance(c, FrequencyCondition)
            assert isinstance(c.policy, ModificationFrequencyPolicy)
            assert c.description

    def test_condition_dataclass_fields(self):
        conditions = build_frequency_conditions()
        c = conditions[0]
        assert hasattr(c, "name")
        assert hasattr(c, "description")
        assert hasattr(c, "policy")


class TestModificationFrequencyExperiment:
    """Test the full modification frequency experiment."""

    def test_get_conditions_returns_7(self, frequency_experiment):
        conditions = frequency_experiment.get_conditions()
        assert len(conditions) == 7

    def test_configure_pipeline(self, frequency_experiment, mock_pipeline):
        conditions = frequency_experiment.get_conditions()
        configured = frequency_experiment.configure_pipeline(conditions[0], mock_pipeline)
        assert configured is mock_pipeline

    def test_measure_returns_condition_result(self, frequency_experiment, mock_pipeline):
        conditions = frequency_experiment.get_conditions()
        # Configure for every_task
        frequency_experiment.configure_pipeline(conditions[0], mock_pipeline)
        mock_pipeline.set_seed(42)
        result = frequency_experiment.measure(mock_pipeline, conditions[0])
        assert result.condition_name == "every_task"
        assert 0.0 <= result.final_accuracy <= 1.0
        assert len(result.accuracy_trajectory) == 10  # iterations_per_condition

    def test_run_full_experiment(self, frequency_experiment, mock_pipeline):
        result = frequency_experiment.run(mock_pipeline, repetitions=2, seed=42)
        assert result.experiment_name == "modification_frequency"
        assert len(result.conditions) == 7
        assert result.repetitions == 2
        for cond_name in result.conditions:
            assert cond_name in result.per_condition_results
            assert len(result.per_condition_results[cond_name]) == 2

    def test_every_task_lower_stability(self, frequency_experiment, mock_pipeline):
        """every_task should have lower stability than never."""
        result = frequency_experiment.run(mock_pipeline, repetitions=3, seed=42)
        every_task_stab = sum(
            r.stability_score for r in result.per_condition_results["every_task"]
        ) / 3
        never_stab = sum(
            r.stability_score for r in result.per_condition_results["never"]
        ) / 3
        # every_task has higher rollback rate, so lower stability
        assert every_task_stab <= never_stab

    def test_never_stays_at_baseline(self, frequency_experiment, mock_pipeline):
        """never should have accuracy around 0.75."""
        result = frequency_experiment.run(mock_pipeline, repetitions=3, seed=42)
        never_acc = sum(
            r.final_accuracy for r in result.per_condition_results["never"]
        ) / 3
        # Should be around 0.75 (baseline), within reasonable range
        assert 0.55 <= never_acc <= 0.85

    def test_every_5_achieves_good_accuracy(self, frequency_experiment, mock_pipeline):
        """every_5 should achieve reasonably high accuracy."""
        result = frequency_experiment.run(mock_pipeline, repetitions=3, seed=42)
        every5_acc = sum(
            r.final_accuracy for r in result.per_condition_results["every_5"]
        ) / 3
        # Should be notably above baseline
        never_acc = sum(
            r.final_accuracy for r in result.per_condition_results["never"]
        ) / 3
        assert every5_acc >= never_acc
