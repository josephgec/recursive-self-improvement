"""Tests for hindsight target experiment."""

import pytest

from src.experiments.hindsight_target import HindsightTargetExperiment
from src.conditions.hindsight_conditions import (
    HindsightCondition,
    HindsightTargetPolicy,
    build_hindsight_conditions,
)
from src.harness.controlled_pipeline import MockPipeline


class TestHindsightTargetPolicy:
    """Test hindsight target policy decisions."""

    def test_weights_only(self):
        policy = HindsightTargetPolicy("weights_only")
        assert policy.get_target(0) == "weights"
        assert policy.get_target(10) == "weights"

    def test_library_only(self):
        policy = HindsightTargetPolicy("library_only")
        assert policy.get_target(0) == "library"
        assert policy.get_target(10) == "library"

    def test_both(self):
        policy = HindsightTargetPolicy("both")
        assert policy.get_target(0) == "both"
        assert policy.get_target(15) == "both"

    def test_none(self):
        policy = HindsightTargetPolicy("none")
        assert policy.get_target(0) == "none"
        assert policy.get_target(15) == "none"

    def test_weights_then_library(self):
        policy = HindsightTargetPolicy("weights_then_library", total_iterations=20)
        assert policy.get_target(0) == "weights"
        assert policy.get_target(9) == "weights"
        assert policy.get_target(10) == "library"
        assert policy.get_target(19) == "library"

    def test_library_then_weights(self):
        policy = HindsightTargetPolicy("library_then_weights", total_iterations=20)
        assert policy.get_target(0) == "library"
        assert policy.get_target(9) == "library"
        assert policy.get_target(10) == "weights"
        assert policy.get_target(19) == "weights"

    def test_unknown_policy_returns_none(self):
        policy = HindsightTargetPolicy("unknown_type")
        assert policy.get_target(0) == "none"


class TestHindsightConditions:
    """Test building hindsight conditions."""

    def test_build_all_6_conditions(self):
        conditions = build_hindsight_conditions()
        assert len(conditions) == 6

    def test_condition_names(self):
        conditions = build_hindsight_conditions()
        names = [c.name for c in conditions]
        expected = [
            "weights_only", "library_only", "both", "none",
            "weights_then_library", "library_then_weights",
        ]
        assert names == expected

    def test_conditions_have_policies(self):
        conditions = build_hindsight_conditions()
        for c in conditions:
            assert isinstance(c, HindsightCondition)
            assert isinstance(c.policy, HindsightTargetPolicy)
            assert c.description

    def test_custom_total_iterations(self):
        conditions = build_hindsight_conditions(total_iterations=10)
        wl = conditions[4]  # weights_then_library
        assert wl.policy.get_target(4) == "weights"
        assert wl.policy.get_target(5) == "library"


class TestHindsightTargetExperiment:
    """Test the full hindsight target experiment."""

    def test_get_conditions_returns_6(self, hindsight_experiment):
        conditions = hindsight_experiment.get_conditions()
        assert len(conditions) == 6

    def test_configure_pipeline(self, hindsight_experiment, mock_pipeline):
        conditions = hindsight_experiment.get_conditions()
        configured = hindsight_experiment.configure_pipeline(conditions[0], mock_pipeline)
        assert configured is mock_pipeline

    def test_measure_returns_condition_result(self, hindsight_experiment, mock_pipeline):
        conditions = hindsight_experiment.get_conditions()
        hindsight_experiment.configure_pipeline(conditions[2], mock_pipeline)  # both
        mock_pipeline.set_seed(42)
        result = hindsight_experiment.measure(mock_pipeline, conditions[2])
        assert result.condition_name == "both"
        assert 0.0 <= result.final_accuracy <= 1.0

    def test_run_full_experiment(self, hindsight_experiment, mock_pipeline):
        result = hindsight_experiment.run(mock_pipeline, repetitions=2, seed=42)
        assert result.experiment_name == "hindsight_target"
        assert len(result.conditions) == 6
        assert result.repetitions == 2
        for cond_name in result.conditions:
            assert cond_name in result.per_condition_results
            assert len(result.per_condition_results[cond_name]) == 2

    def test_both_achieves_highest_accuracy(self, hindsight_experiment, mock_pipeline):
        """'both' should achieve higher accuracy than 'none'."""
        result = hindsight_experiment.run(mock_pipeline, repetitions=3, seed=42)
        both_acc = sum(
            r.final_accuracy for r in result.per_condition_results["both"]
        ) / 3
        none_acc = sum(
            r.final_accuracy for r in result.per_condition_results["none"]
        ) / 3
        assert both_acc > none_acc

    def test_library_generalizes_better(self, hindsight_experiment, mock_pipeline):
        """library_only should have better generalization than weights_only."""
        result = hindsight_experiment.run(mock_pipeline, repetitions=3, seed=42)
        lib_gen = sum(
            r.generalization_score
            for r in result.per_condition_results["library_only"]
        ) / 3
        weights_gen = sum(
            r.generalization_score
            for r in result.per_condition_results["weights_only"]
        ) / 3
        assert lib_gen >= weights_gen

    def test_none_is_baseline(self, hindsight_experiment, mock_pipeline):
        """'none' should have accuracy around 0.75."""
        result = hindsight_experiment.run(mock_pipeline, repetitions=3, seed=42)
        none_acc = sum(
            r.final_accuracy for r in result.per_condition_results["none"]
        ) / 3
        assert 0.55 <= none_acc <= 0.85
