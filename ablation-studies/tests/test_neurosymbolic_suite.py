"""Tests for the neurosymbolic ablation suite."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.suites.neurosymbolic import NeurosymbolicAblation
from src.conditions.neurosymbolic_conditions import NeurosymbolicConditionBuilder
from src.execution.runner import AblationRunner, MockPipeline


class TestNeurosymbolicConditions:
    """Test that all 7 conditions are correctly defined."""

    def test_has_7_conditions(self, neurosymbolic_suite):
        conditions = neurosymbolic_suite.get_conditions()
        assert len(conditions) == 7

    def test_condition_names(self, neurosymbolic_suite):
        conditions = neurosymbolic_suite.get_conditions()
        names = {c.name for c in conditions}
        expected = {
            "full", "symcode_only", "bdm_only", "prose_only",
            "code_no_verify", "hybrid_no_bdm", "integrative",
        }
        assert names == expected

    def test_full_condition_is_marked(self, neurosymbolic_suite):
        full = neurosymbolic_suite.get_full_condition()
        assert full is not None
        assert full.name == "full"
        assert full.is_full is True

    def test_pipeline_configs_differ(self, neurosymbolic_suite):
        conditions = neurosymbolic_suite.get_conditions()
        configs = [str(c.pipeline_config) for c in conditions]
        # At least some configs should be different
        assert len(set(configs)) > 1

    def test_benchmarks(self, neurosymbolic_suite):
        benchmarks = neurosymbolic_suite.get_benchmarks()
        assert len(benchmarks) == 3
        assert "code_generation" in benchmarks

    def test_paper_name(self, neurosymbolic_suite):
        assert "Neurosymbolic" in neurosymbolic_suite.get_paper_name()

    def test_key_comparisons(self, neurosymbolic_suite):
        comparisons = neurosymbolic_suite.get_key_comparisons()
        assert len(comparisons) == 4
        # full vs prose_only should be a key comparison
        assert ("full", "prose_only") in comparisons


class TestNeurosymbolicResults:
    """Test that running produces expected ordering."""

    def test_full_is_best(self, neurosymbolic_result):
        assert neurosymbolic_result.best_condition() == "full"

    def test_full_beats_prose_only(self, neurosymbolic_result):
        full_mean = neurosymbolic_result.get_mean_score("full")
        prose_mean = neurosymbolic_result.get_mean_score("prose_only")
        assert full_mean > prose_mean

    def test_full_beats_bdm_only(self, neurosymbolic_result):
        full_mean = neurosymbolic_result.get_mean_score("full")
        bdm_mean = neurosymbolic_result.get_mean_score("bdm_only")
        assert full_mean > bdm_mean

    def test_full_beats_symcode_only(self, neurosymbolic_result):
        full_mean = neurosymbolic_result.get_mean_score("full")
        sym_mean = neurosymbolic_result.get_mean_score("symcode_only")
        assert full_mean > sym_mean

    def test_all_conditions_have_runs(self, neurosymbolic_result):
        for cond in neurosymbolic_result.get_all_condition_names():
            scores = neurosymbolic_result.get_scores(cond)
            assert len(scores) == 5

    def test_scores_in_valid_range(self, neurosymbolic_result):
        for cond in neurosymbolic_result.get_all_condition_names():
            for score in neurosymbolic_result.get_scores(cond):
                assert 0.0 <= score <= 1.0


class TestNeurosymbolicConditionBuilder:
    """Test the condition builder."""

    def test_build_all_conditions(self):
        builder = NeurosymbolicConditionBuilder()
        conditions = builder.get_all_conditions()
        assert len(conditions) == 7

    def test_expected_accuracy(self):
        builder = NeurosymbolicConditionBuilder()
        assert builder.get_expected_accuracy("full") == 0.85
        assert builder.get_expected_accuracy("prose_only") == 0.72

    def test_build_config(self):
        builder = NeurosymbolicConditionBuilder()
        config = builder.build("prose_only")
        assert config["symbolic_code"] is False
        assert config["prose"] is True
        assert config["bdm"] is False

    def test_unknown_condition_defaults(self):
        builder = NeurosymbolicConditionBuilder()
        acc = builder.get_expected_accuracy("nonexistent")
        assert acc == 0.70
