"""Tests for the SOAR evolutionary search ablation suite."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.suites.soar import SOARAblation
from src.conditions.soar_conditions import SOARConditionBuilder
from src.execution.runner import AblationRunner, MockPipeline


class TestSOARConditions:
    """Test that all 8 conditions are correctly defined."""

    def test_has_8_conditions(self, soar_suite):
        conditions = soar_suite.get_conditions()
        assert len(conditions) == 8

    def test_condition_names(self, soar_suite):
        conditions = soar_suite.get_conditions()
        names = {c.name for c in conditions}
        expected = {
            "full", "no_hindsight", "no_crossover", "no_error_guidance",
            "no_mutation", "random_search", "single_candidate", "hindsight_library",
        }
        assert names == expected

    def test_full_condition_is_marked(self, soar_suite):
        full = soar_suite.get_full_condition()
        assert full is not None
        assert full.name == "full"

    def test_random_search_disables_all_operators(self, soar_suite):
        conditions = soar_suite.get_conditions()
        rs = next(c for c in conditions if c.name == "random_search")
        config = rs.pipeline_config
        assert config["hindsight"] is False
        assert config["crossover"] is False
        assert config["error_guidance"] is False
        assert config["mutation"] is False

    def test_single_candidate_population_1(self, soar_suite):
        conditions = soar_suite.get_conditions()
        sc = next(c for c in conditions if c.name == "single_candidate")
        assert sc.pipeline_config["population_size"] == 1

    def test_benchmarks(self, soar_suite):
        benchmarks = soar_suite.get_benchmarks()
        assert "search_efficiency" in benchmarks
        assert "solution_quality" in benchmarks

    def test_paper_name(self, soar_suite):
        assert "SOAR" in soar_suite.get_paper_name()

    def test_key_comparisons(self, soar_suite):
        comparisons = soar_suite.get_key_comparisons()
        assert ("full", "random_search") in comparisons
        assert ("hindsight_library", "no_hindsight") in comparisons


class TestSOARResults:
    """Test that running produces expected ordering."""

    def test_full_is_best(self, soar_result):
        assert soar_result.best_condition() == "full"

    def test_full_beats_random_search(self, soar_result):
        full_mean = soar_result.get_mean_score("full")
        rs_mean = soar_result.get_mean_score("random_search")
        assert full_mean > rs_mean

    def test_hindsight_helps(self, soar_result):
        """Hindsight library should beat no_hindsight."""
        hl_mean = soar_result.get_mean_score("hindsight_library")
        nh_mean = soar_result.get_mean_score("no_hindsight")
        assert hl_mean > nh_mean

    def test_full_beats_single_candidate(self, soar_result):
        full_mean = soar_result.get_mean_score("full")
        sc_mean = soar_result.get_mean_score("single_candidate")
        assert full_mean > sc_mean

    def test_all_conditions_have_runs(self, soar_result):
        for cond in soar_result.get_all_condition_names():
            scores = soar_result.get_scores(cond)
            assert len(scores) == 5

    def test_scores_in_valid_range(self, soar_result):
        for cond in soar_result.get_all_condition_names():
            for score in soar_result.get_scores(cond):
                assert 0.0 <= score <= 1.0

    def test_random_search_is_worst(self, soar_result):
        """Random search should be among the worst conditions."""
        rs_mean = soar_result.get_mean_score("random_search")
        # At least worse than full and hindsight_library
        assert rs_mean < soar_result.get_mean_score("full")
        assert rs_mean < soar_result.get_mean_score("hindsight_library")


class TestSOARConditionBuilder:
    """Test the SOAR condition builder."""

    def test_build_all_conditions(self):
        builder = SOARConditionBuilder()
        conditions = builder.get_all_conditions()
        assert len(conditions) == 8

    def test_expected_accuracy(self):
        builder = SOARConditionBuilder()
        assert builder.get_expected_accuracy("full") == 0.85
        assert builder.get_expected_accuracy("random_search") == 0.65

    def test_build_config(self):
        builder = SOARConditionBuilder()
        config = builder.build("no_hindsight")
        assert config["hindsight"] is False
        assert config["crossover"] is True

    def test_unknown_condition_defaults(self):
        builder = SOARConditionBuilder()
        acc = builder.get_expected_accuracy("nonexistent")
        assert acc == 0.65
