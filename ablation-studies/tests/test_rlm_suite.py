"""Tests for the RLM recursive language model ablation suite."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.suites.rlm import RLMAblation
from src.conditions.rlm_conditions import RLMConditionBuilder
from src.execution.runner import AblationRunner, MockPipeline


class TestRLMConditions:
    """Test that all 8 conditions are correctly defined."""

    def test_has_8_conditions(self, rlm_suite):
        conditions = rlm_suite.get_conditions()
        assert len(conditions) == 8

    def test_condition_names(self, rlm_suite):
        conditions = rlm_suite.get_conditions()
        names = {c.name for c in conditions}
        expected = {
            "full", "no_recursion", "no_helpers", "depth_1",
            "no_repl", "repl_no_code", "chunked_prompt", "rag_baseline",
        }
        assert names == expected

    def test_full_condition_is_marked(self, rlm_suite):
        full = rlm_suite.get_full_condition()
        assert full is not None
        assert full.name == "full"

    def test_no_repl_disables_code_execution(self, rlm_suite):
        conditions = rlm_suite.get_conditions()
        nr = next(c for c in conditions if c.name == "no_repl")
        assert nr.pipeline_config["repl"] is False
        assert nr.pipeline_config["code_execution"] is False

    def test_depth_1_limits_recursion(self, rlm_suite):
        conditions = rlm_suite.get_conditions()
        d1 = next(c for c in conditions if c.name == "depth_1")
        assert d1.pipeline_config["max_depth"] == 1

    def test_benchmarks(self, rlm_suite):
        benchmarks = rlm_suite.get_benchmarks()
        assert "large_context" in benchmarks
        assert "code_execution" in benchmarks

    def test_paper_name(self, rlm_suite):
        assert "Recursive" in rlm_suite.get_paper_name()

    def test_key_comparisons(self, rlm_suite):
        comparisons = rlm_suite.get_key_comparisons()
        assert ("full", "no_repl") in comparisons
        assert ("full", "no_recursion") in comparisons


class TestRLMResults:
    """Test that running produces expected ordering."""

    def test_full_is_best(self, rlm_result):
        assert rlm_result.best_condition() == "full"

    def test_full_beats_no_repl(self, rlm_result):
        """Full should significantly outperform no_repl (largest gap)."""
        full_mean = rlm_result.get_mean_score("full")
        nr_mean = rlm_result.get_mean_score("no_repl")
        assert full_mean > nr_mean
        # Gap should be substantial
        assert full_mean - nr_mean > 0.15

    def test_full_beats_no_recursion(self, rlm_result):
        full_mean = rlm_result.get_mean_score("full")
        norec_mean = rlm_result.get_mean_score("no_recursion")
        assert full_mean > norec_mean

    def test_full_beats_depth_1(self, rlm_result):
        full_mean = rlm_result.get_mean_score("full")
        d1_mean = rlm_result.get_mean_score("depth_1")
        assert full_mean > d1_mean

    def test_full_beats_rag_baseline(self, rlm_result):
        full_mean = rlm_result.get_mean_score("full")
        rag_mean = rlm_result.get_mean_score("rag_baseline")
        assert full_mean > rag_mean

    def test_no_repl_is_worst(self, rlm_result):
        """no_repl should be among the worst performing conditions."""
        nr_mean = rlm_result.get_mean_score("no_repl")
        conditions = rlm_result.get_all_condition_names()
        # Should be worse than most conditions
        worse_count = sum(
            1 for c in conditions
            if rlm_result.get_mean_score(c) > nr_mean
        )
        assert worse_count >= 5

    def test_all_conditions_have_runs(self, rlm_result):
        for cond in rlm_result.get_all_condition_names():
            scores = rlm_result.get_scores(cond)
            assert len(scores) == 5

    def test_scores_in_valid_range(self, rlm_result):
        for cond in rlm_result.get_all_condition_names():
            for score in rlm_result.get_scores(cond):
                assert 0.0 <= score <= 1.0


class TestRLMConditionBuilder:
    """Test the RLM condition builder."""

    def test_build_all_conditions(self):
        builder = RLMConditionBuilder()
        conditions = builder.get_all_conditions()
        assert len(conditions) == 8

    def test_expected_accuracy(self):
        builder = RLMConditionBuilder()
        assert builder.get_expected_accuracy("full") == 0.85
        assert builder.get_expected_accuracy("no_repl") == 0.60

    def test_build_config(self):
        builder = RLMConditionBuilder()
        config = builder.build("no_repl")
        assert config["repl"] is False
        assert config["code_execution"] is False

    def test_rag_baseline_config(self):
        builder = RLMConditionBuilder()
        config = builder.build("rag_baseline")
        assert config["rag"] is True
        assert config["recursion"] is False

    def test_unknown_condition_defaults(self):
        builder = RLMConditionBuilder()
        acc = builder.get_expected_accuracy("nonexistent")
        assert acc == 0.60
