"""Tests for the Godel Agent ablation suite."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.suites.godel import GodelAgentAblation
from src.conditions.godel_conditions import GodelConditionBuilder
from src.execution.runner import AblationRunner, MockPipeline


class TestGodelConditions:
    """Test that all 8 conditions are correctly defined."""

    def test_has_8_conditions(self, godel_suite):
        conditions = godel_suite.get_conditions()
        assert len(conditions) == 8

    def test_condition_names(self, godel_suite):
        conditions = godel_suite.get_conditions()
        names = {c.name for c in conditions}
        expected = {
            "full", "no_rollback", "no_validation", "no_ceiling",
            "no_cooldown", "no_audit", "no_self_mod", "unrestricted",
        }
        assert names == expected

    def test_full_condition_is_marked(self, godel_suite):
        full = godel_suite.get_full_condition()
        assert full is not None
        assert full.name == "full"

    def test_unrestricted_removes_all_safety(self, godel_suite):
        conditions = godel_suite.get_conditions()
        unrestricted = next(c for c in conditions if c.name == "unrestricted")
        config = unrestricted.pipeline_config
        assert config["rollback"] is False
        assert config["validation"] is False
        assert config["ceiling"] is False
        assert config["cooldown"] is False
        assert config["audit"] is False

    def test_no_self_mod_disables_modification(self, godel_suite):
        conditions = godel_suite.get_conditions()
        no_sm = next(c for c in conditions if c.name == "no_self_mod")
        assert no_sm.pipeline_config["self_modification"] is False

    def test_benchmarks(self, godel_suite):
        benchmarks = godel_suite.get_benchmarks()
        assert "self_modification" in benchmarks
        assert "safety_bounds" in benchmarks

    def test_paper_name(self, godel_suite):
        assert "Godel" in godel_suite.get_paper_name()

    def test_key_comparisons(self, godel_suite):
        comparisons = godel_suite.get_key_comparisons()
        assert ("full", "no_self_mod") in comparisons
        assert ("full", "unrestricted") in comparisons


class TestGodelResults:
    """Test that running produces expected ordering."""

    def test_full_is_best(self, godel_result):
        assert godel_result.best_condition() == "full"

    def test_full_beats_no_self_mod(self, godel_result):
        full_mean = godel_result.get_mean_score("full")
        nsm_mean = godel_result.get_mean_score("no_self_mod")
        assert full_mean > nsm_mean

    def test_unrestricted_degrades(self, godel_result):
        full_mean = godel_result.get_mean_score("full")
        unr_mean = godel_result.get_mean_score("unrestricted")
        assert full_mean > unr_mean

    def test_full_beats_no_rollback(self, godel_result):
        full_mean = godel_result.get_mean_score("full")
        nr_mean = godel_result.get_mean_score("no_rollback")
        assert full_mean > nr_mean

    def test_all_conditions_have_runs(self, godel_result):
        for cond in godel_result.get_all_condition_names():
            scores = godel_result.get_scores(cond)
            assert len(scores) == 5

    def test_scores_in_valid_range(self, godel_result):
        for cond in godel_result.get_all_condition_names():
            for score in godel_result.get_scores(cond):
                assert 0.0 <= score <= 1.0


class TestGodelConditionBuilder:
    """Test the Godel condition builder."""

    def test_build_all_conditions(self):
        builder = GodelConditionBuilder()
        conditions = builder.get_all_conditions()
        assert len(conditions) == 8

    def test_expected_accuracy(self):
        builder = GodelConditionBuilder()
        assert builder.get_expected_accuracy("full") == 0.85
        assert builder.get_expected_accuracy("no_self_mod") == 0.75
        assert builder.get_expected_accuracy("unrestricted") == 0.70

    def test_build_config(self):
        builder = GodelConditionBuilder()
        config = builder.build("no_rollback")
        assert config["rollback"] is False
        assert config["validation"] is True

    def test_unknown_condition_defaults(self):
        builder = GodelConditionBuilder()
        acc = builder.get_expected_accuracy("nonexistent")
        assert acc == 0.70
