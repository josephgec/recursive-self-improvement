"""Tests for StagingEnvironment - pass/fail staging, live agent safety."""

import copy
import pytest
from src.self_mod.staging_env import StagingEnvironment, StagingResult


class TestStagingPass:
    """Tests for modifications that should pass staging."""

    def test_good_modification_passes(self, sample_good_candidate):
        staging = StagingEnvironment()
        agent_state = {"quality": 0.8, "modifier": 0.0}
        result = staging.test_modification(agent_state, sample_good_candidate)
        assert result.passed is True

    def test_improvement_is_positive(self, sample_good_candidate):
        staging = StagingEnvironment()
        agent_state = {"quality": 0.8, "modifier": 0.0}
        result = staging.test_modification(agent_state, sample_good_candidate)
        assert result.improvement > 0

    def test_modified_score_higher(self, sample_good_candidate):
        staging = StagingEnvironment()
        agent_state = {"quality": 0.8, "modifier": 0.0}
        result = staging.test_modification(agent_state, sample_good_candidate)
        assert result.modified_score >= result.original_score


class TestStagingFail:
    """Tests for modifications that should fail staging."""

    def test_bad_modification_fails(self, sample_bad_candidate):
        staging = StagingEnvironment()
        agent_state = {"quality": 0.8, "modifier": 0.0}
        result = staging.test_modification(agent_state, sample_bad_candidate)
        assert result.passed is False

    def test_bad_modification_shows_regression(self, sample_bad_candidate):
        staging = StagingEnvironment()
        agent_state = {"quality": 0.8, "modifier": 0.0}
        result = staging.test_modification(agent_state, sample_bad_candidate)
        assert result.regression is True

    def test_unsafe_modification_fails(self, sample_unsafe_candidate):
        staging = StagingEnvironment()
        agent_state = {"quality": 0.8, "modifier": 0.0}
        result = staging.test_modification(agent_state, sample_unsafe_candidate)
        assert result.passed is False
        assert len(result.errors) > 0


class TestLiveAgentSafety:
    """Tests that the live agent state is never modified."""

    def test_live_agent_unchanged_on_pass(self, sample_good_candidate):
        staging = StagingEnvironment()
        agent_state = {"quality": 0.8, "modifier": 0.0, "version": "1.0"}
        original_state = copy.deepcopy(agent_state)
        staging.test_modification(agent_state, sample_good_candidate)
        assert agent_state == original_state

    def test_live_agent_unchanged_on_fail(self, sample_bad_candidate):
        staging = StagingEnvironment()
        agent_state = {"quality": 0.8, "modifier": 0.0, "version": "1.0"}
        original_state = copy.deepcopy(agent_state)
        staging.test_modification(agent_state, sample_bad_candidate)
        assert agent_state == original_state

    def test_live_agent_unchanged_on_error(self, sample_unsafe_candidate):
        staging = StagingEnvironment()
        agent_state = {"quality": 0.8, "modifier": 0.0}
        original_state = copy.deepcopy(agent_state)
        staging.test_modification(agent_state, sample_unsafe_candidate)
        assert agent_state == original_state


class TestStagingHistory:
    """Tests for staging result history."""

    def test_history_recorded(self, sample_good_candidate):
        staging = StagingEnvironment()
        agent_state = {"quality": 0.8, "modifier": 0.0}
        staging.test_modification(agent_state, sample_good_candidate)
        assert len(staging.get_results_history()) == 1

    def test_multiple_results_tracked(self, sample_good_candidate, sample_bad_candidate):
        staging = StagingEnvironment()
        agent_state = {"quality": 0.8, "modifier": 0.0}
        staging.test_modification(agent_state, sample_good_candidate)
        staging.test_modification(agent_state, sample_bad_candidate)
        assert len(staging.get_results_history()) == 2

    def test_candidate_id_recorded(self, sample_good_candidate):
        staging = StagingEnvironment()
        agent_state = {"quality": 0.8, "modifier": 0.0}
        result = staging.test_modification(agent_state, sample_good_candidate)
        assert result.candidate_id == "good_candidate_001"

    def test_benchmark_results_populated(self, sample_good_candidate):
        staging = StagingEnvironment()
        agent_state = {"quality": 0.8, "modifier": 0.0}
        result = staging.test_modification(agent_state, sample_good_candidate)
        assert len(result.benchmark_results) > 0
