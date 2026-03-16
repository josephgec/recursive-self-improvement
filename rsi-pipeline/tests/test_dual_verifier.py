"""Tests for the Dual Verifier."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.pipeline.state import PipelineState, AgentCodeSnapshot, PerformanceRecord
from src.outer_loop.strategy_evolver import Candidate
from src.verification.dual_verifier import DualVerifier, VerifiedCandidate
from src.verification.empirical_gate import EmpiricalGate, EmpiricalResult
from src.verification.compactness_gate import CompactnessGate, CompactnessResult
from src.verification.pareto_filter import ParetoFilter
from src.verification.verification_cache import VerificationCache


@pytest.fixture
def state():
    return PipelineState(
        agent_code=AgentCodeSnapshot(code="def solve(x): return x + 1"),
        original_code=AgentCodeSnapshot(code="def solve(x): return x + 1"),
        performance=PerformanceRecord(accuracy=0.7),
    )


@pytest.fixture
def good_candidate():
    return Candidate(
        candidate_id="good_001",
        target="strategy_evolver",
        proposed_code="def solve(x): return x * 2 + 1  # improved version with better logic",
        description="Improvement",
        operator="mutate",
    )


@pytest.fixture
def bad_code_candidate():
    return Candidate(
        candidate_id="bad_001",
        target="strategy_evolver",
        proposed_code="x",  # too short to pass empirical
        description="Bad code",
        operator="mutate",
    )


@pytest.fixture
def complex_candidate():
    return Candidate(
        candidate_id="complex_001",
        target="strategy_evolver",
        proposed_code="x " * 1000,  # very high complexity
        description="Complex code",
        operator="mutate",
    )


class TestCandidatePassesBothGates:
    """Test candidates that pass both gates."""

    def test_good_candidate_passes(self, state, good_candidate):
        verifier = DualVerifier(
            empirical_gate=EmpiricalGate(min_pass_rate=0.1),
            compactness_gate=CompactnessGate(max_bdm_score=9999),
        )
        verified = verifier.verify_all([good_candidate], state)

        assert len(verified) == 1
        assert verified[0].empirical.passed is True
        assert verified[0].compactness.passed is True
        assert verified[0].candidate.candidate_id == "good_001"

    def test_verified_has_combined_score(self, state, good_candidate):
        verifier = DualVerifier(
            empirical_gate=EmpiricalGate(min_pass_rate=0.1),
            compactness_gate=CompactnessGate(max_bdm_score=9999),
        )
        verified = verifier.verify_all([good_candidate], state)

        assert len(verified) == 1
        assert verified[0].combined_score > 0

    def test_multiple_candidates_sorted_by_score(self, state):
        c1 = Candidate(candidate_id="c1", target="t", proposed_code="def f(): pass  # short code", operator="mutate")
        c2 = Candidate(candidate_id="c2", target="t", proposed_code="def f(): return 1 + 2 + 3 + 4 + 5  # longer improved code with more content", operator="mutate")

        verifier = DualVerifier(
            empirical_gate=EmpiricalGate(min_pass_rate=0.1),
            compactness_gate=CompactnessGate(max_bdm_score=9999),
        )
        verified = verifier.verify_all([c1, c2], state)

        assert len(verified) >= 1
        # They should be sorted by combined score
        if len(verified) >= 2:
            assert verified[0].combined_score >= verified[1].combined_score


class TestFailsEmpiricalGate:
    """Test candidates failing the empirical gate."""

    def test_short_code_fails_empirical(self, state, bad_code_candidate):
        verifier = DualVerifier(
            empirical_gate=EmpiricalGate(min_pass_rate=0.99),
        )
        verified = verifier.verify_all([bad_code_candidate], state)
        assert len(verified) == 0

    def test_empty_code_fails(self, state):
        empty = Candidate(candidate_id="empty", target="t", proposed_code="", operator="mutate")
        verifier = DualVerifier(
            empirical_gate=EmpiricalGate(min_pass_rate=0.1),
        )
        verified = verifier.verify_all([empty], state)
        assert len(verified) == 0


class TestFailsCompactnessGate:
    """Test candidates failing the compactness gate."""

    def test_complex_code_fails_compactness(self, state):
        # Create code that will pass empirical but fail compactness
        code = " ".join([f"var_{i}" for i in range(500)])  # many unique tokens
        candidate = Candidate(
            candidate_id="complex_fail",
            target="t",
            proposed_code=code,
            operator="mutate",
        )
        verifier = DualVerifier(
            empirical_gate=EmpiricalGate(min_pass_rate=0.1),
            compactness_gate=CompactnessGate(max_bdm_score=1.0),  # very strict
        )
        verified = verifier.verify_all([candidate], state)
        assert len(verified) == 0


class TestParetoFilter:
    """Test the Pareto filter."""

    def test_pareto_filters_dominated(self):
        pf = ParetoFilter()
        candidates = [
            {"accuracy": 0.9, "bdm_score": 50},  # best accuracy, good compactness
            {"accuracy": 0.8, "bdm_score": 100},  # dominated by first
            {"accuracy": 0.7, "bdm_score": 10},   # best compactness
        ]
        result = pf.filter(candidates)

        # First (best accuracy) and third (best compactness) are Pareto-optimal
        assert len(result) >= 1
        accuracies = {c["accuracy"] for c in result}
        assert 0.9 in accuracies  # always Pareto optimal

    def test_pareto_empty_input(self):
        pf = ParetoFilter()
        assert pf.filter([]) == []

    def test_pareto_single_candidate(self):
        pf = ParetoFilter()
        candidates = [{"accuracy": 0.8, "bdm_score": 50}]
        result = pf.filter(candidates)
        assert len(result) == 1

    def test_pareto_all_equal(self):
        pf = ParetoFilter()
        candidates = [
            {"accuracy": 0.8, "bdm_score": 50},
            {"accuracy": 0.8, "bdm_score": 50},
        ]
        result = pf.filter(candidates)
        assert len(result) == 2  # neither dominates the other


class TestVerificationCache:
    """Test verification cache."""

    def test_cache_hit(self, state, good_candidate):
        cache = VerificationCache()
        verifier = DualVerifier(
            empirical_gate=EmpiricalGate(min_pass_rate=0.1),
            compactness_gate=CompactnessGate(max_bdm_score=9999),
            cache=cache,
        )

        # First call populates cache
        verified1 = verifier.verify_all([good_candidate], state)
        assert cache.has(good_candidate.candidate_id)

        # Second call uses cache
        verified2 = verifier.verify_all([good_candidate], state)
        assert len(verified2) == len(verified1)

    def test_cache_size(self):
        cache = VerificationCache()
        assert cache.size == 0
        cache.put("test", {"result": True})
        assert cache.size == 1
        assert cache.has("test")
        assert cache.get("test") == {"result": True}

    def test_cache_clear(self):
        cache = VerificationCache()
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.size == 2
        cache.clear()
        assert cache.size == 0
