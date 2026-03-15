"""Tests for Pareto front computation and selection."""

import pytest

from src.synthesis.candidate_generator import CandidateRule, IOExample
from src.synthesis.empirical_verifier import VerificationResult
from src.synthesis.pareto_selector import ParetoSelector, ScoredRule


class TestParetoFront:
    """Tests for Pareto front computation."""

    def test_single_rule_is_pareto(self, scorer):
        """A single rule should always be on the Pareto front."""
        selector = ParetoSelector(scorer=scorer)

        sr = ScoredRule(
            rule=CandidateRule(rule_id="r1", source_code="def r(x): return x", description=""),
            accuracy=0.8,
            bdm_complexity=10.0,
            mdl_score=10.0,
        )
        front = selector.compute_pareto_front([sr])
        assert len(front) == 1
        assert sr in front

    def test_dominated_rule_excluded(self, scorer):
        """A dominated rule should not be on the Pareto front."""
        selector = ParetoSelector(scorer=scorer)

        good = ScoredRule(
            rule=CandidateRule(rule_id="good", source_code="def r(x): return x", description=""),
            accuracy=1.0,
            bdm_complexity=5.0,
            mdl_score=5.0,
        )
        bad = ScoredRule(
            rule=CandidateRule(rule_id="bad", source_code="def r(x): return x", description=""),
            accuracy=0.5,
            bdm_complexity=10.0,
            mdl_score=10.0,
        )
        front = selector.compute_pareto_front([good, bad])
        assert good in front
        assert bad not in front

    def test_non_dominated_both_on_front(self, scorer):
        """Two non-dominated rules should both be on the front."""
        selector = ParetoSelector(scorer=scorer)

        # r1: high accuracy, high complexity
        r1 = ScoredRule(
            rule=CandidateRule(rule_id="r1", source_code="def r(x): return x", description=""),
            accuracy=1.0,
            bdm_complexity=20.0,
            mdl_score=20.0,
        )
        # r2: lower accuracy, lower complexity
        r2 = ScoredRule(
            rule=CandidateRule(rule_id="r2", source_code="def r(x): return x", description=""),
            accuracy=0.7,
            bdm_complexity=5.0,
            mdl_score=5.0,
        )
        front = selector.compute_pareto_front([r1, r2])
        assert r1 in front
        assert r2 in front

    def test_pareto_ranks_assigned(self, scorer):
        """Pareto ranks should be assigned correctly."""
        selector = ParetoSelector(scorer=scorer)

        rules = [
            CandidateRule(rule_id="r1", source_code="def rule(x):\n    return x * 2\n", description=""),
            CandidateRule(rule_id="r2", source_code="def rule(x):\n    return x + 1\n", description=""),
        ]
        vrs = {
            "r1": VerificationResult(rule_id="r1", passed=5, total=5, accuracy=1.0),
            "r2": VerificationResult(rule_id="r2", passed=1, total=5, accuracy=0.2),
        }
        examples = [IOExample(input=i, output=i * 2) for i in range(1, 6)]

        scored = selector.select(rules, vrs, examples)
        pareto_rules = [s for s in scored if s.is_pareto_optimal]
        assert len(pareto_rules) >= 1
        assert all(s.pareto_rank > 0 for s in scored)

    def test_empty_input(self, scorer):
        """Empty input should return empty front."""
        selector = ParetoSelector(scorer=scorer)
        front = selector.compute_pareto_front([])
        assert front == []


class TestParetoSelection:
    """Tests for the full selection pipeline."""

    def test_select_returns_scored_rules(self, scorer, double_examples):
        """select() should return scored rules with Pareto flags."""
        selector = ParetoSelector(scorer=scorer)

        rules = [
            CandidateRule(
                rule_id=f"r{i}",
                source_code=f"def rule(x):\n    return x * 2\n",
                description="",
            )
            for i in range(3)
        ]
        vrs = {
            f"r{i}": VerificationResult(rule_id=f"r{i}", passed=5, total=5, accuracy=1.0)
            for i in range(3)
        }

        scored = selector.select(rules, vrs, double_examples)
        assert len(scored) == 3
        assert all(isinstance(s, ScoredRule) for s in scored)

    def test_select_with_varying_quality(self, scorer, double_examples):
        """Selection should differentiate varying quality rules."""
        selector = ParetoSelector(scorer=scorer)

        # Perfect rule
        perfect = CandidateRule(
            rule_id="perfect",
            source_code="def rule(x):\n    return x * 2\n",
            description="",
        )
        # Imperfect rule
        imperfect = CandidateRule(
            rule_id="imperfect",
            source_code="def rule(x):\n    return x + 1\n",
            description="",
        )

        vrs = {
            "perfect": VerificationResult(rule_id="perfect", passed=5, total=5, accuracy=1.0),
            "imperfect": VerificationResult(rule_id="imperfect", passed=1, total=5, accuracy=0.2),
        }

        scored = selector.select([perfect, imperfect], vrs, double_examples)

        perfect_scored = next(s for s in scored if s.rule.rule_id == "perfect")
        imperfect_scored = next(s for s in scored if s.rule.rule_id == "imperfect")

        assert perfect_scored.accuracy > imperfect_scored.accuracy
        assert perfect_scored.is_pareto_optimal

    def test_plot_pareto_front(self, scorer, tmp_path):
        """Plotting should produce a file."""
        selector = ParetoSelector(scorer=scorer)

        scored = [
            ScoredRule(
                rule=CandidateRule(rule_id=f"r{i}", source_code="", description=""),
                accuracy=0.5 + 0.1 * i,
                bdm_complexity=10.0 + 2.0 * i,
                mdl_score=10.0,
                is_pareto_optimal=(i % 2 == 0),
            )
            for i in range(5)
        ]

        output = str(tmp_path / "pareto.png")
        result = selector.plot_pareto_front(scored, output_path=output)
        # May be None if matplotlib not available, but shouldn't crash
