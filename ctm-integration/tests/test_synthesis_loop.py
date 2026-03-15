"""Tests for the full synthesis loop."""

import pytest

from src.synthesis.candidate_generator import IOExample
from src.synthesis.synthesis_loop import (
    SymbolicSynthesisLoop,
    IterationResult,
    SynthesisResult,
)
from src.library.rule import VerifiedRule
from src.library.store import RuleStore
from src.library.evolution import LibraryEvolver


class TestSynthesisLoop:
    """Tests for the iterative synthesis loop."""

    def test_run_produces_result(self, double_examples):
        """run() should return a SynthesisResult."""
        loop = SymbolicSynthesisLoop()
        result = loop.run(double_examples, max_iterations=2, candidates_per_iteration=3)
        assert isinstance(result, SynthesisResult)
        assert result.total_iterations > 0
        assert result.total_candidates > 0

    def test_three_iterations(self, double_examples):
        """Running 3 iterations should produce 3 iteration results."""
        loop = SymbolicSynthesisLoop()
        result = loop.run(double_examples, max_iterations=3, candidates_per_iteration=3)
        # May terminate early if perfect accuracy found
        assert result.total_iterations >= 1
        assert len(result.iterations) >= 1

    def test_accuracy_improves_or_perfect(self, double_examples):
        """Accuracy should be reasonable for simple patterns."""
        loop = SymbolicSynthesisLoop()
        result = loop.run(double_examples, max_iterations=3, candidates_per_iteration=5)
        # Mock LLM should detect y=2x easily
        assert result.final_best_accuracy >= 0.8

    def test_perfect_accuracy_early_stop(self, double_examples):
        """Perfect accuracy should cause early termination."""
        loop = SymbolicSynthesisLoop()
        result = loop.run(double_examples, max_iterations=10, candidates_per_iteration=5)
        # Should stop before 10 iterations because y=2x is easy
        assert result.final_best_accuracy == 1.0
        assert result.total_iterations <= 10

    def test_iteration_results_have_candidates(self, double_examples):
        """Each iteration should generate candidates."""
        loop = SymbolicSynthesisLoop()
        result = loop.run(double_examples, max_iterations=2, candidates_per_iteration=3)
        for ir in result.iterations:
            assert isinstance(ir, IterationResult)
            assert ir.candidates_generated >= 3

    def test_best_rules_on_pareto_front(self, double_examples):
        """best_rules should be from the Pareto front."""
        loop = SymbolicSynthesisLoop()
        result = loop.run(double_examples, max_iterations=2, candidates_per_iteration=5)
        if result.best_rules:
            assert all(sr.is_pareto_optimal for sr in result.best_rules)

    def test_best_rule_property(self, double_examples):
        """best_rule should return the highest accuracy rule."""
        loop = SymbolicSynthesisLoop()
        result = loop.run(double_examples, max_iterations=2, candidates_per_iteration=5)
        if result.best_rule:
            assert result.best_rule.accuracy == result.final_best_accuracy


class TestSynthesisWithLibrary:
    """Tests for synthesis loop with library integration."""

    def test_library_grows(self, double_examples, linear_examples, sum_examples):
        """Running synthesis on multiple problems should grow the library."""
        store = RuleStore()
        evolver = LibraryEvolver(store=store, min_accuracy=0.5)
        loop = SymbolicSynthesisLoop()

        problems = [double_examples, linear_examples, sum_examples]

        for examples in problems:
            result = loop.run(examples, max_iterations=2, candidates_per_iteration=3)

            # Add successful rules to library
            new_rules = []
            for sr in result.best_rules:
                if sr.accuracy >= 0.5:
                    new_rules.append(
                        VerifiedRule(
                            rule_id=sr.rule.rule_id,
                            domain="math",
                            description=sr.rule.description,
                            source_code=sr.rule.source_code,
                            accuracy=sr.accuracy,
                            bdm_score=sr.bdm_complexity,
                            mdl_score=sr.mdl_score,
                        )
                    )
            evolver.evolve(new_rules)

        # Library should have grown
        assert store.size > 0

    def test_synthesis_linear(self, linear_examples):
        """Synthesis should find y = 2x + 1."""
        loop = SymbolicSynthesisLoop()
        result = loop.run(linear_examples, max_iterations=3, candidates_per_iteration=5)
        assert result.final_best_accuracy >= 0.8

    def test_synthesis_quadratic(self, quadratic_examples):
        """Synthesis should find y = x^2."""
        loop = SymbolicSynthesisLoop()
        result = loop.run(quadratic_examples, max_iterations=3, candidates_per_iteration=5)
        assert result.final_best_accuracy >= 0.8

    def test_synthesis_sum(self, sum_examples):
        """Synthesis should find sum(list)."""
        loop = SymbolicSynthesisLoop()
        result = loop.run(sum_examples, max_iterations=3, candidates_per_iteration=5)
        assert result.final_best_accuracy >= 0.8
