"""Tests for robustness evaluator: consistency, degradation, perturbation types."""

from __future__ import annotations

import pytest

from src.evaluation.robustness import RobustnessEvaluator, RobustnessResult
from src.utils.task_domains import Task
from src.utils.perturbations import PerturbationGenerator


class TestRobustnessEvaluator:
    def test_evaluate_basic(self, hybrid_pipeline, paired_tasks):
        evaluator = RobustnessEvaluator()
        originals, perturbed = paired_tasks
        result = evaluator.evaluate(
            "hybrid", hybrid_pipeline, originals, perturbed, "rephrase"
        )
        assert isinstance(result, RobustnessResult)
        assert result.system == "hybrid"
        assert result.perturbation_type == "rephrase"
        assert 0.0 <= result.consistency <= 1.0

    def test_consistency_perfect(self):
        """When pipeline gives same answer for original and perturbed."""
        class ConstantPipeline:
            def solve(self, problem):
                from dataclasses import dataclass, field
                @dataclass
                class R:
                    answer: str = "42"
                    correct: bool = False
                    metadata: dict = field(default_factory=dict)
                return R()

        evaluator = RobustnessEvaluator()
        tasks = [Task(task_id="t1", domain="x", problem="p", expected_answer="42")]
        perturbed = [Task(task_id="t1p", domain="x", problem="q", expected_answer="42")]
        result = evaluator.evaluate("const", ConstantPipeline(), tasks, perturbed)
        assert result.consistency == 1.0

    def test_consistency_zero(self):
        """When pipeline always gives different answers."""
        call_count = [0]
        class AlternatingPipeline:
            def solve(self, problem):
                from dataclasses import dataclass, field
                @dataclass
                class R:
                    answer: str = ""
                    correct: bool = False
                    metadata: dict = field(default_factory=dict)
                call_count[0] += 1
                return R(answer=str(call_count[0]))

        evaluator = RobustnessEvaluator()
        tasks = [Task(task_id="t1", domain="x", problem="p", expected_answer="1")]
        perturbed = [Task(task_id="t1p", domain="x", problem="q", expected_answer="1")]
        result = evaluator.evaluate("alt", AlternatingPipeline(), tasks, perturbed)
        assert result.consistency == 0.0

    def test_degradation_no_drop(self, hybrid_pipeline, paired_tasks):
        evaluator = RobustnessEvaluator()
        originals, perturbed = paired_tasks
        result = evaluator.evaluate(
            "hybrid", hybrid_pipeline, originals, perturbed
        )
        # Degradation should be >= 0
        assert result.degradation >= 0.0

    def test_degradation_computation(self):
        evaluator = RobustnessEvaluator()
        assert evaluator._compute_degradation(0.8, 0.6) == pytest.approx(0.2)
        assert evaluator._compute_degradation(0.5, 0.5) == 0.0
        assert evaluator._compute_degradation(0.3, 0.5) == 0.0  # no negative degradation

    def test_worst_case_accuracy(self, hybrid_pipeline, paired_tasks):
        evaluator = RobustnessEvaluator()
        originals, perturbed = paired_tasks
        result = evaluator.evaluate(
            "hybrid", hybrid_pipeline, originals, perturbed
        )
        assert result.worst_case_accuracy == min(
            result.original_accuracy, result.perturbed_accuracy
        )

    def test_per_task_results(self, hybrid_pipeline, paired_tasks):
        evaluator = RobustnessEvaluator()
        originals, perturbed = paired_tasks
        result = evaluator.evaluate(
            "hybrid", hybrid_pipeline, originals, perturbed
        )
        assert len(result.per_task_results) == len(originals)
        for r in result.per_task_results:
            assert "task_id" in r
            assert "consistent" in r
            assert "original_correct" in r
            assert "perturbed_correct" in r

    def test_unequal_lengths_raise(self, hybrid_pipeline):
        evaluator = RobustnessEvaluator()
        orig = [Task(task_id="t1", domain="x", problem="p", expected_answer="a")]
        pert = [
            Task(task_id="t1p", domain="x", problem="q", expected_answer="a"),
            Task(task_id="t2p", domain="x", problem="r", expected_answer="b"),
        ]
        with pytest.raises(AssertionError):
            evaluator.evaluate("hybrid", hybrid_pipeline, orig, pert)

    def test_check_answer_exact(self):
        evaluator = RobustnessEvaluator()
        assert evaluator._check_answer("42", "42")
        assert not evaluator._check_answer("42", "43")

    def test_check_answer_numeric(self):
        evaluator = RobustnessEvaluator()
        assert evaluator._check_answer("7.0", "7")

    def test_check_answer_substring(self):
        evaluator = RobustnessEvaluator()
        assert evaluator._check_answer("the answer is 7", "7")

    def test_answers_match_exact(self):
        evaluator = RobustnessEvaluator()
        assert evaluator._answers_match("42", "42")
        assert not evaluator._answers_match("42", "43")

    def test_answers_match_numeric(self):
        evaluator = RobustnessEvaluator()
        assert evaluator._answers_match("7.0", "7")

    def test_compute_consistency_empty(self):
        evaluator = RobustnessEvaluator()
        assert evaluator._compute_consistency([]) == 0.0

    def test_integrative_robustness(self, integrative_pipeline, paired_tasks):
        evaluator = RobustnessEvaluator()
        originals, perturbed = paired_tasks
        result = evaluator.evaluate(
            "integrative", integrative_pipeline, originals, perturbed
        )
        assert isinstance(result, RobustnessResult)


# ── PerturbationGenerator tests ──

class TestPerturbationGenerator:
    def test_rephrase(self):
        gen = PerturbationGenerator(seed=42)
        task = Task(task_id="t1", domain="arithmetic",
                    problem="What is 3 + 4?", expected_answer="7")
        perturbed = gen.rephrase(task)
        assert perturbed.task_id == "t1_rephrase"
        assert perturbed.expected_answer == "7"
        assert perturbed.problem != task.problem

    def test_add_noise(self):
        gen = PerturbationGenerator(seed=42)
        task = Task(task_id="t1", domain="arithmetic",
                    problem="What is 3 + 4?", expected_answer="7")
        perturbed = gen.add_noise(task)
        assert perturbed.task_id == "t1_noise"
        assert perturbed.expected_answer == "7"
        assert len(perturbed.problem) > len(task.problem)

    def test_domain_shift(self):
        gen = PerturbationGenerator(seed=42)
        task = Task(task_id="t1", domain="arithmetic",
                    problem="Compute 3 + 4", expected_answer="7")
        perturbed = gen.domain_shift(task)
        assert perturbed.task_id == "t1_domain_shift"
        assert perturbed.expected_answer == "7"

    def test_adversarial(self):
        gen = PerturbationGenerator(seed=42)
        task = Task(task_id="t1", domain="arithmetic",
                    problem="What is 3 + 4?", expected_answer="7")
        perturbed = gen.adversarial(task)
        assert perturbed.task_id == "t1_adversarial"
        assert perturbed.expected_answer == "7"
        assert len(perturbed.problem) > len(task.problem)

    def test_generate_all(self):
        gen = PerturbationGenerator(seed=42)
        task = Task(task_id="t1", domain="arithmetic",
                    problem="What is 3 + 4?", expected_answer="7")
        perturbed = gen.generate_all(task)
        assert len(perturbed) == 4
        types = [p.metadata.get("perturbation") for p in perturbed]
        assert "rephrase" in types
        assert "noise" in types
        assert "domain_shift" in types
        assert "adversarial" in types

    def test_rephrase_no_keyword_match(self):
        gen = PerturbationGenerator(seed=42)
        task = Task(task_id="t1", domain="other",
                    problem="An unusual problem statement", expected_answer="x")
        perturbed = gen.rephrase(task)
        assert "Please answer" in perturbed.problem

    def test_domain_shift_no_keyword_match(self):
        gen = PerturbationGenerator(seed=42)
        task = Task(task_id="t1", domain="other",
                    problem="An unusual problem", expected_answer="x")
        perturbed = gen.domain_shift(task)
        assert "different context" in perturbed.problem
