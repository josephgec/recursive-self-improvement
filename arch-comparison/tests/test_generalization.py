"""Tests for generalization evaluator: in/out domain accuracy, gap computation."""

from __future__ import annotations

import pytest

from src.evaluation.generalization import GeneralizationEvaluator, GeneralizationResult
from src.utils.task_domains import Task


class TestGeneralizationEvaluator:
    def test_evaluate_basic(self, hybrid_pipeline, sample_arithmetic_tasks, sample_algebra_tasks):
        evaluator = GeneralizationEvaluator()
        result = evaluator.evaluate(
            "hybrid",
            hybrid_pipeline,
            sample_arithmetic_tasks,
            sample_algebra_tasks,
        )
        assert isinstance(result, GeneralizationResult)
        assert result.system == "hybrid"
        assert 0.0 <= result.in_domain_accuracy <= 1.0
        assert 0.0 <= result.out_of_domain_accuracy <= 1.0

    def test_generalization_gap_computation(self, integrative_pipeline, sample_arithmetic_tasks):
        evaluator = GeneralizationEvaluator()
        # Use same tasks for both — gap should be ~0
        result = evaluator.evaluate(
            "integrative",
            integrative_pipeline,
            sample_arithmetic_tasks,
            sample_arithmetic_tasks,
        )
        assert result.generalization_gap == result.in_domain_accuracy - result.out_of_domain_accuracy

    def test_transfer_ratio(self, hybrid_pipeline, sample_arithmetic_tasks, sample_algebra_tasks):
        evaluator = GeneralizationEvaluator()
        result = evaluator.evaluate(
            "hybrid",
            hybrid_pipeline,
            sample_arithmetic_tasks,
            sample_algebra_tasks,
        )
        if result.in_domain_accuracy > 0:
            expected_ratio = result.out_of_domain_accuracy / result.in_domain_accuracy
            assert abs(result.transfer_ratio - expected_ratio) < 1e-6

    def test_transfer_ratio_zero_in_domain(self):
        """When in-domain accuracy is 0, transfer ratio should be 0."""
        class FailPipeline:
            def solve(self, problem):
                from dataclasses import dataclass, field
                @dataclass
                class R:
                    answer: str = "wrong_answer_xyz"
                    correct: bool = False
                    metadata: dict = field(default_factory=dict)
                return R()

        evaluator = GeneralizationEvaluator()
        tasks = [Task(task_id="t1", domain="x", problem="p", expected_answer="correct")]
        result = evaluator.evaluate("fail", FailPipeline(), tasks, tasks)
        assert result.transfer_ratio == 0.0

    def test_results_contain_task_details(self, hybrid_pipeline, sample_arithmetic_tasks):
        evaluator = GeneralizationEvaluator()
        result = evaluator.evaluate(
            "hybrid",
            hybrid_pipeline,
            sample_arithmetic_tasks,
            sample_arithmetic_tasks,
        )
        assert len(result.in_domain_results) == len(sample_arithmetic_tasks)
        for r in result.in_domain_results:
            assert "task_id" in r
            assert "correct" in r
            assert "predicted" in r

    def test_check_answer_exact_match(self):
        evaluator = GeneralizationEvaluator()
        assert evaluator._check_answer("42", "42")
        assert evaluator._check_answer("  42  ", "42")

    def test_check_answer_numeric(self):
        evaluator = GeneralizationEvaluator()
        assert evaluator._check_answer("42.0", "42")
        assert evaluator._check_answer("7", "7.0")

    def test_check_answer_substring(self):
        evaluator = GeneralizationEvaluator()
        assert evaluator._check_answer("the answer is 7", "7")
        assert evaluator._check_answer("7", "the answer is 7")

    def test_check_answer_mismatch(self):
        evaluator = GeneralizationEvaluator()
        assert not evaluator._check_answer("42", "43")
        assert not evaluator._check_answer("hello", "world")

    def test_compute_accuracy_empty(self):
        evaluator = GeneralizationEvaluator()
        assert evaluator._compute_accuracy([]) == 0.0

    def test_compute_accuracy_all_correct(self):
        evaluator = GeneralizationEvaluator()
        results = [{"correct": True}, {"correct": True}, {"correct": True}]
        assert evaluator._compute_accuracy(results) == 1.0

    def test_compute_accuracy_mixed(self):
        evaluator = GeneralizationEvaluator()
        results = [{"correct": True}, {"correct": False}, {"correct": True}]
        assert abs(evaluator._compute_accuracy(results) - 2 / 3) < 1e-6

    def test_prose_pipeline_evaluation(self, prose_pipeline, sample_arithmetic_tasks):
        evaluator = GeneralizationEvaluator()
        result = evaluator.evaluate(
            "prose",
            prose_pipeline,
            sample_arithmetic_tasks,
            sample_arithmetic_tasks,
        )
        assert isinstance(result, GeneralizationResult)
        assert result.system == "prose"
