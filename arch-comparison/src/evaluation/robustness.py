"""Robustness evaluator: consistency and degradation under perturbations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils.task_domains import Task


@dataclass
class RobustnessResult:
    """Result from robustness evaluation."""
    system: str
    consistency: float = 0.0
    degradation: float = 0.0
    worst_case_accuracy: float = 0.0
    perturbation_type: str = ""
    original_accuracy: float = 0.0
    perturbed_accuracy: float = 0.0
    per_task_results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class RobustnessEvaluator:
    """Evaluates robustness to perturbations.

    Metrics:
    - Consistency: fraction of tasks where original and perturbed give same answer
    - Degradation: drop in accuracy from original to perturbed
    - Worst-case accuracy: minimum accuracy across perturbation types
    """

    def evaluate(
        self,
        system: str,
        pipeline: Any,
        original_tasks: List[Task],
        perturbed_tasks: List[Task],
        perturbation_type: str = "rephrase",
    ) -> RobustnessResult:
        """Evaluate robustness.

        Args:
            system: System name.
            pipeline: Pipeline with solve(problem) method.
            original_tasks: Original tasks.
            perturbed_tasks: Perturbed versions of the same tasks.
            perturbation_type: Type of perturbation applied.

        Returns:
            RobustnessResult with consistency and degradation metrics.
        """
        assert len(original_tasks) == len(perturbed_tasks), \
            "Original and perturbed task lists must be the same length"

        per_task: List[Dict[str, Any]] = []
        orig_correct = 0
        pert_correct = 0
        consistent = 0

        for orig, pert in zip(original_tasks, perturbed_tasks):
            orig_result = pipeline.solve(orig.problem)
            pert_result = pipeline.solve(pert.problem)

            orig_ok = self._check_answer(orig_result.answer, orig.expected_answer)
            pert_ok = self._check_answer(pert_result.answer, pert.expected_answer)

            orig_result.correct = orig_ok
            pert_result.correct = pert_ok

            if orig_ok:
                orig_correct += 1
            if pert_ok:
                pert_correct += 1

            is_consistent = self._answers_match(orig_result.answer, pert_result.answer)
            if is_consistent:
                consistent += 1

            per_task.append({
                "task_id": orig.task_id,
                "original_answer": orig_result.answer,
                "perturbed_answer": pert_result.answer,
                "original_correct": orig_ok,
                "perturbed_correct": pert_ok,
                "consistent": is_consistent,
            })

        n = len(original_tasks) if original_tasks else 1
        orig_acc = orig_correct / n
        pert_acc = pert_correct / n
        consistency = self._compute_consistency(per_task)
        degradation = self._compute_degradation(orig_acc, pert_acc)

        return RobustnessResult(
            system=system,
            consistency=consistency,
            degradation=degradation,
            worst_case_accuracy=min(orig_acc, pert_acc),
            perturbation_type=perturbation_type,
            original_accuracy=orig_acc,
            perturbed_accuracy=pert_acc,
            per_task_results=per_task,
        )

    def _compute_consistency(self, per_task: List[Dict[str, Any]]) -> float:
        """Compute consistency: fraction of tasks with same answer."""
        if not per_task:
            return 0.0
        consistent = sum(1 for t in per_task if t["consistent"])
        return consistent / len(per_task)

    def _compute_degradation(self, orig_acc: float, pert_acc: float) -> float:
        """Compute degradation: drop from original to perturbed accuracy."""
        return max(0.0, orig_acc - pert_acc)

    @staticmethod
    def _check_answer(predicted: str, expected: str) -> bool:
        """Check if predicted answer matches expected."""
        pred = predicted.strip().lower().rstrip(".")
        exp = expected.strip().lower().rstrip(".")
        if pred == exp:
            return True
        try:
            return abs(float(pred) - float(exp)) < 1e-6
        except (ValueError, OverflowError):
            pass
        if exp in pred or pred in exp:
            return True
        return False

    @staticmethod
    def _answers_match(a: str, b: str) -> bool:
        """Check if two answers are equivalent."""
        a_clean = a.strip().lower().rstrip(".")
        b_clean = b.strip().lower().rstrip(".")
        if a_clean == b_clean:
            return True
        try:
            return abs(float(a_clean) - float(b_clean)) < 1e-6
        except (ValueError, OverflowError):
            pass
        return False
