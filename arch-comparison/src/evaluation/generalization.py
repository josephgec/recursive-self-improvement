"""Generalization evaluator: in-domain vs out-of-domain accuracy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.utils.task_domains import Task


@dataclass
class GeneralizationResult:
    """Result from generalization evaluation."""
    system: str
    in_domain_accuracy: float = 0.0
    out_of_domain_accuracy: float = 0.0
    generalization_gap: float = 0.0
    transfer_ratio: float = 0.0
    in_domain_results: List[Dict[str, Any]] = field(default_factory=list)
    out_of_domain_results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class GeneralizationEvaluator:
    """Evaluates how well a system generalises from training to test tasks.

    Measures:
    - In-domain accuracy (same distribution as training)
    - Out-of-domain accuracy (different distribution)
    - Generalization gap (in-domain - out-of-domain)
    - Transfer ratio (out-of-domain / in-domain)
    """

    def evaluate(
        self,
        system: str,
        pipeline: Any,
        train_tasks: List[Task],
        test_tasks: List[Task],
    ) -> GeneralizationResult:
        """Evaluate generalization.

        Args:
            system: System name ("hybrid", "integrative", "prose").
            pipeline: Pipeline with a solve(problem) method.
            train_tasks: In-domain tasks (same distribution as training).
            test_tasks: Out-of-domain tasks (different distribution).

        Returns:
            GeneralizationResult with accuracy metrics.
        """
        in_domain_results = self._evaluate_tasks(pipeline, train_tasks)
        out_of_domain_results = self._evaluate_tasks(pipeline, test_tasks)

        in_acc = self._compute_accuracy(in_domain_results)
        out_acc = self._compute_accuracy(out_of_domain_results)
        gap = in_acc - out_acc
        transfer = out_acc / in_acc if in_acc > 0 else 0.0

        return GeneralizationResult(
            system=system,
            in_domain_accuracy=in_acc,
            out_of_domain_accuracy=out_acc,
            generalization_gap=gap,
            transfer_ratio=transfer,
            in_domain_results=in_domain_results,
            out_of_domain_results=out_of_domain_results,
        )

    def _evaluate_tasks(
        self, pipeline: Any, tasks: List[Task]
    ) -> List[Dict[str, Any]]:
        """Run pipeline on tasks and collect results."""
        results = []
        for task in tasks:
            result = pipeline.solve(task.problem)
            correct = self._check_answer(result.answer, task.expected_answer)
            result.correct = correct
            results.append({
                "task_id": task.task_id,
                "domain": task.domain,
                "problem": task.problem,
                "expected": task.expected_answer,
                "predicted": result.answer,
                "correct": correct,
            })
        return results

    @staticmethod
    def _check_answer(predicted: str, expected: str) -> bool:
        """Check if predicted answer matches expected."""
        pred = predicted.strip().lower().rstrip(".")
        exp = expected.strip().lower().rstrip(".")

        # Direct match
        if pred == exp:
            return True

        # Numeric comparison
        try:
            return abs(float(pred) - float(exp)) < 1e-6
        except (ValueError, OverflowError):
            pass

        # Substring match (expected in predicted)
        if exp in pred or pred in exp:
            return True

        return False

    @staticmethod
    def _compute_accuracy(results: List[Dict[str, Any]]) -> float:
        """Compute accuracy from results."""
        if not results:
            return 0.0
        correct = sum(1 for r in results if r["correct"])
        return correct / len(results)
