"""Iteration evaluator: evaluates agent state at each RSI iteration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.benchmarks.registry import BaseBenchmark, BenchmarkResult, BenchmarkTask


@dataclass
class IterationEvaluation:
    """Result of evaluating one iteration."""
    iteration: int
    benchmark_results: Dict[str, List[BenchmarkResult]]
    accuracy_by_benchmark: Dict[str, float]
    overall_accuracy: float
    category_accuracy: Dict[str, Dict[str, float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class IterationEvaluator:
    """Evaluates agent state at each iteration using consistent tasks."""

    def __init__(
        self,
        benchmarks: Dict[str, BaseBenchmark],
        tasks_per_benchmark: Optional[int] = None,
    ) -> None:
        self._benchmarks = benchmarks
        self._tasks: Dict[str, List[BenchmarkTask]] = {}
        self._setup(tasks_per_benchmark)

    def _setup(self, tasks_per_benchmark: Optional[int]) -> None:
        """Select consistent task sets for each benchmark."""
        for name, bm in self._benchmarks.items():
            all_tasks = bm.tasks
            if tasks_per_benchmark is not None and tasks_per_benchmark < len(all_tasks):
                self._tasks[name] = all_tasks[:tasks_per_benchmark]
            else:
                self._tasks[name] = all_tasks

    def setup(self, tasks_per_benchmark: Optional[int] = None) -> None:
        """Public setup method for re-initialization."""
        self._setup(tasks_per_benchmark)

    @property
    def task_sets(self) -> Dict[str, List[BenchmarkTask]]:
        return dict(self._tasks)

    def evaluate_iteration(
        self,
        agent_state: Any,
        iteration: int,
    ) -> IterationEvaluation:
        """Evaluate agent on all benchmarks for one iteration.

        Uses the same tasks across all iterations for consistency.
        """
        benchmark_results: Dict[str, List[BenchmarkResult]] = {}
        accuracy_by_benchmark: Dict[str, float] = {}
        category_accuracy: Dict[str, Dict[str, float]] = {}

        for name, benchmark in self._benchmarks.items():
            tasks = self._tasks[name]
            results = benchmark.evaluate(agent_state, tasks)
            benchmark_results[name] = results

            # Compute accuracy
            if results:
                correct = sum(1 for r in results if r.correct)
                accuracy_by_benchmark[name] = correct / len(results)
            else:
                accuracy_by_benchmark[name] = 0.0

            # Compute category-level accuracy
            cat_acc: Dict[str, float] = {}
            categories: Dict[str, List[BenchmarkResult]] = {}
            for r in results:
                task = next((t for t in tasks if t.task_id == r.task_id), None)
                if task:
                    categories.setdefault(task.category, []).append(r)
            for cat, cat_results in categories.items():
                cat_correct = sum(1 for r in cat_results if r.correct)
                cat_acc[cat] = cat_correct / len(cat_results)
            category_accuracy[name] = cat_acc

        # Overall accuracy
        all_results = [r for results in benchmark_results.values() for r in results]
        if all_results:
            total_correct = sum(1 for r in all_results if r.correct)
            overall_accuracy = total_correct / len(all_results)
        else:
            overall_accuracy = 0.0

        return IterationEvaluation(
            iteration=iteration,
            benchmark_results=benchmark_results,
            accuracy_by_benchmark=accuracy_by_benchmark,
            overall_accuracy=overall_accuracy,
            category_accuracy=category_accuracy,
        )
