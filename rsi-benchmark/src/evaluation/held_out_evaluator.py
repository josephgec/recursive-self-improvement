"""Held-out evaluator: evaluate on tasks never seen during training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.benchmarks.registry import BaseBenchmark, BenchmarkResult, BenchmarkTask


@dataclass
class HeldOutResult:
    """Result of held-out evaluation."""
    benchmark: str
    accuracy: float
    results: List[BenchmarkResult]
    num_tasks: int
    generalization_gap: float = 0.0  # Difference from training accuracy


class HeldOutEvaluator:
    """Evaluate agent on held-out tasks it has never seen."""

    def __init__(
        self,
        benchmarks: Dict[str, BaseBenchmark],
        held_out_fraction: float = 0.2,
    ) -> None:
        self._benchmarks = benchmarks
        self._held_out_tasks: Dict[str, List[BenchmarkTask]] = {}
        self._training_tasks: Dict[str, List[BenchmarkTask]] = {}
        self._split(held_out_fraction)

    def _split(self, fraction: float) -> None:
        """Split tasks into training and held-out sets."""
        for name, bm in self._benchmarks.items():
            all_tasks = bm.tasks
            n_held_out = max(1, int(len(all_tasks) * fraction))
            self._held_out_tasks[name] = all_tasks[-n_held_out:]
            self._training_tasks[name] = all_tasks[:-n_held_out]

    @property
    def held_out_tasks(self) -> Dict[str, List[BenchmarkTask]]:
        return dict(self._held_out_tasks)

    @property
    def training_tasks(self) -> Dict[str, List[BenchmarkTask]]:
        return dict(self._training_tasks)

    def evaluate(
        self,
        agent: Any,
        training_accuracy: Optional[Dict[str, float]] = None,
    ) -> Dict[str, HeldOutResult]:
        """Evaluate agent on held-out tasks."""
        results: Dict[str, HeldOutResult] = {}
        for name, benchmark in self._benchmarks.items():
            held_out = self._held_out_tasks[name]
            bm_results = benchmark.evaluate(agent, held_out)
            if bm_results:
                correct = sum(1 for r in bm_results if r.correct)
                accuracy = correct / len(bm_results)
            else:
                accuracy = 0.0

            gen_gap = 0.0
            if training_accuracy and name in training_accuracy:
                gen_gap = training_accuracy[name] - accuracy

            results[name] = HeldOutResult(
                benchmark=name,
                accuracy=accuracy,
                results=bm_results,
                num_tasks=len(held_out),
                generalization_gap=gen_gap,
            )
        return results
