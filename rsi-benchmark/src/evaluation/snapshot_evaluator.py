"""Snapshot evaluator: evaluate saved agent snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.benchmarks.registry import BaseBenchmark, BenchmarkResult, BenchmarkTask


@dataclass
class SnapshotEvaluation:
    """Evaluation of a single agent snapshot."""
    snapshot_id: str
    iteration: int
    accuracy_by_benchmark: Dict[str, float]
    overall_accuracy: float
    results: Dict[str, List[BenchmarkResult]] = field(default_factory=dict)


@dataclass
class AgentSnapshot:
    """A saved agent state."""
    snapshot_id: str
    iteration: int
    agent_state: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


class SnapshotEvaluator:
    """Evaluate saved agent snapshots on benchmarks."""

    def __init__(
        self,
        benchmarks: Dict[str, BaseBenchmark],
        tasks: Optional[Dict[str, List[BenchmarkTask]]] = None,
    ) -> None:
        self._benchmarks = benchmarks
        self._tasks = tasks or {name: bm.tasks for name, bm in benchmarks.items()}

    def evaluate(self, snapshot: AgentSnapshot) -> SnapshotEvaluation:
        """Evaluate a single snapshot."""
        accuracy_by_benchmark: Dict[str, float] = {}
        all_results: Dict[str, List[BenchmarkResult]] = {}

        for name, benchmark in self._benchmarks.items():
            tasks = self._tasks.get(name, benchmark.tasks)
            results = benchmark.evaluate(snapshot.agent_state, tasks)
            all_results[name] = results
            if results:
                correct = sum(1 for r in results if r.correct)
                accuracy_by_benchmark[name] = correct / len(results)
            else:
                accuracy_by_benchmark[name] = 0.0

        total = sum(len(r) for r in all_results.values())
        total_correct = sum(
            sum(1 for r in results if r.correct) for results in all_results.values()
        )
        overall = total_correct / total if total > 0 else 0.0

        return SnapshotEvaluation(
            snapshot_id=snapshot.snapshot_id,
            iteration=snapshot.iteration,
            accuracy_by_benchmark=accuracy_by_benchmark,
            overall_accuracy=overall,
            results=all_results,
        )

    def evaluate_all(self, snapshots: List[AgentSnapshot]) -> List[SnapshotEvaluation]:
        """Evaluate multiple snapshots."""
        return [self.evaluate(s) for s in snapshots]

    def compare_snapshots(
        self, evaluations: List[SnapshotEvaluation]
    ) -> Dict[str, Any]:
        """Compare evaluations across snapshots."""
        if not evaluations:
            return {}

        sorted_evals = sorted(evaluations, key=lambda e: e.iteration)
        comparison: Dict[str, Any] = {
            "iterations": [e.iteration for e in sorted_evals],
            "overall_accuracy": [e.overall_accuracy for e in sorted_evals],
            "by_benchmark": {},
        }

        benchmarks = sorted_evals[0].accuracy_by_benchmark.keys()
        for bm in benchmarks:
            comparison["by_benchmark"][bm] = [
                e.accuracy_by_benchmark.get(bm, 0.0) for e in sorted_evals
            ]

        # Improvement from first to last
        if len(sorted_evals) >= 2:
            comparison["total_improvement"] = (
                sorted_evals[-1].overall_accuracy - sorted_evals[0].overall_accuracy
            )
        else:
            comparison["total_improvement"] = 0.0

        return comparison
