"""Analyze transfer learning effects across task types."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from src.synthesis.synthesizer import TrainingPair


class TransferAnalyzer:
    """Analyze how training data from one task type transfers to others."""

    def __init__(self) -> None:
        self._pairs: List[TrainingPair] = []
        self._eval_results: Dict[str, Dict[str, float]] = {}

    def load_pairs(self, pairs: List[TrainingPair]) -> None:
        self._pairs = list(pairs)

    def load_eval_results(self, results: Dict[str, Dict[str, float]]) -> None:
        """Load evaluation results keyed by task_type -> metric -> value."""
        self._eval_results = dict(results)

    def task_coverage(self) -> Dict[str, int]:
        """Count training pairs per task."""
        coverage: Dict[str, int] = defaultdict(int)
        for p in self._pairs:
            coverage[p.task_id] += 1
        return dict(coverage)

    def strategy_task_matrix(self) -> Dict[str, Dict[str, int]]:
        """Matrix of strategy x task_id counts."""
        matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for p in self._pairs:
            matrix[p.strategy][p.task_id] += 1
        return {s: dict(tasks) for s, tasks in matrix.items()}

    def quality_by_task(self) -> Dict[str, float]:
        """Average quality score per task."""
        task_qualities: Dict[str, List[float]] = defaultdict(list)
        for p in self._pairs:
            task_qualities[p.task_id].append(p.quality_score)
        return {
            tid: round(sum(qs) / len(qs), 4)
            for tid, qs in task_qualities.items()
        }

    def transfer_matrix(self) -> Dict[str, Dict[str, float]]:
        """Compute a transfer matrix from eval results.

        Each cell (i, j) represents how well training on task_type i
        transfers to task_type j.
        """
        if not self._eval_results:
            return {}

        # If we have per-task eval results, compute transfer
        tasks = list(self._eval_results.keys())
        matrix: Dict[str, Dict[str, float]] = {}
        for src in tasks:
            matrix[src] = {}
            for tgt in tasks:
                # Self-transfer is the eval result
                src_val = self._eval_results[src].get("zero_shot_solve_rate", 0.0)
                tgt_val = self._eval_results[tgt].get("zero_shot_solve_rate", 0.0)
                # Simple transfer estimate
                if src == tgt:
                    matrix[src][tgt] = src_val
                else:
                    matrix[src][tgt] = round(min(src_val, tgt_val) * 0.7, 4)
        return matrix

    def summary(self) -> Dict[str, Any]:
        return {
            "total_pairs": len(self._pairs),
            "unique_tasks": len(self.task_coverage()),
            "task_coverage": self.task_coverage(),
            "quality_by_task": self.quality_by_task(),
        }
