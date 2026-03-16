"""Corpus statistics for trajectory collections."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List

from src.collection.trajectory import SearchTrajectory


class CorpusStatistics:
    """Compute and report statistics over a corpus of trajectories."""

    def __init__(self, trajectories: List[SearchTrajectory] = None):
        self._trajectories: List[SearchTrajectory] = trajectories or []

    def add(self, trajectory: SearchTrajectory) -> None:
        self._trajectories.append(trajectory)

    def add_many(self, trajectories: List[SearchTrajectory]) -> None:
        self._trajectories.extend(trajectories)

    @property
    def total(self) -> int:
        return len(self._trajectories)

    @property
    def solved_count(self) -> int:
        return sum(1 for t in self._trajectories if t.solved)

    @property
    def solve_rate(self) -> float:
        if not self._trajectories:
            return 0.0
        return self.solved_count / len(self._trajectories)

    def fitness_distribution(self) -> Dict[str, int]:
        """Distribution of best fitness values across buckets."""
        buckets: Counter = Counter()
        for t in self._trajectories:
            bucket = f"{int(t.best_fitness * 10) / 10:.1f}"
            buckets[bucket] += 1
        return dict(sorted(buckets.items()))

    def operator_distribution(self) -> Dict[str, int]:
        """Distribution of operators used across all individuals."""
        ops: Counter = Counter()
        for t in self._trajectories:
            for ind in t.individuals:
                if ind.operator:
                    ops[ind.operator] += 1
        return dict(ops.most_common())

    def generation_stats(self) -> Dict[str, float]:
        """Statistics on generation counts."""
        if not self._trajectories:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
        gens = [t.total_generations for t in self._trajectories]
        mean = sum(gens) / len(gens)
        variance = sum((g - mean) ** 2 for g in gens) / len(gens) if len(gens) > 1 else 0.0
        return {
            "mean": round(mean, 2),
            "min": float(min(gens)),
            "max": float(max(gens)),
            "std": round(math.sqrt(variance), 2),
        }

    def individual_stats(self) -> Dict[str, Any]:
        """Statistics on individuals across all trajectories."""
        total_individuals = sum(len(t.individuals) for t in self._trajectories)
        total_solved = sum(
            sum(1 for i in t.individuals if i.is_solved)
            for t in self._trajectories
        )
        total_errors = sum(
            sum(1 for i in t.individuals if i.error is not None)
            for t in self._trajectories
        )
        return {
            "total": total_individuals,
            "solved": total_solved,
            "errors": total_errors,
        }

    def improvement_chain_stats(self) -> Dict[str, float]:
        """Statistics on improvement chain lengths."""
        chains = []
        for t in self._trajectories:
            if not t.improvement_steps:
                t.extract_improvement_chain()
            chains.append(len(t.improvement_steps))

        if not chains:
            return {"mean": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": round(sum(chains) / len(chains), 2),
            "min": float(min(chains)),
            "max": float(max(chains)),
        }

    def difficulty_distribution(self) -> Dict[str, int]:
        """Distribution of task difficulties."""
        diffs: Counter = Counter()
        for t in self._trajectories:
            if t.task:
                diffs[t.task.difficulty] += 1
        return dict(diffs.most_common())

    def full_report(self) -> Dict[str, Any]:
        """Generate a complete statistics report."""
        return {
            "total_trajectories": self.total,
            "solved_count": self.solved_count,
            "solve_rate": round(self.solve_rate, 4),
            "fitness_distribution": self.fitness_distribution(),
            "operator_distribution": self.operator_distribution(),
            "generation_stats": self.generation_stats(),
            "individual_stats": self.individual_stats(),
            "improvement_chain_stats": self.improvement_chain_stats(),
            "difficulty_distribution": self.difficulty_distribution(),
        }
