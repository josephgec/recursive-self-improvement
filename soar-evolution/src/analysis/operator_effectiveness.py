"""Analysis of operator effectiveness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.population.individual import Individual


@dataclass
class OperatorStats:
    """Statistics for a single operator."""

    name: str
    invocations: int = 0
    improvements: int = 0
    total_fitness_delta: float = 0.0
    children_produced: int = 0
    best_child_fitness: float = 0.0

    @property
    def improvement_rate(self) -> float:
        if self.invocations == 0:
            return 0.0
        return self.improvements / self.invocations

    @property
    def avg_fitness_delta(self) -> float:
        if self.invocations == 0:
            return 0.0
        return self.total_fitness_delta / self.invocations


class OperatorEffectivenessAnalyzer:
    """Tracks and analyzes effectiveness of genetic operators."""

    def __init__(self):
        self._stats: Dict[str, OperatorStats] = {}

    def record(
        self,
        operator_name: str,
        parent_fitness: float,
        child_fitness: float,
    ) -> None:
        """Record the result of an operator application."""
        if operator_name not in self._stats:
            self._stats[operator_name] = OperatorStats(name=operator_name)

        stats = self._stats[operator_name]
        stats.invocations += 1
        stats.children_produced += 1

        delta = child_fitness - parent_fitness
        stats.total_fitness_delta += delta

        if delta > 0:
            stats.improvements += 1

        if child_fitness > stats.best_child_fitness:
            stats.best_child_fitness = child_fitness

    def record_from_individual(
        self,
        individual: Individual,
        parent_fitness: float = 0.0,
    ) -> None:
        """Record from an individual's metadata."""
        self.record(
            individual.operator,
            parent_fitness=individual.metadata.get("parent_fitness", parent_fitness),
            child_fitness=individual.fitness,
        )

    def get_stats(self, operator_name: str) -> Optional[OperatorStats]:
        """Get statistics for a specific operator."""
        return self._stats.get(operator_name)

    def all_stats(self) -> Dict[str, OperatorStats]:
        """Get statistics for all operators."""
        return dict(self._stats)

    def rank_operators(self) -> List[OperatorStats]:
        """Rank operators by improvement rate."""
        return sorted(
            self._stats.values(),
            key=lambda s: s.improvement_rate,
            reverse=True,
        )

    def best_operator(self) -> Optional[str]:
        """Return the name of the most effective operator."""
        ranked = self.rank_operators()
        return ranked[0].name if ranked else None

    def summary(self) -> Dict[str, Any]:
        """Generate effectiveness summary."""
        result = {}
        for name, stats in self._stats.items():
            result[name] = {
                "invocations": stats.invocations,
                "improvement_rate": stats.improvement_rate,
                "avg_fitness_delta": stats.avg_fitness_delta,
                "best_child_fitness": stats.best_child_fitness,
            }
        return result

    def clear(self) -> None:
        """Clear all recorded statistics."""
        self._stats.clear()
