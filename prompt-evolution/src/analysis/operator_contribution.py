"""Per-operator improvement tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.genome.prompt_genome import PromptGenome


@dataclass
class OperatorStats:
    """Statistics for a single operator."""

    operator_name: str
    applications: int = 0
    total_fitness_improvement: float = 0.0
    avg_improvement: float = 0.0
    best_offspring_fitness: float = 0.0
    worst_offspring_fitness: float = 1.0


class OperatorTracker:
    """Track contributions of each genetic operator."""

    def __init__(self):
        self.stats: Dict[str, OperatorStats] = {}
        self._history: List[Dict] = []

    def record(
        self,
        operator: str,
        parent_fitness: float,
        offspring_fitness: float,
    ):
        """Record an operator application and its outcome."""
        if operator not in self.stats:
            self.stats[operator] = OperatorStats(operator_name=operator)

        s = self.stats[operator]
        s.applications += 1
        improvement = offspring_fitness - parent_fitness
        s.total_fitness_improvement += improvement
        s.avg_improvement = s.total_fitness_improvement / s.applications
        s.best_offspring_fitness = max(s.best_offspring_fitness, offspring_fitness)
        s.worst_offspring_fitness = min(s.worst_offspring_fitness, offspring_fitness)

        self._history.append({
            "operator": operator,
            "parent_fitness": parent_fitness,
            "offspring_fitness": offspring_fitness,
            "improvement": improvement,
        })

    def get_summary(self) -> Dict[str, Dict]:
        """Get summary of all operator contributions."""
        summary = {}
        for name, s in self.stats.items():
            summary[name] = {
                "applications": s.applications,
                "avg_improvement": s.avg_improvement,
                "total_improvement": s.total_fitness_improvement,
                "best_offspring": s.best_offspring_fitness,
            }
        return summary

    def get_best_operator(self) -> Optional[str]:
        """Get the operator with the highest average improvement."""
        if not self.stats:
            return None
        return max(
            self.stats.keys(),
            key=lambda k: self.stats[k].avg_improvement,
        )

    def format_report(self) -> str:
        """Format a text report of operator contributions."""
        lines = ["Operator Contribution Report", "=" * 40]

        sorted_ops = sorted(
            self.stats.values(),
            key=lambda s: s.avg_improvement,
            reverse=True,
        )

        for s in sorted_ops:
            lines.append(
                f"{s.operator_name}: "
                f"applications={s.applications}, "
                f"avg_improvement={s.avg_improvement:+.4f}, "
                f"best={s.best_offspring_fitness:.4f}"
            )

        return "\n".join(lines)
