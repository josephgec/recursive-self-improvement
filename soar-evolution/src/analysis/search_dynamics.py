"""Analysis of search dynamics across generations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GenerationSnapshot:
    """Snapshot of population state at one generation."""

    generation: int
    best_fitness: float
    avg_fitness: float
    min_fitness: float = 0.0
    pop_size: int = 0
    diversity: float = 0.0
    num_valid: int = 0


class SearchDynamicsAnalyzer:
    """Analyzes how search dynamics evolve over generations."""

    def __init__(self):
        self._snapshots: List[GenerationSnapshot] = []

    def record(self, snapshot: GenerationSnapshot) -> None:
        """Record a generation snapshot."""
        self._snapshots.append(snapshot)

    def record_from_dict(self, data: Dict[str, Any]) -> None:
        """Record from a dictionary of stats."""
        self._snapshots.append(
            GenerationSnapshot(
                generation=data.get("generation", len(self._snapshots)),
                best_fitness=data.get("best_fitness", 0.0),
                avg_fitness=data.get("avg_fitness", 0.0),
                min_fitness=data.get("min_fitness", 0.0),
                pop_size=data.get("pop_size", 0),
                diversity=data.get("diversity", 0.0),
                num_valid=data.get("num_valid", 0),
            )
        )

    @property
    def snapshots(self) -> List[GenerationSnapshot]:
        return list(self._snapshots)

    def fitness_trajectory(self) -> List[float]:
        """Get best fitness over time."""
        return [s.best_fitness for s in self._snapshots]

    def avg_fitness_trajectory(self) -> List[float]:
        """Get average fitness over time."""
        return [s.avg_fitness for s in self._snapshots]

    def convergence_generation(self, threshold: float = 0.95) -> Optional[int]:
        """Find the generation where fitness first exceeds threshold."""
        for s in self._snapshots:
            if s.best_fitness >= threshold:
                return s.generation
        return None

    def improvement_rate(self) -> float:
        """Overall rate of fitness improvement per generation."""
        if len(self._snapshots) < 2:
            return 0.0
        first = self._snapshots[0].best_fitness
        last = self._snapshots[-1].best_fitness
        gens = len(self._snapshots) - 1
        return (last - first) / max(gens, 1)

    def stagnation_periods(self, min_length: int = 3) -> List[tuple]:
        """Identify periods where fitness didn't improve."""
        periods = []
        start = 0
        best = -1.0

        for i, s in enumerate(self._snapshots):
            if s.best_fitness > best + 1e-6:
                if i - start >= min_length:
                    periods.append((start, i - 1))
                start = i
                best = s.best_fitness

        # Check final period
        if len(self._snapshots) - start >= min_length:
            periods.append((start, len(self._snapshots) - 1))

        return periods

    def summary(self) -> Dict[str, Any]:
        """Generate a summary of search dynamics."""
        if not self._snapshots:
            return {"num_generations": 0}

        trajectory = self.fitness_trajectory()
        return {
            "num_generations": len(self._snapshots),
            "initial_fitness": trajectory[0],
            "final_fitness": trajectory[-1],
            "best_fitness": max(trajectory),
            "improvement_rate": self.improvement_rate(),
            "convergence_gen": self.convergence_generation(),
            "stagnation_periods": len(self.stagnation_periods()),
        }

    def clear(self) -> None:
        """Clear all recorded snapshots."""
        self._snapshots.clear()
