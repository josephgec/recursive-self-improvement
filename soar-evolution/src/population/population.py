"""Population manager with elitism and diversity maintenance."""

from __future__ import annotations

import random
from typing import Callable, List, Optional

from src.population.individual import Individual


class Population:
    """Manages a population of candidate solutions."""

    def __init__(
        self,
        max_size: int = 50,
        elitism_ratio: float = 0.1,
        diversity_threshold: float = 0.3,
    ):
        self.max_size = max_size
        self.elitism_ratio = elitism_ratio
        self.diversity_threshold = diversity_threshold
        self.individuals: List[Individual] = []
        self.generation: int = 0
        self._history: List[dict] = []

    @property
    def size(self) -> int:
        return len(self.individuals)

    @property
    def is_empty(self) -> bool:
        return len(self.individuals) == 0

    @property
    def best(self) -> Optional[Individual]:
        """Return the individual with highest fitness."""
        if not self.individuals:
            return None
        return max(self.individuals, key=lambda ind: ind.fitness)

    @property
    def average_fitness(self) -> float:
        if not self.individuals:
            return 0.0
        return sum(ind.fitness for ind in self.individuals) / len(self.individuals)

    @property
    def best_fitness(self) -> float:
        if not self.individuals:
            return 0.0
        return max(ind.fitness for ind in self.individuals)

    def add(self, individual: Individual) -> None:
        """Add an individual to the population."""
        self.individuals.append(individual)

    def add_all(self, individuals: List[Individual]) -> None:
        """Add multiple individuals."""
        self.individuals.extend(individuals)

    def get_elite(self, count: Optional[int] = None) -> List[Individual]:
        """Get the top-performing individuals."""
        if count is None:
            count = max(1, int(self.size * self.elitism_ratio))
        sorted_pop = sorted(
            self.individuals, key=lambda ind: ind.fitness, reverse=True
        )
        return sorted_pop[:count]

    def get_sorted(self) -> List[Individual]:
        """Get all individuals sorted by fitness descending."""
        return sorted(
            self.individuals, key=lambda ind: ind.fitness, reverse=True
        )

    def truncate(self, keep: Optional[int] = None) -> None:
        """Keep only the top individuals up to max_size."""
        if keep is None:
            keep = self.max_size
        if len(self.individuals) > keep:
            self.individuals = self.get_sorted()[:keep]

    def replace_generation(
        self,
        new_individuals: List[Individual],
        keep_elite: bool = True,
    ) -> None:
        """Replace current population with new generation."""
        self._record_history()
        self.generation += 1

        if keep_elite:
            elite_count = max(1, int(self.max_size * self.elitism_ratio))
            elite = self.get_elite(elite_count)
            combined = elite + new_individuals
        else:
            combined = new_individuals

        # Truncate to max size, keeping best
        combined.sort(key=lambda ind: ind.fitness, reverse=True)
        self.individuals = combined[: self.max_size]

        # Update generation numbers for new individuals
        for ind in self.individuals:
            if ind.generation < self.generation:
                pass  # keep original generation

    def _record_history(self) -> None:
        """Record current generation statistics."""
        if self.individuals:
            self._history.append(
                {
                    "generation": self.generation,
                    "size": self.size,
                    "best_fitness": self.best_fitness,
                    "avg_fitness": self.average_fitness,
                }
            )

    @property
    def history(self) -> List[dict]:
        return list(self._history)

    def random_sample(self, n: int) -> List[Individual]:
        """Get a random sample of individuals."""
        n = min(n, self.size)
        return random.sample(self.individuals, n)

    def remove_duplicates(self) -> int:
        """Remove individuals with identical code. Returns count removed."""
        seen = set()
        unique = []
        for ind in self.individuals:
            if ind.code not in seen:
                seen.add(ind.code)
                unique.append(ind)
        removed = len(self.individuals) - len(unique)
        self.individuals = unique
        return removed

    def clear(self) -> None:
        """Remove all individuals."""
        self.individuals.clear()

    def statistics(self) -> dict:
        """Return population statistics."""
        if not self.individuals:
            return {
                "size": 0,
                "generation": self.generation,
                "best_fitness": 0.0,
                "avg_fitness": 0.0,
                "min_fitness": 0.0,
                "valid_count": 0,
            }

        fitnesses = [ind.fitness for ind in self.individuals]
        return {
            "size": self.size,
            "generation": self.generation,
            "best_fitness": max(fitnesses),
            "avg_fitness": sum(fitnesses) / len(fitnesses),
            "min_fitness": min(fitnesses),
            "valid_count": sum(1 for ind in self.individuals if ind.is_valid),
        }
