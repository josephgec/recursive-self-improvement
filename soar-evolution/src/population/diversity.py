"""Diversity metrics for population management."""

from __future__ import annotations

from typing import List

from src.population.individual import Individual
from src.utils.code_similarity import code_similarity


class DiversityMetrics:
    """Compute diversity metrics for a population."""

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold

    def pairwise_diversity(self, individuals: List[Individual]) -> float:
        """Average pairwise code distance across the population."""
        if len(individuals) < 2:
            return 1.0

        total_distance = 0.0
        count = 0
        for i in range(len(individuals)):
            for j in range(i + 1, len(individuals)):
                sim = code_similarity(
                    individuals[i].code, individuals[j].code
                )
                total_distance += 1.0 - sim
                count += 1

        return total_distance / count if count > 0 else 1.0

    def unique_ratio(self, individuals: List[Individual]) -> float:
        """Ratio of unique programs in the population."""
        if not individuals:
            return 0.0
        unique_codes = set(ind.code for ind in individuals)
        return len(unique_codes) / len(individuals)

    def fitness_spread(self, individuals: List[Individual]) -> float:
        """Standard deviation of fitness values."""
        if len(individuals) < 2:
            return 0.0

        fitnesses = [ind.fitness for ind in individuals]
        mean = sum(fitnesses) / len(fitnesses)
        variance = sum((f - mean) ** 2 for f in fitnesses) / len(fitnesses)
        return variance ** 0.5

    def is_diverse_enough(self, individuals: List[Individual]) -> bool:
        """Check if population has sufficient diversity."""
        return self.pairwise_diversity(individuals) >= self.threshold

    def diversity_report(self, individuals: List[Individual]) -> dict:
        """Generate a full diversity report."""
        return {
            "pairwise_diversity": self.pairwise_diversity(individuals),
            "unique_ratio": self.unique_ratio(individuals),
            "fitness_spread": self.fitness_spread(individuals),
            "is_diverse": self.is_diverse_enough(individuals),
            "population_size": len(individuals),
        }
