"""Selection strategies for evolutionary search."""

from __future__ import annotations

import random
from typing import List, Optional

from src.population.individual import Individual
from src.utils.code_similarity import code_similarity


class TournamentSelection:
    """Tournament selection for choosing parents."""

    def __init__(self, tournament_size: int = 5):
        self.tournament_size = tournament_size

    def select(self, population: List[Individual], n: int = 1) -> List[Individual]:
        """Select n individuals via tournament selection."""
        if not population:
            return []

        selected = []
        for _ in range(n):
            tournament_size = min(self.tournament_size, len(population))
            tournament = random.sample(population, tournament_size)
            winner = max(tournament, key=lambda ind: ind.fitness)
            selected.append(winner)

        return selected

    def select_one(self, population: List[Individual]) -> Optional[Individual]:
        """Select a single individual."""
        result = self.select(population, 1)
        return result[0] if result else None

    def select_pair(
        self, population: List[Individual]
    ) -> Optional[tuple]:
        """Select two distinct individuals for crossover."""
        if len(population) < 2:
            return None

        first = self.select_one(population)
        # Try to select a different individual
        remaining = [ind for ind in population if ind.individual_id != first.individual_id]
        if not remaining:
            remaining = population
        second = self.select_one(remaining)
        return (first, second)


class DiversityAwareSelection:
    """Selection that balances fitness with diversity."""

    def __init__(
        self,
        tournament_size: int = 5,
        diversity_weight: float = 0.3,
    ):
        self.tournament_size = tournament_size
        self.diversity_weight = diversity_weight

    def _diversity_score(
        self,
        candidate: Individual,
        already_selected: List[Individual],
    ) -> float:
        """Compute how different a candidate is from already selected."""
        if not already_selected:
            return 1.0

        similarities = []
        for selected in already_selected:
            sim = code_similarity(candidate.code, selected.code)
            similarities.append(sim)

        avg_similarity = sum(similarities) / len(similarities)
        return 1.0 - avg_similarity

    def select(
        self,
        population: List[Individual],
        n: int = 1,
    ) -> List[Individual]:
        """Select n individuals balancing fitness and diversity."""
        if not population:
            return []

        selected: List[Individual] = []
        for _ in range(n):
            tournament_size = min(self.tournament_size, len(population))
            tournament = random.sample(population, tournament_size)

            # Score each tournament member
            best_score = -1.0
            best_ind = tournament[0]

            for ind in tournament:
                fitness_component = ind.fitness
                diversity_component = self._diversity_score(ind, selected)
                combined = (
                    (1 - self.diversity_weight) * fitness_component
                    + self.diversity_weight * diversity_component
                )
                if combined > best_score:
                    best_score = combined
                    best_ind = ind

            selected.append(best_ind)

        return selected
