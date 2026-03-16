"""Selection strategies for the genetic algorithm."""

from __future__ import annotations

import random
from typing import List, Tuple

from src.genome.prompt_genome import PromptGenome
from src.genome.similarity import genome_similarity


class TournamentSelection:
    """Tournament selection: pick k random individuals, select the fittest."""

    def __init__(self, tournament_size: int = 3):
        self.tournament_size = tournament_size

    def select(
        self, population: List[PromptGenome], n: int = 2
    ) -> List[PromptGenome]:
        """Select n individuals via tournament selection."""
        selected = []
        for _ in range(n):
            candidates = random.sample(
                population, min(self.tournament_size, len(population))
            )
            winner = max(candidates, key=lambda g: g.fitness)
            selected.append(winner)
        return selected

    def select_pair(
        self, population: List[PromptGenome]
    ) -> Tuple[PromptGenome, PromptGenome]:
        """Select two distinct parents for crossover."""
        if len(population) < 2:
            return population[0], population[0]
        parents = self.select(population, n=2)
        # Try to ensure they're different
        attempts = 0
        while (
            parents[0].genome_id == parents[1].genome_id
            and attempts < 5
            and len(population) > 1
        ):
            parents = self.select(population, n=2)
            attempts += 1
        return parents[0], parents[1]


class DiversityAwareSelection:
    """Tournament selection with diversity bonus.

    Adds a diversity bonus to fitness during tournament to maintain variety.
    """

    def __init__(self, tournament_size: int = 3, diversity_weight: float = 0.2):
        self.tournament_size = tournament_size
        self.diversity_weight = diversity_weight

    def select(
        self, population: List[PromptGenome], n: int = 2
    ) -> List[PromptGenome]:
        """Select n individuals with diversity-aware tournament."""
        selected = []
        for _ in range(n):
            candidates = random.sample(
                population, min(self.tournament_size, len(population))
            )

            # Score each candidate: fitness + diversity bonus
            scored = []
            for cand in candidates:
                diversity_bonus = self._diversity_bonus(cand, selected, population)
                effective_fitness = (
                    cand.fitness * (1 - self.diversity_weight)
                    + diversity_bonus * self.diversity_weight
                )
                scored.append((cand, effective_fitness))

            winner = max(scored, key=lambda x: x[1])[0]
            selected.append(winner)

        return selected

    def _diversity_bonus(
        self,
        candidate: PromptGenome,
        already_selected: List[PromptGenome],
        population: List[PromptGenome],
    ) -> float:
        """Compute diversity bonus for a candidate.

        Higher bonus for candidates dissimilar to already-selected and population average.
        """
        if not already_selected:
            return 0.5

        total_dissim = 0.0
        for sel in already_selected:
            sim = genome_similarity(candidate, sel)
            total_dissim += 1.0 - sim

        return total_dissim / len(already_selected)

    def select_pair(
        self, population: List[PromptGenome]
    ) -> Tuple[PromptGenome, PromptGenome]:
        """Select two parents with diversity awareness."""
        if len(population) < 2:
            return population[0], population[0]
        parents = self.select(population, n=2)
        return parents[0], parents[1]
