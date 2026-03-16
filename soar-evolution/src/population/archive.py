"""Elite archive for storing best-ever solutions."""

from __future__ import annotations

from typing import Dict, List, Optional

from src.population.individual import Individual
from src.utils.code_similarity import code_similarity


class EliteArchive:
    """Maintains an archive of elite solutions found during search."""

    def __init__(self, max_size: int = 20, novelty_threshold: float = 0.2):
        self.max_size = max_size
        self.novelty_threshold = novelty_threshold
        self._archive: List[Individual] = []
        self._best_ever: Optional[Individual] = None

    @property
    def size(self) -> int:
        return len(self._archive)

    @property
    def best(self) -> Optional[Individual]:
        return self._best_ever

    @property
    def individuals(self) -> List[Individual]:
        return list(self._archive)

    def is_novel(self, individual: Individual) -> bool:
        """Check if individual is sufficiently different from archive."""
        if not self._archive:
            return True

        for archived in self._archive:
            sim = code_similarity(individual.code, archived.code)
            if sim > (1.0 - self.novelty_threshold):
                return False

        return True

    def try_add(self, individual: Individual) -> bool:
        """Try to add an individual to the archive. Returns True if added."""
        # Update best ever
        if self._best_ever is None or individual.fitness > self._best_ever.fitness:
            self._best_ever = individual

        # Check novelty
        if not self.is_novel(individual):
            # If not novel, only add if it's better than the similar one
            for i, archived in enumerate(self._archive):
                sim = code_similarity(individual.code, archived.code)
                if sim > (1.0 - self.novelty_threshold):
                    if individual.fitness > archived.fitness:
                        self._archive[i] = individual
                        return True
                    return False

        # Add to archive
        if len(self._archive) < self.max_size:
            self._archive.append(individual)
            return True

        # Replace worst if new individual is better
        worst_idx = min(
            range(len(self._archive)),
            key=lambda i: self._archive[i].fitness,
        )
        if individual.fitness > self._archive[worst_idx].fitness:
            self._archive[worst_idx] = individual
            return True

        return False

    def get_top(self, n: int = 5) -> List[Individual]:
        """Get top n individuals from the archive."""
        sorted_archive = sorted(
            self._archive, key=lambda ind: ind.fitness, reverse=True
        )
        return sorted_archive[:n]

    def get_diverse_sample(self, n: int) -> List[Individual]:
        """Get a diverse sample from the archive."""
        if len(self._archive) <= n:
            return list(self._archive)

        # Greedy diversity selection
        selected = [self._archive[0]]  # Start with best
        remaining = list(self._archive[1:])

        while len(selected) < n and remaining:
            best_candidate = None
            best_min_dist = -1.0

            for candidate in remaining:
                min_dist = min(
                    1.0 - code_similarity(candidate.code, s.code)
                    for s in selected
                )
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_candidate = candidate

            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)

        return selected

    def clear(self) -> None:
        """Clear the archive."""
        self._archive.clear()
        self._best_ever = None

    def summary(self) -> Dict:
        """Return archive summary."""
        if not self._archive:
            return {"size": 0, "best_fitness": 0.0}

        fitnesses = [ind.fitness for ind in self._archive]
        return {
            "size": self.size,
            "best_fitness": max(fitnesses),
            "avg_fitness": sum(fitnesses) / len(fitnesses),
            "min_fitness": min(fitnesses),
        }
