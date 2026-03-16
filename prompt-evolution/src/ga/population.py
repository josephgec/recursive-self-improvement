"""Population management for the genetic algorithm."""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional, Tuple

from src.genome.prompt_genome import PromptGenome
from src.genome.serializer import serialize_genome, deserialize_genome
from src.ga.diversity import population_diversity
from src.ga.selection import TournamentSelection


class Population:
    """Manages a population of prompt genomes across generations."""

    def __init__(
        self,
        max_size: int = 20,
        elitism_count: int = 2,
        tournament_size: int = 3,
    ):
        self.max_size = max_size
        self.elitism_count = elitism_count
        self.genomes: List[PromptGenome] = []
        self.generation: int = 0
        self.history: List[Dict] = []
        self.selector = TournamentSelection(tournament_size)
        self._best_fitness_history: List[float] = []

    def add(self, genome: PromptGenome):
        """Add a genome to the population."""
        self.genomes.append(genome)

    def add_all(self, genomes: List[PromptGenome]):
        """Add multiple genomes to the population."""
        self.genomes.extend(genomes)

    @property
    def size(self) -> int:
        return len(self.genomes)

    def advance_generation(self):
        """Record current generation stats and advance counter."""
        stats = {
            "generation": self.generation,
            "size": len(self.genomes),
            "best_fitness": self.get_best().fitness if self.genomes else 0.0,
            "avg_fitness": self._avg_fitness(),
            "diversity": self.get_diversity(),
        }
        self.history.append(stats)
        self._best_fitness_history.append(stats["best_fitness"])
        self.generation += 1

    def select_parents(self, n: int = 2) -> List[PromptGenome]:
        """Select parents using tournament selection."""
        return self.selector.select(self.genomes, n)

    def get_best(self) -> Optional[PromptGenome]:
        """Get the fittest genome in the population."""
        if not self.genomes:
            return None
        return max(self.genomes, key=lambda g: g.fitness)

    def get_elites(self) -> List[PromptGenome]:
        """Get the top-k elites by fitness."""
        sorted_pop = sorted(self.genomes, key=lambda g: g.fitness, reverse=True)
        return sorted_pop[: self.elitism_count]

    def get_diversity(self) -> float:
        """Compute current population diversity."""
        return population_diversity(self.genomes)

    def is_stagnant(self, window: int = 5) -> bool:
        """Check if fitness has stagnated over the last `window` generations."""
        if len(self._best_fitness_history) < window:
            return False

        recent = self._best_fitness_history[-window:]
        # Stagnant if best fitness hasn't improved
        return max(recent) - min(recent) < 0.01

    def truncate(self):
        """Trim population to max_size, keeping the fittest."""
        if len(self.genomes) > self.max_size:
            self.genomes.sort(key=lambda g: g.fitness, reverse=True)
            self.genomes = self.genomes[: self.max_size]

    def save(self, filepath: str):
        """Save population to a JSON file."""
        data = {
            "generation": self.generation,
            "max_size": self.max_size,
            "elitism_count": self.elitism_count,
            "history": self.history,
            "genomes": [json.loads(serialize_genome(g)) for g in self.genomes],
        }
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "Population":
        """Load population from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        pop = cls(
            max_size=data.get("max_size", 20),
            elitism_count=data.get("elitism_count", 2),
        )
        pop.generation = data.get("generation", 0)
        pop.history = data.get("history", [])

        for g_data in data.get("genomes", []):
            genome = deserialize_genome(json.dumps(g_data))
            pop.genomes.append(genome)

        return pop

    def _avg_fitness(self) -> float:
        """Average fitness across the population."""
        if not self.genomes:
            return 0.0
        return sum(g.fitness for g in self.genomes) / len(self.genomes)
