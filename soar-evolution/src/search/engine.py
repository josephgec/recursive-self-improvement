"""Main evolutionary search engine."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.arc.grid import ARCTask
from src.arc.evaluator import ProgramEvaluator
from src.operators.initializer import LLMInitializer
from src.operators.mutator import LLMMutator
from src.operators.crossover import LLMCrossover
from src.population.individual import Individual
from src.population.population import Population
from src.population.fitness import FitnessComputer
from src.population.selection import TournamentSelection
from src.population.archive import EliteArchive
from src.search.early_stopping import EarlyStopping
from src.search.scheduler import BudgetScheduler
from src.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class SearchConfig:
    """Configuration for evolutionary search."""

    population_size: int = 50
    max_generations: int = 100
    mutation_rate: float = 0.7
    crossover_rate: float = 0.3
    tournament_size: int = 5
    elitism_ratio: float = 0.1
    stagnation_limit: int = 15
    target_fitness: float = 1.0


@dataclass
class SearchResult:
    """Result of an evolutionary search run."""

    best_individual: Optional[Individual] = None
    best_fitness: float = 0.0
    generations_run: int = 0
    total_evaluations: int = 0
    solved: bool = False
    history: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    stop_reason: str = ""

    def summary(self) -> str:
        return (
            f"SearchResult(solved={self.solved}, fitness={self.best_fitness:.4f}, "
            f"generations={self.generations_run}, evals={self.total_evaluations}, "
            f"time={self.elapsed_seconds:.1f}s, reason={self.stop_reason})"
        )


class EvolutionarySearchEngine:
    """Main evolutionary search loop for ARC task solving."""

    def __init__(
        self,
        config: Optional[SearchConfig] = None,
        initializer: Optional[LLMInitializer] = None,
        mutator: Optional[LLMMutator] = None,
        crossover: Optional[LLMCrossover] = None,
        fitness_computer: Optional[FitnessComputer] = None,
        llm_call: Optional[Callable[[str], str]] = None,
    ):
        self.config = config or SearchConfig()
        self.fitness_computer = fitness_computer or FitnessComputer()
        self.initializer = initializer or LLMInitializer(llm_call=llm_call)
        self.mutator = mutator or LLMMutator(llm_call=llm_call)
        self.crossover = crossover or LLMCrossover(llm_call=llm_call)
        self.selection = TournamentSelection(self.config.tournament_size)
        self.archive = EliteArchive()
        self.early_stopping = EarlyStopping(
            patience=self.config.stagnation_limit
        )
        self.scheduler = BudgetScheduler(
            max_generations=self.config.max_generations
        )
        self._total_evaluations = 0

    def search(self, task: ARCTask) -> SearchResult:
        """Run evolutionary search for a given ARC task."""
        start_time = time.time()
        result = SearchResult()

        logger.info(f"Starting search for task {task.task_id}")

        # Initialize population
        population = Population(
            max_size=self.config.population_size,
            elitism_ratio=self.config.elitism_ratio,
        )

        initial_individuals = self.initializer.generate(
            task, count=self.config.population_size
        )
        self.fitness_computer.evaluate_population(initial_individuals, task)
        self._total_evaluations += len(initial_individuals)
        population.add_all(initial_individuals)

        # Archive initial best
        if population.best:
            self.archive.try_add(population.best)

        # Main evolution loop
        for gen in range(self.config.max_generations):
            gen_start = time.time()

            # Check termination conditions
            if population.best and population.best.fitness >= self.config.target_fitness:
                result.solved = True
                result.stop_reason = "target_fitness_reached"
                break

            if self.early_stopping.should_stop(population.best_fitness):
                result.stop_reason = "stagnation"
                break

            if not self.scheduler.has_budget(gen):
                result.stop_reason = "budget_exhausted"
                break

            # Generate offspring
            offspring = self._generate_offspring(population, task)
            self.fitness_computer.evaluate_population(offspring, task)
            self._total_evaluations += len(offspring)

            # Update archive
            for ind in offspring:
                self.archive.try_add(ind)

            # Replace generation
            population.replace_generation(offspring)

            # Record history
            gen_time = time.time() - gen_start
            gen_stats = {
                "generation": gen,
                "best_fitness": population.best_fitness,
                "avg_fitness": population.average_fitness,
                "pop_size": population.size,
                "gen_time": gen_time,
            }
            result.history.append(gen_stats)

            logger.debug(
                f"Gen {gen}: best={population.best_fitness:.4f}, "
                f"avg={population.average_fitness:.4f}"
            )

        # Final result
        result.best_individual = population.best or (
            self.archive.best if self.archive.best else None
        )
        result.best_fitness = (
            result.best_individual.fitness if result.best_individual else 0.0
        )
        result.generations_run = len(result.history)
        result.total_evaluations = self._total_evaluations
        result.elapsed_seconds = time.time() - start_time

        if not result.stop_reason:
            result.stop_reason = "max_generations"

        logger.info(result.summary())
        return result

    def _generate_offspring(
        self,
        population: Population,
        task: ARCTask,
    ) -> List[Individual]:
        """Generate offspring via mutation and crossover."""
        offspring = []
        target_count = self.config.population_size

        while len(offspring) < target_count:
            r = random.random()

            if r < self.config.mutation_rate:
                # Mutation
                parents = self.selection.select(population.individuals, 1)
                if parents:
                    child = self.mutator.mutate(parents[0], task)
                    offspring.append(child)

            else:
                # Crossover
                pair = self.selection.select_pair(population.individuals)
                if pair:
                    child = self.crossover.crossover(pair[0], pair[1], task)
                    offspring.append(child)

        return offspring[:target_count]
