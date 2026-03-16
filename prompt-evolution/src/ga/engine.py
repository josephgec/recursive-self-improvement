"""Genetic algorithm engine for prompt evolution."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.genome.prompt_genome import PromptGenome
from src.ga.population import Population
from src.ga.diversity import maintain_diversity
from src.ga.fitness import compute_composite_fitness
from src.operators.thinking_evaluator import FitnessDetails


@dataclass
class GenerationResult:
    """Results from a single generation."""

    generation: int
    best_fitness: float
    avg_fitness: float
    diversity: float
    best_genome_id: str
    num_mutations: int = 0
    num_crossovers: int = 0
    num_elites: int = 0


@dataclass
class EvolutionResult:
    """Results from a complete evolution run."""

    best_genome: Optional[PromptGenome] = None
    best_fitness: float = 0.0
    generations_run: int = 0
    generation_results: List[GenerationResult] = field(default_factory=list)
    final_population: Optional[Population] = None
    stopped_reason: str = ""
    fitness_history: List[float] = field(default_factory=list)


class GAEngine:
    """Genetic algorithm engine that orchestrates prompt evolution.

    Manages the evolution loop with selection, mutation, crossover,
    elitism, and stopping conditions.
    """

    def __init__(
        self,
        initializer: Any,
        mutator: Any,
        crossover_op: Any,
        evaluator: Any,
        population_size: int = 20,
        num_generations: int = 10,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.5,
        elitism_count: int = 2,
        tournament_size: int = 3,
        stagnation_limit: int = 5,
        diversity_threshold: float = 0.3,
    ):
        self.initializer = initializer
        self.mutator = mutator
        self.crossover_op = crossover_op
        self.evaluator = evaluator
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.tournament_size = tournament_size
        self.stagnation_limit = stagnation_limit
        self.diversity_threshold = diversity_threshold

    def evolve(
        self,
        domain_desc: str,
        example_tasks: List[str],
        eval_tasks: List[Dict[str, Any]],
    ) -> EvolutionResult:
        """Run the complete evolution loop.

        Args:
            domain_desc: Description of the target domain
            example_tasks: Example task descriptions for initialization
            eval_tasks: Tasks with expected answers for evaluation

        Returns:
            EvolutionResult with the best genome and evolution history.
        """
        result = EvolutionResult()

        # Initialize population
        population = Population(
            max_size=self.population_size,
            elitism_count=self.elitism_count,
            tournament_size=self.tournament_size,
        )

        initial_genomes = self.initializer.initialize(
            self.population_size, domain_desc, example_tasks
        )
        population.add_all(initial_genomes)

        # Evaluate initial population
        self._evaluate_population(population, eval_tasks)

        for gen in range(self.num_generations):
            gen_result = self._run_generation(population, eval_tasks, gen)
            result.generation_results.append(gen_result)
            result.fitness_history.append(gen_result.best_fitness)

            # Check stopping conditions
            if population.is_stagnant(self.stagnation_limit):
                result.stopped_reason = "stagnation"
                break

            best = population.get_best()
            if best and best.fitness >= 0.99:
                result.stopped_reason = "optimal"
                break
        else:
            result.stopped_reason = "max_generations"

        # Final result
        best = population.get_best()
        if best:
            result.best_genome = best
            result.best_fitness = best.fitness

        result.generations_run = len(result.generation_results)
        result.final_population = population

        return result

    def _run_generation(
        self,
        population: Population,
        eval_tasks: List[Dict[str, Any]],
        gen_number: int,
    ) -> GenerationResult:
        """Run one generation of evolution."""
        # Record and advance
        population.advance_generation()

        # Preserve elites
        elites = population.get_elites()

        # Build next generation
        next_gen: List[PromptGenome] = []

        # Add elites directly
        for elite in elites:
            elite_copy = elite.copy()
            elite_copy.fitness = elite.fitness
            elite_copy.generation = gen_number + 1
            next_gen.append(elite_copy)

        num_mutations = 0
        num_crossovers = 0

        # Fill the rest of the population
        while len(next_gen) < self.population_size:
            r = random.random()

            if r < self.crossover_rate and self.crossover_op is not None:
                # Crossover
                parents = population.select_parents(2)
                offspring = self.crossover_op.crossover(
                    parents[0],
                    parents[1],
                    parents[0].fitness,
                    parents[1].fitness,
                )
                offspring.generation = gen_number + 1

                # Maybe also mutate the offspring
                if random.random() < self.mutation_rate:
                    offspring = self.mutator.mutate(offspring)
                    offspring.generation = gen_number + 1
                    num_mutations += 1

                next_gen.append(offspring)
                num_crossovers += 1

            elif r < self.crossover_rate + self.mutation_rate:
                # Mutation only
                parents = population.select_parents(1)
                mutant = self.mutator.mutate(parents[0])
                mutant.generation = gen_number + 1
                next_gen.append(mutant)
                num_mutations += 1

            else:
                # Reproduction (copy a selected individual)
                parents = population.select_parents(1)
                child = parents[0].copy()
                child.generation = gen_number + 1
                child.fitness = 0.0
                next_gen.append(child)

        # Replace population
        population.genomes = next_gen[: self.population_size]

        # Maintain diversity
        def _inject():
            genomes = self.initializer.initialize(1, "", [])
            return genomes[0]

        maintain_diversity(
            population.genomes,
            threshold=self.diversity_threshold,
            inject_fn=_inject,
        )

        # Evaluate new population
        self._evaluate_population(population, eval_tasks)

        best = population.get_best()
        return GenerationResult(
            generation=gen_number + 1,
            best_fitness=best.fitness if best else 0.0,
            avg_fitness=sum(g.fitness for g in population.genomes) / max(len(population.genomes), 1),
            diversity=population.get_diversity(),
            best_genome_id=best.genome_id if best else "",
            num_mutations=num_mutations,
            num_crossovers=num_crossovers,
            num_elites=len(elites),
        )

    def _evaluate_population(
        self,
        population: Population,
        eval_tasks: List[Dict[str, Any]],
    ):
        """Evaluate all genomes in the population."""
        for genome in population.genomes:
            if genome.fitness == 0.0:  # Only evaluate unevaluated genomes
                details = self.evaluator.evaluate(genome, eval_tasks)
                genome.fitness = details.composite_fitness
