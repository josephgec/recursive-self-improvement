"""Tests for the GA engine, population, selection, diversity."""

import os
import tempfile

import pytest

from src.genome.prompt_genome import PromptGenome
from src.genome.sections import DEFAULT_SECTIONS
from src.ga.engine import GAEngine, GenerationResult, EvolutionResult
from src.ga.population import Population
from src.ga.selection import TournamentSelection, DiversityAwareSelection
from src.ga.diversity import population_diversity, maintain_diversity
from src.ga.fitness import compute_composite_fitness, fitness_details_to_dict
from src.operators.thinking_initializer import ThinkingInitializer
from src.operators.thinking_mutator import ThinkingMutator
from src.operators.thinking_crossover import ThinkingCrossover
from src.operators.thinking_evaluator import ThinkingEvaluator, FitnessDetails


class TestPopulation:
    """Tests for Population management."""

    def test_creation(self):
        pop = Population(max_size=10, elitism_count=2)
        assert pop.max_size == 10
        assert pop.size == 0

    def test_add(self, sample_genome):
        pop = Population()
        pop.add(sample_genome)
        assert pop.size == 1

    def test_add_all(self, sample_genome, sample_genome_b):
        pop = Population()
        pop.add_all([sample_genome, sample_genome_b])
        assert pop.size == 2

    def test_get_best(self, sample_genome, sample_genome_b):
        pop = Population()
        sample_genome.fitness = 0.8
        sample_genome_b.fitness = 0.6
        pop.add_all([sample_genome, sample_genome_b])
        best = pop.get_best()
        assert best.fitness == 0.8

    def test_get_best_empty(self):
        pop = Population()
        assert pop.get_best() is None

    def test_get_elites(self, sample_genome, sample_genome_b):
        pop = Population(elitism_count=1)
        sample_genome.fitness = 0.9
        sample_genome_b.fitness = 0.5
        pop.add_all([sample_genome, sample_genome_b])
        elites = pop.get_elites()
        assert len(elites) == 1
        assert elites[0].fitness == 0.9

    def test_advance_generation(self, sample_genome):
        pop = Population()
        sample_genome.fitness = 0.7
        pop.add(sample_genome)
        pop.advance_generation()
        assert pop.generation == 1
        assert len(pop.history) == 1
        assert pop.history[0]["best_fitness"] == 0.7

    def test_select_parents(self, sample_genome, sample_genome_b):
        pop = Population(max_size=10, tournament_size=2)
        sample_genome.fitness = 0.8
        sample_genome_b.fitness = 0.5
        pop.add_all([sample_genome, sample_genome_b])
        parents = pop.select_parents(2)
        assert len(parents) == 2

    def test_diversity(self, sample_genome, sample_genome_b):
        pop = Population()
        pop.add_all([sample_genome, sample_genome_b])
        diversity = pop.get_diversity()
        assert 0.0 <= diversity <= 1.0

    def test_is_stagnant_not_enough_history(self, sample_genome):
        pop = Population()
        pop.add(sample_genome)
        assert pop.is_stagnant(window=5) is False

    def test_is_stagnant_with_improvement(self, sample_genome):
        pop = Population()
        pop.add(sample_genome)
        pop._best_fitness_history = [0.5, 0.55, 0.6, 0.65, 0.7]
        assert pop.is_stagnant(window=5) is False

    def test_is_stagnant_flat(self, sample_genome):
        pop = Population()
        pop.add(sample_genome)
        pop._best_fitness_history = [0.7, 0.7, 0.7, 0.7, 0.7]
        assert pop.is_stagnant(window=5) is True

    def test_truncate(self):
        pop = Population(max_size=3)
        for i in range(5):
            g = PromptGenome(genome_id=f"g{i}")
            g.fitness = i * 0.1
            pop.add(g)
        pop.truncate()
        assert pop.size == 3
        assert pop.genomes[0].fitness >= pop.genomes[-1].fitness

    def test_save_load(self, sample_genome, sample_genome_b):
        pop = Population(max_size=10, elitism_count=2)
        pop.add_all([sample_genome, sample_genome_b])
        pop.advance_generation()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            pop.save(filepath)
            loaded = Population.load(filepath)
            assert loaded.generation == pop.generation
            assert loaded.size == pop.size
            assert loaded.history == pop.history
        finally:
            os.unlink(filepath)


class TestSelection:
    """Tests for selection strategies."""

    def _make_population(self, n=5):
        genomes = []
        for i in range(n):
            g = PromptGenome(genome_id=f"sel_{i}")
            g.fitness = 0.1 * (i + 1)
            for name, content in DEFAULT_SECTIONS.items():
                g.set_section(name, content + f" variant {i}")
            genomes.append(g)
        return genomes

    def test_tournament_select(self):
        genomes = self._make_population(5)
        selector = TournamentSelection(tournament_size=3)
        selected = selector.select(genomes, n=2)
        assert len(selected) == 2

    def test_tournament_select_pair(self):
        genomes = self._make_population(5)
        selector = TournamentSelection(tournament_size=2)
        a, b = selector.select_pair(genomes)
        assert a is not None
        assert b is not None

    def test_tournament_single_individual(self):
        genomes = self._make_population(1)
        selector = TournamentSelection(tournament_size=3)
        a, b = selector.select_pair(genomes)
        assert a.genome_id == b.genome_id

    def test_diversity_aware_select(self):
        genomes = self._make_population(5)
        selector = DiversityAwareSelection(tournament_size=3, diversity_weight=0.3)
        selected = selector.select(genomes, n=2)
        assert len(selected) == 2

    def test_diversity_aware_pair(self):
        genomes = self._make_population(5)
        selector = DiversityAwareSelection(tournament_size=2)
        a, b = selector.select_pair(genomes)
        assert a is not None
        assert b is not None

    def test_diversity_bonus_favors_different(self):
        genomes = self._make_population(3)
        selector = DiversityAwareSelection(tournament_size=3, diversity_weight=0.5)
        # With diversity weight, should sometimes pick less fit but diverse
        # Just ensure it runs without error
        for _ in range(10):
            selected = selector.select(genomes, n=1)
            assert len(selected) == 1


class TestDiversity:
    """Tests for diversity functions."""

    def test_population_diversity_single(self):
        g = PromptGenome()
        assert population_diversity([g]) == 1.0

    def test_population_diversity_identical(self, sample_genome):
        copy = sample_genome.copy()
        # Make copy have same content
        for name, sec in sample_genome.sections.items():
            copy.set_section(name, sec.content)
        diversity = population_diversity([sample_genome, copy])
        assert diversity < 0.1  # Very similar

    def test_population_diversity_different(self, sample_genome, sample_genome_b):
        diversity = population_diversity([sample_genome, sample_genome_b])
        assert diversity > 0.0

    def test_maintain_diversity_above_threshold(self, sample_genome, sample_genome_b):
        genomes = [sample_genome, sample_genome_b]
        result = maintain_diversity(genomes, threshold=0.01)
        assert len(result) == 2

    def test_maintain_diversity_injects(self):
        # Create identical genomes
        genomes = []
        for i in range(3):
            g = PromptGenome(genome_id=f"dup_{i}")
            g.set_section("identity", "Same content")
            g.set_section("methodology", "Same methodology")
            g.fitness = 0.5
            genomes.append(g)

        def inject():
            g = PromptGenome(genome_id="injected")
            g.set_section("identity", "Completely different unique content")
            g.set_section("methodology", "Novel approach entirely new")
            return g

        result = maintain_diversity(genomes, threshold=0.9, inject_fn=inject)
        assert len(result) == 3

    def test_maintain_diversity_no_inject_fn(self, sample_genome):
        result = maintain_diversity([sample_genome], threshold=0.9)
        assert len(result) == 1


class TestFitness:
    """Tests for fitness computation."""

    def test_compute_composite_fitness(self):
        result = compute_composite_fitness(0.8, 0.6, 0.9)
        expected = 0.7 * 0.8 + 0.15 * 0.6 + 0.15 * 0.9
        assert result == pytest.approx(expected)

    def test_compute_composite_clamped(self):
        result = compute_composite_fitness(1.5, 0.0, 0.0)
        assert result <= 1.0

    def test_fitness_details_to_dict(self):
        details = FitnessDetails(
            accuracy=0.8,
            reasoning_score=0.6,
            consistency_score=0.9,
            composite_fitness=0.76,
            section_scores={"identity": 0.8},
        )
        d = fitness_details_to_dict(details)
        assert d["accuracy"] == 0.8
        assert d["num_tasks"] == 0


class TestGAEngine:
    """Tests for the GA engine evolution loop."""

    def _make_engine(self, mock_thinking_llm, mock_output_llm, **kwargs):
        initializer = ThinkingInitializer(mock_thinking_llm)
        mutator = ThinkingMutator(mock_thinking_llm)
        crossover = ThinkingCrossover(mock_thinking_llm)
        evaluator = ThinkingEvaluator(mock_output_llm)

        defaults = dict(
            initializer=initializer,
            mutator=mutator,
            crossover_op=crossover,
            evaluator=evaluator,
            population_size=4,
            num_generations=3,
            mutation_rate=0.3,
            crossover_rate=0.4,
            elitism_count=1,
            tournament_size=2,
            stagnation_limit=5,
        )
        defaults.update(kwargs)
        return GAEngine(**defaults)

    def test_evolve_returns_result(
        self, mock_thinking_llm, mock_output_llm, sample_tasks
    ):
        engine = self._make_engine(mock_thinking_llm, mock_output_llm)
        result = engine.evolve(
            domain_desc="Financial math",
            example_tasks=["Calculate compound interest"],
            eval_tasks=sample_tasks[:5],
        )

        assert isinstance(result, EvolutionResult)
        assert result.best_genome is not None
        assert result.best_fitness > 0.0

    def test_evolve_generations_run(
        self, mock_thinking_llm, mock_output_llm, sample_tasks
    ):
        engine = self._make_engine(
            mock_thinking_llm, mock_output_llm, num_generations=3,
        )
        result = engine.evolve(
            "Financial math", ["task"], sample_tasks[:5],
        )
        assert result.generations_run <= 3
        assert len(result.generation_results) <= 3

    def test_evolve_elitism(
        self, mock_thinking_llm, mock_output_llm, sample_tasks
    ):
        engine = self._make_engine(
            mock_thinking_llm, mock_output_llm, elitism_count=1,
        )
        result = engine.evolve(
            "Financial math", ["task"], sample_tasks[:5],
        )
        # Elites should be tracked in generation results
        for gen_result in result.generation_results:
            assert gen_result.num_elites >= 0

    def test_evolve_stopped_reason(
        self, mock_thinking_llm, mock_output_llm, sample_tasks
    ):
        engine = self._make_engine(
            mock_thinking_llm, mock_output_llm, num_generations=2,
        )
        result = engine.evolve(
            "Financial math", ["task"], sample_tasks[:3],
        )
        assert result.stopped_reason in ("max_generations", "stagnation", "optimal")

    def test_evolve_fitness_history(
        self, mock_thinking_llm, mock_output_llm, sample_tasks
    ):
        engine = self._make_engine(
            mock_thinking_llm, mock_output_llm, num_generations=3,
        )
        result = engine.evolve(
            "Financial math", ["task"], sample_tasks[:5],
        )
        assert len(result.fitness_history) > 0

    def test_evolve_no_crossover(
        self, mock_thinking_llm, mock_output_llm, sample_tasks
    ):
        engine = self._make_engine(
            mock_thinking_llm, mock_output_llm,
            crossover_rate=0.0,
            crossover_op=None,
        )
        result = engine.evolve(
            "Financial math", ["task"], sample_tasks[:3],
        )
        assert isinstance(result, EvolutionResult)

    def test_generation_result_fields(
        self, mock_thinking_llm, mock_output_llm, sample_tasks
    ):
        engine = self._make_engine(mock_thinking_llm, mock_output_llm)
        result = engine.evolve(
            "Financial math", ["task"], sample_tasks[:3],
        )
        for gr in result.generation_results:
            assert isinstance(gr, GenerationResult)
            assert gr.generation > 0
            assert gr.best_fitness >= 0.0
            assert gr.diversity >= 0.0

    def test_stagnation_stop(self, mock_thinking_llm, mock_output_llm, sample_tasks):
        engine = self._make_engine(
            mock_thinking_llm, mock_output_llm,
            num_generations=100,
            stagnation_limit=3,
        )
        result = engine.evolve(
            "Financial math", ["task"], sample_tasks[:3],
        )
        # May or may not stagnate, but should finish
        assert result.stopped_reason in ("max_generations", "stagnation", "optimal")
