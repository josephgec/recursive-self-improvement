"""Tests for Population, Individual, Selection, Diversity, and Archive."""

import pytest
from src.population.individual import Individual
from src.population.population import Population
from src.population.selection import TournamentSelection, DiversityAwareSelection
from src.population.diversity import DiversityMetrics
from src.population.archive import EliteArchive


class TestIndividual:
    def test_create(self):
        ind = Individual(code="def transform(g): return g", generation=0)
        assert ind.code == "def transform(g): return g"
        assert ind.generation == 0
        assert ind.fitness == 0.0
        assert not ind.evaluated

    def test_is_valid(self):
        ind = Individual(code="code")
        assert ind.is_valid

        ind2 = Individual(code="code", compile_error="error")
        assert not ind2.is_valid

        ind3 = Individual(code="code", runtime_errors=["err"])
        assert not ind3.is_valid

    def test_code_length(self):
        ind = Individual(code="abc\ndef")
        assert ind.code_length == 7

    def test_line_count(self):
        ind = Individual(code="line1\nline2\nline3")
        assert ind.line_count == 3

    def test_copy(self):
        ind = Individual(code="test", generation=5, fitness=0.8)
        ind2 = ind.copy()
        assert ind2.code == ind.code
        assert ind2.parent_ids == [ind.individual_id]
        assert ind2.individual_id != ind.individual_id

    def test_summary(self):
        ind = Individual(code="c", fitness=0.5, train_accuracy=0.3)
        s = ind.summary()
        assert "0.500" in s
        assert "0.300" in s

    def test_repr(self):
        ind = Individual(code="c")
        assert "Individual(" in repr(ind)


class TestPopulation:
    def _make_pop(self, n=5):
        pop = Population(max_size=10)
        for i in range(n):
            ind = Individual(code=f"code_{i}", fitness=i * 0.2)
            pop.add(ind)
        return pop

    def test_create(self):
        pop = Population()
        assert pop.is_empty
        assert pop.size == 0

    def test_add(self):
        pop = Population()
        ind = Individual(code="test")
        pop.add(ind)
        assert pop.size == 1
        assert not pop.is_empty

    def test_add_all(self):
        pop = Population()
        inds = [Individual(code=f"c{i}") for i in range(5)]
        pop.add_all(inds)
        assert pop.size == 5

    def test_best(self):
        pop = self._make_pop(5)
        assert pop.best.fitness == 0.8

    def test_best_empty(self):
        pop = Population()
        assert pop.best is None

    def test_average_fitness(self):
        pop = self._make_pop(5)
        avg = pop.average_fitness
        assert abs(avg - 0.4) < 0.01

    def test_average_fitness_empty(self):
        pop = Population()
        assert pop.average_fitness == 0.0

    def test_best_fitness(self):
        pop = self._make_pop(5)
        assert pop.best_fitness == 0.8

    def test_best_fitness_empty(self):
        pop = Population()
        assert pop.best_fitness == 0.0

    def test_get_elite(self):
        pop = self._make_pop(10)
        elite = pop.get_elite(3)
        assert len(elite) == 3
        assert elite[0].fitness >= elite[1].fitness

    def test_get_sorted(self):
        pop = self._make_pop(5)
        sorted_pop = pop.get_sorted()
        for i in range(len(sorted_pop) - 1):
            assert sorted_pop[i].fitness >= sorted_pop[i + 1].fitness

    def test_truncate(self):
        pop = Population(max_size=3)
        for i in range(10):
            pop.add(Individual(code=f"c{i}", fitness=i * 0.1))
        pop.truncate()
        assert pop.size == 3

    def test_replace_generation(self):
        pop = self._make_pop(5)
        new_inds = [Individual(code=f"new_{i}", fitness=i * 0.3) for i in range(5)]
        pop.replace_generation(new_inds)
        assert pop.generation == 1

    def test_random_sample(self):
        pop = self._make_pop(10)
        sample = pop.random_sample(3)
        assert len(sample) == 3

    def test_random_sample_larger_than_pop(self):
        pop = self._make_pop(3)
        sample = pop.random_sample(10)
        assert len(sample) == 3

    def test_remove_duplicates(self):
        pop = Population()
        pop.add(Individual(code="same"))
        pop.add(Individual(code="same"))
        pop.add(Individual(code="different"))
        removed = pop.remove_duplicates()
        assert removed == 1
        assert pop.size == 2

    def test_clear(self):
        pop = self._make_pop(5)
        pop.clear()
        assert pop.is_empty

    def test_statistics(self):
        pop = self._make_pop(5)
        stats = pop.statistics()
        assert stats["size"] == 5
        assert stats["best_fitness"] == 0.8

    def test_statistics_empty(self):
        pop = Population()
        stats = pop.statistics()
        assert stats["size"] == 0

    def test_history(self):
        pop = self._make_pop(5)
        new_inds = [Individual(code=f"n{i}", fitness=0.5) for i in range(3)]
        pop.replace_generation(new_inds)
        assert len(pop.history) == 1


class TestTournamentSelection:
    def test_select(self):
        ts = TournamentSelection(tournament_size=3)
        inds = [Individual(code=f"c{i}", fitness=i * 0.1) for i in range(10)]
        selected = ts.select(inds, 5)
        assert len(selected) == 5

    def test_select_empty(self):
        ts = TournamentSelection()
        assert ts.select([], 5) == []

    def test_select_one(self):
        ts = TournamentSelection(tournament_size=3)
        inds = [Individual(code=f"c{i}", fitness=i * 0.1) for i in range(10)]
        one = ts.select_one(inds)
        assert one is not None

    def test_select_one_empty(self):
        ts = TournamentSelection()
        assert ts.select_one([]) is None

    def test_select_pair(self):
        ts = TournamentSelection(tournament_size=3)
        inds = [Individual(code=f"c{i}", fitness=i * 0.1) for i in range(10)]
        pair = ts.select_pair(inds)
        assert pair is not None
        assert len(pair) == 2

    def test_select_pair_too_small(self):
        ts = TournamentSelection()
        assert ts.select_pair([Individual(code="c")]) is None


class TestDiversityAwareSelection:
    def test_select(self):
        das = DiversityAwareSelection(tournament_size=3, diversity_weight=0.3)
        inds = [Individual(code=f"code_variant_{i}\ndef f(): pass", fitness=i * 0.1) for i in range(10)]
        selected = das.select(inds, 5)
        assert len(selected) == 5

    def test_select_empty(self):
        das = DiversityAwareSelection()
        assert das.select([], 5) == []


class TestDiversityMetrics:
    def test_pairwise_diversity(self):
        dm = DiversityMetrics()
        inds = [
            Individual(code="def a(): pass"),
            Individual(code="def b(): return 1"),
            Individual(code="x = 42; y = 99"),
        ]
        div = dm.pairwise_diversity(inds)
        assert 0.0 <= div <= 1.0

    def test_single_individual(self):
        dm = DiversityMetrics()
        assert dm.pairwise_diversity([Individual(code="c")]) == 1.0

    def test_unique_ratio(self):
        dm = DiversityMetrics()
        inds = [
            Individual(code="same"),
            Individual(code="same"),
            Individual(code="diff"),
        ]
        assert abs(dm.unique_ratio(inds) - 2 / 3) < 0.01

    def test_unique_ratio_empty(self):
        dm = DiversityMetrics()
        assert dm.unique_ratio([]) == 0.0

    def test_fitness_spread(self):
        dm = DiversityMetrics()
        inds = [Individual(code="c", fitness=f) for f in [0.1, 0.5, 0.9]]
        spread = dm.fitness_spread(inds)
        assert spread > 0

    def test_fitness_spread_single(self):
        dm = DiversityMetrics()
        assert dm.fitness_spread([Individual(code="c")]) == 0.0

    def test_diversity_report(self):
        dm = DiversityMetrics()
        inds = [Individual(code=f"c{i}", fitness=i * 0.1) for i in range(5)]
        report = dm.diversity_report(inds)
        assert "pairwise_diversity" in report
        assert "unique_ratio" in report
        assert "is_diverse" in report

    def test_is_diverse_enough(self):
        dm = DiversityMetrics(threshold=0.0)
        inds = [Individual(code="same"), Individual(code="same")]
        # With threshold=0.0, everything is diverse enough
        assert dm.is_diverse_enough(inds)


class TestEliteArchive:
    def test_add(self):
        archive = EliteArchive(max_size=5)
        ind = Individual(code="test", fitness=0.5)
        assert archive.try_add(ind)
        assert archive.size == 1
        assert archive.best.fitness == 0.5

    def test_novelty_check(self):
        archive = EliteArchive(max_size=5, novelty_threshold=0.2)
        ind1 = Individual(code="def transform(g): return g", fitness=0.5)
        ind2 = Individual(code="def transform(g): return g", fitness=0.6)
        archive.try_add(ind1)
        # Same code should update if better fitness
        result = archive.try_add(ind2)
        assert result  # updated in place
        assert archive.size == 1

    def test_max_size(self):
        archive = EliteArchive(max_size=3, novelty_threshold=0.01)
        for i in range(5):
            ind = Individual(code=f"unique_code_variant_{i}" * 10, fitness=i * 0.2)
            archive.try_add(ind)
        assert archive.size <= 3

    def test_get_top(self):
        archive = EliteArchive(max_size=10, novelty_threshold=0.01)
        for i in range(5):
            ind = Individual(code=f"unique_program_{i}_" * 20, fitness=i * 0.2)
            archive.try_add(ind)
        top = archive.get_top(3)
        assert len(top) <= 3
        if len(top) >= 2:
            assert top[0].fitness >= top[1].fitness

    def test_get_diverse_sample(self):
        archive = EliteArchive(max_size=10, novelty_threshold=0.01)
        for i in range(5):
            ind = Individual(code=f"program_variation_{i}_" * 20, fitness=i * 0.2)
            archive.try_add(ind)
        sample = archive.get_diverse_sample(3)
        assert len(sample) <= 3

    def test_get_diverse_sample_small_archive(self):
        archive = EliteArchive(max_size=10)
        ind = Individual(code="single", fitness=0.5)
        archive.try_add(ind)
        sample = archive.get_diverse_sample(5)
        assert len(sample) == 1

    def test_clear(self):
        archive = EliteArchive()
        archive.try_add(Individual(code="c", fitness=0.5))
        archive.clear()
        assert archive.size == 0
        assert archive.best is None

    def test_summary(self):
        archive = EliteArchive()
        assert archive.summary() == {"size": 0, "best_fitness": 0.0}

        archive.try_add(Individual(code="c", fitness=0.5))
        s = archive.summary()
        assert s["size"] == 1
        assert s["best_fitness"] == 0.5

    def test_is_novel_empty(self):
        archive = EliteArchive()
        assert archive.is_novel(Individual(code="c"))

    def test_replace_worst(self):
        archive = EliteArchive(max_size=2, novelty_threshold=0.01)
        archive.try_add(Individual(code="aaa" * 30, fitness=0.1))
        archive.try_add(Individual(code="bbb" * 30, fitness=0.2))
        # This should replace the worst
        result = archive.try_add(Individual(code="ccc" * 30, fitness=0.3))
        assert result
        assert archive.size == 2

    def test_reject_worse_than_all(self):
        archive = EliteArchive(max_size=2, novelty_threshold=0.01)
        archive.try_add(Individual(code="aaa" * 30, fitness=0.5))
        archive.try_add(Individual(code="bbb" * 30, fitness=0.6))
        result = archive.try_add(Individual(code="ccc" * 30, fitness=0.1))
        assert not result

    def test_not_novel_not_better(self):
        archive = EliteArchive(max_size=5, novelty_threshold=0.2)
        ind1 = Individual(code="def transform(g): return g", fitness=0.8)
        archive.try_add(ind1)
        ind2 = Individual(code="def transform(g): return g", fitness=0.3)
        result = archive.try_add(ind2)
        assert not result

    def test_individuals_property(self):
        archive = EliteArchive()
        ind = Individual(code="c", fitness=0.5)
        archive.try_add(ind)
        assert len(archive.individuals) == 1
