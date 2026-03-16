"""Tests for FitnessComputer."""

import pytest
from src.population.fitness import FitnessComputer, FitnessWeights
from src.population.individual import Individual
from src.arc.evaluator import ProgramEvalResult, EvalResult, ProgramEvaluator
from src.arc.grid import Grid


class TestFitnessWeights:
    def test_defaults(self):
        w = FitnessWeights()
        assert w.pixel_accuracy == 0.7
        assert w.simplicity == 0.15
        assert w.consistency == 0.15


class TestFitnessComputer:
    def test_compute_simplicity_short(self):
        fc = FitnessComputer()
        score = fc.compute_simplicity("def f(): pass")
        assert 0.5 < score <= 1.0

    def test_compute_simplicity_long(self):
        fc = FitnessComputer()
        long_code = "x = 1\n" * 500
        score = fc.compute_simplicity(long_code)
        assert score < 0.5

    def test_compute_simplicity_empty(self):
        fc = FitnessComputer()
        assert fc.compute_simplicity("") == 0.0

    def test_compute_simplicity_deep_nesting(self):
        fc = FitnessComputer()
        code = "if True:\n" + "    " * 10 + "pass\n"
        score = fc.compute_simplicity(code)
        assert 0.0 <= score <= 1.0

    def test_compute_consistency_empty(self):
        fc = FitnessComputer()
        result = ProgramEvalResult()
        assert fc.compute_consistency(result) == 0.0

    def test_compute_consistency_single(self):
        fc = FitnessComputer()
        r = EvalResult(
            success=True,
            output_grid=Grid.from_list([[1]]),
            expected_grid=Grid.from_list([[1]]),
        )
        result = ProgramEvalResult(train_results=[r])
        assert fc.compute_consistency(result) == 1.0

    def test_compute_consistency_varied(self):
        fc = FitnessComputer()
        r1 = EvalResult(
            success=True,
            output_grid=Grid.from_list([[1]]),
            expected_grid=Grid.from_list([[1]]),
        )
        r2 = EvalResult(
            success=True,
            output_grid=Grid.from_list([[0]]),
            expected_grid=Grid.from_list([[1]]),
        )
        result = ProgramEvalResult(train_results=[r1, r2])
        consistency = fc.compute_consistency(result)
        assert 0.0 <= consistency <= 1.0
        # High variance should reduce consistency
        assert consistency < 1.0

    def test_evaluate_individual(self, color_swap_task, good_color_swap_individual):
        fc = FitnessComputer()
        ind = fc.evaluate_individual(good_color_swap_individual, color_swap_task)
        assert ind.evaluated
        assert ind.fitness > 0.0
        assert ind.train_accuracy == 1.0
        assert ind.test_accuracy == 1.0

    def test_evaluate_individual_bad(self, color_swap_task, bad_individual):
        fc = FitnessComputer()
        ind = fc.evaluate_individual(bad_individual, color_swap_task)
        assert ind.evaluated
        assert len(ind.runtime_errors) > 0

    def test_evaluate_individual_error(self, color_swap_task, error_individual):
        fc = FitnessComputer()
        ind = fc.evaluate_individual(error_individual, color_swap_task)
        assert ind.evaluated
        assert len(ind.runtime_errors) > 0

    def test_evaluate_individual_compile_error(self, color_swap_task):
        fc = FitnessComputer()
        ind = Individual(code="not valid python!!!")
        fc.evaluate_individual(ind, color_swap_task)
        assert ind.compile_error is not None
        # pixel_accuracy and consistency are 0, but simplicity may be nonzero
        assert ind.pixel_accuracy == 0.0
        assert ind.consistency_score == 0.0
        # fitness = 0.7*0 + 0.15*simplicity + 0.15*0 = small positive
        assert ind.fitness < 0.2

    def test_evaluate_population(self, color_swap_task):
        fc = FitnessComputer()
        inds = [
            Individual(code="def transform(g): return [row[:] for row in g]"),
            Individual(code="def transform(g): return g"),
        ]
        fc.evaluate_population(inds, color_swap_task)
        assert all(ind.evaluated for ind in inds)

    def test_fitness_weights_custom(self, color_swap_task, good_color_swap_individual):
        weights = FitnessWeights(pixel_accuracy=1.0, simplicity=0.0, consistency=0.0)
        fc = FitnessComputer(weights=weights)
        ind = fc.evaluate_individual(good_color_swap_individual, color_swap_task)
        # With only pixel accuracy weight, fitness should equal train accuracy
        assert abs(ind.fitness - ind.train_accuracy) < 0.01
