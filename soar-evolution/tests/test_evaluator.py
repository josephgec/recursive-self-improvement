"""Tests for ProgramEvaluator."""

import pytest
from src.arc.evaluator import ProgramEvaluator, compute_pixel_accuracy, EvalResult, ProgramEvalResult
from src.arc.grid import Grid, ARCExample, ARCTask


class TestComputePixelAccuracy:
    def test_perfect_match(self):
        g1 = Grid.from_list([[1, 2], [3, 4]])
        g2 = Grid.from_list([[1, 2], [3, 4]])
        assert compute_pixel_accuracy(g1, g2) == 1.0

    def test_no_match(self):
        g1 = Grid.from_list([[1, 1], [1, 1]])
        g2 = Grid.from_list([[2, 2], [2, 2]])
        assert compute_pixel_accuracy(g1, g2) == 0.0

    def test_partial_match(self):
        g1 = Grid.from_list([[1, 2], [3, 4]])
        g2 = Grid.from_list([[1, 2], [3, 5]])
        assert compute_pixel_accuracy(g1, g2) == 0.75

    def test_different_shapes(self):
        g1 = Grid.from_list([[1, 2]])
        g2 = Grid.from_list([[1, 2], [3, 4]])
        assert compute_pixel_accuracy(g1, g2) == 0.0


class TestEvalResult:
    def test_successful_correct(self):
        out = Grid.from_list([[1, 2], [3, 4]])
        exp = Grid.from_list([[1, 2], [3, 4]])
        r = EvalResult(success=True, output_grid=out, expected_grid=exp)
        assert r.correct
        assert r.pixel_accuracy == 1.0

    def test_successful_wrong(self):
        out = Grid.from_list([[1, 2], [3, 5]])
        exp = Grid.from_list([[1, 2], [3, 4]])
        r = EvalResult(success=True, output_grid=out, expected_grid=exp)
        assert not r.correct
        assert r.pixel_accuracy == 0.75

    def test_failed(self):
        r = EvalResult(success=False, error="boom")
        assert not r.correct
        assert r.pixel_accuracy == 0.0


class TestProgramEvalResult:
    def test_train_accuracy(self):
        r1 = EvalResult(success=True, output_grid=Grid.from_list([[1]]), expected_grid=Grid.from_list([[1]]))
        r2 = EvalResult(success=True, output_grid=Grid.from_list([[0]]), expected_grid=Grid.from_list([[1]]))
        result = ProgramEvalResult(train_results=[r1, r2])
        assert result.train_accuracy == 0.5

    def test_empty_results(self):
        result = ProgramEvalResult()
        assert result.train_accuracy == 0.0
        assert result.test_accuracy == 0.0
        assert not result.all_train_correct
        assert not result.all_test_correct
        assert not result.fully_correct

    def test_fully_correct(self):
        r = EvalResult(success=True, output_grid=Grid.from_list([[1]]), expected_grid=Grid.from_list([[1]]))
        result = ProgramEvalResult(train_results=[r], test_results=[r])
        assert result.fully_correct


class TestProgramEvaluator:
    def test_compile_valid(self):
        ev = ProgramEvaluator()
        ok, fn, err = ev.compile_program("def transform(x): return x")
        assert ok
        assert fn is not None
        assert err is None

    def test_compile_syntax_error(self):
        ev = ProgramEvaluator()
        ok, fn, err = ev.compile_program("def transform(x) return x")
        assert not ok
        assert fn is None
        assert "Compilation error" in err

    def test_compile_no_transform(self):
        ev = ProgramEvaluator()
        ok, fn, err = ev.compile_program("def foo(x): return x")
        assert not ok
        assert "No 'transform'" in err

    def test_compile_not_callable(self):
        ev = ProgramEvaluator()
        ok, fn, err = ev.compile_program("transform = 42")
        assert not ok
        assert "not callable" in err

    def test_evaluate_correct(self, color_swap_task):
        ev = ProgramEvaluator()
        code = """def transform(input_grid):
    result = []
    for row in input_grid:
        new_row = []
        for cell in row:
            if cell == 1:
                new_row.append(2)
            else:
                new_row.append(cell)
        result.append(new_row)
    return result
"""
        result = ev.evaluate_task(code, color_swap_task)
        assert result.compile_error is None
        assert result.train_accuracy == 1.0
        assert result.all_train_correct
        assert result.test_accuracy == 1.0

    def test_evaluate_wrong(self, color_swap_task):
        ev = ProgramEvaluator()
        code = "def transform(g): return [row[:] for row in g]\n"
        result = ev.evaluate_task(code, color_swap_task)
        assert result.train_accuracy < 1.0

    def test_evaluate_compile_error(self, color_swap_task):
        ev = ProgramEvaluator()
        result = ev.evaluate_task("not valid python!!!", color_swap_task)
        assert result.compile_error is not None
        assert result.train_accuracy == 0.0

    def test_evaluate_runtime_error(self, color_swap_task):
        ev = ProgramEvaluator()
        code = "def transform(g): return g[999]\n"
        result = ev.evaluate_task(code, color_swap_task)
        assert any(r.error for r in result.train_results)

    def test_evaluate_returns_none(self, color_swap_task):
        ev = ProgramEvaluator()
        code = "def transform(g): return None\n"
        result = ev.evaluate_task(code, color_swap_task)
        assert any("None" in (r.error or "") for r in result.train_results)

    def test_evaluate_returns_wrong_type(self, color_swap_task):
        ev = ProgramEvaluator()
        code = "def transform(g): return 42\n"
        result = ev.evaluate_task(code, color_swap_task)
        assert any(r.error for r in result.train_results)

    def test_quick_check(self, color_swap_task):
        ev = ProgramEvaluator()
        code = "def transform(g): return [row[:] for row in g]\n"
        acc = ev.quick_check(code, color_swap_task)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_evaluate_no_test(self, color_swap_task):
        ev = ProgramEvaluator()
        code = "def transform(g): return [row[:] for row in g]\n"
        result = ev.evaluate_task(code, color_swap_task, eval_test=False)
        assert len(result.test_results) == 0
