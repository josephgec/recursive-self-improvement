"""Test benchmarks: load all 6 benchmarks, verify tasks, answer checking."""

import pytest

from src.benchmarks.registry import BenchmarkRegistry, register_all_benchmarks, BaseBenchmark
from src.benchmarks.math500 import MATH500Benchmark
from src.benchmarks.arc_agi import ARCAGIBenchmark
from src.benchmarks.oolong import OOLONGBenchmark
from src.benchmarks.humaneval import HumanEvalBenchmark
from src.benchmarks.swebench import SWEBenchBenchmark
from src.benchmarks.financial import FinancialBenchmark
from src.benchmarks.answer_checking import (
    NumericChecker, ExactChecker, CodeChecker, SymbolicChecker,
)


class TestBenchmarkRegistry:
    """Test benchmark registry operations."""

    def test_register_and_load(self):
        BenchmarkRegistry.clear()
        register_all_benchmarks()
        assert len(BenchmarkRegistry.available()) == 6

    def test_load_unknown_raises(self):
        BenchmarkRegistry.clear()
        with pytest.raises(KeyError):
            BenchmarkRegistry.load("nonexistent")

    def test_load_all(self, registered_benchmarks):
        assert len(registered_benchmarks) == 6
        for name, bm in registered_benchmarks.items():
            assert isinstance(bm, BaseBenchmark)
            assert len(bm.tasks) > 0

    def test_evaluate_via_registry(self, mock_agent):
        BenchmarkRegistry.clear()
        register_all_benchmarks()
        results = BenchmarkRegistry.evaluate("math500", mock_agent)
        assert len(results) > 0
        assert all(r.benchmark == "math500" for r in results)

    def test_clear_registry(self):
        BenchmarkRegistry.clear()
        register_all_benchmarks()
        assert len(BenchmarkRegistry.available()) == 6
        BenchmarkRegistry.clear()
        assert len(BenchmarkRegistry.available()) == 0


class TestMATH500:
    """Test MATH500 benchmark."""

    def test_task_count(self):
        bm = MATH500Benchmark()
        assert len(bm.tasks) >= 30

    def test_categories(self):
        bm = MATH500Benchmark()
        cats = bm.categories
        assert "algebra" in cats
        assert "number_theory" in cats
        assert "calculus" in cats
        assert "geometry" in cats
        assert "counting" in cats

    def test_get_tasks_by_category(self):
        bm = MATH500Benchmark()
        algebra = bm.get_tasks("algebra")
        assert len(algebra) >= 8
        assert all(t.category == "algebra" for t in algebra)

    def test_numeric_answer_checking(self):
        bm = MATH500Benchmark()
        task = bm.get_tasks("algebra")[0]  # 2x + 5 = 13, answer = 4.0
        assert bm.check_answer(task, 4.0)
        assert bm.check_answer(task, 4)
        assert bm.check_answer(task, 3.995)  # Within tolerance
        assert not bm.check_answer(task, 5.0)

    def test_string_answer_checking(self):
        bm = MATH500Benchmark()
        # Find a string-answer task
        nt_tasks = bm.get_tasks("number_theory")
        is_prime_task = [t for t in nt_tasks if "prime" in t.prompt.lower() and "yes" in str(t.expected_answer).lower()]
        if is_prime_task:
            task = is_prime_task[0]
            assert bm.check_answer(task, "yes")
            assert bm.check_answer(task, "Yes")
            assert not bm.check_answer(task, "no")

    def test_evaluate_with_agent(self, mock_agent):
        bm = MATH500Benchmark()
        results = bm.evaluate(mock_agent)
        assert len(results) == len(bm.tasks)
        correct = sum(1 for r in results if r.correct)
        assert 0 <= correct <= len(results)


class TestARCAGI:
    """Test ARC-AGI benchmark."""

    def test_task_count(self):
        bm = ARCAGIBenchmark()
        assert len(bm.tasks) >= 15

    def test_categories(self):
        bm = ARCAGIBenchmark()
        cats = bm.categories
        assert "color_swap" in cats
        assert "pattern" in cats
        assert "transform" in cats

    def test_answer_checking(self):
        bm = ARCAGIBenchmark()
        task = bm.get_tasks("color_swap")[0]
        assert bm.check_answer(task, task.expected_answer)
        assert not bm.check_answer(task, [[9, 9], [9, 9]])

    def test_evaluate(self, mock_agent):
        bm = ARCAGIBenchmark()
        results = bm.evaluate(mock_agent)
        assert len(results) == len(bm.tasks)


class TestOOLONG:
    """Test OOLONG benchmark."""

    def test_task_count(self):
        bm = OOLONGBenchmark()
        assert len(bm.tasks) >= 15

    def test_categories(self):
        bm = OOLONGBenchmark()
        cats = bm.categories
        assert "retrieval" in cats
        assert "aggregation" in cats
        assert "reasoning" in cats
        assert "counting" in cats

    def test_answer_checking_numeric(self):
        bm = OOLONGBenchmark()
        agg_tasks = bm.get_tasks("aggregation")
        task = agg_tasks[0]  # Sum of [10, 20, 30, 40] = 100
        assert bm.check_answer(task, 100)
        assert bm.check_answer(task, 100.0)

    def test_answer_checking_string(self):
        bm = OOLONGBenchmark()
        ret_tasks = bm.get_tasks("retrieval")
        task = ret_tasks[0]  # Capital of France = Paris
        assert bm.check_answer(task, "Paris")
        assert bm.check_answer(task, "paris")

    def test_evaluate(self, mock_agent):
        bm = OOLONGBenchmark()
        results = bm.evaluate(mock_agent)
        assert len(results) == len(bm.tasks)


class TestHumanEval:
    """Test HumanEval benchmark."""

    def test_task_count(self):
        bm = HumanEvalBenchmark()
        assert len(bm.tasks) >= 15

    def test_all_function_completion(self):
        bm = HumanEvalBenchmark()
        assert all(t.category == "function_completion" for t in bm.tasks)

    def test_answer_checking_exact(self):
        bm = HumanEvalBenchmark()
        task = bm.tasks[0]  # add function
        assert bm.check_answer(task, task.expected_answer)

    def test_answer_checking_via_execution(self):
        bm = HumanEvalBenchmark()
        task = bm.tasks[0]  # def add(a, b): return a + b
        # Equivalent but different code
        alt_code = "def add(a, b): return b + a"
        assert bm.check_answer(task, alt_code)

    def test_evaluate(self, mock_agent):
        bm = HumanEvalBenchmark()
        results = bm.evaluate(mock_agent)
        assert len(results) == len(bm.tasks)


class TestSWEBench:
    """Test SWE-Bench benchmark."""

    def test_task_count(self):
        bm = SWEBenchBenchmark()
        assert len(bm.tasks) >= 10

    def test_categories(self):
        bm = SWEBenchBenchmark()
        cats = bm.categories
        assert "bug_fix" in cats
        assert "feature" in cats

    def test_answer_checking(self):
        bm = SWEBenchBenchmark()
        task = bm.get_tasks("bug_fix")[0]
        assert bm.check_answer(task, task.expected_answer)
        assert not bm.check_answer(task, "wrong answer")

    def test_evaluate(self, mock_agent):
        bm = SWEBenchBenchmark()
        results = bm.evaluate(mock_agent)
        assert len(results) == len(bm.tasks)


class TestFinancial:
    """Test Financial benchmark."""

    def test_task_count(self):
        bm = FinancialBenchmark()
        assert len(bm.tasks) >= 15

    def test_categories(self):
        bm = FinancialBenchmark()
        cats = bm.categories
        assert "compound_interest" in cats
        assert "options" in cats
        assert "risk" in cats
        assert "bonds" in cats

    def test_numeric_tolerance(self):
        bm = FinancialBenchmark()
        task = bm.get_tasks("compound_interest")[0]  # 1102.50
        assert bm.check_answer(task, 1102.50)
        assert bm.check_answer(task, 1102.0)  # Within 2% tolerance
        assert not bm.check_answer(task, 1200.0)

    def test_evaluate(self, mock_agent):
        bm = FinancialBenchmark()
        results = bm.evaluate(mock_agent)
        assert len(results) == len(bm.tasks)


class TestAnswerCheckers:
    """Test individual answer checker classes."""

    def test_numeric_checker_exact(self):
        assert NumericChecker.check(5.0, 5.0)
        assert NumericChecker.check(5, 5.0)

    def test_numeric_checker_tolerance(self):
        assert NumericChecker.check(5.05, 5.0, tolerance=0.02)
        assert not NumericChecker.check(5.5, 5.0, tolerance=0.02)

    def test_numeric_checker_zero(self):
        assert NumericChecker.check(0, 0)
        assert NumericChecker.check(0.005, 0, tolerance=0.01)

    def test_numeric_checker_invalid(self):
        assert not NumericChecker.check("abc", 5.0)
        assert not NumericChecker.check(None, 5.0)

    def test_exact_checker(self):
        assert ExactChecker.check(42, 42)
        assert ExactChecker.check("hello", "hello")
        assert not ExactChecker.check(42, 43)
        assert ExactChecker.check([1, 2], [1, 2])
        assert not ExactChecker.check([1, 2], [2, 1])

    def test_code_checker_exact(self):
        code = "def foo(): return 1"
        assert CodeChecker.check(code, code)

    def test_code_checker_test_cases(self):
        code = "def add(a, b): return a + b"
        test_cases = [(2, 3, 5), (0, 0, 0)]
        assert CodeChecker.check(code, "different", test_cases)

    def test_code_checker_failing(self):
        code = "def add(a, b): return a - b"
        test_cases = [(2, 3, 5)]
        assert not CodeChecker.check(code, "different", test_cases)

    def test_code_checker_invalid_code(self):
        assert not CodeChecker.check("not valid python {{", "expected", [(1, 2, 3)])

    def test_symbolic_checker(self):
        assert SymbolicChecker.check("x^2", "x**2")
        assert SymbolicChecker.check("2*x", "2x")
        assert SymbolicChecker.check("X + Y", "x+y")
        assert not SymbolicChecker.check("x^2", "x^3")
