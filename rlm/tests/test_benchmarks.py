"""Tests for OOLONG, LoCoDiff, and Synthetic benchmarks."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.evaluation.oolong import OOLONGBenchmark, OOLONGTask
from src.evaluation.locodiff import LoCoDiffBenchmark, LoCoDiffTask
from src.evaluation.synthetic import SyntheticLongContextGenerator, SyntheticTask
from src.evaluation.metrics import RLMMetrics, MetricResult
from src.evaluation.runner import BenchmarkRunner, ComparisonResult, ScalingResult
from src.core.session import SessionResult
from src.core.result_protocol import RLMResult
from tests.conftest import MockLLM


class TestOOLONG:
    def test_task_count(self):
        bench = OOLONGBenchmark()
        assert len(bench) >= 20

    def test_task_categories(self):
        bench = OOLONGBenchmark()
        categories = {t.category for t in bench.tasks}
        assert "retrieval" in categories
        assert "aggregation" in categories
        assert "reasoning" in categories

    def test_get_by_category(self):
        bench = OOLONGBenchmark()
        retrieval = bench.get_by_category("retrieval")
        assert len(retrieval) > 0
        assert all(t.category == "retrieval" for t in retrieval)

    def test_get_by_difficulty(self):
        bench = OOLONGBenchmark()
        easy = bench.get_by_difficulty("easy")
        assert len(easy) > 0
        assert all(t.difficulty == "easy" for t in easy)

    def test_task_fields(self):
        bench = OOLONGBenchmark()
        task = bench.tasks[0]
        assert isinstance(task, OOLONGTask)
        assert task.task_id
        assert task.query
        assert task.context
        assert task.expected_answer

    def test_iteration(self):
        bench = OOLONGBenchmark()
        tasks = list(bench)
        assert len(tasks) == len(bench)


class TestLoCoDiff:
    def test_task_count(self):
        bench = LoCoDiffBenchmark()
        assert len(bench) >= 10

    def test_task_categories(self):
        bench = LoCoDiffBenchmark()
        categories = {t.category for t in bench.tasks}
        assert "function_find" in categories
        assert "bug_detect" in categories
        assert "diff_apply" in categories
        assert "dependency" in categories

    def test_get_by_category(self):
        bench = LoCoDiffBenchmark()
        funcs = bench.get_by_category("function_find")
        assert len(funcs) > 0

    def test_task_fields(self):
        bench = LoCoDiffBenchmark()
        task = bench.tasks[0]
        assert isinstance(task, LoCoDiffTask)
        assert task.task_id
        assert task.context

    def test_iteration(self):
        bench = LoCoDiffBenchmark()
        tasks = list(bench)
        assert len(tasks) == len(bench)


class TestSyntheticGenerator:
    def test_needle_in_haystack(self):
        gen = SyntheticLongContextGenerator()
        task = gen.needle_in_haystack()
        assert isinstance(task, SyntheticTask)
        assert "OPEN_SESAME_42" in task.context
        assert task.expected_answer == "OPEN_SESAME_42"

    def test_needle_position(self):
        gen = SyntheticLongContextGenerator()
        task = gen.needle_in_haystack(position=0.1)
        assert task.metadata["position"] == 0.1

    def test_multi_needle(self):
        gen = SyntheticLongContextGenerator()
        task = gen.multi_needle()
        assert "NEEDLE_A" in task.context
        assert "NEEDLE_B" in task.context
        assert "NEEDLE_C" in task.context

    def test_multi_needle_custom(self):
        gen = SyntheticLongContextGenerator()
        needles = ["KEY_X: val1", "KEY_Y: val2"]
        task = gen.multi_needle(needles=needles)
        assert task.metadata["num_needles"] == 2

    def test_counting_task(self):
        gen = SyntheticLongContextGenerator()
        task = gen.counting_task(target_word="marker", count=5)
        assert "marker" in task.context
        assert task.metadata["target_word"] == "marker"

    def test_summarization_task(self):
        gen = SyntheticLongContextGenerator()
        task = gen.summarization_task(num_sections=3)
        assert task.metadata["num_sections"] == 3
        assert "##" in task.context


class TestRLMMetrics:
    def _make_result(self, value: str, iterations: int = 3) -> SessionResult:
        return SessionResult(
            result=RLMResult(value=value, source="FINAL", raw_argument=value),
            trajectory=[],
            total_iterations=iterations,
            depth=0,
        )

    def test_accuracy_exact(self):
        metrics = RLMMetrics()
        results = [self._make_result("42"), self._make_result("hello")]
        expected = ["42", "hello"]
        acc = metrics.accuracy(results, expected, exact=True)
        assert acc.value == 1.0

    def test_accuracy_partial(self):
        metrics = RLMMetrics()
        results = [self._make_result("42"), self._make_result("wrong")]
        expected = ["42", "correct"]
        acc = metrics.accuracy(results, expected, exact=True)
        assert acc.value == 0.5

    def test_accuracy_fuzzy(self):
        metrics = RLMMetrics()
        results = [self._make_result("The answer is 42")]
        expected = ["42"]
        acc = metrics.accuracy(results, expected, exact=False)
        assert acc.value == 1.0

    def test_accuracy_empty(self):
        metrics = RLMMetrics()
        acc = metrics.accuracy([], [])
        assert acc.value == 0.0

    def test_cost_per_query(self):
        metrics = RLMMetrics()
        results = [self._make_result("x", 5), self._make_result("y", 3)]
        cost = metrics.cost_per_query(results, cost_per_iteration=0.01)
        assert cost.value == pytest.approx(0.04)

    def test_cost_empty(self):
        metrics = RLMMetrics()
        cost = metrics.cost_per_query([])
        assert cost.value == 0.0

    def test_context_utilization(self):
        metrics = RLMMetrics()
        results = [self._make_result("x")]
        util = metrics.context_utilization(results)
        assert 0.0 <= util.value <= 1.0

    def test_recursion_depth_distribution(self):
        metrics = RLMMetrics()
        results = [self._make_result("x"), self._make_result("y")]
        dist = metrics.recursion_depth_distribution(results)
        assert dist.details["distribution"][0] == 2

    def test_strategy_distribution(self):
        metrics = RLMMetrics()
        results = [self._make_result("x")]
        strat = metrics.strategy_distribution(results)
        assert strat.value >= 1

    def test_accuracy_per_dollar(self):
        metrics = RLMMetrics()
        results = [self._make_result("42", 5)]
        expected = ["42"]
        apd = metrics.accuracy_per_dollar(results, expected)
        assert apd.value > 0


class TestBenchmarkRunner:
    def test_run_tasks(self):
        runner = BenchmarkRunner()
        bench = OOLONGBenchmark()
        tasks = bench.tasks[:2]
        llm = MockLLM()
        results = runner.run_tasks(tasks, llm, max_iterations=3)
        assert len(results) == 2
        assert all(isinstance(r, SessionResult) for r in results)

    def test_run_comparison(self):
        runner = BenchmarkRunner()
        bench = OOLONGBenchmark()
        tasks = bench.tasks[:2]
        llm_a = MockLLM()
        llm_b = MockLLM()
        comp = runner.run_comparison(tasks, llm_a, llm_b, "test_bench")
        assert isinstance(comp, ComparisonResult)
        assert comp.benchmark_name == "test_bench"
        assert len(comp.summary()) > 0

    def test_run_context_scaling(self):
        gen = SyntheticLongContextGenerator()
        runner = BenchmarkRunner()
        llm = MockLLM()

        def factory(size):
            return gen.needle_in_haystack(haystack_size=size)

        result = runner.run_context_scaling(
            task_factory=factory,
            context_sizes=[1000, 2000],
            llm=llm,
            max_iterations=3,
        )
        assert isinstance(result, ScalingResult)
        assert len(result.context_sizes) == 2
        assert len(result.summary()) > 0
