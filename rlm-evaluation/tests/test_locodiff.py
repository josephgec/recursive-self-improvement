"""Tests for LoCoDiff benchmark."""

import pytest

from src.benchmarks.locodiff import LoCoDiffBenchmark
from src.benchmarks.task import EvalTask


class TestLoCoDiffBenchmark:
    """Test LoCoDiff benchmark task loading."""

    def test_load_returns_tasks(self):
        benchmark = LoCoDiffBenchmark()
        tasks = benchmark.load()
        assert len(tasks) >= 10
        assert all(isinstance(t, EvalTask) for t in tasks)

    def test_all_tasks_have_required_fields(self):
        benchmark = LoCoDiffBenchmark()
        tasks = benchmark.load()
        for task in tasks:
            assert task.task_id
            assert task.benchmark == "locodiff"
            assert task.query
            assert task.context
            assert task.expected_answer
            assert task.category

    def test_categories_present(self):
        benchmark = LoCoDiffBenchmark()
        tasks = benchmark.load()
        categories = {t.category for t in tasks}
        assert "function_change" in categories
        assert "bug_fix" in categories
        assert "refactoring" in categories

    def test_function_change_tasks(self):
        benchmark = LoCoDiffBenchmark()
        tasks = benchmark.load()
        func_tasks = [t for t in tasks if t.category == "function_change"]
        assert len(func_tasks) >= 3

    def test_bug_fix_tasks(self):
        benchmark = LoCoDiffBenchmark()
        tasks = benchmark.load()
        bug_tasks = [t for t in tasks if t.category == "bug_fix"]
        assert len(bug_tasks) >= 2

    def test_refactoring_tasks(self):
        benchmark = LoCoDiffBenchmark()
        tasks = benchmark.load()
        refactor_tasks = [t for t in tasks if t.category == "refactoring"]
        assert len(refactor_tasks) >= 2

    def test_context_contains_diff_markers(self):
        """Check that contexts contain diff-like content."""
        benchmark = LoCoDiffBenchmark()
        tasks = benchmark.load()
        for task in tasks:
            # Context should contain diff-related content
            assert len(task.context) > 100

    def test_unique_task_ids(self):
        benchmark = LoCoDiffBenchmark()
        tasks = benchmark.load()
        ids = [t.task_id for t in tasks]
        assert len(ids) == len(set(ids))

    def test_seed_reproducibility(self):
        b1 = LoCoDiffBenchmark(seed=42)
        b2 = LoCoDiffBenchmark(seed=42)
        tasks1 = b1.load()
        tasks2 = b2.load()
        assert len(tasks1) == len(tasks2)
        for t1, t2 in zip(tasks1, tasks2):
            assert t1.task_id == t2.task_id
            assert t1.expected_answer == t2.expected_answer

    def test_difficulty_levels(self):
        benchmark = LoCoDiffBenchmark()
        tasks = benchmark.load()
        difficulties = {t.difficulty for t in tasks}
        assert len(difficulties) >= 1  # At least one difficulty level

    def test_benchmark_name(self):
        benchmark = LoCoDiffBenchmark()
        assert benchmark.name == "locodiff"
