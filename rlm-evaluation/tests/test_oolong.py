"""Tests for OOLONG benchmark."""

import pytest

from src.benchmarks.oolong import OOLONGBenchmark
from src.benchmarks.task import EvalTask


class TestOOLONGBenchmark:
    """Test OOLONG benchmark task loading and validation."""

    def test_load_returns_tasks(self):
        benchmark = OOLONGBenchmark()
        tasks = benchmark.load()
        assert len(tasks) >= 20
        assert all(isinstance(t, EvalTask) for t in tasks)

    def test_all_tasks_have_required_fields(self):
        benchmark = OOLONGBenchmark()
        tasks = benchmark.load()
        for task in tasks:
            assert task.task_id
            assert task.benchmark == "oolong"
            assert task.query
            assert task.context
            assert task.expected_answer
            assert task.category

    def test_categories_present(self):
        benchmark = OOLONGBenchmark()
        tasks = benchmark.load()
        categories = {t.category for t in tasks}
        assert "retrieval" in categories
        assert "aggregation" in categories
        assert "reasoning" in categories
        assert "counting" in categories

    def test_retrieval_tasks(self):
        benchmark = OOLONGBenchmark()
        tasks = benchmark.load()
        retrieval = [t for t in tasks if t.category == "retrieval"]
        assert len(retrieval) >= 5
        # Check that expected answers are in the context
        for t in retrieval:
            assert t.expected_answer.lower() in t.context.lower()

    def test_aggregation_tasks(self):
        benchmark = OOLONGBenchmark()
        tasks = benchmark.load()
        agg = [t for t in tasks if t.category == "aggregation"]
        assert len(agg) >= 2

    def test_reasoning_tasks(self):
        benchmark = OOLONGBenchmark()
        tasks = benchmark.load()
        reason = [t for t in tasks if t.category == "reasoning"]
        assert len(reason) >= 2

    def test_counting_tasks(self):
        benchmark = OOLONGBenchmark()
        tasks = benchmark.load()
        counting = [t for t in tasks if t.category == "counting"]
        assert len(counting) >= 2

    def test_unique_task_ids(self):
        benchmark = OOLONGBenchmark()
        tasks = benchmark.load()
        ids = [t.task_id for t in tasks]
        assert len(ids) == len(set(ids)), "Task IDs must be unique"

    def test_context_has_content(self):
        benchmark = OOLONGBenchmark()
        tasks = benchmark.load()
        for task in tasks:
            assert len(task.context) > 100

    def test_answer_checking_retrieval(self):
        """Check answers are programmatically verifiable for retrieval tasks."""
        benchmark = OOLONGBenchmark()
        tasks = benchmark.load()
        for task in [t for t in tasks if t.category == "retrieval"]:
            # The expected answer should appear somewhere in the context
            assert task.expected_answer in task.context or \
                   task.expected_answer.lower() in task.context.lower()

    def test_seed_reproducibility(self):
        b1 = OOLONGBenchmark(seed=42)
        b2 = OOLONGBenchmark(seed=42)
        tasks1 = b1.load()
        tasks2 = b2.load()
        assert len(tasks1) == len(tasks2)
        for t1, t2 in zip(tasks1, tasks2):
            assert t1.task_id == t2.task_id
            assert t1.expected_answer == t2.expected_answer

    def test_different_seeds_same_structure(self):
        b1 = OOLONGBenchmark(seed=42)
        b2 = OOLONGBenchmark(seed=99)
        tasks1 = b1.load()
        tasks2 = b2.load()
        assert len(tasks1) == len(tasks2)
