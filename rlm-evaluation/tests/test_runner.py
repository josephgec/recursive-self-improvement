"""Tests for benchmark runner with checkpointing."""

import os
import pytest

from src.benchmarks.task import EvalTask, EvalResult
from src.execution.runner import BenchmarkRunner, BenchmarkRun, ScalingResults
from src.execution.rlm_executor import RLMExecutor
from src.execution.checkpoint import CheckpointManager
from src.execution.budget_tracker import BudgetTracker
from src.execution.parallel import ParallelExecutor


class TestBenchmarkRunner:
    """Test benchmark runner."""

    @pytest.fixture
    def executor(self):
        return RLMExecutor(seed=42)

    @pytest.fixture
    def runner(self, executor, tmp_checkpoint_dir):
        return BenchmarkRunner(
            executor_fn=executor.execute,
            checkpoint_dir=tmp_checkpoint_dir,
            checkpoint_interval=3,
        )

    def test_run_benchmark(self, runner, sample_tasks):
        run = runner.run_benchmark(sample_tasks, "test_bench", run_id="test_run")
        assert isinstance(run, BenchmarkRun)
        assert run.total_tasks == len(sample_tasks)
        assert 0.0 <= run.accuracy <= 1.0
        assert run.total_cost >= 0

    def test_run_benchmark_creates_checkpoint(self, runner, sample_tasks, tmp_checkpoint_dir):
        runner.run_benchmark(sample_tasks, "test_bench", run_id="ckpt_test")
        # Checkpoint should exist
        assert runner.checkpoint_mgr.exists("ckpt_test")

    def test_run_benchmark_resume(self, runner, sample_tasks, tmp_checkpoint_dir):
        # Run first time
        run1 = runner.run_benchmark(sample_tasks, "test_bench", run_id="resume_test")
        total_results_1 = run1.total_tasks

        # Run again - should resume (no new work needed since all done)
        run2 = runner.run_benchmark(sample_tasks, "test_bench", run_id="resume_test", resume=True)
        assert run2.total_tasks == total_results_1

    def test_run_benchmark_no_resume(self, runner, sample_tasks):
        run1 = runner.run_benchmark(sample_tasks, "test_bench", run_id="no_resume")
        # Run without resume
        run2 = runner.run_benchmark(sample_tasks, "test_bench", run_id="no_resume", resume=False)
        assert run2.total_tasks == len(sample_tasks)

    def test_run_all_benchmarks(self, runner, sample_tasks):
        tasks_by_bench = {
            "bench_a": sample_tasks[:4],
            "bench_b": sample_tasks[4:],
        }
        runs = runner.run_all_benchmarks(tasks_by_bench)
        assert "bench_a" in runs
        assert "bench_b" in runs
        assert runs["bench_a"].total_tasks == 4
        assert runs["bench_b"].total_tasks == 4

    def test_budget_tracking(self, executor, tmp_checkpoint_dir, sample_tasks):
        runner = BenchmarkRunner(
            executor_fn=executor.execute,
            checkpoint_dir=tmp_checkpoint_dir,
            budget_limit=1000.0,
        )
        runner.run_benchmark(sample_tasks[:3], "test", run_id="budget_test")
        assert runner.budget_tracker.total_cost > 0
        assert runner.budget_tracker.total_calls > 0


class TestBenchmarkRun:
    """Test BenchmarkRun dataclass."""

    def test_compute_stats(self):
        results = [
            EvalResult(task_id="t1", benchmark="b", answer="a", correct=True, cost=0.01),
            EvalResult(task_id="t2", benchmark="b", answer="b", correct=False, cost=0.02),
            EvalResult(task_id="t3", benchmark="b", answer="c", correct=True, cost=0.015),
        ]
        run = BenchmarkRun(run_id="test", benchmark="b", results=results)
        run.compute_stats()
        assert run.total_tasks == 3
        assert run.correct_count == 2
        assert abs(run.accuracy - 2/3) < 0.01
        assert abs(run.total_cost - 0.045) < 0.001

    def test_empty_compute_stats(self):
        run = BenchmarkRun(run_id="empty", benchmark="b")
        run.compute_stats()
        assert run.total_tasks == 0
        assert run.accuracy == 0.0


class TestScalingResults:
    """Test ScalingResults dataclass."""

    def test_find_crossover(self):
        results = ScalingResults(
            context_sizes=[1000, 4000, 8000],
            rlm_accuracies=[0.6, 0.7, 0.8],
            standard_accuracies=[0.9, 0.65, 0.4],
        )
        crossover = results.find_crossover()
        assert crossover == 4000

    def test_no_crossover(self):
        results = ScalingResults(
            context_sizes=[1000, 4000],
            rlm_accuracies=[0.5, 0.5],
            standard_accuracies=[0.9, 0.9],
        )
        crossover = results.find_crossover()
        assert crossover is None


class TestCheckpointManager:
    """Test checkpoint save/load/resume."""

    def test_save_and_load(self, tmp_checkpoint_dir):
        mgr = CheckpointManager(tmp_checkpoint_dir)
        results = [
            EvalResult(task_id="t1", benchmark="b", answer="a", correct=True),
        ]
        mgr.save("test_ckpt", results)
        loaded = mgr.load("test_ckpt")
        assert loaded is not None
        assert loaded["run_id"] == "test_ckpt"
        assert len(loaded["results"]) == 1

    def test_resume(self, tmp_checkpoint_dir):
        mgr = CheckpointManager(tmp_checkpoint_dir)
        results = [
            EvalResult(task_id="t1", benchmark="b", answer="a", correct=True),
            EvalResult(task_id="t2", benchmark="b", answer="b", correct=False),
        ]
        mgr.save("resume_ckpt", results)
        resumed = mgr.resume("resume_ckpt")
        assert len(resumed) == 2

    def test_resume_nonexistent(self, tmp_checkpoint_dir):
        mgr = CheckpointManager(tmp_checkpoint_dir)
        resumed = mgr.resume("nonexistent")
        assert resumed == []

    def test_exists(self, tmp_checkpoint_dir):
        mgr = CheckpointManager(tmp_checkpoint_dir)
        assert not mgr.exists("nope")
        mgr.save("yes", [])
        assert mgr.exists("yes")

    def test_delete(self, tmp_checkpoint_dir):
        mgr = CheckpointManager(tmp_checkpoint_dir)
        mgr.save("del_me", [])
        assert mgr.exists("del_me")
        assert mgr.delete("del_me")
        assert not mgr.exists("del_me")

    def test_delete_nonexistent(self, tmp_checkpoint_dir):
        mgr = CheckpointManager(tmp_checkpoint_dir)
        assert not mgr.delete("nope")

    def test_list_checkpoints(self, tmp_checkpoint_dir):
        mgr = CheckpointManager(tmp_checkpoint_dir)
        mgr.save("ckpt_a", [])
        mgr.save("ckpt_b", [])
        checkpoints = mgr.list_checkpoints()
        assert "ckpt_a" in checkpoints
        assert "ckpt_b" in checkpoints

    def test_get_completed_task_ids(self, tmp_checkpoint_dir):
        mgr = CheckpointManager(tmp_checkpoint_dir)
        results = [
            EvalResult(task_id="t1", benchmark="b", answer="a", correct=True),
            EvalResult(task_id="t2", benchmark="b", answer="b", correct=False),
        ]
        mgr.save("ids_ckpt", results)
        ids = mgr.get_completed_task_ids("ids_ckpt")
        assert ids == {"t1", "t2"}


class TestBudgetTracker:
    """Test budget tracking."""

    def test_track_usage(self):
        tracker = BudgetTracker()
        tracker.track("rlm", "t1", 1000, 500, 0.025, 3)
        assert tracker.total_cost == 0.025
        assert tracker.total_input_tokens == 1000
        assert tracker.total_output_tokens == 500
        assert tracker.total_tokens == 1500
        assert tracker.total_calls == 3

    def test_budget_limit(self):
        tracker = BudgetTracker(budget_limit=0.05)
        tracker.track("rlm", "t1", 1000, 500, 0.03, 1)
        assert not tracker.over_budget
        assert tracker.budget_remaining == pytest.approx(0.02)

        tracker.track("rlm", "t2", 1000, 500, 0.03, 1)
        assert tracker.over_budget

    def test_cost_by_system(self):
        tracker = BudgetTracker()
        tracker.track("rlm", "t1", 1000, 500, 0.03, 1)
        tracker.track("standard", "t2", 500, 200, 0.01, 1)
        costs = tracker.cost_by_system()
        assert costs["rlm"] == 0.03
        assert costs["standard"] == 0.01

    def test_summary(self):
        tracker = BudgetTracker()
        tracker.track("rlm", "t1", 1000, 500, 0.03, 2)
        summary = tracker.summary()
        assert summary["total_cost"] == 0.03
        assert summary["total_calls"] == 2
        assert summary["num_records"] == 1

    def test_tokens_by_system(self):
        tracker = BudgetTracker()
        tracker.track("rlm", "t1", 1000, 500, 0.03, 1)
        tokens = tracker.tokens_by_system()
        assert tokens["rlm"] == 1500

    def test_calls_by_system(self):
        tracker = BudgetTracker()
        tracker.track("rlm", "t1", 1000, 500, 0.03, 3)
        tracker.track("rlm", "t2", 500, 200, 0.01, 2)
        calls = tracker.calls_by_system()
        assert calls["rlm"] == 5

    def test_records_property(self):
        tracker = BudgetTracker()
        tracker.track("rlm", "t1", 1000, 500, 0.03, 1)
        records = tracker.records
        assert len(records) == 1
        assert records[0].system == "rlm"


class TestParallelExecutor:
    """Test parallel execution."""

    def test_execute_all(self, sample_tasks):
        executor = RLMExecutor(seed=42)
        parallel = ParallelExecutor(max_workers=2)
        results = parallel.execute_all(sample_tasks[:3], executor.execute)
        assert len(results) == 3
        assert all(isinstance(r, EvalResult) for r in results)

    def test_execute_all_with_callback(self, sample_tasks):
        executor = RLMExecutor(seed=42)
        parallel = ParallelExecutor(max_workers=2)
        completed = []
        results = parallel.execute_all(
            sample_tasks[:3],
            executor.execute,
            on_complete=lambda r: completed.append(r.task_id),
        )
        assert len(results) == 3
        assert len(completed) == 3

    def test_execute_empty(self):
        executor = RLMExecutor(seed=42)
        parallel = ParallelExecutor(max_workers=2)
        results = parallel.execute_all([], executor.execute)
        assert results == []

    def test_execute_batch(self, sample_tasks):
        executor = RLMExecutor(seed=42)
        parallel = ParallelExecutor(max_workers=2)
        results = parallel.execute_batch(sample_tasks[:6], executor.execute, batch_size=3)
        assert len(results) == 6

    def test_error_handling(self):
        def failing_executor(task):
            raise ValueError("Test error")

        task = EvalTask(
            task_id="fail",
            benchmark="test",
            query="q",
            context="c",
            expected_answer="a",
        )
        parallel = ParallelExecutor(max_workers=1)
        results = parallel.execute_all([task], failing_executor)
        assert len(results) == 1
        assert results[0].error is not None
        assert not results[0].correct
