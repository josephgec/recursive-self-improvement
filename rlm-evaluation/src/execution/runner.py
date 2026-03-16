"""Benchmark runner: orchestrates evaluation runs with checkpointing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from src.benchmarks.task import EvalTask, EvalResult
from src.execution.budget_tracker import BudgetTracker
from src.execution.checkpoint import CheckpointManager
from src.execution.parallel import ParallelExecutor


@dataclass
class BenchmarkRun:
    """Results from a single benchmark run."""
    run_id: str
    benchmark: str
    results: List[EvalResult] = field(default_factory=list)
    total_tasks: int = 0
    correct_count: int = 0
    accuracy: float = 0.0
    total_cost: float = 0.0
    metadata: Dict[str, object] = field(default_factory=dict)

    def compute_stats(self) -> None:
        """Compute accuracy and cost stats from results."""
        self.total_tasks = len(self.results)
        self.correct_count = sum(1 for r in self.results if r.correct)
        self.accuracy = self.correct_count / self.total_tasks if self.total_tasks > 0 else 0.0
        self.total_cost = sum(r.cost for r in self.results)


@dataclass
class ScalingResults:
    """Results from a context scaling experiment."""
    context_sizes: List[int] = field(default_factory=list)
    rlm_accuracies: List[float] = field(default_factory=list)
    standard_accuracies: List[float] = field(default_factory=list)
    rlm_costs: List[float] = field(default_factory=list)
    standard_costs: List[float] = field(default_factory=list)
    crossover_point: Optional[int] = None

    def find_crossover(self) -> Optional[int]:
        """Find the context size where RLM surpasses standard."""
        for i, size in enumerate(self.context_sizes):
            if i < len(self.rlm_accuracies) and i < len(self.standard_accuracies):
                if self.rlm_accuracies[i] > self.standard_accuracies[i]:
                    self.crossover_point = size
                    return size
        return None


class BenchmarkRunner:
    """Orchestrate benchmark evaluation runs."""

    def __init__(
        self,
        executor_fn: Callable[[EvalTask], EvalResult],
        checkpoint_dir: str = "data/results",
        max_workers: int = 4,
        checkpoint_interval: int = 10,
        budget_limit: float = float("inf"),
    ) -> None:
        self.executor_fn = executor_fn
        self.checkpoint_mgr = CheckpointManager(checkpoint_dir)
        self.parallel = ParallelExecutor(max_workers=max_workers)
        self.checkpoint_interval = checkpoint_interval
        self.budget_tracker = BudgetTracker(budget_limit)
        self._results_buffer: List[EvalResult] = []

    def run_benchmark(
        self,
        tasks: List[EvalTask],
        benchmark_name: str,
        run_id: Optional[str] = None,
        resume: bool = True,
    ) -> BenchmarkRun:
        """Run a benchmark evaluation.

        Args:
            tasks: Tasks to evaluate.
            benchmark_name: Name of the benchmark.
            run_id: Unique run identifier.
            resume: Whether to resume from checkpoint.

        Returns:
            BenchmarkRun with all results.
        """
        run_id = run_id or f"{benchmark_name}_run"
        run = BenchmarkRun(run_id=run_id, benchmark=benchmark_name)

        # Resume from checkpoint if available
        completed_results: List[EvalResult] = []
        completed_ids: set = set()
        if resume and self.checkpoint_mgr.exists(run_id):
            completed_results = self.checkpoint_mgr.resume(run_id)
            completed_ids = {r.task_id for r in completed_results}

        # Filter out already-completed tasks
        remaining_tasks = [t for t in tasks if t.task_id not in completed_ids]

        # Execute remaining tasks
        self._results_buffer = list(completed_results)
        new_results: List[EvalResult] = []

        for i, task in enumerate(remaining_tasks):
            if self.budget_tracker.over_budget:
                break

            result = self.executor_fn(task)
            new_results.append(result)
            self._results_buffer.append(result)

            # Track budget
            self.budget_tracker.track(
                system=benchmark_name,
                task_id=task.task_id,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cost=result.cost,
                num_calls=result.num_calls,
            )

            # Periodic checkpoint
            if (i + 1) % self.checkpoint_interval == 0:
                self.checkpoint_mgr.save(run_id, self._results_buffer)

        # Final checkpoint
        self.checkpoint_mgr.save(run_id, self._results_buffer)

        run.results = self._results_buffer
        run.compute_stats()
        return run

    def run_all_benchmarks(
        self,
        benchmark_tasks: Dict[str, List[EvalTask]],
    ) -> Dict[str, BenchmarkRun]:
        """Run multiple benchmarks."""
        runs: Dict[str, BenchmarkRun] = {}
        for name, tasks in benchmark_tasks.items():
            runs[name] = self.run_benchmark(tasks, name)
        return runs

    def run_scaling_experiment(
        self,
        base_tasks: List[EvalTask],
        context_sizes: List[int],
        rlm_executor_fn: Callable[[EvalTask], EvalResult],
        standard_executor_fn: Callable[[EvalTask], EvalResult],
    ) -> ScalingResults:
        """Run a context scaling experiment comparing RLM vs standard."""
        results = ScalingResults(context_sizes=context_sizes)

        for size in context_sizes:
            # Create tasks with this context size
            sized_tasks = [t.with_context_size(size) for t in base_tasks]

            # Run RLM
            rlm_results = [rlm_executor_fn(t) for t in sized_tasks]
            rlm_correct = sum(1 for r in rlm_results if r.correct)
            rlm_accuracy = rlm_correct / len(rlm_results) if rlm_results else 0
            rlm_cost = sum(r.cost for r in rlm_results)
            results.rlm_accuracies.append(rlm_accuracy)
            results.rlm_costs.append(rlm_cost)

            # Run standard
            std_results = [standard_executor_fn(t) for t in sized_tasks]
            std_correct = sum(1 for r in std_results if r.correct)
            std_accuracy = std_correct / len(std_results) if std_results else 0
            std_cost = sum(r.cost for r in std_results)
            results.standard_accuracies.append(std_accuracy)
            results.standard_costs.append(std_cost)

        results.find_crossover()
        return results
