#!/usr/bin/env python3
"""Run a single benchmark evaluation."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmarks.registry import create_default_registry
from src.execution.rlm_executor import RLMExecutor
from src.execution.runner import BenchmarkRunner


def main(benchmark_name: str = "oolong") -> None:
    """Run a single benchmark."""
    registry = create_default_registry()
    tasks = registry.load(benchmark_name)
    executor = RLMExecutor()
    runner = BenchmarkRunner(executor_fn=executor.execute)

    print(f"Running {benchmark_name} benchmark with {len(tasks)} tasks...")
    run = runner.run_benchmark(tasks, benchmark_name)
    run.compute_stats()

    print(f"\nResults:")
    print(f"  Tasks: {run.total_tasks}")
    print(f"  Correct: {run.correct_count}")
    print(f"  Accuracy: {run.accuracy:.1%}")
    print(f"  Total cost: ${run.total_cost:.4f}")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "oolong"
    main(name)
