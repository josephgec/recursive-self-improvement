#!/usr/bin/env python3
"""Run all benchmarks."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmarks.registry import create_default_registry
from src.execution.rlm_executor import RLMExecutor
from src.execution.runner import BenchmarkRunner


def main() -> None:
    """Run all benchmarks."""
    registry = create_default_registry()
    executor = RLMExecutor()
    runner = BenchmarkRunner(executor_fn=executor.execute)

    benchmark_tasks = {}
    for name in registry.available_benchmarks:
        benchmark_tasks[name] = registry.load(name)

    print(f"Running {len(benchmark_tasks)} benchmarks...")
    runs = runner.run_all_benchmarks(benchmark_tasks)

    for name, run in runs.items():
        print(f"\n{name}:")
        print(f"  Tasks: {run.total_tasks}")
        print(f"  Accuracy: {run.accuracy:.1%}")
        print(f"  Cost: ${run.total_cost:.4f}")


if __name__ == "__main__":
    main()
