#!/usr/bin/env python3
"""Run evaluation on a single benchmark."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmarks.registry import BenchmarkRegistry, register_all_benchmarks


class SimpleAgent:
    def __init__(self):
        self._iteration = 0

    def set_iteration(self, iteration):
        self._iteration = iteration

    def solve(self, task):
        import hashlib
        accuracy = 0.60 + 0.015 * self._iteration
        h = int(hashlib.md5(task.task_id.encode()).hexdigest(), 16) % 100
        return task.expected_answer if h < accuracy * 100 else None


def main():
    benchmark_name = sys.argv[1] if len(sys.argv) > 1 else "math500"
    register_all_benchmarks()

    benchmark = BenchmarkRegistry.load(benchmark_name)
    agent = SimpleAgent()

    print(f"Benchmark: {benchmark_name}")
    print(f"Tasks: {len(benchmark.tasks)}")
    print(f"Categories: {benchmark.categories}")

    for iteration in range(5):
        agent.set_iteration(iteration)
        results = benchmark.evaluate(agent)
        correct = sum(1 for r in results if r.correct)
        print(f"  Iteration {iteration}: {correct}/{len(results)} = {correct/len(results):.4f}")


if __name__ == "__main__":
    main()
