#!/usr/bin/env python3
"""Run benchmarks against mock LLMs."""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.oolong import OOLONGBenchmark
from src.evaluation.locodiff import LoCoDiffBenchmark
from src.evaluation.runner import BenchmarkRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmarks")
    parser.add_argument(
        "--benchmark", choices=["oolong", "locodiff", "all"], default="all"
    )
    parser.add_argument("--max-tasks", type=int, default=5)
    args = parser.parse_args()

    from tests.conftest import MockLLM
    llm = MockLLM()
    runner = BenchmarkRunner()

    if args.benchmark in ("oolong", "all"):
        bench = OOLONGBenchmark()
        tasks = bench.tasks[:args.max_tasks]
        results = runner.run_tasks(tasks, llm, max_iterations=5)
        expected = [t.expected_answer for t in tasks]
        acc = runner.metrics.accuracy(results, expected, exact=False)
        print(f"OOLONG accuracy: {acc.value:.2%}")

    if args.benchmark in ("locodiff", "all"):
        bench = LoCoDiffBenchmark()
        tasks = bench.tasks[:args.max_tasks]
        results = runner.run_tasks(tasks, llm, max_iterations=5)
        expected = [t.expected_answer for t in tasks]
        acc = runner.metrics.accuracy(results, expected, exact=False)
        print(f"LoCoDiff accuracy: {acc.value:.2%}")


if __name__ == "__main__":
    main()
