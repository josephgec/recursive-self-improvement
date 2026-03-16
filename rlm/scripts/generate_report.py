#!/usr/bin/env python3
"""Generate a comprehensive evaluation report."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.oolong import OOLONGBenchmark
from src.evaluation.runner import BenchmarkRunner
from src.analysis.report import generate_report


def main() -> None:
    from tests.conftest import MockLLM

    llm = MockLLM()
    bench = OOLONGBenchmark()
    tasks = bench.tasks[:10]
    runner = BenchmarkRunner()
    results = runner.run_tasks(tasks, llm, max_iterations=5)

    expected = [t.expected_answer for t in tasks]
    task_types = [t.category for t in tasks]

    report = generate_report(
        results=results,
        expected=expected,
        task_types=task_types,
        title="RLM Evaluation Report",
    )
    print(report)


if __name__ == "__main__":
    main()
