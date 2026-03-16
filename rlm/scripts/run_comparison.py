#!/usr/bin/env python3
"""Run a comparison between two LLM approaches."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.oolong import OOLONGBenchmark
from src.evaluation.runner import BenchmarkRunner


def main() -> None:
    from tests.conftest import MockLLM

    llm_a = MockLLM()
    llm_b = MockLLM()

    bench = OOLONGBenchmark()
    tasks = bench.tasks[:5]
    runner = BenchmarkRunner()

    comparison = runner.run_comparison(
        tasks=tasks,
        llm_a=llm_a,
        llm_b=llm_b,
        benchmark_name="OOLONG",
    )
    print(comparison.summary())


if __name__ == "__main__":
    main()
