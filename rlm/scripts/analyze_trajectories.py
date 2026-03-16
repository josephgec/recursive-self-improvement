#!/usr/bin/env python3
"""Analyze trajectories from benchmark runs."""

from __future__ import annotations

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.oolong import OOLONGBenchmark
from src.evaluation.runner import BenchmarkRunner
from src.strategies.trajectory_logger import TrajectoryLogger
from src.analysis.trajectory_analysis import efficiency_by_strategy


def main() -> None:
    from tests.conftest import MockLLM

    llm = MockLLM()
    bench = OOLONGBenchmark()
    tasks = bench.tasks[:5]
    runner = BenchmarkRunner()
    results = runner.run_tasks(tasks, llm, max_iterations=5)

    logger = TrajectoryLogger()
    for r in results:
        logger.log_session(r)

    trajectories = logger.export_all()
    print(f"Logged {len(trajectories)} trajectories")

    eff = efficiency_by_strategy(results)
    for strategy, info in eff.items():
        print(f"{strategy}: avg_iters={info['avg_iterations']:.1f}, n={info['count']}")


if __name__ == "__main__":
    main()
