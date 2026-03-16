#!/usr/bin/env python3
"""Run evolutionary search on a single ARC task."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arc.loader import ARCLoader
from src.search.engine import EvolutionarySearchEngine, SearchConfig
from src.analysis.report import ReportGenerator
from src.utils.logging import setup_logging


def main():
    setup_logging(level="DEBUG")

    task_id = sys.argv[1] if len(sys.argv) > 1 else "simple_color_swap"

    loader = ARCLoader()
    task = loader.load_task(task_id)
    print(f"Task: {task_id} ({task.num_train} train, {task.num_test} test)")

    config = SearchConfig(
        population_size=10,
        max_generations=10,
        stagnation_limit=5,
    )

    engine = EvolutionarySearchEngine(config=config)
    result = engine.search(task)

    print(result.summary())
    if result.best_individual:
        print(f"\nBest program:\n{result.best_individual.code}")

    gen = ReportGenerator()
    report = gen.generate_search_report(result, task_id=task_id)
    gen.save_report(report, f"data/results/task_{task_id}.json")


if __name__ == "__main__":
    main()
