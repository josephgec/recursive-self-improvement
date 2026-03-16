#!/usr/bin/env python3
"""Run benchmark suite on all ARC tasks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arc.loader import ARCLoader
from src.search.engine import EvolutionarySearchEngine, SearchConfig
from src.analysis.report import ReportGenerator
from src.analysis.task_difficulty import TaskDifficultyAnalyzer
from src.utils.logging import setup_logging


def main():
    setup_logging(level="INFO")

    loader = ARCLoader()
    tasks = loader.load_all()

    config = SearchConfig(
        population_size=10,
        max_generations=5,
        stagnation_limit=5,
    )

    results = {}
    difficulty_analyzer = TaskDifficultyAnalyzer()

    for task in tasks:
        print(f"Benchmarking task: {task.task_id}")
        engine = EvolutionarySearchEngine(config=config)
        result = engine.search(task)
        results[task.task_id] = result
        difficulty_analyzer.analyze_task(task, result)

    gen = ReportGenerator(difficulty_analyzer=difficulty_analyzer)
    report = gen.generate_full_report(results)
    print(gen.format_text_report(report))
    gen.save_report(report, "data/results/benchmark_results.json")
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
