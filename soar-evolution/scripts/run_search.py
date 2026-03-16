#!/usr/bin/env python3
"""Run evolutionary search on all built-in ARC tasks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arc.loader import ARCLoader
from src.search.engine import EvolutionarySearchEngine, SearchConfig
from src.search.parallel import ParallelTaskSearch
from src.analysis.report import ReportGenerator
from src.utils.logging import setup_logging


def main():
    setup_logging(level="INFO")

    loader = ARCLoader()
    tasks = loader.load_all()

    config = SearchConfig(
        population_size=10,
        max_generations=10,
        stagnation_limit=5,
    )

    searcher = ParallelTaskSearch(config=config, max_workers=1)
    result = searcher.search_all(tasks)

    print(result.summary())

    gen = ReportGenerator()
    report = gen.generate_benchmark_report(result.results)
    print(gen.format_text_report(report))

    gen.save_report(report, "data/results/search_results.json")
    print("Results saved to data/results/search_results.json")


if __name__ == "__main__":
    main()
