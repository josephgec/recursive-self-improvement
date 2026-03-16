#!/usr/bin/env python3
"""Analyze search results from a previous run."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.search_dynamics import SearchDynamicsAnalyzer, GenerationSnapshot
from src.utils.logging import setup_logging


def main():
    setup_logging(level="INFO")

    results_file = sys.argv[1] if len(sys.argv) > 1 else "data/results/search_results.json"
    results_path = Path(results_file)

    if not results_path.exists():
        print(f"No results file found at {results_path}")
        print("Run `make run-search` first.")
        return

    with open(results_path) as f:
        data = json.load(f)

    print(f"Tasks: {data.get('num_tasks', 'N/A')}")
    print(f"Solved: {data.get('num_solved', 'N/A')}")
    print(f"Solve rate: {data.get('solve_rate', 0):.1%}")
    print(f"Total time: {data.get('total_time', 0):.1f}s")

    if "tasks" in data:
        for tid, tdata in data["tasks"].items():
            status = "SOLVED" if tdata.get("solved") else "UNSOLVED"
            print(f"  {tid}: {status} (fitness={tdata.get('best_fitness', 0):.4f})")


if __name__ == "__main__":
    main()
