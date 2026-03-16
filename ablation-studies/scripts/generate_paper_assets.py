#!/usr/bin/env python3
"""Generate paper-ready assets (tables, figures, narrative) for all suites."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.suites import NeurosymbolicAblation, GodelAgentAblation, SOARAblation, RLMAblation
from src.execution.runner import AblationRunner


def main():
    repetitions = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    suites = [
        NeurosymbolicAblation(),
        GodelAgentAblation(),
        SOARAblation(),
        RLMAblation(),
    ]

    runner = AblationRunner()

    for suite in suites:
        print(f"\n=== Generating assets for {suite.get_paper_name()} ===")
        result = runner.run_suite(suite, repetitions=repetitions, seed=seed)
        assets = suite.generate_paper_assets(result)

        print("\n--- Main Results Table ---")
        print(assets.tables.get("main_results", "(none)"))

        print("\n--- Pairwise Comparisons ---")
        print(assets.tables.get("pairwise", "(none)"))

        print("\n--- Narrative ---")
        print(assets.narrative)


if __name__ == "__main__":
    main()
