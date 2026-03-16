#!/usr/bin/env python3
"""Run all ablation suites."""

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
    all_results = runner.run_all_suites(suites, repetitions=repetitions, seed=seed)

    for suite_name, result in all_results.items():
        print(f"\n=== {suite_name} ===")
        for cond in result.get_all_condition_names():
            mean = result.get_mean_score(cond)
            print(f"  {cond}: {mean:.4f}")
        print(f"  Best: {result.best_condition()}")


if __name__ == "__main__":
    main()
