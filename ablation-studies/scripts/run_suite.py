#!/usr/bin/env python3
"""Run a single ablation suite."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.suites import NeurosymbolicAblation, GodelAgentAblation, SOARAblation, RLMAblation
from src.execution.runner import AblationRunner


SUITES = {
    "neurosymbolic": NeurosymbolicAblation,
    "godel": GodelAgentAblation,
    "soar": SOARAblation,
    "rlm": RLMAblation,
}


def main():
    suite_name = sys.argv[1] if len(sys.argv) > 1 else "neurosymbolic"
    repetitions = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42

    if suite_name not in SUITES:
        print(f"Unknown suite: {suite_name}. Available: {list(SUITES.keys())}")
        sys.exit(1)

    suite = SUITES[suite_name]()
    runner = AblationRunner()

    print(f"Running {suite_name} ablation suite ({repetitions} reps, seed={seed})")
    result = runner.run_suite(suite, repetitions=repetitions, seed=seed)

    for cond in result.get_all_condition_names():
        mean = result.get_mean_score(cond)
        print(f"  {cond}: {mean:.4f}")

    print(f"\nBest condition: {result.best_condition()}")


if __name__ == "__main__":
    main()
