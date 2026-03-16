#!/usr/bin/env python3
"""Check statistical power for planned ablation comparisons."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.power_analysis import required_repetitions, achieved_power, minimum_detectable_effect


def main():
    print("=== Power Analysis for Ablation Studies ===\n")

    # Expected effect sizes based on mock data
    effect_sizes = {
        "full vs prose_only (neuro)": 6.5,
        "full vs random_search (SOAR)": 10.0,
        "full vs no_repl (RLM)": 12.5,
        "full vs no_self_mod (Godel)": 5.0,
        "full vs integrative (neuro)": 1.0,
    }

    for comparison, d in effect_sizes.items():
        n_needed = required_repetitions(d, alpha=0.05, power=0.80)
        print(f"{comparison}:")
        print(f"  Expected d = {d:.1f}")
        print(f"  Required n = {n_needed}")
        for n in [3, 5, 10]:
            pwr = achieved_power(d, n, alpha=0.05)
            print(f"  Power at n={n}: {pwr:.3f}")
        print()

    print("=== Minimum Detectable Effects ===\n")
    for n in [3, 5, 10, 20]:
        mde = minimum_detectable_effect(n, alpha=0.05, power=0.80)
        print(f"  n={n}: MDE = {mde:.3f}")


if __name__ == "__main__":
    main()
