#!/usr/bin/env python3
"""Run context scaling experiment."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmarks.synthetic import SyntheticTaskGenerator
from src.execution.rlm_executor import RLMExecutor
from src.execution.standard_executor import StandardExecutor
from src.comparison.scaling_experiment import ContextScalingExperiment


def main() -> None:
    """Run scaling experiment."""
    gen = SyntheticTaskGenerator()
    base_tasks = gen.needle_in_haystack(context_tokens=1000, num_tasks=3)

    rlm = RLMExecutor()
    std = StandardExecutor(context_window=4096)

    sizes = [1000, 2000, 4000, 8000, 16000, 32000]
    experiment = ContextScalingExperiment(rlm.execute, std.execute)

    print("Running scaling experiment...")
    results = experiment.run(base_tasks, sizes)

    print("\n" + experiment.plot_scaling_curves(results))

    crossover = experiment.find_crossover_point(results)
    if crossover:
        print(f"\nCrossover point: {crossover} tokens")


if __name__ == "__main__":
    main()
