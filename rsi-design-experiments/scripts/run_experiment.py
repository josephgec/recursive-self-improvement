#!/usr/bin/env python3
"""Run a single experiment."""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments.modification_frequency import ModificationFrequencyExperiment
from src.experiments.hindsight_target import HindsightTargetExperiment
from src.experiments.rlm_depth import RLMDepthExperiment
from src.harness.runner import ExperimentRunner
from src.reporting.experiment_report import generate_experiment_report


EXPERIMENTS = {
    "modification_frequency": ModificationFrequencyExperiment,
    "hindsight_target": HindsightTargetExperiment,
    "rlm_depth": RLMDepthExperiment,
}


def main():
    parser = argparse.ArgumentParser(description="Run a single RSI design experiment")
    parser.add_argument(
        "--experiment",
        choices=list(EXPERIMENTS.keys()),
        required=True,
        help="Which experiment to run",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="Number of repetitions per condition (default: 5)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of iterations per condition (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    config = {"iterations_per_condition": args.iterations}
    experiment_class = EXPERIMENTS[args.experiment]
    experiment = experiment_class(config)

    runner = ExperimentRunner(seed=args.seed)
    result = runner.run_experiment(experiment, repetitions=args.repetitions)

    report = generate_experiment_report(result)
    print(report)

    # Save report
    os.makedirs("data/reports", exist_ok=True)
    report_path = f"data/reports/{args.experiment}_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
