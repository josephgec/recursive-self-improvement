#!/usr/bin/env python3
"""Run all experiments and generate combined report."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments.modification_frequency import ModificationFrequencyExperiment
from src.experiments.hindsight_target import HindsightTargetExperiment
from src.experiments.rlm_depth import RLMDepthExperiment
from src.harness.runner import ExperimentRunner
from src.reporting.combined_report import generate_combined_report


def main():
    config = {"iterations_per_condition": 20}
    experiments = [
        ModificationFrequencyExperiment(config),
        HindsightTargetExperiment(config),
        RLMDepthExperiment(config),
    ]

    runner = ExperimentRunner(seed=42)
    results = runner.run_all(experiments, repetitions=5)

    report = generate_combined_report(results)
    print(report)

    os.makedirs("data/reports", exist_ok=True)
    report_path = "data/reports/combined_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
