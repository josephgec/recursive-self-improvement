#!/usr/bin/env python3
"""Generate optimal pipeline configuration from experiment results."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments.modification_frequency import ModificationFrequencyExperiment
from src.experiments.hindsight_target import HindsightTargetExperiment
from src.experiments.rlm_depth import RLMDepthExperiment
from src.harness.runner import ExperimentRunner
from src.analysis.recommendation import RecommendationGenerator
from src.reporting.config_generator import ConfigGenerator


def main():
    config = {"iterations_per_condition": 20}
    experiments = [
        ModificationFrequencyExperiment(config),
        HindsightTargetExperiment(config),
        RLMDepthExperiment(config),
    ]

    runner = ExperimentRunner(seed=42)
    results = runner.run_all(experiments, repetitions=5)

    # Generate recommendation
    rec_gen = RecommendationGenerator()
    recommendation = rec_gen.generate(results)

    # Generate config
    config_gen = ConfigGenerator()
    yaml_config = config_gen.generate_optimal_config(recommendation)

    print("# Optimal Pipeline Configuration\n")
    print(yaml_config)

    # Save
    os.makedirs("data/reports", exist_ok=True)
    config_path = "data/reports/optimal_config.yaml"
    with open(config_path, "w") as f:
        f.write(yaml_config)
    print(f"Configuration saved to {config_path}")


if __name__ == "__main__":
    main()
