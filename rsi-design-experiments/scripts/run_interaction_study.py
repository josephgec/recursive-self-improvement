#!/usr/bin/env python3
"""Run interaction study between experimental factors."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments.modification_frequency import ModificationFrequencyExperiment
from src.experiments.hindsight_target import HindsightTargetExperiment
from src.experiments.rlm_depth import RLMDepthExperiment
from src.harness.runner import ExperimentRunner
from src.analysis.interaction_effects import InteractionAnalyzer


def main():
    config = {"iterations_per_condition": 20}
    experiments = [
        ModificationFrequencyExperiment(config),
        HindsightTargetExperiment(config),
        RLMDepthExperiment(config),
    ]

    runner = ExperimentRunner(seed=42)
    results = runner.run_all(experiments, repetitions=5)

    # Collect composite scores
    all_scores = {}
    for result in results:
        all_scores[result.experiment_name] = result.get_all_scores("composite_score")

    # Analyze interactions
    analyzer = InteractionAnalyzer()
    interactions = analyzer.detect_interactions(all_scores)

    print("# Interaction Study Results\n")
    for ir in interactions:
        print(f"## {ir.factor_a_name} x {ir.factor_b_name}")
        print(f"  Main effect A: F={ir.main_effect_a.f_statistic:.4f}, "
              f"p={ir.main_effect_a.p_value:.4f}, "
              f"significant={ir.main_effect_a.significant}")
        print(f"  Main effect B: F={ir.main_effect_b.f_statistic:.4f}, "
              f"p={ir.main_effect_b.p_value:.4f}, "
              f"significant={ir.main_effect_b.significant}")
        print(f"  Interaction: significant={ir.interaction_significant}, "
              f"strength={ir.interaction_strength:.4f}")
        print()


if __name__ == "__main__":
    main()
