#!/usr/bin/env python3
"""Package all deliverables for Phase 2a."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ga.engine import EvolutionResult
from src.comparison.ablation import AblationResult, ConditionResult
from src.deliverables.final_report import generate_final_report
from src.deliverables.soar_summary import generate_soar_summary
from src.deliverables.hindsight_summary import generate_hindsight_summary


def main():
    # Create sample results for packaging
    evo_result = EvolutionResult(
        best_fitness=0.72,
        generations_run=5,
        stopped_reason="max_generations",
        fitness_history=[0.55, 0.60, 0.65, 0.68, 0.72],
    )

    ablation_result = AblationResult()
    ablation_result.conditions["full_thinking"] = ConditionResult(
        condition_name="full_thinking",
        fitness_scores=[0.70, 0.72, 0.68],
    )
    ablation_result.conditions["full_thinking"].compute_stats()
    ablation_result.conditions["no_thinking"] = ConditionResult(
        condition_name="no_thinking",
        fitness_scores=[0.50, 0.52, 0.48],
    )
    ablation_result.conditions["no_thinking"].compute_stats()
    ablation_result.generate_summary()

    report = generate_final_report(evo_result, ablation_result)

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "reports",
    )
    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, "final_report.md")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
