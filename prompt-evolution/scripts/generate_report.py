#!/usr/bin/env python3
"""Generate comprehensive analysis report."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ga.engine import EvolutionResult, GenerationResult
from src.comparison.ablation import AblationResult, ConditionResult
from src.analysis.report import generate_report


def main():
    # Create sample results
    gen_results = [
        GenerationResult(
            generation=i + 1,
            best_fitness=0.5 + i * 0.05,
            avg_fitness=0.4 + i * 0.04,
            diversity=0.8 - i * 0.05,
            best_genome_id=f"gen_{i}",
            num_mutations=3,
            num_crossovers=2,
            num_elites=1,
        )
        for i in range(5)
    ]

    evo_result = EvolutionResult(
        best_fitness=0.75,
        generations_run=5,
        stopped_reason="max_generations",
        fitness_history=[0.5, 0.55, 0.60, 0.65, 0.75],
        generation_results=gen_results,
    )

    ablation_result = AblationResult()
    ablation_result.conditions["full_thinking"] = ConditionResult(
        condition_name="full_thinking",
        fitness_scores=[0.70, 0.72, 0.68],
    )
    ablation_result.conditions["full_thinking"].compute_stats()
    ablation_result.generate_summary()

    report = generate_report(evo_result, ablation_result)

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "reports",
    )
    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, "analysis_report.md")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
