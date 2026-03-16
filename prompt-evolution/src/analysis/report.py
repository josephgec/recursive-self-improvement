"""Comprehensive report generation combining all analyses."""

from __future__ import annotations

from typing import Dict, List, Optional

from src.ga.engine import EvolutionResult
from src.comparison.ablation import AblationResult
from src.analysis.evolution_dynamics import (
    fitness_trajectory,
    operator_contribution,
    convergence_speed,
    plot_fitness_ascii,
)
from src.deliverables.final_report import generate_final_report


def generate_report(
    evolution_result: Optional[EvolutionResult] = None,
    ablation_result: Optional[AblationResult] = None,
) -> str:
    """Generate comprehensive markdown report.

    Combines evolution dynamics, ablation results, and deliverables.
    """
    lines = []

    # Main report
    lines.append(generate_final_report(evolution_result, ablation_result))
    lines.append("")

    # Detailed evolution dynamics
    if evolution_result:
        lines.append("# Detailed Evolution Dynamics")
        lines.append("")

        # Fitness plot
        lines.append("## Fitness Trajectory")
        lines.append("```")
        lines.append(plot_fitness_ascii(evolution_result))
        lines.append("```")
        lines.append("")

        # Operator contributions
        lines.append("## Operator Contributions")
        contrib = operator_contribution(evolution_result)
        for op_type, stats in contrib.items():
            lines.append(
                f"- **{op_type}**: {stats['count']} applications, "
                f"avg {stats['avg_per_generation']:.1f}/generation"
            )
        lines.append("")

        # Convergence
        speed = convergence_speed(evolution_result, threshold=0.8)
        if speed is not None:
            lines.append(f"Reached 80% fitness at generation {speed}")
        else:
            lines.append("Did not reach 80% fitness threshold")
        lines.append("")

    # Ablation details
    if ablation_result:
        lines.append("# Ablation Study Details")
        lines.append("")
        lines.append(ablation_result.summary)
        lines.append("")

    return "\n".join(lines)
