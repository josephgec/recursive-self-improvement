"""Final report generation packaging Phase 2a deliverables."""

from __future__ import annotations

from typing import Optional

from src.ga.engine import EvolutionResult
from src.comparison.ablation import AblationResult
from src.deliverables.soar_summary import generate_soar_summary
from src.deliverables.hindsight_summary import generate_hindsight_summary


def generate_final_report(
    evolution_result: Optional[EvolutionResult] = None,
    ablation_result: Optional[AblationResult] = None,
) -> str:
    """Generate the complete final report packaging all deliverables.

    Combines SOAR summary, hindsight analysis, and technical details.
    """
    lines = ["# Prompt Evolution - Final Report", ""]
    lines.append("## Executive Summary")
    lines.append("")

    if evolution_result:
        lines.append(
            f"Evolution completed {evolution_result.generations_run} generations "
            f"achieving best fitness of {evolution_result.best_fitness:.4f}."
        )
        lines.append(f"Stopped due to: {evolution_result.stopped_reason}")
    lines.append("")

    if ablation_result:
        lines.append(f"Ablation study tested {len(ablation_result.conditions)} conditions.")
        ranking = ablation_result.get_ranking()
        if ranking:
            lines.append(f"Best condition: {ranking[0]}")
    lines.append("")

    # Best Evolved Prompt
    lines.append("## Best Evolved Prompt")
    if evolution_result and evolution_result.best_genome:
        lines.append("```")
        lines.append(evolution_result.best_genome.to_system_prompt())
        lines.append("```")
    else:
        lines.append("No evolved prompt available.")
    lines.append("")

    # Evolution Trajectory
    lines.append("## Evolution Trajectory")
    if evolution_result and evolution_result.fitness_history:
        for i, fitness in enumerate(evolution_result.fitness_history):
            lines.append(f"- Generation {i + 1}: {fitness:.4f}")
    lines.append("")

    # SOAR Summary
    soar = generate_soar_summary(evolution_result, ablation_result)
    lines.append(soar)
    lines.append("")

    # Hindsight Analysis
    hindsight = generate_hindsight_summary(evolution_result, ablation_result)
    lines.append(hindsight)

    return "\n".join(lines)
