"""SOAR summary generation for evolution results."""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.ga.engine import EvolutionResult
from src.comparison.ablation import AblationResult


def generate_soar_summary(
    evolution_result: Optional[EvolutionResult] = None,
    ablation_result: Optional[AblationResult] = None,
) -> str:
    """Generate a SOAR (Strengths, Opportunities, Aspirations, Results) summary.

    Args:
        evolution_result: Results from evolution run
        ablation_result: Results from ablation study

    Returns:
        Formatted SOAR summary string.
    """
    lines = ["# SOAR Summary", ""]

    # Strengths
    lines.append("## Strengths")
    if evolution_result and evolution_result.best_genome:
        lines.append(
            f"- Best fitness achieved: {evolution_result.best_fitness:.4f}"
        )
        lines.append(
            f"- Evolution ran for {evolution_result.generations_run} generations"
        )
        if evolution_result.stopped_reason == "optimal":
            lines.append("- Reached near-optimal fitness")
    lines.append(
        "- Thinking-model operators provide structured prompt optimization"
    )
    lines.append("")

    # Opportunities
    lines.append("## Opportunities")
    if ablation_result:
        ranking = ablation_result.get_ranking()
        if ranking:
            lines.append(f"- Best condition: {ranking[0]}")
            if len(ranking) > 1:
                lines.append(
                    f"- Room for improvement: {ranking[-1]} "
                    f"(mean={ablation_result.conditions[ranking[-1]].mean_fitness:.4f})"
                )
    lines.append("- Extend to additional domains beyond financial math")
    lines.append("- Explore larger population sizes and more generations")
    lines.append("")

    # Aspirations
    lines.append("## Aspirations")
    lines.append("- Achieve > 90% accuracy on financial math benchmarks")
    lines.append("- Demonstrate transfer learning of evolved prompts")
    lines.append("- Build a library of domain-specific evolved prompts")
    lines.append("")

    # Results
    lines.append("## Results")
    if evolution_result:
        lines.append(
            f"- Final best fitness: {evolution_result.best_fitness:.4f}"
        )
        if evolution_result.fitness_history:
            initial = evolution_result.fitness_history[0]
            final = evolution_result.fitness_history[-1]
            improvement = ((final - initial) / max(initial, 0.001)) * 100
            lines.append(f"- Fitness improvement: {improvement:.1f}%")
    if ablation_result:
        lines.append(f"- {len(ablation_result.conditions)} conditions tested")

    return "\n".join(lines)
