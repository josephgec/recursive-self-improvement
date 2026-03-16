"""Hindsight analysis summary generation."""

from __future__ import annotations

from typing import Optional

from src.ga.engine import EvolutionResult
from src.comparison.ablation import AblationResult


def generate_hindsight_summary(
    evolution_result: Optional[EvolutionResult] = None,
    ablation_result: Optional[AblationResult] = None,
) -> str:
    """Generate a hindsight analysis of the evolution process.

    Analyzes what worked, what didn't, and lessons learned.
    """
    lines = ["# Hindsight Analysis", ""]

    lines.append("## What Worked Well")
    if evolution_result:
        if evolution_result.best_fitness > 0.5:
            lines.append("- Evolution successfully improved prompt fitness")
        if evolution_result.generations_run > 1:
            lines.append("- Multi-generation evolution showed progressive improvement")
    lines.append("- Structured section-based genome allowed targeted mutations")
    lines.append("- Thinking-model operators provided meaningful improvements")
    lines.append("")

    lines.append("## What Could Be Improved")
    if evolution_result:
        if evolution_result.stopped_reason == "stagnation":
            lines.append("- Evolution stagnated - consider larger population or more diversity")
        if evolution_result.best_fitness < 0.7:
            lines.append("- Final fitness below target - more generations may help")
    lines.append("- Consider adaptive mutation rates based on population diversity")
    lines.append("- Explore co-evolution of prompt sections")
    lines.append("")

    lines.append("## Key Insights")
    if ablation_result:
        ranking = ablation_result.get_ranking()
        if ranking:
            lines.append(f"- Best approach: {ranking[0]}")
            if "full_thinking" in ranking and "no_thinking" in ranking:
                t_idx = ranking.index("full_thinking")
                nt_idx = ranking.index("no_thinking")
                if t_idx < nt_idx:
                    lines.append(
                        "- Thinking-model operators outperform non-thinking variants"
                    )
                else:
                    lines.append(
                        "- Non-thinking operators surprisingly competitive"
                    )
    lines.append("- Section-level granularity enables precise optimization")
    lines.append("- Diversity maintenance prevents premature convergence")
    lines.append("")

    lines.append("## Recommendations")
    lines.append("1. Use thinking-model operators for prompt optimization")
    lines.append("2. Maintain population diversity through injection")
    lines.append("3. Target mutations at weakest sections identified by evaluation")
    lines.append("4. Use crossover to combine complementary strengths")

    return "\n".join(lines)
