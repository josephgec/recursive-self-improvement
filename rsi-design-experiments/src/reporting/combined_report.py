"""Generate combined markdown report across all experiments."""

from typing import List

from src.experiments.base import ExperimentResult
from src.analysis.sensitivity import SensitivityAnalyzer
from src.analysis.interaction_effects import InteractionAnalyzer
from src.analysis.recommendation import RecommendationGenerator
from src.reporting.experiment_report import generate_experiment_report
from src.reporting.config_generator import ConfigGenerator


def generate_combined_report(
    all_results: List[ExperimentResult],
    significance_level: float = 0.05,
) -> str:
    """Generate a combined markdown report for all experiments.

    Includes:
    - Executive summary
    - Sensitivity ranking
    - Interaction effects
    - Individual experiment reports
    - Final recommended configuration
    """
    lines = []
    lines.append("# RSI Design Experiments: Combined Report")
    lines.append("")

    # Executive summary
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"Total experiments: {len(all_results)}")
    for result in all_results:
        lines.append(
            f"- {result.experiment_name}: {len(result.conditions)} conditions, "
            f"{result.repetitions} repetitions"
        )
    lines.append("")

    # Sensitivity ranking
    lines.append("## Sensitivity Ranking")
    lines.append("")
    lines.append("Which design decisions matter most?")
    lines.append("")

    sensitivity = SensitivityAnalyzer()
    rankings = sensitivity.rank_experiments(all_results)
    for i, sr in enumerate(rankings, 1):
        lines.append(
            f"{i}. **{sr.experiment_name}** - sensitivity: {sr.sensitivity:.4f} "
            f"(range: {sr.min_value:.4f} - {sr.max_value:.4f})"
        )
    lines.append("")

    # Interaction effects
    lines.append("## Interaction Effects")
    lines.append("")

    if len(all_results) >= 2:
        interaction_analyzer = InteractionAnalyzer(significance_level)
        all_scores = {
            r.experiment_name: r.get_all_scores("composite_score")
            for r in all_results
        }
        interactions = interaction_analyzer.detect_interactions(all_scores)

        for ir in interactions:
            status = "SIGNIFICANT" if ir.interaction_significant else "not significant"
            lines.append(
                f"- {ir.factor_a_name} x {ir.factor_b_name}: "
                f"{status} (strength: {ir.interaction_strength:.4f})"
            )
        lines.append("")
    else:
        lines.append("Not enough experiments for interaction analysis.")
        lines.append("")

    # Individual experiment reports
    lines.append("## Individual Experiment Details")
    lines.append("")
    for result in all_results:
        report = generate_experiment_report(result, significance_level)
        lines.append(report)
        lines.append("---")
        lines.append("")

    # Final configuration
    lines.append("## Recommended Configuration")
    lines.append("")

    rec_gen = RecommendationGenerator()
    recommendation = rec_gen.generate(all_results)

    lines.append("### Settings")
    lines.append("")
    if recommendation.modification_frequency:
        lines.append(
            f"- Modification frequency: **{recommendation.modification_frequency}** "
            f"(confidence: {recommendation.confidence_levels.get('modification_frequency', 'unknown')})"
        )
    if recommendation.hindsight_target:
        lines.append(
            f"- Hindsight target: **{recommendation.hindsight_target}** "
            f"(confidence: {recommendation.confidence_levels.get('hindsight_target', 'unknown')})"
        )
    if recommendation.rlm_depth is not None:
        lines.append(
            f"- RLM depth: **{recommendation.rlm_depth}** "
            f"(confidence: {recommendation.confidence_levels.get('rlm_depth', 'unknown')})"
        )
    lines.append("")

    # Export config
    lines.append("### YAML Configuration")
    lines.append("")
    config_gen = ConfigGenerator()
    yaml_config = config_gen.generate_optimal_config(recommendation)
    lines.append("```yaml")
    lines.append(yaml_config)
    lines.append("```")
    lines.append("")

    return "\n".join(lines)
