"""Generate markdown reports for individual experiments."""

from typing import Optional

from src.experiments.base import ExperimentResult
from src.analysis.anova import ANOVAAnalyzer
from src.analysis.optimal_finder import OptimalFinder


def generate_experiment_report(
    result: ExperimentResult,
    significance_level: float = 0.05,
) -> str:
    """Generate a markdown report for a single experiment.

    Includes:
    - Conditions table with mean metrics
    - ANOVA results
    - Accuracy trajectories summary
    - Recommendation
    """
    lines = []
    lines.append(f"# Experiment: {result.experiment_name}")
    lines.append("")
    lines.append(f"Repetitions: {result.repetitions}")
    lines.append(f"Conditions: {len(result.conditions)}")
    lines.append("")

    # Conditions table
    lines.append("## Conditions Summary")
    lines.append("")
    lines.append(
        "| Condition | Mean Accuracy | Mean Stability | Mean Cost | "
        "Mean Composite |"
    )
    lines.append(
        "|-----------|--------------|----------------|-----------|"
        "----------------|"
    )

    for cond_name in result.conditions:
        cond_results = result.per_condition_results.get(cond_name, [])
        if not cond_results:
            continue
        n = len(cond_results)
        mean_acc = sum(r.final_accuracy for r in cond_results) / n
        mean_stab = sum(r.stability_score for r in cond_results) / n
        mean_cost = sum(r.total_cost for r in cond_results) / n
        mean_comp = sum(r.composite_score for r in cond_results) / n
        lines.append(
            f"| {cond_name} | {mean_acc:.4f} | {mean_stab:.4f} | "
            f"{mean_cost:.4f} | {mean_comp:.4f} |"
        )
    lines.append("")

    # ANOVA
    lines.append("## ANOVA Analysis")
    lines.append("")

    anova = ANOVAAnalyzer(significance_level)
    scores = result.get_all_scores("composite_score")
    anova_result = anova.one_way_anova(scores)

    lines.append(f"- F-statistic: {anova_result.f_statistic:.4f}")
    lines.append(f"- p-value: {anova_result.p_value:.4f}")
    lines.append(f"- Significant: {anova_result.significant}")
    lines.append(f"- Eta-squared: {anova_result.eta_squared:.4f}")
    lines.append("")

    # Tukey HSD
    if anova_result.significant:
        lines.append("### Tukey HSD Pairwise Comparisons")
        lines.append("")
        tukey_results = anova.tukey_hsd(scores)
        sig_pairs = [t for t in tukey_results if t.significant]
        if sig_pairs:
            for t in sig_pairs:
                lines.append(
                    f"- {t.group_a} vs {t.group_b}: "
                    f"diff={t.mean_diff:.4f}, q={t.q_statistic:.4f} *"
                )
        else:
            lines.append("No significant pairwise differences found.")
        lines.append("")

    # Trajectories summary
    lines.append("## Accuracy Trajectories")
    lines.append("")
    for cond_name in result.conditions:
        cond_results = result.per_condition_results.get(cond_name, [])
        if cond_results and cond_results[0].accuracy_trajectory:
            traj = cond_results[0].accuracy_trajectory
            lines.append(
                f"- {cond_name}: start={traj[0]:.4f}, "
                f"end={traj[-1]:.4f}, len={len(traj)}"
            )
    lines.append("")

    # Recommendation
    lines.append("## Recommendation")
    lines.append("")
    finder = OptimalFinder()
    best = finder.find_best_condition(result, "composite_score")
    lines.append(f"**Best condition: {best.best_condition}** (score: {best.best_score:.4f})")
    lines.append("")

    return "\n".join(lines)
