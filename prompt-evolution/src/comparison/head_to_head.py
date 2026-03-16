"""Head-to-head comparison analysis."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.comparison.ablation import AblationResult, ConditionResult
from src.comparison.statistical_tests import StatisticalComparator, PairwiseTest


class HeadToHeadAnalyzer:
    """Analyze head-to-head comparisons between conditions."""

    def __init__(self):
        self.comparator = StatisticalComparator()

    def compare_conditions(
        self,
        ablation_result: AblationResult,
    ) -> Dict[Tuple[str, str], PairwiseTest]:
        """Run pairwise comparisons between all conditions.

        Returns:
            Dict mapping (condition_a, condition_b) to PairwiseTest results.
        """
        results = {}
        conditions = list(ablation_result.conditions.keys())

        for i, cond_a in enumerate(conditions):
            for cond_b in conditions[i + 1 :]:
                scores_a = ablation_result.conditions[cond_a].fitness_scores
                scores_b = ablation_result.conditions[cond_b].fitness_scores

                test = self.comparator.pairwise_test(
                    scores_a,
                    scores_b,
                    condition_a=cond_a,
                    condition_b=cond_b,
                )
                results[(cond_a, cond_b)] = test

        return results

    def plot_comparison(
        self,
        ablation_result: AblationResult,
    ) -> str:
        """Generate ASCII comparison chart.

        Returns a text-based bar chart of mean fitness by condition.
        """
        ranking = ablation_result.get_ranking()
        max_fitness = max(
            cr.mean_fitness
            for cr in ablation_result.conditions.values()
        ) if ablation_result.conditions else 1.0

        lines = ["Condition Comparison", "-" * 60]
        bar_width = 40

        for cond in ranking:
            cr = ablation_result.conditions[cond]
            if max_fitness > 0:
                bar_len = int((cr.mean_fitness / max_fitness) * bar_width)
            else:
                bar_len = 0
            bar = "#" * bar_len
            lines.append(
                f"{cond:25s} |{bar:<{bar_width}s}| "
                f"{cr.mean_fitness:.4f} (+/-{cr.std_fitness:.4f})"
            )

        return "\n".join(lines)

    def generate_ranking_table(
        self,
        ablation_result: AblationResult,
        pairwise_results: Optional[Dict[Tuple[str, str], PairwiseTest]] = None,
    ) -> str:
        """Generate a ranking table with pairwise significance markers.

        Returns a formatted markdown table.
        """
        ranking = ablation_result.get_ranking()

        lines = [
            "| Rank | Condition | Mean Fitness | Std | Best |",
            "|------|-----------|-------------|-----|------|",
        ]

        for i, cond in enumerate(ranking, 1):
            cr = ablation_result.conditions[cond]
            lines.append(
                f"| {i} | {cond} | {cr.mean_fitness:.4f} | "
                f"{cr.std_fitness:.4f} | {cr.best_fitness:.4f} |"
            )

        if pairwise_results:
            lines.append("")
            lines.append("Significant pairwise differences (p < 0.05):")
            for (ca, cb), test in pairwise_results.items():
                if test.significant:
                    better = ca if test.mean_a > test.mean_b else cb
                    lines.append(
                        f"  - {better} > {ca if better == cb else cb} "
                        f"(p={test.p_value:.4f}, d={test.effect_size:.3f})"
                    )

        return "\n".join(lines)
