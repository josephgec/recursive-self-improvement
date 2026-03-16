"""LaTeX table generation for publication."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from src.suites.base import AblationSuiteResult
from src.analysis.statistical_tests import PairwiseResult
from src.publication.significance_stars import add_stars


class LaTeXTableGenerator:
    """Generate publication-quality LaTeX tables with booktabs style."""

    def main_results_table(
        self,
        result: AblationSuiteResult,
        caption: Optional[str] = None,
        label: Optional[str] = None,
    ) -> str:
        """Generate the main results table showing all conditions.

        Formats with \\toprule/\\midrule/\\bottomrule, bold best result.
        """
        conditions = result.get_all_condition_names()
        if not conditions:
            return ""

        # Find best condition
        best_name = result.best_condition()

        if caption is None:
            caption = f"Ablation results for {result.suite_name}"
        if label is None:
            safe_name = result.suite_name.replace(" ", "_").lower()
            label = f"tab:{safe_name}_results"

        lines = []
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append(f"\\caption{{{caption}}}")
        lines.append(f"\\label{{{label}}}")
        lines.append("\\begin{tabular}{lcc}")
        lines.append("\\toprule")
        lines.append("Condition & Mean Accuracy & Std Dev \\\\")
        lines.append("\\midrule")

        for cond in conditions:
            scores = result.get_scores(cond)
            if not scores:
                continue

            mean_score = sum(scores) / len(scores)
            if len(scores) > 1:
                variance = sum((x - mean_score) ** 2 for x in scores) / (len(scores) - 1)
                std_score = math.sqrt(variance)
            else:
                std_score = 0.0

            # Format condition name
            display_name = cond.replace("_", "\\_")

            # Bold the best
            if cond == best_name:
                lines.append(
                    f"\\textbf{{{display_name}}} & "
                    f"\\textbf{{{mean_score:.3f}}} & "
                    f"\\textbf{{{std_score:.3f}}} \\\\"
                )
            else:
                lines.append(
                    f"{display_name} & {mean_score:.3f} & {std_score:.3f} \\\\"
                )

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

        return "\n".join(lines)

    def pairwise_comparison_table(
        self,
        comparisons: List[PairwiseResult],
        caption: Optional[str] = None,
        label: Optional[str] = None,
    ) -> str:
        """Generate a pairwise comparison table with significance stars.

        Includes difference, CI, effect size, and significance stars.
        """
        if not comparisons:
            return ""

        if caption is None:
            caption = "Pairwise comparisons"
        if label is None:
            label = "tab:pairwise"

        lines = []
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append(f"\\caption{{{caption}}}")
        lines.append(f"\\label{{{label}}}")
        lines.append("\\begin{tabular}{llccccc}")
        lines.append("\\toprule")
        lines.append(
            "A & B & $\\Delta$ & 95\\% CI & $d$ & $p$ & Sig. \\\\"
        )
        lines.append("\\midrule")

        for comp in comparisons:
            a_name = comp.condition_a.replace("_", "\\_")
            b_name = comp.condition_b.replace("_", "\\_")
            ci_str = f"[{comp.ci_lower:.3f}, {comp.ci_upper:.3f}]"
            p_str = f"{comp.p_value:.4f}" if comp.p_value >= 0.0001 else "$<$0.0001"
            stars = comp.stars

            lines.append(
                f"{a_name} & {b_name} & {comp.difference:.3f} & "
                f"{ci_str} & {comp.effect_size:.2f} & {p_str} & {stars} \\\\"
            )

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

        return "\n".join(lines)

    def condition_detail_table(
        self,
        result: AblationSuiteResult,
        condition_name: str,
    ) -> str:
        """Generate a detailed table for a single condition showing per-run scores."""
        scores = result.get_scores(condition_name)
        if not scores:
            return ""

        display_name = condition_name.replace("_", "\\_")
        lines = []
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append(f"\\caption{{Per-run scores for {display_name}}}")
        lines.append("\\begin{tabular}{cc}")
        lines.append("\\toprule")
        lines.append("Run & Accuracy \\\\")
        lines.append("\\midrule")

        for i, score in enumerate(scores):
            lines.append(f"{i + 1} & {score:.4f} \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

        return "\n".join(lines)
