"""Narrative text generation for results sections."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from src.suites.base import AblationSuiteResult
from src.analysis.statistical_tests import PairwiseResult


class NarrativeGenerator:
    """Generate publication-ready narrative text for results sections."""

    def generate_results_section(
        self,
        result: AblationSuiteResult,
        analyses: Dict[Any, PairwiseResult],
        paper_name: str,
    ) -> str:
        """Generate a complete results section paragraph."""
        paragraphs = []

        # Opening sentence
        n_conditions = len(result.get_all_condition_names())
        best = result.best_condition()
        best_score = result.get_mean_score(best)
        paragraphs.append(
            f"We evaluated {n_conditions} conditions for the {paper_name} approach. "
            f"The {best} configuration achieved the highest accuracy "
            f"({best_score:.3f})."
        )

        # Key comparisons
        comparison_sentences = []
        for key, pw_result in analyses.items():
            sentence = self.format_comparison_sentence(pw_result)
            comparison_sentences.append(sentence)

        if comparison_sentences:
            paragraphs.append(" ".join(comparison_sentences))

        # Summary
        significant_count = sum(
            1 for pw in analyses.values() if pw.significant
        )
        paragraphs.append(
            f"Of {len(analyses)} key comparisons, {significant_count} "
            f"reached statistical significance (p < 0.05)."
        )

        return "\n\n".join(paragraphs)

    def format_comparison_sentence(self, pw: PairwiseResult) -> str:
        """Format a single comparison as a sentence with inline statistics.

        Example: "The full condition outperformed prose_only by 0.130
        (95% CI [0.098, 0.162], p < 0.001, d = 6.50)."
        """
        a = pw.condition_a
        b = pw.condition_b
        diff = pw.difference
        direction = "outperformed" if diff > 0 else "underperformed"

        # Format p-value
        if pw.p_value < 0.001:
            p_str = "p < 0.001"
        elif pw.p_value < 0.01:
            p_str = f"p = {pw.p_value:.3f}"
        else:
            p_str = f"p = {pw.p_value:.3f}"

        sentence = (
            f"The {a} condition {direction} {b} by {abs(diff):.3f} "
            f"(95% CI [{pw.ci_lower:.3f}, {pw.ci_upper:.3f}], "
            f"{p_str}, d = {pw.effect_size:.2f})."
        )

        return sentence

    def generate_summary_paragraph(
        self,
        all_results: Dict[str, AblationSuiteResult],
    ) -> str:
        """Generate a cross-suite summary paragraph."""
        lines = []
        lines.append(
            f"Across {len(all_results)} ablation suites, "
            "we identified the contribution of each component."
        )

        for suite_name, result in all_results.items():
            best = result.best_condition()
            best_score = result.get_mean_score(best)
            conditions = result.get_all_condition_names()
            worst = min(conditions, key=lambda c: result.get_mean_score(c))
            worst_score = result.get_mean_score(worst)
            gap = best_score - worst_score

            lines.append(
                f"In {suite_name}, the best-to-worst gap was {gap:.3f} "
                f"({best}: {best_score:.3f} vs {worst}: {worst_score:.3f})."
            )

        return " ".join(lines)
