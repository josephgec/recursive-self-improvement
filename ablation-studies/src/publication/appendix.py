"""Appendix generation for extended results."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from src.suites.base import AblationSuiteResult
from src.analysis.statistical_tests import PairwiseResult, PublicationStatistics
from src.publication.latex_tables import LaTeXTableGenerator


class AppendixGenerator:
    """Generate appendix content with detailed per-category tables."""

    def __init__(self):
        self.table_gen = LaTeXTableGenerator()
        self.stats = PublicationStatistics()

    def generate_per_category_tables(
        self,
        result: AblationSuiteResult,
        categories: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """Generate detailed tables for each category/benchmark.

        If no categories specified, generates one table per condition
        showing per-run details.
        """
        tables = {}

        if categories:
            for cat in categories:
                table = self._category_table(result, cat)
                tables[cat] = table
        else:
            # Generate per-condition detail tables
            for cond in result.get_all_condition_names():
                table = self.table_gen.condition_detail_table(result, cond)
                tables[cond] = table

        return tables

    def generate_extended_comparisons(
        self,
        result: AblationSuiteResult,
    ) -> str:
        """Generate extended pairwise comparisons (all pairs)."""
        conditions = result.get_all_condition_names()
        comparisons = []

        for i in range(len(conditions)):
            for j in range(i + 1, len(conditions)):
                a_name = conditions[i]
                b_name = conditions[j]
                a_scores = result.get_scores(a_name)
                b_scores = result.get_scores(b_name)
                if a_scores and b_scores:
                    pw = self.stats.pairwise_comparison(
                        a_scores, b_scores, a_name, b_name
                    )
                    comparisons.append(pw)

        if not comparisons:
            return ""

        return self.table_gen.pairwise_comparison_table(
            comparisons,
            caption=f"Extended pairwise comparisons for {result.suite_name}",
            label=f"tab:extended_{result.suite_name.replace(' ', '_').lower()}",
        )

    def generate_full_appendix(
        self,
        result: AblationSuiteResult,
        categories: Optional[List[str]] = None,
    ) -> str:
        """Generate the complete appendix section."""
        sections = []

        sections.append(f"\\section{{Extended Results: {result.suite_name}}}")
        sections.append("")

        # Per-category tables
        tables = self.generate_per_category_tables(result, categories)
        for name, table in tables.items():
            sections.append(f"\\subsection{{{name}}}")
            sections.append(table)
            sections.append("")

        # Extended comparisons
        ext_table = self.generate_extended_comparisons(result)
        if ext_table:
            sections.append("\\subsection{All Pairwise Comparisons}")
            sections.append(ext_table)

        return "\n".join(sections)

    def _category_table(
        self,
        result: AblationSuiteResult,
        category: str,
    ) -> str:
        """Generate a table for a specific category."""
        conditions = result.get_all_condition_names()
        best_name = result.best_condition()

        lines = []
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append(f"\\caption{{Results for {category}}}")
        lines.append("\\begin{tabular}{lc}")
        lines.append("\\toprule")
        lines.append("Condition & Accuracy \\\\")
        lines.append("\\midrule")

        for cond in conditions:
            mean_score = result.get_mean_score(cond)
            display = cond.replace("_", "\\_")
            if cond == best_name:
                lines.append(f"\\textbf{{{display}}} & \\textbf{{{mean_score:.3f}}} \\\\")
            else:
                lines.append(f"{display} & {mean_score:.3f} \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

        return "\n".join(lines)
