"""Paper draft generator for publication preparation.

Generates results sections, tables, and figures from experiment data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class PaperDraftGenerator:
    """Generates paper draft components from experiment results.

    Produces results sections, tables, and figure descriptions
    in a format suitable for academic papers.
    """

    def __init__(self, paper_title: str = "Untitled Paper"):
        self.paper_title = paper_title
        self._generated_sections: List[str] = []

    def generate_results_section(
        self,
        experiments: List[Dict[str, Any]],
        baseline_name: str = "baseline",
    ) -> str:
        """Generate a results section from experiment data.

        Args:
            experiments: List of experiment result dicts.
            baseline_name: Name of the baseline method.

        Returns:
            Formatted results section text.
        """
        if not experiments:
            section = "\\section{Results}\n\nNo experimental results available.\n"
            self._generated_sections.append(section)
            return section

        lines = ["\\section{Results}\n"]

        # Summary paragraph
        best = max(experiments, key=lambda e: e.get("score", 0.0))
        worst = min(experiments, key=lambda e: e.get("score", 0.0))
        avg_score = sum(e.get("score", 0.0) for e in experiments) / len(experiments)

        lines.append(
            f"We evaluated {len(experiments)} configurations. "
            f"The best performing method ({best.get('name', 'unknown')}) "
            f"achieved a score of {best.get('score', 0.0):.3f}, "
            f"while the average across all methods was {avg_score:.3f}.\n"
        )

        # Per-experiment details
        for exp in experiments:
            name = exp.get("name", "unknown")
            score = exp.get("score", 0.0)
            details = exp.get("details", "")
            lines.append(f"\\textbf{{{name}}}: {score:.3f}. {details}\n")

        section = "\n".join(lines)
        self._generated_sections.append(section)
        return section

    def generate_tables(
        self,
        data: List[Dict[str, Any]],
        caption: str = "Results comparison",
        columns: Optional[List[str]] = None,
    ) -> str:
        """Generate a LaTeX table from data.

        Args:
            data: List of row dicts.
            caption: Table caption.
            columns: Column names (auto-detected if not provided).

        Returns:
            LaTeX table string.
        """
        if not data:
            return "% No data for table\n"

        if columns is None:
            columns = list(data[0].keys())

        n_cols = len(columns)
        col_spec = "|".join(["c"] * n_cols)

        lines = [
            "\\begin{table}[h]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\begin{{tabular}}{{|{col_spec}|}}",
            "\\hline",
            " & ".join(f"\\textbf{{{c}}}" for c in columns) + " \\\\",
            "\\hline",
        ]

        for row in data:
            values = []
            for col in columns:
                val = row.get(col, "")
                if isinstance(val, float):
                    values.append(f"{val:.3f}")
                else:
                    values.append(str(val))
            lines.append(" & ".join(values) + " \\\\")

        lines.extend([
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
        ])

        table = "\n".join(lines)
        self._generated_sections.append(table)
        return table

    def generate_figures(
        self,
        figure_specs: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate LaTeX figure environments.

        Args:
            figure_specs: List of dicts with 'filename', 'caption', 'label'.

        Returns:
            List of LaTeX figure strings.
        """
        figures = []
        for spec in figure_specs:
            filename = spec.get("filename", "figure.png")
            caption = spec.get("caption", "Figure")
            label = spec.get("label", "fig:figure")
            width = spec.get("width", "0.8\\textwidth")

            fig = "\n".join([
                "\\begin{figure}[h]",
                "\\centering",
                f"\\includegraphics[width={width}]{{{filename}}}",
                f"\\caption{{{caption}}}",
                f"\\label{{{label}}}",
                "\\end{figure}",
            ])
            figures.append(fig)
            self._generated_sections.append(fig)

        return figures

    def get_all_sections(self) -> List[str]:
        """Return all generated sections."""
        return list(self._generated_sections)
