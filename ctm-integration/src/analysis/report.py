"""Report generation: comprehensive markdown report with all analyses."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from src.bdm.calibration import BDMCalibrator, CalibrationReport
from src.bdm.scorer import BDMScorer
from src.library.evolution import LibraryEvolver, LibraryMetrics
from src.library.store import RuleStore
from src.analysis.escape_analysis import CollapseEscapeAnalyzer, CollapseEscapeResult


def generate_report(
    calibration_report: Optional[CalibrationReport] = None,
    library_metrics: Optional[LibraryMetrics] = None,
    synthesis_trajectory: Optional[List[Dict[str, Any]]] = None,
    escape_results: Optional[List[CollapseEscapeResult]] = None,
    output_path: Optional[str] = None,
) -> str:
    """Generate a comprehensive markdown report.

    Args:
        calibration_report: BDM calibration results.
        library_metrics: Current library quality metrics.
        synthesis_trajectory: History of synthesis iterations.
        escape_results: Collapse/escape analysis results.
        output_path: Path to save the report.

    Returns:
        Markdown report string.
    """
    sections = []

    sections.append("# CTM Integration Report\n")

    # --- BDM Calibration ---
    sections.append("## 1. BDM Calibration\n")
    if calibration_report:
        sections.append(f"**Tests run**: {calibration_report.num_tests}\n")
        sections.append(f"**Ordering correct**: {calibration_report.ordering_correct}\n")
        sections.append(f"\n{calibration_report.summary}\n")

        sections.append("\n| Test Case | BDM Score | Category |")
        sections.append("|-----------|-----------|----------|")
        for result in calibration_report.results:
            sections.append(
                f"| {result.name} | {result.bdm_score:.2f} | {result.expected_ordering} |"
            )
        sections.append("")
    else:
        sections.append("*No calibration data available.*\n")

    # --- Synthesis Trajectory ---
    sections.append("## 2. Synthesis Trajectory\n")
    if synthesis_trajectory:
        sections.append("| Iteration | Candidates | Best Accuracy | Pareto Front |")
        sections.append("|-----------|------------|---------------|--------------|")
        for entry in synthesis_trajectory:
            sections.append(
                f"| {entry.get('iteration', '?')} "
                f"| {entry.get('candidates', '?')} "
                f"| {entry.get('best_accuracy', 0):.2%} "
                f"| {entry.get('pareto_size', '?')} |"
            )
        sections.append("")
    else:
        sections.append("*No synthesis data available.*\n")

    # --- Library Quality ---
    sections.append("## 3. Library Quality\n")
    if library_metrics:
        sections.append(f"- **Total rules**: {library_metrics.total_rules}")
        sections.append(f"- **Unique domains**: {library_metrics.unique_domains}")
        sections.append(f"- **Average accuracy**: {library_metrics.avg_accuracy:.2%}")
        sections.append(f"- **Average BDM score**: {library_metrics.avg_bdm_score:.2f}")
        sections.append(f"- **Average MDL score**: {library_metrics.avg_mdl_score:.2f}")
        sections.append(f"- **Coverage**: {library_metrics.coverage:.2%}")
        sections.append(f"- **Quality score**: {library_metrics.quality_score:.4f}")
        sections.append("")
    else:
        sections.append("*No library metrics available.*\n")

    # --- Collapse/Escape Analysis ---
    sections.append("## 4. Collapse/Escape Analysis\n")
    if escape_results:
        sections.append("| Metric | Status | Trend |")
        sections.append("|--------|--------|-------|")
        for result in escape_results:
            sections.append(
                f"| {result.metric_name} | {result.status} | {result.trend:.4f} |"
            )
        sections.append("")

        collapsing = any(r.is_collapsing for r in escape_results)
        escaping = any(r.is_escaping for r in escape_results)

        if collapsing:
            sections.append(
                "**Warning**: Some metrics show collapse patterns. "
                "Consider increasing diversity pressure.\n"
            )
        elif escaping:
            sections.append(
                "**Good**: Metrics show escape/improvement patterns. "
                "The library is growing in quality.\n"
            )
        else:
            sections.append("**Stable**: Metrics are relatively stable.\n")
    else:
        sections.append("*No escape analysis data available.*\n")

    # --- Summary ---
    sections.append("## 5. Summary\n")
    sections.append(
        "This report summarizes the CTM integration pipeline: "
        "BDM calibration validates the complexity measure, "
        "synthesis trajectory shows rule generation progress, "
        "library quality tracks the verified rule collection, "
        "and collapse/escape analysis monitors long-term health.\n"
    )

    report = "\n".join(sections)

    if output_path:
        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )
        with open(output_path, "w") as f:
            f.write(report)

    return report
