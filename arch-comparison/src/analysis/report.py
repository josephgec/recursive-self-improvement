"""Report generation: comprehensive markdown report from benchmark results."""

from __future__ import annotations

from typing import Dict, Optional

from src.analysis.head_to_head import HeadToHeadAnalyzer, HeadToHeadReport
from src.analysis.failure_modes import FailureModeAnalyzer
from src.analysis.cost_analysis import CostAnalyzer
from src.analysis.rsi_suitability import RSISuitabilityAssessor
from src.evaluation.benchmark_suite import BenchmarkResults


def generate_report(
    results: BenchmarkResults,
    title: str = "Architecture Comparison Report",
) -> str:
    """Generate a comprehensive markdown report from benchmark results.

    Args:
        results: BenchmarkResults from a full benchmark run.
        title: Report title.

    Returns:
        Markdown-formatted report string.
    """
    sections = []
    sections.append(f"# {title}\n")
    sections.append("## Executive Summary\n")

    # Head-to-head
    h2h_analyzer = HeadToHeadAnalyzer()
    h2h_report = h2h_analyzer.analyze(results)
    sections.append("### Head-to-Head Comparison\n")
    sections.append(h2h_analyzer.generate_winner_table(h2h_report))
    sections.append("")

    # Generalization
    sections.append("\n## Generalization Results\n")
    for system, gen in results.generalization.items():
        sections.append(f"### {system}\n")
        sections.append(f"- In-domain accuracy: {gen.in_domain_accuracy:.3f}")
        sections.append(f"- Out-of-domain accuracy: {gen.out_of_domain_accuracy:.3f}")
        sections.append(f"- Generalization gap: {gen.generalization_gap:.3f}")
        sections.append(f"- Transfer ratio: {gen.transfer_ratio:.3f}")
        sections.append("")

    # Interpretability
    sections.append("\n## Interpretability Results\n")
    for system, interp in results.interpretability.items():
        sections.append(f"### {system}\n")
        sections.append(f"- Step verifiability: {interp.step_verifiability:.3f}")
        sections.append(f"- Faithfulness: {interp.faithfulness:.3f}")
        sections.append(f"- Readability: {interp.readability:.3f}")
        sections.append(f"- Overall: {interp.overall_score:.3f}")
        sections.append("")

    # Robustness
    sections.append("\n## Robustness Results\n")
    for system, robust in results.robustness.items():
        sections.append(f"### {system}\n")
        sections.append(f"- Consistency: {robust.consistency:.3f}")
        sections.append(f"- Degradation: {robust.degradation:.3f}")
        sections.append(f"- Original accuracy: {robust.original_accuracy:.3f}")
        sections.append(f"- Perturbed accuracy: {robust.perturbed_accuracy:.3f}")
        sections.append("")

    # Cost analysis
    cost_analyzer = CostAnalyzer()
    cost_report = cost_analyzer.analyze(results)
    sections.append("\n## Cost Analysis\n")
    for system, costs in cost_report.system_costs.items():
        sections.append(f"### {system}\n")
        for metric, value in costs.items():
            sections.append(f"- {metric}: {value:.4f}")
        sections.append("")
    sections.append(f"\nMost efficient system: **{cost_report.most_efficient}**\n")

    # RSI suitability
    rsi_assessor = RSISuitabilityAssessor()
    rsi_results = rsi_assessor.assess(results)
    sections.append("\n## RSI Suitability Assessment\n")
    for system, assessment in rsi_results.items():
        sections.append(f"### {system}\n")
        sections.append(f"- Overall RSI score: {assessment.overall_score:.3f}")
        sections.append(f"- Modularity: {assessment.modularity:.3f}")
        sections.append(f"- Verifiability: {assessment.verifiability:.3f}")
        sections.append(f"- Composability: {assessment.composability:.3f}")
        sections.append(f"- Contamination resistance: {assessment.contamination_resistance:.3f}")
        sections.append(f"- Transparency: {assessment.transparency:.3f}")
        sections.append(f"- Recommendation: {assessment.recommendation}")
        if assessment.strengths:
            sections.append(f"- Strengths: {', '.join(assessment.strengths)}")
        if assessment.weaknesses:
            sections.append(f"- Weaknesses: {', '.join(assessment.weaknesses)}")
        sections.append("")

    # Failure modes
    failure_analyzer = FailureModeAnalyzer()
    failure_reports = failure_analyzer.analyze(results)
    sections.append("\n## Failure Mode Analysis\n")
    for system, fm_report in failure_reports.items():
        sections.append(f"### {system}\n")
        sections.append(f"- Total failures: {fm_report.total_failures}")
        sections.append(f"- Failure rate: {fm_report.failure_rate:.3f}")
        for cat in fm_report.categories:
            if cat.count > 0:
                sections.append(f"  - {cat.name}: {cat.count} ({cat.description})")
        sections.append("")

    return "\n".join(sections)
