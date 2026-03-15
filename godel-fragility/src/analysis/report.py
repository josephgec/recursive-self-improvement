"""Generate comprehensive markdown reports from stress test results."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.analysis.complexity_ceiling import CeilingAnalysis, ComplexityCeilingAnalyzer
from src.analysis.failure_landscape import FailureLandscape, FailureLandscapeAnalyzer
from src.analysis.fragility_score import FragilityReport, FragilityScorer
from src.analysis.recovery_patterns import RecoveryPatternAnalyzer
from src.harness.stress_runner import StressTestResults
from src.measurement.recovery_tracker import RecoveryTracker


def generate_report(
    results: StressTestResults,
    recovery_tracker: Optional[RecoveryTracker] = None,
    ceiling_analysis: Optional[CeilingAnalysis] = None,
    fragility_report: Optional[FragilityReport] = None,
    landscape: Optional[FailureLandscape] = None,
    output_path: Optional[str] = None,
) -> str:
    """Generate a comprehensive markdown report.

    Args:
        results: Stress test results.
        recovery_tracker: Recovery event tracker.
        ceiling_analysis: Complexity ceiling analysis.
        fragility_report: Fragility score report.
        landscape: Failure landscape analysis.
        output_path: If provided, write report to this file.

    Returns:
        Markdown report string.
    """
    sections: List[str] = []

    # Title
    sections.append("# Godel Agent Fragility Report\n")
    sections.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Executive Summary
    sections.append(_executive_summary(results, fragility_report))

    # Stress Test Coverage
    sections.append(_stress_test_coverage(results))

    # Failure Landscape
    if landscape:
        sections.append(_failure_landscape_section(landscape))

    # Recovery Analysis
    if recovery_tracker:
        sections.append(_recovery_analysis_section(recovery_tracker))

    # Complexity Ceiling
    if ceiling_analysis:
        sections.append(_complexity_ceiling_section(ceiling_analysis))

    # Vulnerabilities
    if landscape and landscape.vulnerabilities:
        sections.append(_vulnerabilities_section(landscape))

    # Fragility Score
    if fragility_report:
        sections.append(_fragility_score_section(fragility_report))

    # Detailed Results
    sections.append(_detailed_results(results))

    report = "\n".join(sections)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)

    return report


def _executive_summary(
    results: StressTestResults,
    fragility_report: Optional[FragilityReport],
) -> str:
    """Generate executive summary section."""
    lines = ["## Executive Summary\n"]

    lines.append(f"- **Scenarios tested:** {results.total_scenarios}")
    lines.append(f"- **Passed:** {results.passed} ({results.pass_rate:.0%})")
    lines.append(f"- **Failed:** {results.failed}")
    lines.append(f"- **Timed out:** {results.timed_out}")
    lines.append(f"- **Duration:** {results.duration_seconds:.1f}s")

    if fragility_report:
        lines.append(f"- **Fragility score:** {fragility_report.overall_score:.2f} (Grade: {fragility_report.grade})")
        lines.append(f"\n> {fragility_report.interpretation}")

    lines.append("")
    return "\n".join(lines)


def _stress_test_coverage(results: StressTestResults) -> str:
    """Generate stress test coverage section."""
    lines = ["## Stress Test Coverage\n"]

    # Category breakdown
    lines.append("### By Category\n")
    lines.append("| Category | Total | Passed | Failed | Pass Rate |")
    lines.append("|----------|-------|--------|--------|-----------|")

    cat_results = results.category_results
    for cat, stats in sorted(cat_results.items()):
        rate = stats["passed"] / stats["total"] if stats["total"] > 0 else 0
        lines.append(
            f"| {cat} | {stats['total']} | {stats['passed']} | "
            f"{stats['failed']} | {rate:.0%} |"
        )

    # Failure mode distribution
    lines.append("\n### Failure Mode Distribution\n")
    lines.append("| Mode | Count |")
    lines.append("|------|-------|")
    for mode, count in sorted(
        results.failure_mode_distribution.items(), key=lambda x: -x[1]
    ):
        lines.append(f"| {mode} | {count} |")

    lines.append("")
    return "\n".join(lines)


def _failure_landscape_section(landscape: FailureLandscape) -> str:
    """Generate failure landscape section."""
    lines = ["## Failure Landscape\n"]

    lines.append(f"- Total failures: {landscape.total_failures} / {landscape.total_scenarios}")

    if landscape.category_failure_rates:
        lines.append("\n### Category Failure Rates\n")
        for cat, rate in sorted(
            landscape.category_failure_rates.items(), key=lambda x: -x[1]
        ):
            bar = "#" * int(rate * 20)
            lines.append(f"- **{cat}**: {rate:.0%} {bar}")

    if landscape.severity_distribution:
        lines.append("\n### Severity Distribution\n")
        for sev in sorted(landscape.severity_distribution.keys()):
            count = landscape.severity_distribution[sev]
            lines.append(f"- Severity {sev}: {count} failure(s)")

    lines.append("")
    return "\n".join(lines)


def _recovery_analysis_section(tracker: RecoveryTracker) -> str:
    """Generate recovery analysis section."""
    lines = ["## Recovery Analysis\n"]

    lines.append(f"- **Recovery rate:** {tracker.get_recovery_rate():.0%}")
    lines.append(f"- **Detection rate:** {tracker.get_detection_rate():.0%}")

    mttd = tracker.get_mean_time_to_detect()
    mttr = tracker.get_mean_time_to_recover()
    if mttd is not None:
        lines.append(f"- **Mean time to detect:** {mttd:.1f} iterations")
    if mttr is not None:
        lines.append(f"- **Mean time to recover:** {mttr:.1f} iterations")

    by_type = tracker.recovery_by_fault_type()
    if by_type:
        lines.append("\n### Recovery by Fault Type\n")
        lines.append("| Fault Type | Recovery Rate |")
        lines.append("|------------|---------------|")
        for ft, rate in sorted(by_type.items(), key=lambda x: -x[1]):
            lines.append(f"| {ft} | {rate:.0%} |")

    lines.append("")
    return "\n".join(lines)


def _complexity_ceiling_section(analysis: CeilingAnalysis) -> str:
    """Generate complexity ceiling section."""
    lines = ["## Complexity Ceiling\n"]

    if analysis.ceiling_estimate is not None:
        lines.append(f"- **Estimated ceiling:** {analysis.ceiling_estimate:.0f} AST nodes")
    else:
        lines.append("- **Estimated ceiling:** Not determined (insufficient data)")

    lines.append(f"- **Decline pattern:** {analysis.decline_characterization}")
    lines.append(f"- **Cliff detected:** {'Yes' if analysis.cliff_detected else 'No'}")

    if analysis.cliff_location is not None:
        lines.append(f"- **Cliff location:** {analysis.cliff_location:.0f} AST nodes")

    if analysis.safe_operating_range:
        lo, hi = analysis.safe_operating_range
        lines.append(f"- **Safe operating range:** {lo:.0f} - {hi:.0f} AST nodes")

    if analysis.sigmoid_fit:
        sf = analysis.sigmoid_fit
        lines.append(f"\n### Sigmoid Fit")
        lines.append(f"- R-squared: {sf.r_squared:.3f}")
        lines.append(f"- Midpoint (ceiling): {sf.x0:.0f}")
        lines.append(f"- Steepness: {sf.k:.4f}")

    lines.append("")
    return "\n".join(lines)


def _vulnerabilities_section(landscape: FailureLandscape) -> str:
    """Generate vulnerabilities section."""
    lines = ["## Critical Vulnerabilities\n"]

    for i, vuln in enumerate(landscape.vulnerabilities, 1):
        severity_icon = {"1": "Low", "2": "Medium", "3": "High", "4": "Critical", "5": "Catastrophic"}.get(
            str(vuln.severity), "Unknown"
        )
        lines.append(f"### {i}. {vuln.name} (Severity: {severity_icon})\n")
        lines.append(f"- **Category:** {vuln.category}")
        lines.append(f"- **Failure mode:** {vuln.failure_mode.value}")
        lines.append(f"- **Description:** {vuln.description}")
        lines.append(f"- **Affected scenarios:** {', '.join(vuln.affected_scenarios)}")
        if vuln.remediation:
            lines.append(f"- **Remediation:** {vuln.remediation}")
        lines.append("")

    return "\n".join(lines)


def _fragility_score_section(report: FragilityReport) -> str:
    """Generate fragility score section."""
    lines = ["## Fragility Score\n"]

    lines.append(f"**Overall: {report.overall_score:.2f} / 1.00 (Grade: {report.grade})**\n")

    lines.append("### Component Breakdown\n")
    lines.append("| Component | Score | Interpretation |")
    lines.append("|-----------|-------|----------------|")
    for comp, score in sorted(report.components.items()):
        interp = "Good" if score < 0.3 else "Moderate" if score < 0.6 else "Poor"
        bar = "#" * int(score * 10)
        lines.append(f"| {comp} | {score:.2f} {bar} | {interp} |")

    if report.recommendations:
        lines.append("\n### Recommendations\n")
        for rec in report.recommendations:
            lines.append(f"- {rec}")

    lines.append("")
    return "\n".join(lines)


def _detailed_results(results: StressTestResults) -> str:
    """Generate detailed per-scenario results."""
    lines = ["## Detailed Scenario Results\n"]

    lines.append("| Scenario | Category | Severity | Pass | Failure Mode | Iterations | Duration |")
    lines.append("|----------|----------|----------|------|--------------|------------|----------|")

    for r in sorted(results.results, key=lambda x: (x.category, x.scenario_name)):
        status = "PASS" if r.success else "FAIL"
        lines.append(
            f"| {r.scenario_name} | {r.category} | {r.severity} | "
            f"{status} | {r.failure_mode.value} | {r.iterations_run} | "
            f"{r.duration_seconds:.1f}s |"
        )

    lines.append("")
    return "\n".join(lines)
