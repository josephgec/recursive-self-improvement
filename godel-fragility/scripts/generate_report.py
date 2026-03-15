#!/usr/bin/env python3
"""Generate a report from saved stress test results."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.complexity_ceiling import CeilingAnalysis, ComplexityCeilingAnalyzer
from src.analysis.failure_landscape import FailureLandscapeAnalyzer
from src.analysis.fragility_score import FragilityScorer
from src.analysis.report import generate_report
from src.harness.stress_runner import ScenarioResult, StressTestResults
from src.measurement.failure_classifier import FailureMode

app = typer.Typer(help="Generate a report from saved results.")


@app.command()
def main(
    results_path: str = typer.Option(
        "data/results/stress_test_results.json",
        "--results",
        "-r",
        help="Path to results JSON",
    ),
    output_path: str = typer.Option(
        "data/reports/fragility_report.md",
        "--output",
        "-o",
        help="Output report path",
    ),
) -> None:
    """Generate a markdown report from saved results."""
    from rich.console import Console

    console = Console()

    if not os.path.exists(results_path):
        console.print(f"[red]Results file not found: {results_path}[/red]")
        raise typer.Exit(1)

    with open(results_path) as f:
        data = json.load(f)

    # Reconstruct StressTestResults
    scenario_results = []
    for s in data.get("scenarios", []):
        scenario_results.append(
            ScenarioResult(
                scenario_name=s["name"],
                category=s["category"],
                severity=s["severity"],
                success=s["success"],
                failure_mode=FailureMode(s["failure_mode"]),
                accuracies=s.get("accuracies", []),
                iterations_run=s.get("iterations_run", 0),
                duration_seconds=s.get("duration_seconds", 0),
            )
        )

    results = StressTestResults(
        results=scenario_results,
        total_scenarios=data.get("total_scenarios", len(scenario_results)),
        passed=data.get("passed", sum(1 for r in scenario_results if r.success)),
        failed=data.get("failed", sum(1 for r in scenario_results if not r.success)),
        timed_out=data.get("timed_out", 0),
        duration_seconds=data.get("duration_seconds", 0),
    )

    # Analyze
    landscape_analyzer = FailureLandscapeAnalyzer()
    landscape = landscape_analyzer.compute_landscape(results)

    scorer = FragilityScorer()
    fragility_report = scorer.compute(results)

    # Generate report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report_text = generate_report(
        results=results,
        fragility_report=fragility_report,
        landscape=landscape,
        output_path=output_path,
    )

    console.print(f"[green]Report generated: {output_path}[/green]")
    console.print(f"Fragility score: {fragility_report.overall_score:.2f} (Grade: {fragility_report.grade})")


if __name__ == "__main__":
    app()
