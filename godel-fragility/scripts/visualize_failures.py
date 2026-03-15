#!/usr/bin/env python3
"""Generate failure visualization plots from saved results."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.failure_landscape import FailureLandscapeAnalyzer
from src.analysis.recovery_patterns import RecoveryPatternAnalyzer
from src.harness.stress_runner import ScenarioResult, StressTestResults
from src.measurement.failure_classifier import FailureMode
from src.measurement.recovery_tracker import RecoveryEvent

app = typer.Typer(help="Generate failure visualization plots.")


@app.command()
def main(
    results_path: str = typer.Option(
        "data/results/stress_test_results.json",
        "--results",
        "-r",
        help="Path to results JSON",
    ),
    output_dir: str = typer.Option(
        "data/reports/plots",
        "--output-dir",
        "-o",
        help="Output directory for plots",
    ),
) -> None:
    """Generate failure visualization plots."""
    from rich.console import Console

    console = Console()

    if not os.path.exists(results_path):
        console.print(f"[red]Results file not found: {results_path}[/red]")
        raise typer.Exit(1)

    with open(results_path) as f:
        data = json.load(f)

    # Reconstruct results
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
        passed=data.get("passed", 0),
        failed=data.get("failed", 0),
    )

    os.makedirs(output_dir, exist_ok=True)

    # Failure heatmap
    landscape_analyzer = FailureLandscapeAnalyzer()
    landscape = landscape_analyzer.compute_landscape(results)

    fig = landscape_analyzer.plot_failure_heatmap(
        landscape,
        output_path=os.path.join(output_dir, "failure_heatmap.png"),
    )
    if fig:
        console.print("[green]Generated failure_heatmap.png[/green]")
    else:
        console.print("[yellow]Could not generate heatmap (matplotlib may not be available)[/yellow]")

    # Recovery trajectories (from scenario accuracies as proxy)
    recovery_analyzer = RecoveryPatternAnalyzer()
    events = []
    for s in data.get("scenarios", []):
        if not s["success"] and s.get("accuracies"):
            event = RecoveryEvent(
                scenario_name=s["name"],
                fault_type=s["category"],
                iteration_injected=0,
                accuracies=s["accuracies"],
            )
            events.append(event)

    if events:
        fig2 = recovery_analyzer.plot_recovery_trajectories(
            events,
            output_path=os.path.join(output_dir, "recovery_trajectories.png"),
        )
        if fig2:
            console.print("[green]Generated recovery_trajectories.png[/green]")

    console.print(f"\nPlots saved to {output_dir}")


if __name__ == "__main__":
    app()
