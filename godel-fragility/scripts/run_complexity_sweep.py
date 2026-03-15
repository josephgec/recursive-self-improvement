#!/usr/bin/env python3
"""Run a complexity sweep to find the agent's complexity ceiling."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.complexity_ceiling import ComplexityCeilingAnalyzer
from src.harness.controlled_env import ControlledEnvironment
from src.measurement.complexity_curve import ComplexityCurveTracker, ComplexityDataPoint
from src.measurement.comprehension_probe import ComprehensionProbe
from src.utils.code_mutators import inflate_complexity, add_dead_code

app = typer.Typer(help="Run a complexity sweep to find the agent's complexity ceiling.")


@app.command()
def main(
    initial_complexity: int = typer.Option(30, "--initial-complexity", help="Starting complexity (AST nodes)"),
    max_complexity: int = typer.Option(500, "--max-complexity", help="Maximum complexity to test"),
    step_size: int = typer.Option(20, "--step-size", help="Complexity increment per step"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    output_dir: str = typer.Option("data", "--output-dir", help="Output directory"),
) -> None:
    """Sweep through complexity levels and measure performance."""
    from rich.console import Console
    from rich.progress import Progress

    console = Console()
    console.print("[bold]Complexity Ceiling Sweep[/bold]\n")

    env = ControlledEnvironment(seed=seed)
    tracker = ComplexityCurveTracker()
    probe = ComprehensionProbe(seed=seed)

    complexity = initial_complexity
    steps = list(range(initial_complexity, max_complexity + 1, step_size))

    with Progress() as progress:
        task = progress.add_task("Sweeping complexity...", total=len(steps))

        for complexity in steps:
            agent = env.create_fresh_agent()

            # Inflate the agent's code to target complexity
            code = agent.get_all_code()
            inflated = add_dead_code(code, lines_to_add=complexity // 5)
            inflated = inflate_complexity(inflated, factor=max(1, complexity // 50))

            # Run the agent at this complexity level
            accuracies = []
            for _ in range(5):
                acc = agent.run_iteration()
                accuracies.append(acc)

            avg_accuracy = sum(accuracies) / len(accuracies)

            # Probe comprehension
            comp_result = probe.probe(inflated, complexity=complexity)

            tracker.record(
                ComplexityDataPoint(
                    complexity=complexity,
                    accuracy=avg_accuracy,
                    comprehension_score=comp_result.overall_score,
                    iteration=complexity,
                )
            )

            progress.update(task, advance=1)

    # Analyze
    analyzer = ComplexityCeilingAnalyzer()
    complexities = [d.complexity for d in tracker.data]
    accuracies_list = [d.accuracy for d in tracker.data]
    comprehension = [d.comprehension_score for d in tracker.data if d.comprehension_score is not None]

    analysis = analyzer.analyze(complexities, accuracies_list, comprehension)

    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"  Complexity range: {analysis.complexity_range}")
    console.print(f"  Ceiling estimate: {analysis.ceiling_estimate}")
    console.print(f"  Decline pattern: {analysis.decline_characterization}")
    console.print(f"  Cliff detected: {analysis.cliff_detected}")
    if analysis.safe_operating_range:
        console.print(f"  Safe range: {analysis.safe_operating_range[0]:.0f} - {analysis.safe_operating_range[1]:.0f}")

    # Save results
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    result_path = os.path.join(output_dir, "results", "complexity_sweep.json")
    with open(result_path, "w") as f:
        json.dump(
            {
                "ceiling_estimate": analysis.ceiling_estimate,
                "cliff_detected": analysis.cliff_detected,
                "cliff_location": analysis.cliff_location,
                "decline_characterization": analysis.decline_characterization,
                "safe_operating_range": analysis.safe_operating_range,
                "data_points": [
                    {
                        "complexity": d.complexity,
                        "accuracy": d.accuracy,
                        "comprehension": d.comprehension_score,
                    }
                    for d in tracker.data
                ],
            },
            f,
            indent=2,
        )
    console.print(f"\nResults saved to {result_path}")


if __name__ == "__main__":
    app()
