#!/usr/bin/env python3
"""Run a single adversarial scenario."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adversarial.adversarial_tasks import AdversarialTaskScenarios
from src.adversarial.boundary_pusher import ComplexityEscalation
from src.adversarial.circular_deps import CircularDependencyScenarios
from src.adversarial.rollback_corruptor import RollbackCorruptionScenarios
from src.adversarial.scenario_registry import ScenarioRegistry
from src.adversarial.self_reference import SelfReferenceAttacks
from src.harness.stress_runner import StressTestRunner

app = typer.Typer(help="Run a single adversarial scenario.")


def _register_all(registry: ScenarioRegistry) -> None:
    providers = [
        SelfReferenceAttacks(),
        ComplexityEscalation(),
        RollbackCorruptionScenarios(),
        CircularDependencyScenarios(),
        AdversarialTaskScenarios(),
    ]
    for provider in providers:
        for attr_name in dir(provider):
            if attr_name.startswith("scenario_"):
                method = getattr(provider, attr_name)
                scenario = method()
                registry.register(scenario)


@app.command()
def main(
    scenario: str = typer.Option(..., "--scenario", "-s", help="Scenario name to run"),
    repetitions: int = typer.Option(1, "--repetitions", "-r", help="Number of repetitions"),
    timeout: float = typer.Option(60.0, "--timeout", "-t", help="Timeout per run"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    output_dir: str = typer.Option("data", "--output-dir", help="Output directory"),
) -> None:
    """Run a single adversarial scenario and display results."""
    from rich.console import Console

    console = Console()
    console.print(f"[bold]Running scenario: {scenario}[/bold]\n")

    registry = ScenarioRegistry()
    _register_all(registry)

    if scenario not in registry:
        console.print(f"[red]Scenario '{scenario}' not found.[/red]")
        console.print(f"Available: {[s.name for s in registry.get_all()]}")
        raise typer.Exit(1)

    runner = StressTestRunner(
        registry=registry,
        seed=seed,
        timeout_seconds=timeout,
        repetitions=repetitions,
    )

    result = runner.run_scenario(scenario)

    console.print(f"  Success: {'[green]Yes[/green]' if result.success else '[red]No[/red]'}")
    console.print(f"  Failure mode: {result.failure_mode.value}")
    console.print(f"  Iterations: {result.iterations_run}")
    console.print(f"  Duration: {result.duration_seconds:.2f}s")
    console.print(f"  Modifications: {result.modification_count}")

    if result.accuracies:
        console.print(f"  Final accuracy: {result.accuracies[-1]:.3f}")
        console.print(f"  Accuracy range: [{min(result.accuracies):.3f}, {max(result.accuracies):.3f}]")

    if result.error:
        console.print(f"  [red]Error: {result.error}[/red]")

    # Save result
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    result_path = os.path.join(output_dir, "results", f"scenario_{scenario}.json")
    with open(result_path, "w") as f:
        json.dump(
            {
                "scenario": result.scenario_name,
                "category": result.category,
                "success": result.success,
                "failure_mode": result.failure_mode.value,
                "iterations_run": result.iterations_run,
                "duration_seconds": result.duration_seconds,
                "accuracies": result.accuracies,
                "error": result.error,
            },
            f,
            indent=2,
        )
    console.print(f"\nResult saved to {result_path}")


if __name__ == "__main__":
    app()
