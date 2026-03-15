#!/usr/bin/env python3
"""Run the full adversarial stress test suite."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import typer
import yaml

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adversarial.adversarial_tasks import AdversarialTaskScenarios
from src.adversarial.boundary_pusher import ComplexityEscalation
from src.adversarial.circular_deps import CircularDependencyScenarios
from src.adversarial.rollback_corruptor import RollbackCorruptionScenarios
from src.adversarial.scenario_registry import ScenarioRegistry
from src.adversarial.self_reference import SelfReferenceAttacks
from src.analysis.complexity_ceiling import ComplexityCeilingAnalyzer
from src.analysis.failure_landscape import FailureLandscapeAnalyzer
from src.analysis.fragility_score import FragilityScorer
from src.analysis.report import generate_report
from src.harness.stress_runner import StressTestRunner
from src.measurement.complexity_curve import ComplexityCurveTracker, ComplexityDataPoint

app = typer.Typer(help="Run adversarial stress tests on the Godel Agent.")


def _register_all_scenarios(registry: ScenarioRegistry) -> None:
    """Register all built-in adversarial scenarios."""
    for scenario in SelfReferenceAttacks().__class__.__mro__[0].__dict__:
        pass  # Just checking the class is importable

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


def _load_config(config_path: str) -> dict:
    """Load YAML config file, with defaults."""
    defaults = {
        "project": {"seed": 42, "output_dir": "data"},
        "stress_test": {
            "scenarios": "all",
            "repetitions": 1,
            "timeout_per_scenario": 600,
        },
        "complexity_sweep": {
            "initial_nodes": 30,
            "max_nodes": 500,
            "step_size": 20,
            "modification_attempts": 10,
        },
        "analysis": {
            "compute_fragility_score": True,
            "generate_plots": False,
        },
    }

    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            user_config = yaml.safe_load(f) or {}
        # Merge
        for key, val in user_config.items():
            if isinstance(val, dict) and key in defaults:
                defaults[key].update(val)
            else:
                defaults[key] = val

    return defaults


@app.command()
def main(
    config: str = typer.Option("configs/default.yaml", "--config", "-c", help="Config file path"),
    scenarios: Optional[str] = typer.Option(None, "--scenarios", "-s", help="Comma-separated scenario names or 'all'"),
    repetitions: int = typer.Option(0, "--repetitions", "-r", help="Override repetitions (0 = use config)"),
    timeout: float = typer.Option(0, "--timeout", "-t", help="Override timeout per scenario (0 = use config)"),
    report: bool = typer.Option(False, "--report", help="Generate report after testing"),
    complexity_sweep: bool = typer.Option(False, "--complexity-sweep", help="Run complexity sweep"),
) -> None:
    """Run adversarial stress tests."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print("[bold]Godel Agent Fragility Testing[/bold]\n")

    # Load config
    cfg = _load_config(config)
    seed = cfg.get("project", {}).get("seed", 42)
    output_dir = cfg.get("project", {}).get("output_dir", "data")

    stress_cfg = cfg.get("stress_test", {})
    reps = repetitions if repetitions > 0 else stress_cfg.get("repetitions", 1)
    tout = timeout if timeout > 0 else stress_cfg.get("timeout_per_scenario", 600)

    # Register scenarios
    registry = ScenarioRegistry()
    _register_all_scenarios(registry)

    console.print(f"Registered {len(registry)} scenarios")

    # Filter scenarios if requested
    scenario_filter = scenarios or stress_cfg.get("scenarios", "all")
    if scenario_filter != "all":
        if isinstance(scenario_filter, str):
            scenario_names = [s.strip() for s in scenario_filter.split(",")]
        else:
            scenario_names = list(scenario_filter)
        filtered_registry = ScenarioRegistry()
        for name in scenario_names:
            if name in registry:
                filtered_registry.register(registry.get(name))
            else:
                console.print(f"[yellow]Warning: scenario '{name}' not found[/yellow]")
        registry = filtered_registry

    console.print(f"Running {len(registry)} scenarios x {reps} repetitions (timeout: {tout}s)\n")

    # Run stress tests
    runner = StressTestRunner(
        registry=registry,
        seed=seed,
        timeout_seconds=tout,
        repetitions=reps,
    )
    results = runner.run_all()

    # Display results table
    table = Table(title="Stress Test Results")
    table.add_column("Scenario", style="cyan")
    table.add_column("Category")
    table.add_column("Result", style="bold")
    table.add_column("Failure Mode")
    table.add_column("Iterations", justify="right")

    for r in results.results:
        status = "[green]PASS[/green]" if r.success else "[red]FAIL[/red]"
        table.add_row(
            r.scenario_name,
            r.category,
            status,
            r.failure_mode.value,
            str(r.iterations_run),
        )

    console.print(table)
    console.print(f"\nTotal: {results.total_scenarios} | Passed: {results.passed} | Failed: {results.failed}")
    console.print(f"Duration: {results.duration_seconds:.1f}s\n")

    # Complexity sweep
    ceiling_analysis = None
    if complexity_sweep:
        console.print("[bold]Running complexity sweep...[/bold]")
        sweep_cfg = cfg.get("complexity_sweep", {})
        curve_tracker = ComplexityCurveTracker()

        from src.harness.controlled_env import ControlledEnvironment, MockAgent

        env = ControlledEnvironment(seed=seed)
        complexity = sweep_cfg.get("initial_nodes", 30)
        max_c = sweep_cfg.get("max_nodes", 500)
        step = sweep_cfg.get("step_size", 20)

        while complexity <= max_c:
            agent = env.create_fresh_agent()
            # Simulate running at this complexity level
            for _ in range(sweep_cfg.get("modification_attempts", 10)):
                accuracy = agent.run_iteration()
                curve_tracker.record(
                    ComplexityDataPoint(
                        complexity=complexity,
                        accuracy=accuracy,
                        iteration=complexity,
                    )
                )
            complexity += step

        ceiling = curve_tracker.find_complexity_ceiling()
        analyzer = ComplexityCeilingAnalyzer()
        complexities = [d.complexity for d in curve_tracker.data]
        accuracies = [d.accuracy for d in curve_tracker.data]
        ceiling_analysis = analyzer.analyze(complexities, accuracies)
        console.print(f"Complexity ceiling estimate: {ceiling_analysis.ceiling_estimate}")

    # Generate report
    if report:
        console.print("[bold]Generating report...[/bold]")

        landscape_analyzer = FailureLandscapeAnalyzer()
        landscape = landscape_analyzer.compute_landscape(results)

        scorer = FragilityScorer()
        fragility_report = scorer.compute(
            results,
            recovery_tracker=runner.recovery_tracker,
            complexity_ceiling=ceiling_analysis.ceiling_estimate if ceiling_analysis else None,
        )

        os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)
        report_path = os.path.join(output_dir, "reports", "fragility_report.md")

        report_text = generate_report(
            results=results,
            recovery_tracker=runner.recovery_tracker,
            ceiling_analysis=ceiling_analysis,
            fragility_report=fragility_report,
            landscape=landscape,
            output_path=report_path,
        )
        console.print(f"Report saved to {report_path}")

    # Save raw results
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    results_path = os.path.join(output_dir, "results", "stress_test_results.json")
    results_data = {
        "total_scenarios": results.total_scenarios,
        "passed": results.passed,
        "failed": results.failed,
        "timed_out": results.timed_out,
        "duration_seconds": results.duration_seconds,
        "failure_mode_distribution": results.failure_mode_distribution,
        "scenarios": [
            {
                "name": r.scenario_name,
                "category": r.category,
                "severity": r.severity,
                "success": r.success,
                "failure_mode": r.failure_mode.value,
                "iterations_run": r.iterations_run,
                "duration_seconds": r.duration_seconds,
                "accuracies": r.accuracies,
            }
            for r in results.results
        ],
    }
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    console.print(f"Results saved to {results_path}")


if __name__ == "__main__":
    app()
