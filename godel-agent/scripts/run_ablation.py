#!/usr/bin/env python3
"""Run ablation study: static vs self-modifying vs no-rollback."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import typer
import yaml

from src.core.agent import GodelAgent
from src.tasks.loader import TaskSuiteLoader

app = typer.Typer(help="Run ablation comparison.")


def load_config() -> dict:
    """Load default config."""
    with open("configs/default.yaml") as f:
        return yaml.safe_load(f) or {}


@app.command()
def main(
    tasks: str = typer.Option("math", "--tasks", "-t", help="Task domain"),
    iterations: int = typer.Option(10, "--iterations", "-n", help="Iterations per condition"),
) -> None:
    """Run static vs self-modifying vs no-rollback comparison."""
    base_config = load_config()
    base_config["agent"]["llm_provider"] = "mock"
    base_config["meta_learning"]["max_iterations"] = iterations

    task_list = TaskSuiteLoader.load(tasks)
    typer.echo(f"Loaded {len(task_list)} tasks")

    results_summary: dict[str, dict] = {}

    # Condition 1: Static (no modification)
    typer.echo("\n--- Condition 1: Static (no modification) ---")
    cfg = copy.deepcopy(base_config)
    agent = GodelAgent(cfg)
    results = agent.run(task_list, allow_modification=False)
    accs = [r.accuracy for r in results]
    results_summary["static"] = {
        "accuracies": accs,
        "mean": sum(accs) / len(accs) if accs else 0,
        "modifications": 0,
    }
    typer.echo(f"  Mean accuracy: {results_summary['static']['mean']:.3f}")

    # Condition 2: Self-modifying (with rollback)
    typer.echo("\n--- Condition 2: Self-modifying (with rollback) ---")
    cfg = copy.deepcopy(base_config)
    agent = GodelAgent(cfg)
    results = agent.run(task_list, allow_modification=True)
    accs = [r.accuracy for r in results]
    mods = sum(1 for r in results if r.modification_applied)
    results_summary["self_modifying"] = {
        "accuracies": accs,
        "mean": sum(accs) / len(accs) if accs else 0,
        "modifications": mods,
    }
    typer.echo(f"  Mean accuracy: {results_summary['self_modifying']['mean']:.3f}")
    typer.echo(f"  Modifications: {mods}")

    # Condition 3: No rollback
    typer.echo("\n--- Condition 3: Self-modifying (no rollback) ---")
    cfg = copy.deepcopy(base_config)
    cfg["validation"]["auto_rollback"] = False
    agent = GodelAgent(cfg)
    results = agent.run(task_list, allow_modification=True)
    accs = [r.accuracy for r in results]
    mods = sum(1 for r in results if r.modification_applied)
    results_summary["no_rollback"] = {
        "accuracies": accs,
        "mean": sum(accs) / len(accs) if accs else 0,
        "modifications": mods,
    }
    typer.echo(f"  Mean accuracy: {results_summary['no_rollback']['mean']:.3f}")
    typer.echo(f"  Modifications: {mods}")

    # Summary
    typer.echo("\n=== Ablation Summary ===")
    for cond, data in results_summary.items():
        typer.echo(f"  {cond}: mean_acc={data['mean']:.3f}, mods={data['modifications']}")

    # Save results
    output_dir = Path("data/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(results_summary, f, indent=2, default=str)
    typer.echo(f"\nResults saved to {output_dir / 'ablation_results.json'}")


if __name__ == "__main__":
    app()
