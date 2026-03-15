#!/usr/bin/env python3
"""Run the Godel Agent."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
import yaml

from src.core.agent import GodelAgent
from src.tasks.loader import TaskSuiteLoader
from src.analysis.report import generate_report

app = typer.Typer(help="Run the Godel Agent self-improvement loop.")


def load_config(config_path: str) -> dict:
    """Load and merge config with defaults."""
    default_path = Path("configs/default.yaml")
    config = {}
    if default_path.exists():
        with open(default_path) as f:
            config = yaml.safe_load(f) or {}

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            override = yaml.safe_load(f) or {}
        # Deep merge
        for key, value in override.items():
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value

    return config


@app.command()
def main(
    config: str = typer.Option("configs/default.yaml", "--config", "-c", help="Config file path"),
    tasks: str = typer.Option("math", "--tasks", "-t", help="Task domain: math, code, all"),
    iterations: int = typer.Option(0, "--iterations", "-n", help="Override max iterations (0=use config)"),
    no_modification: bool = typer.Option(False, "--no-modification", help="Disable self-modification"),
    report: bool = typer.Option(False, "--report", "-r", help="Generate report after run"),
    resume: str = typer.Option("", "--resume", help="Resume from checkpoint path"),
) -> None:
    """Run the Godel Agent."""
    cfg = load_config(config)

    if iterations > 0:
        cfg.setdefault("meta_learning", {})["max_iterations"] = iterations

    # Load tasks
    task_list = TaskSuiteLoader.load(tasks)
    typer.echo(f"Loaded {len(task_list)} tasks from '{tasks}' domain")

    # Create agent
    agent = GodelAgent(cfg)

    # Resume if specified
    if resume:
        state = agent.state_manager.load_from_disk(resume)
        typer.echo(f"Resumed from checkpoint: {state.state_id}")

    # Run
    typer.echo(f"Running for {cfg.get('meta_learning', {}).get('max_iterations', 50)} iterations...")
    results = agent.run(
        task_list,
        allow_modification=not no_modification,
    )

    # Summary
    typer.echo("\n=== Run Complete ===")
    for r in results:
        status = ""
        if r.deliberated:
            status += " [deliberated]"
        if r.modification_applied:
            status += " [modified]"
        if r.modification_rolled_back:
            status += " [rolled back]"
        typer.echo(f"  Iteration {r.iteration}: accuracy={r.accuracy:.3f}{status}")

    # Report
    if report:
        entries = agent.audit.entries
        report_text = generate_report(entries)
        output_dir = Path(cfg.get("project", {}).get("output_dir", "data")) / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "run_report.md"
        with open(report_path, "w") as f:
            f.write(report_text)
        typer.echo(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    app()
