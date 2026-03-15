#!/usr/bin/env python3
"""Run a single M_0 -> M_n lineage experiment.

Usage::

    python scripts/run_lineage.py --config configs/debug.yaml --schedule zero_alpha
    python scripts/run_lineage.py --config configs/default.yaml --schedule linear_decay --output-dir data/linear
    python scripts/run_lineage.py --config configs/debug.yaml --resume
    python scripts/run_lineage.py --config configs/debug.yaml --dry-run
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Train a single M_0 -> M_n model-collapse lineage.")

logger = logging.getLogger(__name__)


def _load_config(config_path: Path) -> dict:
    """Load and merge YAML config files."""
    import yaml

    # Load default config first, then overlay.
    project_root = Path(__file__).resolve().parent.parent
    default_path = project_root / "configs" / "default.yaml"

    cfg = {}
    if default_path.exists() and config_path.resolve() != default_path.resolve():
        with open(default_path) as f:
            cfg = yaml.safe_load(f) or {}

    with open(config_path) as f:
        overlay = yaml.safe_load(f) or {}

    _deep_merge(cfg, overlay)
    return cfg


def _deep_merge(base: dict, overlay: dict) -> None:
    """Recursively merge *overlay* into *base* in place."""
    for key, value in overlay.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _apply_schedule(config: dict, schedule_name: str) -> None:
    """Load a schedule YAML and merge it into *config*."""
    import yaml

    project_root = Path(__file__).resolve().parent.parent
    schedule_path = project_root / "configs" / "schedules" / f"{schedule_name}.yaml"
    if not schedule_path.exists():
        typer.echo(f"Schedule file not found: {schedule_path}", err=True)
        raise typer.Exit(1)

    with open(schedule_path) as f:
        sched_cfg = yaml.safe_load(f) or {}

    _deep_merge(config, sched_cfg)


@app.command()
def main(
    config: Path = typer.Option(
        "configs/default.yaml", "--config", "-c",
        help="Path to the experiment config YAML.",
    ),
    schedule: Optional[str] = typer.Option(
        None, "--schedule", "-s",
        help="Schedule name (e.g. zero_alpha, linear_decay). "
             "Overrides the schedule section in the config.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o",
        help="Override the output directory.",
    ),
    resume: bool = typer.Option(
        False, "--resume", "-r",
        help="Resume from the latest checkpoint.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Print the resolved config and exit without training.",
    ),
) -> None:
    """Train a single lineage from M_0 to M_n."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = _load_config(config)

    if schedule:
        _apply_schedule(cfg, schedule)

    if output_dir:
        cfg.setdefault("experiment", {})["output_dir"] = str(output_dir)

    if dry_run:
        typer.echo("Resolved configuration:")
        typer.echo(json.dumps(cfg, indent=2, default=str))
        raise typer.Exit(0)

    # Save resolved config for reproducibility.
    out = Path(cfg["experiment"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    from src.training.lineage import LineageOrchestrator

    orchestrator = LineageOrchestrator(cfg)

    if resume:
        orchestrator.resume()
    else:
        orchestrator.run()


if __name__ == "__main__":
    app()
