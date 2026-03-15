#!/usr/bin/env python3
"""Full experiment orchestrator.

Runs all lineages in the experiment matrix (scales x schedules),
then generates scale comparisons and phase diagrams.

Usage::

    python scripts/run_full_experiment.py --scale 1b --schedules zero_alpha
    python scripts/run_full_experiment.py --scale both --schedules all
    python scripts/run_full_experiment.py --scale 1b --schedules all --config configs/debug.yaml --dry-run
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Run the full model-collapse experiment matrix.")

logger = logging.getLogger(__name__)

ALL_SCHEDULES = ["zero_alpha", "constant_alpha", "linear_decay", "exponential_decay"]
SCALE_CONFIGS = {
    "1b": "configs/1b_baseline.yaml",
    "7b": "configs/7b_baseline.yaml",
}


def _load_config(config_path: Path) -> dict:
    """Load a YAML config file with defaults."""
    import yaml

    project_root = Path(__file__).resolve().parent.parent
    default_path = project_root / "configs" / "default.yaml"

    cfg = {}
    if default_path.exists():
        with open(default_path) as f:
            cfg = yaml.safe_load(f) or {}

    if config_path.resolve() != default_path.resolve():
        with open(config_path) as f:
            overlay = yaml.safe_load(f) or {}
        _deep_merge(cfg, overlay)

    return cfg


def _deep_merge(base: dict, overlay: dict) -> None:
    for key, value in overlay.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _apply_schedule(config: dict, schedule_name: str) -> None:
    """Load a schedule YAML and merge it into config."""
    import yaml

    project_root = Path(__file__).resolve().parent.parent
    schedule_path = project_root / "configs" / "schedules" / f"{schedule_name}.yaml"
    if not schedule_path.exists():
        typer.echo(f"Schedule file not found: {schedule_path}", err=True)
        raise typer.Exit(1)

    with open(schedule_path) as f:
        sched_cfg = yaml.safe_load(f) or {}

    _deep_merge(config, sched_cfg)


def _apply_scale(config: dict, scale_config_path: str) -> None:
    """Load a scale-specific config overlay."""
    import yaml

    project_root = Path(__file__).resolve().parent.parent
    scale_path = project_root / scale_config_path
    if scale_path.exists():
        with open(scale_path) as f:
            overlay = yaml.safe_load(f) or {}
        _deep_merge(config, overlay)


@app.command()
def main(
    scale: str = typer.Option(
        "1b", "--scale", "-s",
        help="Model scale: '1b', '7b', or 'both'.",
    ),
    schedules: str = typer.Option(
        "all", "--schedules",
        help="Comma-separated schedule names or 'all'.",
    ),
    config: Path = typer.Option(
        "configs/default.yaml", "--config", "-c",
        help="Base config file.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o",
        help="Root output directory.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Print the experiment plan and exit.",
    ),
    resume: bool = typer.Option(
        False, "--resume",
        help="Resume interrupted lineages.",
    ),
) -> None:
    """Run the full experiment matrix."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Resolve scales.
    if scale == "both":
        scales = ["1b", "7b"]
    else:
        scales = [scale]

    # Resolve schedules.
    if schedules == "all":
        schedule_list = ALL_SCHEDULES
    else:
        schedule_list = [s.strip() for s in schedules.split(",")]

    root_output = output_dir or Path("data/experiments")

    # Build the experiment plan.
    plan: list[dict] = []
    for sc in scales:
        for sched in schedule_list:
            run_dir = root_output / sc / sched
            plan.append({
                "scale": sc,
                "schedule": sched,
                "output_dir": str(run_dir),
            })

    if dry_run:
        typer.echo(f"Experiment plan: {len(plan)} lineages")
        for entry in plan:
            typer.echo(f"  {entry['scale']} / {entry['schedule']} -> {entry['output_dir']}")
        raise typer.Exit(0)

    # Run each lineage.
    from src.training.lineage import LineageOrchestrator

    for entry in plan:
        typer.echo(f"\n{'='*60}")
        typer.echo(f"Running: {entry['scale']} / {entry['schedule']}")
        typer.echo(f"{'='*60}")

        cfg = _load_config(config)

        # Apply scale-specific overrides.
        if entry["scale"] in SCALE_CONFIGS:
            _apply_scale(cfg, SCALE_CONFIGS[entry["scale"]])

        # Apply schedule.
        _apply_schedule(cfg, entry["schedule"])

        # Set output dir.
        cfg.setdefault("experiment", {})["output_dir"] = entry["output_dir"]

        # Save resolved config.
        run_dir = Path(entry["output_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2, default=str)

        orchestrator = LineageOrchestrator(cfg)
        try:
            if resume:
                orchestrator.resume()
            else:
                orchestrator.run()
        except Exception as e:
            logger.error("Lineage %s/%s failed: %s", entry["scale"],
                         entry["schedule"], e)
            continue

    # Post-processing: generate scale comparison and phase diagrams.
    _generate_post_analysis(root_output, scales, schedule_list)


def _generate_post_analysis(
    root_output: Path,
    scales: list[str],
    schedule_list: list[str],
) -> None:
    """Generate scale comparison plots and phase diagrams after all lineages complete."""
    import pandas as pd

    from src.analysis.phase_diagrams import plot_collapse_boundary, plot_phase_diagram

    plots_dir = root_output / "analysis"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for sc in scales:
        # Collect all schedule metrics for this scale.
        all_runs: dict[str, pd.DataFrame] = {}
        for sched in schedule_list:
            metrics_path = root_output / sc / sched / "metrics" / "metrics.json"
            if metrics_path.exists():
                try:
                    with open(metrics_path) as f:
                        records = json.load(f)
                    if records:
                        all_runs[sched] = pd.DataFrame(records)
                except Exception:
                    continue

        if all_runs:
            for metric in ["kl_divergence", "entropy"]:
                plot_phase_diagram(
                    all_runs, metric,
                    plots_dir / f"phase_{sc}_{metric}.png",
                )
            plot_collapse_boundary(
                all_runs, collapse_threshold=0.5,
                output_path=plots_dir / f"collapse_boundary_{sc}.png",
            )

    # Scale comparison if both scales were run.
    if "1b" in scales and "7b" in scales:
        _generate_scale_comparison(root_output, schedule_list, plots_dir)


def _generate_scale_comparison(
    root_output: Path,
    schedule_list: list[str],
    plots_dir: Path,
) -> None:
    """Generate scale comparison plots for matching schedules."""
    import pandas as pd

    from src.analysis.scale_comparison import (
        compute_scale_interaction_stats,
        plot_scale_comparison_panel,
    )

    for sched in schedule_list:
        m1b_path = root_output / "1b" / sched / "metrics" / "metrics.json"
        m7b_path = root_output / "7b" / sched / "metrics" / "metrics.json"

        if m1b_path.exists() and m7b_path.exists():
            try:
                with open(m1b_path) as f:
                    m1b = pd.DataFrame(json.load(f))
                with open(m7b_path) as f:
                    m7b = pd.DataFrame(json.load(f))

                plot_scale_comparison_panel(
                    m1b, m7b,
                    plots_dir / f"scale_comparison_{sched}.png",
                )
                stats = compute_scale_interaction_stats(m1b, m7b)
                with open(plots_dir / f"scale_stats_{sched}.json", "w") as f:
                    json.dump(stats, f, indent=2, default=str)
            except Exception as e:
                logger.warning("Scale comparison for %s failed: %s", sched, e)


if __name__ == "__main__":
    app()
