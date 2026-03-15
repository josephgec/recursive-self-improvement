"""Automated report generator for model-collapse experiments.

Reads metrics and config from an experiment directory, generates all
plots, and produces a Markdown report summarising findings.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def generate_report(
    experiment_dir: str | Path,
    output_path: str | Path,
) -> Path:
    """Generate a comprehensive Markdown report for an experiment.

    The report includes:
    - Configuration summary
    - Per-schedule collapse curves
    - Scale comparison (if both 1B and 7B data exist)
    - Phase diagrams
    - Fixed-point analysis
    - Collapse rate table
    - Recommendations

    Args:
        experiment_dir: Root directory of the experiment (contains
            ``metrics/``, ``checkpoints/``, and optionally ``config.json``).
        output_path: Path for the output Markdown file.

    Returns:
        Path to the generated report.
    """
    experiment_dir = Path(experiment_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Gather data
    # ------------------------------------------------------------------

    config = _load_config(experiment_dir)
    schedule_metrics = _load_all_schedule_metrics(experiment_dir)
    plots_dir = output_path.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Generate plots
    # ------------------------------------------------------------------

    plot_paths: dict[str, list[Path]] = {}

    for sched_name, df in schedule_metrics.items():
        from src.analysis.collapse_curves import plot_all_curves

        sched_plot_dir = plots_dir / sched_name
        paths = plot_all_curves(df, sched_plot_dir)
        plot_paths[sched_name] = paths

    # Phase diagrams.
    phase_paths: list[Path] = []
    if len(schedule_metrics) > 1:
        from src.analysis.phase_diagrams import (
            plot_collapse_boundary,
            plot_phase_diagram,
        )

        for metric in ["kl_divergence", "entropy"]:
            p = plot_phase_diagram(
                schedule_metrics, metric,
                plots_dir / f"phase_{metric}.png",
            )
            phase_paths.append(p)

        p = plot_collapse_boundary(
            schedule_metrics, collapse_threshold=0.5,
            output_path=plots_dir / "collapse_boundary.png",
        )
        phase_paths.append(p)

    # Scale comparison.
    scale_paths: list[Path] = []
    metrics_1b, metrics_7b = _find_scale_data(experiment_dir)
    if metrics_1b is not None and metrics_7b is not None:
        from src.analysis.scale_comparison import (
            compute_scale_interaction_stats,
            plot_scale_comparison_panel,
        )

        p = plot_scale_comparison_panel(
            metrics_1b, metrics_7b,
            plots_dir / "scale_comparison_panel.png",
        )
        scale_paths.append(p)
        scale_stats = compute_scale_interaction_stats(metrics_1b, metrics_7b)
    else:
        scale_stats = None

    # ------------------------------------------------------------------
    # Build Markdown
    # ------------------------------------------------------------------

    lines: list[str] = []
    lines.append("# Model Collapse Experiment Report")
    lines.append("")

    # Config summary.
    lines.append("## Configuration")
    lines.append("")
    if config:
        lines.append("```json")
        lines.append(json.dumps(config, indent=2, default=str))
        lines.append("```")
    else:
        lines.append("_No configuration file found._")
    lines.append("")

    # Per-schedule curves.
    lines.append("## Per-Schedule Collapse Curves")
    lines.append("")
    for sched_name in sorted(schedule_metrics.keys()):
        lines.append(f"### {sched_name}")
        lines.append("")
        sched_plot_dir = plots_dir / sched_name
        for img in sorted(sched_plot_dir.glob("*.png")):
            rel = img.relative_to(output_path.parent)
            lines.append(f"![{img.stem}]({rel})")
            lines.append("")

    # Scale comparison.
    if scale_paths:
        lines.append("## Scale Comparison (1B vs 7B)")
        lines.append("")
        for img in scale_paths:
            rel = img.relative_to(output_path.parent)
            lines.append(f"![scale comparison]({rel})")
            lines.append("")
        if scale_stats:
            lines.append("### Scale Interaction Statistics")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(scale_stats, indent=2, default=str))
            lines.append("```")
            lines.append("")

    # Phase diagrams.
    if phase_paths:
        lines.append("## Phase Diagrams")
        lines.append("")
        for img in phase_paths:
            rel = img.relative_to(output_path.parent)
            lines.append(f"![phase diagram]({rel})")
            lines.append("")

    # Fixed-point analysis.
    lines.append("## Fixed-Point Analysis")
    lines.append("")
    fp_table = _build_fixed_point_table(schedule_metrics)
    lines.extend(fp_table)
    lines.append("")

    # Collapse rate table.
    lines.append("## Collapse Rate Table")
    lines.append("")
    rate_table = _build_collapse_rate_table(schedule_metrics)
    lines.extend(rate_table)
    lines.append("")

    # Recommendations.
    lines.append("## Recommendations")
    lines.append("")
    recs = _generate_recommendations(schedule_metrics, scale_stats)
    for rec in recs:
        lines.append(f"- {rec}")
    lines.append("")

    report_text = "\n".join(lines)
    output_path.write_text(report_text, encoding="utf-8")
    logger.info("Report generated at %s", output_path)
    return output_path


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _load_config(experiment_dir: Path) -> dict[str, Any] | None:
    """Load experiment config from experiment_dir."""
    for name in ("config.json", "config.yaml", "config.yml"):
        cfg_path = experiment_dir / name
        if cfg_path.exists():
            if name.endswith(".json"):
                with open(cfg_path) as f:
                    return json.load(f)
            else:
                try:
                    import yaml
                    with open(cfg_path) as f:
                        return yaml.safe_load(f)
                except ImportError:
                    pass
    return None


def _load_all_schedule_metrics(
    experiment_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Load metrics for every schedule sub-directory.

    Looks for ``<experiment_dir>/<schedule>/metrics/metrics.json``
    or ``<experiment_dir>/metrics/metrics.json`` (single-schedule).
    """
    results: dict[str, pd.DataFrame] = {}

    # Pattern 1: per-schedule sub-dirs.
    for subdir in sorted(experiment_dir.iterdir()):
        if not subdir.is_dir():
            continue
        metrics_file = subdir / "metrics" / "metrics.json"
        if not metrics_file.exists():
            metrics_file = subdir / "metrics.json"
        if metrics_file.exists():
            df = _load_metrics_json(metrics_file)
            if df is not None and len(df) > 0:
                results[subdir.name] = df

    # Pattern 2: single-schedule experiment.
    if not results:
        for candidate in [
            experiment_dir / "metrics" / "metrics.json",
            experiment_dir / "metrics.json",
        ]:
            if candidate.exists():
                df = _load_metrics_json(candidate)
                if df is not None and len(df) > 0:
                    results["default"] = df
                    break

    return results


def _load_metrics_json(path: Path) -> pd.DataFrame | None:
    """Load a metrics.json file into a DataFrame."""
    try:
        with open(path) as f:
            records = json.load(f)
        if not records:
            return None
        return pd.DataFrame(records)
    except Exception as e:
        logger.warning("Failed to load %s: %s", path, e)
        return None


def _find_scale_data(
    experiment_dir: Path,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Look for 1b and 7b sub-directories with metrics."""
    metrics_1b = None
    metrics_7b = None

    for name in ("1b", "1B", "1b_baseline"):
        p = experiment_dir / name / "metrics" / "metrics.json"
        if p.exists():
            metrics_1b = _load_metrics_json(p)
            break

    for name in ("7b", "7B", "7b_baseline"):
        p = experiment_dir / name / "metrics" / "metrics.json"
        if p.exists():
            metrics_7b = _load_metrics_json(p)
            break

    return metrics_1b, metrics_7b


def _build_fixed_point_table(
    schedule_metrics: dict[str, pd.DataFrame],
) -> list[str]:
    """Build a Markdown table of fixed-point detection results."""
    lines = [
        "| Schedule | Converged | Generation | Final KL | Final Entropy |",
        "|----------|-----------|------------|----------|---------------|",
    ]
    for sched, df in sorted(schedule_metrics.items()):
        kl_vals = df.get("kl_divergence", pd.Series(dtype=float))
        ent_vals = df.get("entropy", pd.Series(dtype=float))

        # Simple convergence test: KL delta < 0.01 for last 2 generations.
        converged = False
        conv_gen = "N/A"
        if len(kl_vals) >= 2:
            for i in range(1, len(kl_vals)):
                if abs(kl_vals.iloc[i] - kl_vals.iloc[i - 1]) < 0.01:
                    converged = True
                    conv_gen = str(i)
                    break

        final_kl = f"{kl_vals.iloc[-1]:.4f}" if len(kl_vals) > 0 else "N/A"
        final_ent = f"{ent_vals.iloc[-1]:.4f}" if len(ent_vals) > 0 else "N/A"

        lines.append(
            f"| {sched} | {'Yes' if converged else 'No'} | {conv_gen} "
            f"| {final_kl} | {final_ent} |"
        )
    return lines


def _build_collapse_rate_table(
    schedule_metrics: dict[str, pd.DataFrame],
) -> list[str]:
    """Build a Markdown table of collapse rates (KL growth per generation)."""
    lines = [
        "| Schedule | Gens | Avg KL/Gen | Max KL/Gen | Total KL |",
        "|----------|------|------------|------------|----------|",
    ]
    import numpy as np

    for sched, df in sorted(schedule_metrics.items()):
        kl_vals = df.get("kl_divergence", pd.Series(dtype=float)).values
        n_gens = len(kl_vals)
        if n_gens < 2:
            lines.append(f"| {sched} | {n_gens} | N/A | N/A | N/A |")
            continue
        deltas = np.diff(kl_vals)
        avg_delta = float(np.mean(deltas))
        max_delta = float(np.max(deltas))
        total_kl = float(kl_vals[-1] - kl_vals[0])
        lines.append(
            f"| {sched} | {n_gens} | {avg_delta:.4f} | {max_delta:.4f} "
            f"| {total_kl:.4f} |"
        )
    return lines


def _generate_recommendations(
    schedule_metrics: dict[str, pd.DataFrame],
    scale_stats: dict[str, Any] | None,
) -> list[str]:
    """Generate text recommendations based on the experimental results."""
    recs: list[str] = []

    if not schedule_metrics:
        recs.append("No metrics data available. Run the experiment first.")
        return recs

    # Check which schedules show collapse.
    for sched, df in schedule_metrics.items():
        kl_vals = df.get("kl_divergence", pd.Series(dtype=float)).values
        if len(kl_vals) >= 2 and kl_vals[-1] > 1.0:
            recs.append(
                f"Schedule '{sched}' shows significant collapse "
                f"(KL={kl_vals[-1]:.3f}). Consider increasing alpha (real data fraction)."
            )

    if scale_stats:
        ratio = scale_stats.get("collapse_rate_ratio", 1.0)
        if ratio > 1.5:
            recs.append(
                f"The 1B model collapses {ratio:.1f}x faster than the 7B model. "
                f"Larger models are more resilient to collapse."
            )
        elif ratio < 0.67:
            recs.append(
                f"The 7B model collapses faster (ratio={ratio:.2f}). "
                f"Check if LoRA fidelity is sufficient."
            )

    if not recs:
        recs.append("All schedules appear stable. No immediate action needed.")

    return recs
