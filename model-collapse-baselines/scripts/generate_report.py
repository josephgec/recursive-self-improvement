#!/usr/bin/env python3
"""Generate a Markdown report from experiment results.

Usage::

    python scripts/generate_report.py --experiment-dir data/experiments
    python scripts/generate_report.py --experiment-dir data/experiments --output report/report.md
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Generate a Markdown report from experiment results.")

logger = logging.getLogger(__name__)


@app.command()
def main(
    experiment_dir: Path = typer.Option(
        ..., "--experiment-dir", "-e",
        help="Root directory of the experiment.",
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output Markdown file. Defaults to <experiment-dir>/report.md.",
    ),
) -> None:
    """Build the final experiment report."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_path = output or (experiment_dir / "report.md")

    from src.analysis.report import generate_report

    report_path = generate_report(experiment_dir, output_path)
    typer.echo(f"Report generated: {report_path}")


if __name__ == "__main__":
    app()
