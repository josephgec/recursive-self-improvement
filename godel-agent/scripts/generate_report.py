#!/usr/bin/env python3
"""Generate report from audit log."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from src.analysis.report import generate_report

app = typer.Typer(help="Generate analysis report.")


@app.command()
def main(
    log_dir: str = typer.Option("data/audit_logs/latest/", "--log-dir", "-l", help="Audit log directory"),
    output: str = typer.Option("data/reports/report.md", "--output", "-o", help="Output file path"),
) -> None:
    """Generate a markdown report from audit logs."""
    log_path = Path(log_dir) / "audit_log.jsonl"

    if not log_path.exists():
        typer.echo(f"No audit log found at {log_path}")
        raise typer.Exit(1)

    entries = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    typer.echo(f"Loaded {len(entries)} entries")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = generate_report(entries, str(output_path))
    typer.echo(f"Report saved to {output_path}")
    typer.echo(f"\nReport preview:\n{report[:500]}")


if __name__ == "__main__":
    app()
