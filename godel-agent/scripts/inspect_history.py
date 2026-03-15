#!/usr/bin/env python3
"""Browse and inspect audit log history."""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(help="Inspect agent audit history.")


@app.command()
def main(
    log_dir: str = typer.Argument("data/audit_logs/latest/", help="Audit log directory"),
    event_type: str = typer.Option("all", "--type", "-t", help="Filter by event type"),
    limit: int = typer.Option(50, "--limit", "-n", help="Max entries to show"),
) -> None:
    """Browse audit log entries."""
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

    if event_type != "all":
        entries = [e for e in entries if e.get("type") == event_type]

    typer.echo(f"Found {len(entries)} entries")
    typer.echo("")

    for entry in entries[:limit]:
        etype = entry.get("type", "unknown")
        iteration = entry.get("iteration", "?")

        if etype == "iteration":
            acc = entry.get("accuracy", 0)
            correct = entry.get("correct_tasks", 0)
            total = entry.get("total_tasks", 0)
            typer.echo(f"[iter {iteration}] accuracy={acc:.3f} ({correct}/{total})")

        elif etype == "deliberation":
            data = entry.get("data", {})
            proceed = data.get("should_proceed", False)
            typer.echo(f"[iter {iteration}] deliberation: proceed={proceed}")

        elif etype == "modification":
            proposal = entry.get("proposal", {})
            target = proposal.get("target", "?")
            accepted = entry.get("accepted", False)
            typer.echo(f"[iter {iteration}] modification: {target} accepted={accepted}")

        elif etype == "rollback":
            reason = entry.get("reason", "unknown")
            typer.echo(f"[iter {iteration}] ROLLBACK: {reason}")

        else:
            typer.echo(f"[iter {iteration}] {etype}: {json.dumps(entry)[:100]}")


if __name__ == "__main__":
    app()
