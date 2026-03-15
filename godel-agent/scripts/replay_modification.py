#!/usr/bin/env python3
"""Replay a specific modification step from the audit log."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from src.modification.modifier import CodeModifier, ModificationProposal
from src.meta.registry import ComponentRegistry
from src.meta.prompt_strategy import DefaultPromptStrategy
from src.modification.diff_engine import DiffEngine
from src.audit.diff_formatter import DiffFormatter

app = typer.Typer(help="Replay a modification step.")


@app.command()
def main(
    log_dir: str = typer.Option("data/audit_logs/latest/", "--log-dir", "-l", help="Audit log directory"),
    index: int = typer.Option(0, "--index", "-i", help="Modification index to replay"),
) -> None:
    """Replay a specific modification from the audit log."""
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

    modifications = [e for e in entries if e.get("type") == "modification"]

    if not modifications:
        typer.echo("No modifications found in log")
        raise typer.Exit(1)

    if index >= len(modifications):
        typer.echo(f"Index {index} out of range. Found {len(modifications)} modifications.")
        raise typer.Exit(1)

    mod_entry = modifications[index]
    proposal_data = mod_entry.get("proposal", {})
    accepted = mod_entry.get("accepted", False)
    iteration = mod_entry.get("iteration", "?")

    typer.echo(f"=== Replay Modification {index} (iteration {iteration}) ===")
    typer.echo(f"Target: {proposal_data.get('target', '?')}")
    typer.echo(f"Description: {proposal_data.get('description', '?')}")
    typer.echo(f"Risk: {proposal_data.get('risk', '?')}")
    typer.echo(f"Originally accepted: {accepted}")
    typer.echo(f"\nProposed code:\n{proposal_data.get('code', 'N/A')}")

    # Validate
    proposal = ModificationProposal.from_dict(proposal_data)
    modifier = CodeModifier()
    validation = modifier.validate_proposal(proposal)
    typer.echo(f"\nValidation: {validation}")

    # Try to apply to a fresh component
    registry = ComponentRegistry()
    strategy = DefaultPromptStrategy()
    registry.register("prompt_strategy", strategy)

    if proposal.target in registry:
        typer.echo("\nAttempting to apply modification...")
        result = modifier.apply_modification(proposal, registry)
        typer.echo(f"Result: success={result.success}, error={result.error}")

        if result.success and result.old_code and result.new_code:
            diff_engine = DiffEngine()
            diff = diff_engine.compute_diff(result.old_code, result.new_code)
            formatter = DiffFormatter()
            typer.echo(f"\nDiff:\n{formatter.format_unified(diff)}")

            # Revert
            modifier.revert(result)
            typer.echo("Reverted modification.")
    else:
        typer.echo(f"\nTarget '{proposal.target}' not available for replay.")


if __name__ == "__main__":
    app()
