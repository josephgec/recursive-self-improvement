"""Report generation for agent runs."""

from __future__ import annotations

from typing import Any

from src.analysis.modification_history import ModificationHistoryAnalyzer


def generate_report(
    audit_entries: list[dict[str, Any]],
    output_path: str | None = None,
) -> str:
    """Generate a markdown report from audit log entries.

    Covers:
    - Performance trajectory
    - Modification timeline
    - Complexity analysis
    - Summary statistics
    """
    analyzer = ModificationHistoryAnalyzer(audit_entries)
    convergence = analyzer.convergence_analysis()
    success_rates = analyzer.success_rate_by_component()

    iterations = [e for e in audit_entries if e.get("type") == "iteration"]
    modifications = [e for e in audit_entries if e.get("type") == "modification"]
    rollbacks = [e for e in audit_entries if e.get("type") == "rollback"]

    lines: list[str] = []
    lines.append("# Godel Agent Run Report")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append(f"- Total iterations: {convergence['total_iterations']}")
    lines.append(f"- Total modifications: {convergence['total_modifications']}")
    lines.append(f"- Total rollbacks: {convergence['total_rollbacks']}")
    lines.append(f"- Acceptance rate: {convergence['acceptance_rate']:.1%}")
    lines.append(f"- Final accuracy: {convergence['final_accuracy']:.3f}")
    lines.append(f"- Best accuracy: {convergence['best_accuracy']:.3f}")
    lines.append("")

    # Performance trajectory
    lines.append("## Performance Trajectory")
    accuracies = convergence["accuracy_trajectory"]
    if accuracies:
        lines.append("")
        lines.append("| Iteration | Accuracy |")
        lines.append("|-----------|----------|")
        for i, acc in enumerate(accuracies):
            lines.append(f"| {i} | {acc:.3f} |")
    lines.append("")

    # Modification timeline
    lines.append("## Modification Timeline")
    if modifications:
        lines.append("")
        lines.append("| Iteration | Target | Accepted | Description |")
        lines.append("|-----------|--------|----------|-------------|")
        for mod in modifications:
            it = mod.get("iteration", "?")
            proposal = mod.get("proposal", {})
            target = proposal.get("target", "?")
            accepted = mod.get("accepted", False)
            desc = proposal.get("description", "")[:50]
            lines.append(f"| {it} | {target} | {accepted} | {desc} |")
    else:
        lines.append("No modifications applied.")
    lines.append("")

    # Success rates by component
    if success_rates:
        lines.append("## Success Rate by Component")
        lines.append("")
        for component, rate in success_rates.items():
            lines.append(f"- {component}: {rate:.1%}")
        lines.append("")

    # Rollbacks
    if rollbacks:
        lines.append("## Rollbacks")
        lines.append("")
        for rb in rollbacks:
            it = rb.get("iteration", "?")
            reason = rb.get("reason", "unknown")
            lines.append(f"- Iteration {it}: {reason}")
        lines.append("")

    report = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)

    return report
